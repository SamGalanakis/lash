import type {
  Thread,
  ThreadItem,
  TokenUsageInfo,
  Turn,
  TurnInputItem,
  ModelInfo,
} from "./types";

type NotificationHandler = (method: string, params: unknown) => void;

export class LashClient {
  private ws: WebSocket | null = null;
  private requestId = 0;
  private pendingRequests = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();
  private notificationHandler: NotificationHandler | null = null;
  private onConnectionChange: ((connected: boolean) => void) | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private intentionallyClosed = false;
  private ready: Promise<void> = Promise.resolve();
  private resolveReady: (() => void) | null = null;
  private rejectReady: ((e: Error) => void) | null = null;

  connect(
    onNotification: NotificationHandler,
    onConnectionChange?: (connected: boolean) => void,
  ): void {
    this.notificationHandler = onNotification;
    this.onConnectionChange = onConnectionChange ?? null;
    this.intentionallyClosed = false;
    this._connect();
  }

  private _connect(): void {
    this.rejectReady?.(new Error("WebSocket reconnecting"));

    this.ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/api/ws`;
    const ws = new WebSocket(url);
    this.ws = ws;

    const isStale = () => this.ws !== ws;

    ws.onopen = async () => {
      if (isStale()) return;
      console.log("[lash-client] connected");
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
      try {
        await this.rawRequest("initialize", {
          clientInfo: { name: "lash-web", version: "0.1.0" },
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "";
        if (!msg.includes("Already initialized")) {
          console.error("[lash-client] initialize failed", e);
          return;
        }
      }
      this.resolveReady?.();
      this.resolveReady = null;
      this.rejectReady = null;
      this.onConnectionChange?.(true);
    };

    ws.onmessage = (event) => {
      if (isStale()) return;
      try {
        console.log(`[ws:recv] ${event.data}`);
        const msg = JSON.parse(event.data as string);

        if (msg.id !== undefined && msg.id !== null) {
          const pending = this.pendingRequests.get(msg.id);
          if (pending) {
            this.pendingRequests.delete(msg.id);
            if (msg.error) {
              pending.reject(
                new Error(msg.error.message || JSON.stringify(msg.error)),
              );
            } else {
              pending.resolve(msg.result);
            }
          }
          return;
        }

        if (msg.method && this.notificationHandler) {
          this.notificationHandler(msg.method, msg.params);
        }
      } catch {
        console.warn("[lash-client] failed to parse message");
      }
    };

    ws.onclose = () => {
      if (isStale()) return;
      this.onConnectionChange?.(false);
      this.rejectPending("WebSocket disconnected");
      if (!this.intentionallyClosed) {
        console.log("[lash-client] disconnected, reconnecting in 2s...");
        this.reconnectTimer = setTimeout(() => this._connect(), 2000);
      }
    };

    ws.onerror = () => {
      if (isStale()) return;
      ws.close();
    };
  }

  disconnect(): void {
    this.intentionallyClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    this.rejectReady?.(new Error("Disconnected"));
    this.rejectPending("Disconnected");
    this.ws?.close();
    this.ws = null;
  }

  private rejectPending(reason: string): void {
    for (const [, { reject }] of this.pendingRequests) {
      reject(new Error(reason));
    }
    this.pendingRequests.clear();
  }

  /** Send a request without waiting for the initialize handshake. */
  private rawRequest<T>(method: string, params?: unknown): Promise<T> {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        reject(new Error("WebSocket not connected"));
        return;
      }
      const id = ++this.requestId;
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request timed out: ${method}`));
      }, 30_000);
      this.pendingRequests.set(id, {
        resolve: (v) => { clearTimeout(timer); (resolve as (v: unknown) => void)(v); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });
      const payload = JSON.stringify({ method, id, params });
      console.log(`[ws:send] ${payload}`);
      this.ws.send(payload);
    });
  }

  private async request<T>(method: string, params?: unknown): Promise<T> {
    const maxAttempts = 3;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        await this.ready;
        return await this.rawRequest<T>(method, params);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "";
        const isRetryable =
          msg === "WebSocket reconnecting" || msg === "WebSocket disconnected";
        if (isRetryable && attempt < maxAttempts - 1) {
          continue;
        }
        throw e;
      }
    }
    throw new Error("WebSocket not connected");
  }

  async threadStart(params?: {
    model?: string;
    cwd?: string;
  }): Promise<{ thread: Thread }> {
    return this.request("thread/start", params);
  }

  async threadResume(threadId: string): Promise<{ thread: Thread }> {
    return this.request("thread/resume", { threadId });
  }

  async threadList(params?: {
    limit?: number;
    archived?: boolean;
  }): Promise<{ data: Thread[]; nextCursor?: string }> {
    return this.request("thread/list", params);
  }

  async threadRead(
    threadId: string,
    includeTurns = false,
  ): Promise<{ thread: Thread }> {
    return this.request("thread/read", { threadId, includeTurns });
  }

  async threadArchive(threadId: string): Promise<void> {
    await this.request("thread/archive", { threadId });
  }

  async turnStart(params: {
    threadId: string;
    input: TurnInputItem[];
    model?: string;
    mode?: string;
  }): Promise<{ turn: Turn }> {
    return this.request("turn/start", params);
  }

  async turnInterrupt(threadId: string, turnId: string): Promise<void> {
    await this.request("turn/interrupt", { threadId, turnId });
  }

  async modelList(): Promise<{ data: ModelInfo[] }> {
    return this.request("model/list", {});
  }

  async skillsList(): Promise<{
    data: { name: string; description: string; enabled: boolean }[];
  }> {
    return this.request("skills/list", {});
  }
}

export const lashClient = new LashClient();

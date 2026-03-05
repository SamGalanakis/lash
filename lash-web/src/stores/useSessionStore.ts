import { create } from "zustand";
import type {
  Thread,
  ThreadItem,
  TokenUsageInfo,
  Turn,
  TurnInputItem,
  AgentMessageItem,
} from "@/lib/types";
import { lashClient } from "@/lib/lash-client";

function upsertThreadAtTop(threads: Thread[], thread: Thread): Thread[] {
  return [thread, ...threads.filter((t) => t.id !== thread.id)];
}

function dedupeThreads(threads: Thread[]): Thread[] {
  const seen = new Set<string>();
  return threads.filter((thread) => {
    if (seen.has(thread.id)) return false;
    seen.add(thread.id);
    return true;
  });
}

interface ActiveTurn {
  turnId: string;
  threadId: string;
  items: ThreadItem[];
  streamingText: string;
  streamingItemId: string | null;
  toolOutputLines: string[];
  activeDelegate: { name: string; task: string } | null;
}

interface SessionState {
  threads: Thread[];
  currentThreadId: string | null;
  activeTurn: ActiveTurn | null;
  tokenUsage: TokenUsageInfo | null;
  connected: boolean;
  loading: boolean;
  error: string | null;
  threadMessages: Record<string, ThreadItem[]>;

  setConnected: (connected: boolean) => void;
  loadThreads: () => Promise<void>;
  startThread: () => Promise<string>;
  selectThread: (threadId: string) => void;
  loadThreadHistory: (threadId: string) => Promise<void>;
  sendMessage: (text: string) => Promise<void>;
  interruptTurn: () => Promise<void>;
  archiveThread: (threadId: string) => Promise<void>;

  handleNotification: (method: string, params: unknown) => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  threads: [],
  currentThreadId: null,
  activeTurn: null,
  tokenUsage: null,
  connected: false,
  loading: false,
  error: null,
  threadMessages: {},

  setConnected: (connected) => set({ connected }),

  loadThreads: async () => {
    set({ loading: true, error: null });
    try {
      const { data } = await lashClient.threadList({ limit: 50 });
      set({ threads: dedupeThreads(data), loading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : String(err),
        loading: false,
      });
    }
  },

  startThread: async () => {
    if (!get().connected) {
      const msg = "Not connected to server";
      set({ error: msg });
      throw new Error(msg);
    }
    try {
      const { thread } = await lashClient.threadStart();
      set((s) => ({
        threads: upsertThreadAtTop(s.threads, thread),
        currentThreadId: thread.id,
        activeTurn: null,
        tokenUsage: null,
        error: null,
      }));
      return thread.id;
    } catch (err) {
      set({ error: err instanceof Error ? err.message : String(err) });
      throw err;
    }
  },

  selectThread: (threadId) => {
    set({ currentThreadId: threadId, activeTurn: null, tokenUsage: null });
    get().loadThreadHistory(threadId);
  },

  loadThreadHistory: async (threadId) => {
    const cached = get().threadMessages[threadId];
    if (cached && cached.length > 0) return;
    try {
      const { thread } = await lashClient.threadRead(threadId, true);
      const items: ThreadItem[] = [];
      if (thread.turns) {
        for (const turn of thread.turns) {
          items.push(...turn.items);
        }
      }
      set((s) => ({
        threadMessages: { ...s.threadMessages, [threadId]: items },
      }));
    } catch {
      set((s) => ({
        threadMessages: { ...s.threadMessages, [threadId]: [] },
      }));
    }
  },

  sendMessage: async (text) => {
    const { currentThreadId, activeTurn, connected } = get();
    if (!currentThreadId || activeTurn || !connected) return;

    const input: TurnInputItem[] = [{ type: "text", text }];

    const userItem: ThreadItem = {
      type: "userMessage",
      id: `local-${Date.now()}`,
      content: input,
    };

    set({
      error: null,
      activeTurn: {
        turnId: "",
        threadId: currentThreadId,
        items: [userItem],
        streamingText: "",
        streamingItemId: null,
        toolOutputLines: [],
        activeDelegate: null,
      },
    });

    try {
      const { turn } = await lashClient.turnStart({
        threadId: currentThreadId,
        input,
      });

      set((s) => {
        if (!s.activeTurn) return s;
        return {
          activeTurn: { ...s.activeTurn, turnId: turn.id },
        };
      });
    } catch (err) {
      set({ activeTurn: null, error: err instanceof Error ? err.message : String(err) });
    }
  },

  interruptTurn: async () => {
    const { activeTurn } = get();
    if (!activeTurn) return;
    await lashClient.turnInterrupt(activeTurn.threadId, activeTurn.turnId);
  },

  archiveThread: async (threadId) => {
    await lashClient.threadArchive(threadId);
    set((s) => ({
      threads: s.threads.filter((t) => t.id !== threadId),
      currentThreadId:
        s.currentThreadId === threadId ? null : s.currentThreadId,
    }));
  },

  handleNotification: (method, params) => {
    const p = params as Record<string, unknown>;

    switch (method) {
      case "thread/started": {
        const thread = p.thread as Thread;
        set((s) => ({
          threads: upsertThreadAtTop(s.threads, thread),
        }));
        break;
      }

      case "thread/status/changed": {
        const threadId = p.threadId as string;
        const status = p.status as Thread["status"];
        set((s) => ({
          threads: s.threads.map((t) =>
            t.id === threadId ? { ...t, status } : t,
          ),
        }));
        break;
      }

      case "item/started": {
        const item = p.item as ThreadItem;
        set((s) => {
          if (!s.activeTurn) return s;
          if (item.type === "agentMessage") {
            return {
              activeTurn: {
                ...s.activeTurn,
                streamingItemId: item.id,
                streamingText: "",
              },
            };
          }
          const clearDelegate =
            item.type === "toolCall" && item.name.startsWith("delegate_");
          return {
            activeTurn: {
              ...s.activeTurn,
              items: [...s.activeTurn.items, item],
              activeDelegate: clearDelegate ? null : s.activeTurn.activeDelegate,
            },
          };
        });
        break;
      }

      case "item/agentMessage/delta": {
        const delta = p.delta as string;
        set((s) => {
          if (!s.activeTurn) return s;
          return {
            activeTurn: {
              ...s.activeTurn,
              streamingText: s.activeTurn.streamingText + delta,
            },
          };
        });
        break;
      }

      case "turn/toolOutput/delta": {
        const threadId = p.threadId as string | undefined;
        const turnId = p.turnId as string | undefined;
        const delta = p.delta as string;
        set((s) => {
          if (!s.activeTurn) return s;
          if (
            (threadId && s.activeTurn.threadId !== threadId) ||
            (turnId && s.activeTurn.turnId && s.activeTurn.turnId !== turnId)
          ) {
            return s;
          }
          return {
            activeTurn: {
              ...s.activeTurn,
              toolOutputLines: [...s.activeTurn.toolOutputLines, delta],
            },
          };
        });
        break;
      }

      case "turn/toolOutput/cleared": {
        const threadId = p.threadId as string | undefined;
        const turnId = p.turnId as string | undefined;
        set((s) => {
          if (!s.activeTurn) return s;
          if (
            (threadId && s.activeTurn.threadId !== threadId) ||
            (turnId && s.activeTurn.turnId && s.activeTurn.turnId !== turnId)
          ) {
            return s;
          }
          return {
            activeTurn: {
              ...s.activeTurn,
              toolOutputLines: [],
            },
          };
        });
        break;
      }

      case "turn/delegate/started": {
        const threadId = p.threadId as string | undefined;
        const turnId = p.turnId as string | undefined;
        const name = (p.name as string) || "delegate";
        const task = (p.task as string) || "";
        set((s) => {
          if (!s.activeTurn) return s;
          if (
            (threadId && s.activeTurn.threadId !== threadId) ||
            (turnId && s.activeTurn.turnId && s.activeTurn.turnId !== turnId)
          ) {
            return s;
          }
          return {
            activeTurn: {
              ...s.activeTurn,
              activeDelegate: { name, task },
            },
          };
        });
        break;
      }

      case "item/completed": {
        const item = p.item as ThreadItem;
        set((s) => {
          if (!s.activeTurn) return s;
          if (
            item.type === "agentMessage" &&
            s.activeTurn.streamingItemId === item.id
          ) {
            return {
              activeTurn: {
                ...s.activeTurn,
                items: [...s.activeTurn.items, item],
                streamingText: "",
                streamingItemId: null,
              },
            };
          }
          const existing = s.activeTurn.items.findIndex(
            (i) => i.id === item.id,
          );
          if (existing >= 0) {
            const items = [...s.activeTurn.items];
            items[existing] = item;
            return { activeTurn: { ...s.activeTurn, items } };
          }
          return {
            activeTurn: {
              ...s.activeTurn,
              items: [...s.activeTurn.items, item],
            },
          };
        });
        break;
      }

      case "turn/completed": {
        const completedTurn = p.turn as Turn | undefined;
        set((s) => {
          const at = s.activeTurn;
          const threadId = at?.threadId ?? (p.threadId as string | undefined);
          if (!threadId) return { activeTurn: null };

          const localUserItems = (at?.items ?? []).filter(
            (i) => i.type === "userMessage",
          );

          let agentItems: ThreadItem[];
          if (completedTurn?.items?.length) {
            agentItems = completedTurn.items;
          } else {
            agentItems = (at?.items ?? []).filter(
              (i) => i.type !== "userMessage",
            );
            if (at?.streamingText && at?.streamingItemId) {
              agentItems.push({
                type: "agentMessage",
                id: at.streamingItemId,
                text: at.streamingText,
              } as AgentMessageItem);
            }
          }

          const turnItems = [...localUserItems, ...agentItems];
          const existing = s.threadMessages[threadId] ?? [];
          return {
            activeTurn: null,
            threadMessages: {
              ...s.threadMessages,
              [threadId]: [...existing, ...turnItems],
            },
          };
        });
        get().loadThreads();
        break;
      }

      case "thread/tokenUsage/updated": {
        const cumulative = p.cumulative as TokenUsageInfo;
        set({ tokenUsage: cumulative });
        break;
      }

      case "thread/archived": {
        const threadId = p.threadId as string;
        set((s) => ({
          threads: s.threads.filter((t) => t.id !== threadId),
          currentThreadId:
            s.currentThreadId === threadId ? null : s.currentThreadId,
        }));
        break;
      }
    }
  },
}));

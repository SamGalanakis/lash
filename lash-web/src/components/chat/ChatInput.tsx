import { useState, useRef, useCallback, useEffect } from "react";
import { useSessionStore } from "@/stores/useSessionStore";
import { RiSendPlane2Fill, RiStopCircleLine } from "@remixicon/react";

interface ChatInputProps {
  threadId: string;
  disabled?: boolean;
}

export function ChatInput({ threadId, disabled }: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const sendMessage = useSessionStore((s) => s.sendMessage);
  const interruptTurn = useSessionStore((s) => s.interruptTurn);
  const activeTurn = useSessionStore((s) => s.activeTurn);
  const error = useSessionStore((s) => s.error);
  const connected = useSessionStore((s) => s.connected);

  const isActive = activeTurn?.threadId === threadId;

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [input, adjustHeight]);

  const submitInput = useCallback(() => {
    const text = input.trim();
    if (!text || disabled) return;
    setInput("");
    sendMessage(text);
  }, [input, disabled, sendMessage]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      submitInput();
    },
    [submitInput],
  );

  return (
    <div className="border-t border-border/70 bg-background/95 px-4 py-3 backdrop-blur-sm">
      {!connected && (
        <div className="chat-column mb-2 rounded-xl border border-destructive/30 bg-destructive/12 px-3 py-2 text-xs text-destructive">
          Not connected to server. Retrying...
        </div>
      )}
      {error && (
        <div className="chat-column mb-2 rounded-xl border border-destructive/30 bg-destructive/12 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="chat-column">
        <div className="overflow-hidden rounded-2xl border border-input/80 bg-card shadow-sm transition-shadow focus-within:shadow-[0_0_0_2px_color-mix(in_srgb,var(--color-ring)_18%,transparent)]">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitInput();
              }
            }}
            placeholder={isActive ? "Agent is working..." : "Send a message..."}
            rows={1}
            disabled={disabled}
            className="min-h-[56px] w-full resize-none border-0 bg-transparent px-4 py-3 text-sm leading-relaxed placeholder:text-muted-foreground focus:outline-none disabled:opacity-55"
          />

          <div className="flex items-center justify-between gap-2 border-t border-border/70 bg-muted/40 px-3 py-2">
            <span className="text-[11px] text-muted-foreground">
              Enter to send, Shift+Enter for newline
            </span>

            {isActive ? (
              <button
                type="button"
                onClick={() => interruptTurn()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-destructive/35 bg-destructive/12 px-2.5 py-1.5 text-xs font-medium text-destructive transition-colors hover:bg-destructive/18"
                title="Stop"
              >
                <RiStopCircleLine className="h-4 w-4" />
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-primary/35 bg-primary px-2.5 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-35"
              >
                <RiSendPlane2Fill className="h-4 w-4" />
                Send
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}

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

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || disabled) return;
      setInput("");
      sendMessage(text);
    },
    [input, disabled, sendMessage],
  );

  return (
    <div className="border-t border-border bg-background px-4 py-3">
      {!connected && (
        <div className="mx-auto max-w-3xl mb-2 rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive">
          Not connected to server. Retrying...
        </div>
      )}
      {error && (
        <div className="mx-auto max-w-3xl mb-2 rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}
      <form
        onSubmit={handleSubmit}
        className="mx-auto max-w-3xl relative"
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          placeholder={isActive ? "Agent is working..." : "Send a message..."}
          rows={1}
          disabled={disabled}
          className="w-full resize-none rounded-xl border border-border bg-card px-4 py-2.5 pr-12 text-sm shadow-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/20 disabled:opacity-50 transition-shadow"
        />

        {isActive ? (
          <button
            type="button"
            onClick={() => interruptTurn()}
            className="absolute bottom-2.5 right-3 rounded-lg bg-destructive p-2 text-destructive-foreground transition-opacity hover:opacity-90"
            title="Stop"
          >
            <RiStopCircleLine className="h-4 w-4" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="absolute bottom-2.5 right-3 rounded-lg bg-primary p-2 text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-30"
          >
            <RiSendPlane2Fill className="h-4 w-4" />
          </button>
        )}
      </form>
    </div>
  );
}

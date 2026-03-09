import { useState, useCallback } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useSessionStore } from "@/stores/useSessionStore";
import { RiSendPlane2Fill, RiTerminalLine } from "@remixicon/react";

export function WelcomeView() {
  const [input, setInput] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const startThread = useSessionStore((s) => s.startThread);
  const sendMessage = useSessionStore((s) => s.sendMessage);
  const error = useSessionStore((s) => s.error);
  const connected = useSessionStore((s) => s.connected);
  const navigate = useNavigate();

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || submitting) return;

      setSubmitting(true);
      try {
        const threadId = await startThread();
        setInput("");
        navigate({ to: "/chat/$threadId", params: { threadId } });
        setTimeout(() => sendMessage(text), 100);
      } catch {
        setSubmitting(false);
      }
    },
    [input, submitting, startThread, sendMessage, navigate],
  );

  return (
    <div className="flex h-full items-center justify-center">
      <div className="chat-column">
        <div className="mb-8 text-center">
          <div className="mb-4 inline-flex items-center justify-center rounded-2xl bg-muted p-4">
            <RiTerminalLine className="h-8 w-8 text-muted-foreground" />
          </div>
          <h1 className="mb-2 text-2xl font-semibold tracking-tight">lash</h1>
          <p className="text-sm text-muted-foreground">
            AI coding agent. Ask anything to get started.
          </p>
        </div>

        {!connected && (
          <div className="mb-3 rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive text-center">
            Not connected to server. Retrying...
          </div>
        )}
        {error && (
          <div className="mb-3 rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive text-center">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="overflow-hidden rounded-2xl border border-input/80 bg-card shadow-sm transition-shadow focus-within:shadow-[0_0_0_2px_color-mix(in_srgb,var(--color-ring)_18%,transparent)]">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask lash something..."
              rows={3}
              disabled={submitting}
              className="w-full resize-none border-0 bg-transparent px-4 py-3 text-sm leading-relaxed placeholder:text-muted-foreground focus:outline-none disabled:opacity-55"
            />
            <div className="flex items-center justify-end border-t border-border/70 bg-muted/40 px-3 py-2">
              <button
                type="submit"
                disabled={!input.trim() || submitting}
                className="inline-flex items-center gap-1.5 rounded-lg border border-primary/35 bg-primary px-2.5 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-35"
              >
                <RiSendPlane2Fill className="h-4 w-4" />
                Send
              </button>
            </div>
          </div>
        </form>

        <div className="mt-6 grid grid-cols-2 gap-2">
          {[
            "Explain this codebase",
            "Fix the failing tests",
            "Add a new feature",
            "Review my changes",
          ].map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => setInput(suggestion)}
              className="rounded-lg border border-border bg-card px-3 py-2 text-left text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

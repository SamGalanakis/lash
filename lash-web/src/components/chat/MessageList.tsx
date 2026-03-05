import { useEffect, useRef } from "react";
import { useSessionStore } from "@/stores/useSessionStore";
import { MessageBubble } from "./MessageBubble";
import { StreamingMessage } from "./StreamingMessage";
import {
  RiLoader4Line,
  RiGroupLine,
  RiTerminalBoxLine,
} from "@remixicon/react";
import type { ThreadItem } from "@/lib/types";

const EMPTY_ITEMS: ThreadItem[] = [];

interface MessageListProps {
  threadId: string;
}

export function MessageList({ threadId }: MessageListProps) {
  const activeTurn = useSessionStore((s) => s.activeTurn);
  const threadMessages = useSessionStore(
    (s) => s.threadMessages[threadId] ?? EMPTY_ITEMS,
  );
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeItems =
    activeTurn?.threadId === threadId ? activeTurn.items : [];
  const allItems = [...threadMessages, ...activeItems];
  const streamingText =
    activeTurn?.threadId === threadId ? activeTurn.streamingText : "";
  const toolOutputLines =
    activeTurn?.threadId === threadId ? activeTurn.toolOutputLines : [];
  const activeDelegate =
    activeTurn?.threadId === threadId ? activeTurn.activeDelegate : null;
  const isActive = activeTurn?.threadId === threadId;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [allItems.length, streamingText, toolOutputLines.length, activeDelegate]);

  if (allItems.length === 0 && !isActive) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-muted-foreground">
          Send a message to start the conversation.
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-6 space-y-4">
      {allItems.map((item) => (
        <MessageBubble key={item.id} item={item} />
      ))}

      {streamingText && <StreamingMessage text={streamingText} />}

      {activeDelegate && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <RiGroupLine className="h-4 w-4" />
          <span className="text-xs">
            {activeDelegate.name}: {activeDelegate.task}
          </span>
        </div>
      )}

      {toolOutputLines.length > 0 && (
        <div className="space-y-1">
          {toolOutputLines.map((line, idx) => (
            <div
              key={`tool-output-${idx}`}
              className="flex items-start gap-2 text-xs font-mono text-muted-foreground"
            >
              <RiTerminalBoxLine className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <span className="whitespace-pre-wrap break-words">{line}</span>
            </div>
          ))}
        </div>
      )}

      {isActive && !streamingText && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <RiLoader4Line className="h-4 w-4 animate-spin" />
          <span className="text-xs">Thinking...</span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

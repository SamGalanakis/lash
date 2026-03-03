import { useEffect, useRef } from "react";
import { useSessionStore } from "@/stores/useSessionStore";
import { MessageBubble } from "./MessageBubble";
import { StreamingMessage } from "./StreamingMessage";
import { RiLoader4Line } from "@remixicon/react";
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
  const isActive = activeTurn?.threadId === threadId;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [allItems.length, streamingText]);

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

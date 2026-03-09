import { useEffect, useMemo, useRef } from "react";
import { useSessionStore } from "@/stores/useSessionStore";
import { MessageBubble } from "./MessageBubble";
import { StreamingMessage } from "./StreamingMessage";
import {
  RiLoader4Line,
  RiGroupLine,
  RiTerminalBoxLine,
} from "@remixicon/react";
import type { ThreadItem, UserMessageItem } from "@/lib/types";

const EMPTY_ITEMS: ThreadItem[] = [];

interface MessageListProps {
  threadId: string;
}

interface MessageTurn {
  id: string;
  user: UserMessageItem;
  assistantItems: ThreadItem[];
}

function groupItemsByTurn(items: ThreadItem[]): {
  leadingItems: ThreadItem[];
  turns: MessageTurn[];
} {
  const leadingItems: ThreadItem[] = [];
  const turns: MessageTurn[] = [];
  let currentTurn: MessageTurn | null = null;

  for (const item of items) {
    if (item.type === "userMessage") {
      currentTurn = {
        id: item.id,
        user: item,
        assistantItems: [],
      };
      turns.push(currentTurn);
      continue;
    }

    if (currentTurn) {
      currentTurn.assistantItems.push(item);
      continue;
    }

    leadingItems.push(item);
  }

  return { leadingItems, turns };
}

export function MessageList({ threadId }: MessageListProps) {
  const activeTurn = useSessionStore((s) => s.activeTurn);
  const threadMessages = useSessionStore(
    (s) => s.threadMessages[threadId] ?? EMPTY_ITEMS,
  );
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeItems =
    activeTurn?.threadId === threadId ? activeTurn.items : EMPTY_ITEMS;
  const allItems = useMemo(
    () => [...threadMessages, ...activeItems],
    [threadMessages, activeItems],
  );

  const streamingText =
    activeTurn?.threadId === threadId ? activeTurn.streamingText : "";
  const toolOutputLines =
    activeTurn?.threadId === threadId ? activeTurn.toolOutputLines : [];
  const activeDelegate =
    activeTurn?.threadId === threadId ? activeTurn.activeDelegate : null;
  const isActive = activeTurn?.threadId === threadId;

  const { leadingItems, turns } = useMemo(
    () => groupItemsByTurn(allItems),
    [allItems],
  );

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
    <div className="chat-message-column py-6">
      <div className="space-y-4">
        {leadingItems.map((item) => (
          <MessageBubble key={item.id} item={item} />
        ))}

        {turns.map((turn, idx) => {
          const isLastTurn = idx === turns.length - 1;
          const showActiveState = isLastTurn && isActive;

          return (
            <section key={turn.id} className="relative">
              <div className="relative z-20 md:sticky md:top-0 md:bg-background/95 md:backdrop-blur-sm">
                <div className="relative z-10 py-1">
                  <MessageBubble item={turn.user} />
                </div>
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-x-0 top-full hidden h-6 bg-gradient-to-b from-background/90 to-transparent md:block"
                />
              </div>

              <div className="space-y-3 pt-2">
                {turn.assistantItems.map((item) => (
                  <MessageBubble key={item.id} item={item} />
                ))}

                {showActiveState && streamingText && (
                  <StreamingMessage text={streamingText} />
                )}

                {showActiveState && activeDelegate && (
                  <div className="inline-flex items-center gap-2 rounded-lg border border-border/70 bg-muted/35 px-3 py-2 text-muted-foreground">
                    <RiGroupLine className="h-4 w-4" />
                    <span className="text-xs">
                      {activeDelegate.name}: {activeDelegate.task}
                    </span>
                  </div>
                )}

                {showActiveState && toolOutputLines.length > 0 && (
                  <div className="space-y-1.5 rounded-xl border border-border/70 bg-card/80 p-3">
                    {toolOutputLines.map((line, lineIdx) => (
                      <div
                        key={`tool-output-${lineIdx}`}
                        className="flex items-start gap-2 font-mono text-xs text-muted-foreground"
                      >
                        <RiTerminalBoxLine className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                        <span className="whitespace-pre-wrap break-words">
                          {line}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {showActiveState && !streamingText && (
                  <div className="inline-flex items-center gap-2 rounded-lg border border-border/70 bg-muted/35 px-3 py-2 text-muted-foreground">
                    <RiLoader4Line className="h-4 w-4 animate-spin" />
                    <span className="text-xs">Thinking...</span>
                  </div>
                )}
              </div>
            </section>
          );
        })}

        {turns.length === 0 && isActive && streamingText && (
          <StreamingMessage text={streamingText} />
        )}

        {turns.length === 0 && isActive && !streamingText && (
          <div className="inline-flex items-center gap-2 rounded-lg border border-border/70 bg-muted/35 px-3 py-2 text-muted-foreground">
            <RiLoader4Line className="h-4 w-4 animate-spin" />
            <span className="text-xs">Thinking...</span>
          </div>
        )}
      </div>

      <div ref={bottomRef} />
    </div>
  );
}

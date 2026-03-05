import { useEffect, useRef } from "react";
import { useParams } from "@tanstack/react-router";
import { useSessionStore } from "@/stores/useSessionStore";
import { lashClient } from "@/lib/lash-client";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";

export function ChatView() {
  const { threadId } = useParams({ from: "/chat/$threadId" });
  const currentThreadId = useSessionStore((s) => s.currentThreadId);
  const selectThread = useSessionStore((s) => s.selectThread);
  const activeTurn = useSessionStore((s) => s.activeTurn);
  const resumeAttempted = useRef(false);

  useEffect(() => {
    if (threadId && threadId !== currentThreadId) {
      selectThread(threadId);
      if (!resumeAttempted.current) {
        resumeAttempted.current = true;
        lashClient.threadResume(threadId).catch(() => {});
      }
    }
  }, [threadId, currentThreadId, selectThread]);

  useEffect(() => {
    resumeAttempted.current = false;
  }, [threadId]);

  const isLoading = activeTurn !== null;

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto">
        <MessageList threadId={threadId} />
      </div>
      <ChatInput threadId={threadId} disabled={isLoading} />
    </div>
  );
}

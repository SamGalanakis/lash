import { useNavigate } from "@tanstack/react-router";
import { useSessionStore } from "@/stores/useSessionStore";
import { useUIStore } from "@/stores/useUIStore";
import { cn } from "@/lib/utils";
import {
  RiAddLine,
  RiChat1Line,
  RiDeleteBinLine,
  RiArrowLeftSLine,
} from "@remixicon/react";

export function Sidebar() {
  const threads = useSessionStore((s) => s.threads);
  const currentThreadId = useSessionStore((s) => s.currentThreadId);
  const startThread = useSessionStore((s) => s.startThread);
  const selectThread = useSessionStore((s) => s.selectThread);
  const archiveThread = useSessionStore((s) => s.archiveThread);
  const setSidebarOpen = useUIStore((s) => s.setSidebarOpen);
  const navigate = useNavigate();

  const handleNewThread = async () => {
    try {
      const threadId = await startThread();
      navigate({ to: "/chat/$threadId", params: { threadId } });
    } catch {
      // error is already set in the store by startThread
    }
  };

  const handleSelectThread = (threadId: string) => {
    selectThread(threadId);
    navigate({ to: "/chat/$threadId", params: { threadId } });
  };

  const handleArchive = async (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation();
    await archiveThread(threadId);
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <aside className="flex w-64 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center justify-between px-3 py-2 border-b border-sidebar-border">
        <span className="text-sm font-medium text-sidebar-foreground/80">
          Threads
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={handleNewThread}
            className="rounded-md p-1.5 hover:bg-sidebar-accent transition-colors"
            title="New thread"
          >
            <RiAddLine className="h-4 w-4" />
          </button>
          <button
            onClick={() => setSidebarOpen(false)}
            className="rounded-md p-1.5 hover:bg-sidebar-accent transition-colors"
            title="Close sidebar"
          >
            <RiArrowLeftSLine className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {threads.length === 0 && (
          <div className="px-3 py-8 text-center text-xs text-muted-foreground">
            No threads yet
          </div>
        )}
        {threads.map((thread) => (
          <button
            key={thread.id}
            onClick={() => handleSelectThread(thread.id)}
            className={cn(
              "group flex w-full items-start gap-2 rounded-md mx-1 px-2.5 py-2 text-left text-sm transition-colors",
              "hover:bg-sidebar-accent",
              currentThreadId === thread.id &&
                "bg-sidebar-accent text-sidebar-accent-foreground",
            )}
            style={{ width: "calc(100% - 0.5rem)" }}
          >
            <RiChat1Line className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <div className="flex-1 min-w-0">
              <p className="truncate text-[13px] leading-snug">
                {thread.preview || "New thread"}
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                {formatDate(thread.createdAt)}
              </p>
            </div>
            <button
              onClick={(e) => handleArchive(e, thread.id)}
              className="shrink-0 rounded p-1 opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all"
              title="Archive"
            >
              <RiDeleteBinLine className="h-3 w-3" />
            </button>
          </button>
        ))}
      </div>
    </aside>
  );
}

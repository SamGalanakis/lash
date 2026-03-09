import type { ThreadItem } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  RiCodeSSlashLine,
  RiTerminalBoxLine,
  RiToolsLine,
  RiErrorWarningLine,
  RiCheckLine,
  RiCloseLine,
  RiGroupLine,
  RiLoader4Line,
  RiTimeLine,
} from "@remixicon/react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MessageBubbleProps {
  item: ThreadItem;
}

export function MessageBubble({ item }: MessageBubbleProps) {
  switch (item.type) {
    case "userMessage":
      return <UserMessage item={item} />;
    case "agentMessage":
      return <AgentMessage item={item} />;
    case "codeBlock":
      return <CodeBlock item={item} />;
    case "codeOutput":
      return <CodeOutput item={item} />;
    case "toolCall":
      return <ToolCall item={item} />;
    case "subAgentResult":
      return <SubAgentResult item={item} />;
    case "retryStatus":
      return <RetryStatus item={item} />;
    case "error":
      return <ErrorMessage item={item} />;
    default:
      return null;
  }
}

function UserMessage({ item }: { item: Extract<ThreadItem, { type: "userMessage" }> }) {
  const text = item.content
    .map((part) => {
      if (part.type === "text") return part.text;
      if (part.type === "fileRef") return `[File] ${part.path}`;
      if (part.type === "dirRef") return `[Folder] ${part.path}`;
      if (part.type === "skill") return `[Skill] ${part.name}`;
      if (part.type === "image" || part.type === "localImage") return "[Image]";
      return "";
    })
    .filter(Boolean)
    .join("\n");

  return (
    <div className="flex justify-end">
      <div
        style={{ backgroundColor: "var(--chat-user-message-bg)" }}
        className="max-w-[85%] rounded-2xl rounded-br-sm border border-primary/15 px-5 py-3 text-sm leading-relaxed text-foreground"
      >
        <p className="whitespace-pre-wrap break-words">
          {text || "Sent message"}
        </p>
      </div>
    </div>
  );
}

function AgentMessage({ item }: { item: Extract<ThreadItem, { type: "agentMessage" }> }) {
  return (
    <div className="max-w-[85%]">
      <div className="prose prose-sm max-w-none prose-headings:my-2 prose-p:my-1.5 prose-pre:my-2 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5 prose-code:font-mono prose-code:text-[13px] prose-pre:border prose-pre:border-border/70 prose-pre:bg-card prose-pre:rounded-xl prose-blockquote:border-l-border prose-blockquote:text-muted-foreground">
        <Markdown remarkPlugins={[remarkGfm]}>{item.text}</Markdown>
      </div>
    </div>
  );
}

function CodeBlock({ item }: { item: Extract<ThreadItem, { type: "codeBlock" }> }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border/80 bg-card/80">
      <div className="flex items-center gap-1.5 border-b border-border/70 bg-muted/55 px-3 py-1.5">
        <RiCodeSSlashLine className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-[11px] font-medium tracking-wide text-muted-foreground">
          Code
        </span>
      </div>
      <pre className="overflow-x-auto p-3 text-xs leading-relaxed">
        <code>{item.code}</code>
      </pre>
    </div>
  );
}

function CodeOutput({ item }: { item: Extract<ThreadItem, { type: "codeOutput" }> }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border/80 bg-card/80">
      <div className="flex items-center gap-1.5 border-b border-border/70 bg-muted/55 px-3 py-1.5">
        <RiTerminalBoxLine className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-[11px] font-medium tracking-wide text-muted-foreground">
          Output
        </span>
      </div>
      <pre
        className={cn(
          "overflow-x-auto p-3 font-mono text-xs leading-relaxed",
          item.error ? "bg-destructive/8 text-destructive" : "",
        )}
      >
        {item.output}
        {item.error && (
          <span className="text-destructive">{"\n"}{item.error}</span>
        )}
      </pre>
    </div>
  );
}

function ToolCall({ item }: { item: Extract<ThreadItem, { type: "toolCall" }> }) {
  const StatusIcon = item.status === "inProgress"
    ? RiLoader4Line
    : item.success
      ? RiCheckLine
      : RiCloseLine;

  return (
    <div className="overflow-hidden rounded-xl border border-border/80 bg-card/90">
      <div className="flex items-center gap-2 px-3 py-2">
        <RiToolsLine className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        <span className="text-xs font-medium truncate">{item.name}</span>
        <div className="ml-auto flex items-center gap-1.5">
          {item.durationMs !== undefined && (
            <span className="text-[10px] text-muted-foreground tabular-nums">
              {item.durationMs}ms
            </span>
          )}
          <StatusIcon
            className={cn(
              "h-3.5 w-3.5",
              item.status === "inProgress" && "animate-spin text-muted-foreground",
              item.status === "completed" && item.success && "text-emerald-500",
              item.status === "failed" && "text-destructive",
            )}
          />
        </div>
      </div>
    </div>
  );
}

function SubAgentResult({ item }: { item: Extract<ThreadItem, { type: "subAgentResult" }> }) {
  return (
    <div className="rounded-xl border border-border/80 bg-card/90 px-3 py-2">
      <div className="flex items-center gap-2">
        <RiGroupLine className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs font-medium truncate flex-1">{item.task}</span>
        {item.success ? (
          <RiCheckLine className="h-3.5 w-3.5 text-emerald-500" />
        ) : (
          <RiCloseLine className="h-3.5 w-3.5 text-destructive" />
        )}
      </div>
      <div className="mt-1 text-[10px] text-muted-foreground">
        {item.toolCalls} tool calls · {item.iterations} iterations
      </div>
    </div>
  );
}

function RetryStatus({ item }: { item: Extract<ThreadItem, { type: "retryStatus" }> }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-border/60 bg-muted/35 px-3 py-2 text-muted-foreground">
      <RiTimeLine className="h-3.5 w-3.5" />
      <span className="text-xs">
        Retrying ({item.attempt}/{item.maxAttempts}) — {item.reason}
      </span>
    </div>
  );
}

function ErrorMessage({ item }: { item: Extract<ThreadItem, { type: "error" }> }) {
  return (
    <div className="rounded-xl border border-destructive/35 bg-destructive/10 px-3 py-2">
      <div className="flex items-start gap-2">
        <RiErrorWarningLine className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
        <div>
          <p className="text-xs font-medium text-destructive">{item.message}</p>
          {item.errorInfo && (
            <p className="text-[11px] text-destructive/70 mt-0.5">
              {item.errorInfo}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

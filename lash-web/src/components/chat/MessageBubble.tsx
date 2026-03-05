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
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text)
    .join("\n");

  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-primary-foreground">
        <p className="text-sm whitespace-pre-wrap">{text}</p>
      </div>
    </div>
  );
}

function AgentMessage({ item }: { item: Extract<ThreadItem, { type: "agentMessage" }> }) {
  return (
    <div className="max-w-[85%]">
      <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1.5 prose-pre:my-2 prose-headings:my-2 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5">
        <Markdown remarkPlugins={[remarkGfm]}>{item.text}</Markdown>
      </div>
    </div>
  );
}

function CodeBlock({ item }: { item: Extract<ThreadItem, { type: "codeBlock" }> }) {
  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <div className="flex items-center gap-1.5 bg-muted px-3 py-1.5 border-b border-border">
        <RiCodeSSlashLine className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-[11px] font-medium text-muted-foreground">Code</span>
      </div>
      <pre className="overflow-x-auto bg-card p-3 text-xs leading-relaxed">
        <code>{item.code}</code>
      </pre>
    </div>
  );
}

function CodeOutput({ item }: { item: Extract<ThreadItem, { type: "codeOutput" }> }) {
  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <div className="flex items-center gap-1.5 bg-muted px-3 py-1.5 border-b border-border">
        <RiTerminalBoxLine className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-[11px] font-medium text-muted-foreground">Output</span>
      </div>
      <pre
        className={cn(
          "overflow-x-auto p-3 text-xs leading-relaxed font-mono",
          item.error ? "bg-destructive/5 text-destructive" : "bg-card",
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
    <div className="rounded-lg border border-border bg-card overflow-hidden">
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
    <div className="rounded-lg border border-border bg-card px-3 py-2">
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
    <div className="flex items-center gap-2 text-muted-foreground">
      <RiTimeLine className="h-3.5 w-3.5" />
      <span className="text-xs">
        Retrying ({item.attempt}/{item.maxAttempts}) — {item.reason}
      </span>
    </div>
  );
}

function ErrorMessage({ item }: { item: Extract<ThreadItem, { type: "error" }> }) {
  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2">
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

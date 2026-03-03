export interface Thread {
  id: string;
  preview: string;
  model: string;
  provider: string;
  createdAt: number;
  updatedAt?: number;
  status?: ThreadStatus;
  turns?: Turn[];
}

export type ThreadStatus =
  | { type: "notLoaded" }
  | { type: "idle" }
  | { type: "active"; activeFlags: string[] }
  | { type: "systemError" };

export interface Turn {
  id: string;
  status: "inProgress" | "completed" | "interrupted" | "failed";
  items: ThreadItem[];
  error?: TurnError;
}

export interface TurnError {
  message: string;
  errorInfo?: string;
}

export type ThreadItem =
  | UserMessageItem
  | AgentMessageItem
  | CodeBlockItem
  | CodeOutputItem
  | ToolCallItem
  | SubAgentResultItem
  | RetryStatusItem
  | ErrorItem;

export interface UserMessageItem {
  type: "userMessage";
  id: string;
  content: TurnInputItem[];
}

export interface AgentMessageItem {
  type: "agentMessage";
  id: string;
  text: string;
}

export interface CodeBlockItem {
  type: "codeBlock";
  id: string;
  code: string;
}

export interface CodeOutputItem {
  type: "codeOutput";
  id: string;
  output: string;
  error?: string;
}

export interface ToolCallItem {
  type: "toolCall";
  id: string;
  name: string;
  args: unknown;
  result?: unknown;
  success: boolean;
  durationMs?: number;
  status: "inProgress" | "completed" | "failed";
}

export interface SubAgentResultItem {
  type: "subAgentResult";
  id: string;
  task: string;
  success: boolean;
  toolCalls: number;
  iterations: number;
}

export interface RetryStatusItem {
  type: "retryStatus";
  id: string;
  waitSeconds: number;
  attempt: number;
  maxAttempts: number;
  reason: string;
}

export interface ErrorItem {
  type: "error";
  id: string;
  message: string;
  errorInfo?: string;
}

export type TurnInputItem =
  | { type: "text"; text: string }
  | { type: "image"; url: string }
  | { type: "localImage"; path: string }
  | { type: "skill"; name: string; path: string }
  | { type: "fileRef"; path: string }
  | { type: "dirRef"; path: string };

export interface TokenUsageInfo {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
}

export interface ModelInfo {
  id: string;
  contextWindow?: number;
  reasoningEffort?: string;
}

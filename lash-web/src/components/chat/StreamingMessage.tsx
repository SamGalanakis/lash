import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface StreamingMessageProps {
  text: string;
}

export function StreamingMessage({ text }: StreamingMessageProps) {
  return (
    <div className="max-w-[85%]">
      <div className="prose prose-sm max-w-none prose-headings:my-2 prose-p:my-1.5 prose-pre:my-2 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5 prose-code:font-mono prose-code:text-[13px] prose-pre:border prose-pre:border-border/70 prose-pre:bg-card prose-pre:rounded-xl">
        <Markdown remarkPlugins={[remarkGfm]}>{text}</Markdown>
        <span className="inline-block h-4 w-[2px] ml-0.5 bg-foreground animate-pulse" />
      </div>
    </div>
  );
}

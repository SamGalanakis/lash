import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface StreamingMessageProps {
  text: string;
}

export function StreamingMessage({ text }: StreamingMessageProps) {
  return (
    <div className="max-w-[85%]">
      <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1.5 prose-pre:my-2 prose-headings:my-2 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5">
        <Markdown remarkPlugins={[remarkGfm]}>{text}</Markdown>
        <span className="inline-block h-4 w-[2px] ml-0.5 bg-foreground animate-pulse" />
      </div>
    </div>
  );
}

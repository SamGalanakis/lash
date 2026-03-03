import { useSessionStore } from "@/stores/useSessionStore";
import { useUIStore } from "@/stores/useUIStore";
import {
  RiMenuLine,
  RiMoonLine,
  RiSunLine,
  RiComputerLine,
} from "@remixicon/react";

export function Header() {
  const tokenUsage = useSessionStore((s) => s.tokenUsage);
  const connected = useSessionStore((s) => s.connected);
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const theme = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);

  const cycleTheme = () => {
    const order: Array<"light" | "dark" | "system"> = [
      "light",
      "dark",
      "system",
    ];
    const next = order[(order.indexOf(theme) + 1) % order.length];
    setTheme(next);
  };

  const ThemeIcon =
    theme === "light" ? RiSunLine : theme === "dark" ? RiMoonLine : RiComputerLine;

  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-border bg-background px-3">
      <div className="flex items-center gap-2">
        {!sidebarOpen && (
          <button
            onClick={toggleSidebar}
            className="rounded-md p-1.5 hover:bg-accent transition-colors"
          >
            <RiMenuLine className="h-4 w-4" />
          </button>
        )}
        <span className="text-sm font-semibold tracking-tight">lash</span>
        <div className="flex items-center gap-1.5 ml-2">
          <div
            className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"}`}
          />
          <span className="text-[11px] text-muted-foreground">
            {connected ? "connected" : "disconnected"}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {tokenUsage && (
          <div className="text-[11px] text-muted-foreground font-mono tabular-nums">
            {tokenUsage.inputTokens.toLocaleString()}↓{" "}
            {tokenUsage.outputTokens.toLocaleString()}↑
          </div>
        )}
        <button
          onClick={cycleTheme}
          className="rounded-md p-1.5 hover:bg-accent transition-colors"
          title={`Theme: ${theme}`}
        >
          <ThemeIcon className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}

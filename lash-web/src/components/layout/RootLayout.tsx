import { useEffect, type ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useSessionStore } from "@/stores/useSessionStore";
import { useUIStore } from "@/stores/useUIStore";
import { lashClient } from "@/lib/lash-client";
import { Toaster } from "sonner";

export function RootLayout({ children }: { children: ReactNode }) {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const handleNotification = useSessionStore((s) => s.handleNotification);
  const setConnected = useSessionStore((s) => s.setConnected);
  const loadThreads = useSessionStore((s) => s.loadThreads);

  useEffect(() => {
    lashClient.connect(
      (method, params) => {
        handleNotification(method, params);
      },
      (connected) => {
        setConnected(connected);
        if (connected) loadThreads();
      },
    );

    return () => {
      lashClient.disconnect();
      setConnected(false);
    };
  }, [handleNotification, setConnected, loadThreads]);

  return (
    <div className="flex h-full overflow-hidden">
      {sidebarOpen && <Sidebar />}
      <div className="flex flex-1 flex-col min-w-0">
        <Header />
        <main className="flex-1 overflow-hidden">{children}</main>
      </div>
      <Toaster
        position="bottom-right"
        toastOptions={{
          className: "bg-card text-card-foreground border-border",
        }}
      />
    </div>
  );
}

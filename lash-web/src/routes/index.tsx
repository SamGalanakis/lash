import { createFileRoute } from "@tanstack/react-router";
import { WelcomeView } from "@/components/chat/WelcomeView";

export const Route = createFileRoute("/")({
  component: WelcomeView,
});

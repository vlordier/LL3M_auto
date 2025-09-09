import React from "react";
import { SunIcon, MoonIcon } from "@heroicons/react/24/outline";
import { useUIStore } from "@/stores/uiStore";
import { Button } from "./Button";

export function ThemeToggle() {
  const { theme, cycleTheme } = useUIStore();
  const systemDark =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;

  const label =
    theme === "system" ? "System" : theme === "dark" ? "Dark" : "Light";
  const isDark = theme === "dark" || (theme === "system" && systemDark);

  return (
    <Button
      variant="ghost"
      size="sm"
      aria-label="Cycle theme (Light/Dark/System)"
      onClick={cycleTheme}
      leftIcon={
        isDark ? (
          <SunIcon className="w-4 h-4" />
        ) : (
          <MoonIcon className="w-4 h-4" />
        )
      }
    >
      {label}
    </Button>
  );
}

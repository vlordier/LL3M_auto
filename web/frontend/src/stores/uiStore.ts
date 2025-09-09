import { create } from "zustand";

type Theme = "light" | "dark" | "system";

interface UIStore {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  cycleTheme: () => void;
  applyTheme: () => void;
}

function getSystemPrefersDark(): boolean {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  );
}

function applyThemeClass(theme: Theme) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  const shouldDark =
    theme === "dark" || (theme === "system" && getSystemPrefersDark());
  if (shouldDark) root.classList.add("dark");
  else root.classList.remove("dark");
}

export const useUIStore = create<UIStore>((set, get) => ({
  theme: ((): Theme => {
    if (typeof localStorage === "undefined") return "system";
    const saved = localStorage.getItem("theme") as Theme | null;
    return saved ?? "system";
  })(),
  setTheme: (theme) => {
    set({ theme });
    try {
      localStorage.setItem("theme", theme);
    } catch {}
    applyThemeClass(theme);
  },
  toggleTheme: () => {
    const current = get().theme;
    const next: Theme = current === "dark" ? "light" : "dark";
    get().setTheme(next);
  },
  cycleTheme: () => {
    const order: Theme[] = ["light", "dark", "system"];
    const idx = order.indexOf(get().theme);
    const next = order[(idx + 1) % order.length];
    get().setTheme(next);
  },
  applyTheme: () => {
    applyThemeClass(get().theme);
  },
}));

// Auto-apply theme on script load (client)
if (typeof window !== "undefined") {
  try {
    const saved = (localStorage.getItem("theme") as Theme | null) ?? "system";
    applyThemeClass(saved);
  } catch {
    applyThemeClass("system");
  }
}

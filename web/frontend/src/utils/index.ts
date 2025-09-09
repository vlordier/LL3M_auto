import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number | undefined): string {
  if (!seconds) return "N/A";

  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

export function formatFileSize(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

export function formatRelativeTime(date: string | Date): string {
  const now = new Date();
  const target = new Date(date);
  const diffMs = now.getTime() - target.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);

  if (diffSeconds < 60) return "just now";
  if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
  if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
  if (diffSeconds < 604800) return `${Math.floor(diffSeconds / 86400)}d ago`;

  return target.toLocaleDateString();
}

export function getAssetUrl(relativePath: string): string {
  return `/api/assets/${relativePath}`;
}

export function getPreviewUrl(assetPath: string): string {
  // Convert asset path to preview URL
  if (assetPath.endsWith(".blend")) {
    return assetPath.replace(".blend", "_screenshot.png");
  }
  if (assetPath.endsWith("_screenshot.png")) {
    return assetPath;
  }
  return assetPath;
}

export function getAssetType(
  filename: string,
): "gltf" | "obj" | "blend" | "unknown" {
  const ext = filename.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "gltf":
    case "glb":
      return "gltf";
    case "obj":
      return "obj";
    case "blend":
      return "blend";
    default:
      return "unknown";
  }
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function downloadFile(url: string, filename: string): void {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

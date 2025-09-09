import React, { useEffect, useState } from "react";

type Service = {
  name: string;
  ok: boolean;
  status: string;
  message?: string;
  details?: Record<string, any>;
};

type Diagnostics = {
  ok: boolean;
  timestamp: string;
  services: Record<string, Service>;
};

function Dot({ ok, skipped }: { ok: boolean; skipped?: boolean }) {
  const cls = skipped ? "bg-yellow-400" : ok ? "bg-green-500" : "bg-red-500";
  return <span className={`inline-block w-2 h-2 rounded-full ${cls}`} />;
}

export function BackendStatus({
  onOpenDetails,
}: {
  onOpenDetails?: () => void;
}) {
  const [diag, setDiag] = useState<Diagnostics | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    try {
      setLoading(true);
      const r = await fetch("/api/diagnostics");
      if (r.ok) {
        const data: Diagnostics = await r.json();
        setDiag(data);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  const svc = (key: string): Service | undefined => diag?.services?.[key];
  const pill = (key: string, label: string) => {
    const s = svc(key);
    const skipped = s?.status === "skipped";
    const title = s
      ? `${s.name}: ${s.status}${s.message ? " — " + s.message : ""}`
      : "Loading...";
    return (
      <span
        key={key}
        className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200"
        title={title}
      >
        <Dot ok={!!s?.ok} skipped={skipped} /> {label}
      </span>
    );
  };

  return (
    <div className="flex items-center gap-2">
      {loading && (
        <span
          className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"
          title="Checking backend status"
        />
      )}
      {pill("openai", "OpenAI")}
      {pill("lmstudio", "LM")}
      {pill("blender", "Blender")}
      {pill("context7_mcp", "C7")}
      {pill("vector_db", "VectorDB")}
      <button
        type="button"
        className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700"
        onClick={onOpenDetails}
        title="View detailed diagnostics"
      >
        Details
      </button>
    </div>
  );
}

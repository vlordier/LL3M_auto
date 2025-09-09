import React, { useEffect, useMemo, useState } from "react";

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

function Row({
  label,
  svc,
  links,
}: {
  label: string;
  svc?: Service;
  links?: { label: string; href: string }[];
}) {
  const skipped = svc?.status === "skipped";
  return (
    <div className="flex items-start justify-between gap-3 py-2 border-b border-gray-200 dark:border-gray-800">
      <div className="flex items-start gap-2 min-w-0">
        <Dot ok={!!svc?.ok} skipped={skipped} />
        <div className="truncate">
          <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {label}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300 truncate">
            {svc?.status || "unknown"}
            {svc?.message ? ` — ${svc.message}` : ""}
          </div>
        </div>
      </div>
      {!!links?.length && (
        <div className="flex items-center gap-2 flex-wrap">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              target="_blank"
              rel="noreferrer"
              className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              {l.label}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export function DiagnosticsDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [diag, setDiag] = useState<Diagnostics | null>(null);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<{
    exists: boolean;
    path?: string;
    lines: string[];
  } | null>(null);
  const [logsLoading, setLogsLoading] = useState(false);
  const [mcpBusy, setMcpBusy] = useState(false);
  const [blenderBusy, setBlenderBusy] = useState(false);
  const [blenderResult, setBlenderResult] = useState<{
    ok: boolean;
    path?: string;
    version?: string;
    headless_ok: boolean;
    message: string;
    suggestions?: string[];
  } | null>(null);
  const [vrmBusy, setVrmBusy] = useState(false);
  const [vrmMessage, setVrmMessage] = useState<string | null>(null);

  const load = async () => {
    try {
      setLoading(true);
      const r = await fetch("/api/diagnostics");
      if (r.ok) setDiag(await r.json());
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  const loadLogs = async () => {
    try {
      setLogsLoading(true);
      const r = await fetch("/api/logs/mcp?limit=300");
      if (r.ok) setLogs(await r.json());
    } catch {
      // ignore
    } finally {
      setLogsLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    load();
    loadLogs();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [open]);

  const baseLM = useMemo(
    () =>
      typeof window !== "undefined"
        ? `${window.location.protocol}//${window.location.hostname}:1234`
        : "http://localhost:1234",
    [],
  );
  const baseAPI = useMemo(
    () =>
      typeof window !== "undefined"
        ? `${window.location.origin}`
        : "http://localhost:8000",
    [],
  );

  if (!open) return null;

  const content = (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full w-full max-w-xl bg-white dark:bg-gray-900 shadow-xl flex flex-col">
        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Backend Diagnostics
            </div>
            <div className="text-xs text-gray-400 dark:text-gray-500">
              Last update: {diag?.timestamp || "—"}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {loading && (
              <span className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            )}
            <button
              className="px-2 py-1 text-xs rounded bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              onClick={load}
              title="Refresh"
            >
              Refresh
            </button>
            <button
              className="px-2 py-1 text-xs rounded bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              onClick={() => {
                try {
                  navigator.clipboard.writeText(JSON.stringify(diag, null, 2));
                } catch {}
              }}
              title="Copy JSON"
            >
              Copy JSON
            </button>
            <button
              className="px-2 py-1 text-xs rounded bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              onClick={onClose}
            >
              Close
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <Row
            label="Web Backend"
            svc={diag?.services?.web_backend}
            links={[{ label: "Health", href: `${baseAPI}/health` }]}
          />
          <Row
            label="Production API"
            svc={diag?.services?.prod_api}
            links={[
              { label: "Health", href: `${baseAPI}/api/v1/health` },
              { label: "Docs", href: `${baseAPI}/api/docs` },
            ]}
          />
          <Row
            label="OpenAI"
            svc={diag?.services?.openai}
            links={[
              {
                label: "API Keys",
                href: "https://platform.openai.com/api-keys",
              },
            ]}
          />
          <Row
            label="LM Studio"
            svc={diag?.services?.lmstudio}
            links={[{ label: "UI", href: baseLM }]}
          />
          <Row label="Blender" svc={diag?.services?.blender} />
          <div className="flex items-center gap-2 mb-2">
            <button
              className="px-2 py-0.5 text-xs rounded bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              onClick={async () => {
                try {
                  setBlenderBusy(true);
                  const r = await fetch("/api/blender/ensure", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ set_env: true }),
                  });
                  if (r.ok) setBlenderResult(await r.json());
                  await load();
                } finally {
                  setBlenderBusy(false);
                }
              }}
              disabled={blenderBusy}
              title="Detect and verify Blender in headless mode"
            >
              Ensure Blender
            </button>
            {blenderBusy && (
              <span className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            )}
            <button
              className="px-2 py-0.5 text-xs rounded bg-purple-100 text-purple-700 hover:bg-purple-200 dark:bg-purple-900/30 dark:text-purple-300"
              onClick={async () => {
                try {
                  setVrmBusy(true);
                  setVrmMessage(null);
                  const r = await fetch("/api/blender/install-vrm-addon", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({}),
                  });
                  const js = await r.json();
                  setVrmMessage(js?.message || (r.ok ? "Installed" : "Failed"));
                  await load();
                } finally {
                  setVrmBusy(false);
                }
              }}
              disabled={vrmBusy}
              title="Download and install VRM add-on"
            >
              Install VRM Add‑on
            </button>
            {vrmBusy && (
              <span className="w-3 h-3 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
            )}
          </div>
          {blenderResult && (
            <div
              className={`text-xs p-2 rounded ${blenderResult.ok ? "bg-green-50 text-green-700 border border-green-200" : "bg-red-50 text-red-700 border border-red-200"}`}
            >
              <div className="font-medium">{blenderResult.message}</div>
              {blenderResult.path && (
                <div>
                  Path: <code>{blenderResult.path}</code>
                </div>
              )}
              {blenderResult.version && (
                <div>Version: {blenderResult.version}</div>
              )}
              {!!blenderResult.suggestions?.length && (
                <ul className="mt-1 list-disc pl-5">
                  {blenderResult.suggestions.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
          {vrmMessage && (
            <div className="text-xs p-2 rounded bg-purple-50 text-purple-700 border border-purple-200">
              {vrmMessage}
            </div>
          )}
          <Row
            label="Context7 MCP"
            svc={diag?.services?.context7_mcp}
            links={[
              {
                label: "Health",
                href:
                  diag?.services?.context7_mcp?.message?.split(" ")[0] || "",
              },
            ]}
          />
          <div className="border-b border-gray-200 dark:border-gray-800 py-2">
            <Row label="Vector Snippet DB" svc={diag?.services?.vector_db} />
            <div className="mt-2 flex items-center gap-2">
              <button
                className="px-2 py-1 text-xs rounded bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-300"
                onClick={async () => {
                  if (!confirm("Clear all entries from vector DB?")) return;
                  const r = await fetch("/api/vector-db/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ confirm: true, remove_file: false }),
                  });
                  if (r.ok) await load();
                }}
              >
                Clear Entries
              </button>
              <button
                className="px-2 py-1 text-xs rounded bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-300"
                onClick={async () => {
                  if (
                    !confirm("Recreate vector DB file (will delete the file)?")
                  )
                    return;
                  const r = await fetch("/api/vector-db/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ confirm: true, remove_file: true }),
                  });
                  if (r.ok) await load();
                }}
              >
                Recreate DB File
              </button>
            </div>
          </div>

          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Blender MCP Control
              </div>
              <div className="flex items-center gap-2 text-xs">
                <button
                  className="px-2 py-0.5 rounded bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-300"
                  onClick={async () => {
                    try {
                      setMcpBusy(true);
                      const r = await fetch("/api/mcp/start", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({}),
                      });
                      if (r.ok) await load();
                    } finally {
                      setMcpBusy(false);
                    }
                  }}
                  disabled={mcpBusy}
                  title="Start MCP via backend script"
                >
                  Run MCP
                </button>
                <button
                  className="px-2 py-0.5 rounded bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-300"
                  onClick={async () => {
                    try {
                      setMcpBusy(true);
                      const r = await fetch("/api/mcp/stop", {
                        method: "POST",
                      });
                      if (r.ok) await load();
                    } finally {
                      setMcpBusy(false);
                    }
                  }}
                  disabled={mcpBusy}
                  title="Stop MCP process by PID file"
                >
                  Stop MCP
                </button>
              </div>
            </div>

            <div className="text-xs text-gray-600 dark:text-gray-300 mb-2">
              Command hint:
            </div>
            <div className="border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-950 p-2 text-xs font-mono select-all">
              ./setup/start_blender_mcp.sh &gt; blender_mcp.log 2&gt;&amp;1
              &amp;
            </div>
          </div>

          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Blender MCP Logs
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                {logs?.path ? (
                  <span className="truncate max-w-[260px]" title={logs.path}>
                    Path: {logs.path}
                  </span>
                ) : (
                  <span>No log file detected</span>
                )}
                <button
                  className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
                  onClick={loadLogs}
                  title="Refresh logs"
                >
                  Refresh
                </button>
              </div>
            </div>
            <div className="border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-950 p-2 max-h-64 overflow-auto text-xs font-mono whitespace-pre-wrap">
              {logsLoading
                ? "Loading..."
                : logs?.exists
                  ? logs.lines.join("")
                  : "No logs found. Set BLENDER_MCP_LOG_PATH or start MCP with logs redirected to blender_mcp.log"}
            </div>
          </div>

          <div className="mt-4">
            <a
              href={`${baseAPI}/api/diagnostics`}
              className="text-xs underline text-blue-600 dark:text-blue-400"
              target="_blank"
              rel="noreferrer"
            >
              Open raw diagnostics JSON
            </a>
          </div>
        </div>
      </div>
    </div>
  );

  return content;
}

import React, { useEffect, useMemo, useState } from "react";
import { Layout } from "@/components/Layout/Layout";
import { ChatInterface } from "@/components/Chat/ChatInterface";
import { ThreeViewer } from "@/components/Viewer3D/ThreeViewer";
import { AssetManager } from "@/components/Assets/AssetManager";
import { Button } from "@/components/ui/Button";
import { useSessionStore } from "@/stores/sessionStore";
import {
  EyeIcon,
  ChatBubbleLeftRightIcon,
  FolderOpenIcon,
} from "@heroicons/react/24/outline";
import { useUIStore } from "@/stores/uiStore";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { BackendStatus } from "@/components/ui/BackendStatus";
import { DiagnosticsDrawer } from "@/components/ui/DiagnosticsDrawer";

function App() {
  const initialView = useMemo<"chat" | "split" | "3d" | "assets">(() => {
    try {
      const saved = localStorage.getItem("viewMode") as
        | "chat"
        | "split"
        | "3d"
        | "assets"
        | null;
      if (saved) return saved;
    } catch {}
    const isWide =
      typeof window !== "undefined" ? window.innerWidth >= 1024 : true;
    return isWide ? "split" : "chat";
  }, []);

  const [viewMode, setViewMode] = useState<"chat" | "split" | "3d" | "assets">(
    initialView,
  );
  const { currentSession, isGenerating, progressMessages, loadSession, loadProgressMessages, loadStepControl } = useSessionStore();
  const [diagOpen, setDiagOpen] = useState(false);
  const { applyTheme } = useUIStore();

  useEffect(() => {
    applyTheme();
  }, [applyTheme]);

  // Handle loading session from URL parameter
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('sessionId');
    
    if (sessionId && !currentSession) {
      console.log('🔗 Loading session from URL parameter:', sessionId);
      loadSession(sessionId).then(async () => {
        console.log('✅ Session loaded from URL, now loading progress messages...');
        await loadProgressMessages(sessionId);
        await loadStepControl(sessionId);
        console.log('✅ Progress messages and step control loaded');
      }).catch((error) => {
        console.error('❌ Failed to load session from URL:', error);
      });
    }
  }, [loadSession, loadProgressMessages, loadStepControl, currentSession]);

  useEffect(() => {
    try {
      localStorage.setItem("viewMode", viewMode);
    } catch {}
  }, [viewMode]);

  // Get the latest asset from current session
  const latestAsset =
    currentSession?.artifacts?.[currentSession.artifacts.length - 1];

  // Handle asset selection from AssetManager
  const handleAssetSelect = (asset: any) => {
    // Could integrate with the current session or 3D viewer
    console.log("Asset selected:", asset);
  };

  // Handle edit completion from AssetManager
  const handleEditComplete = (sessionId: string) => {
    // Switch to chat view to see the editing progress
    setViewMode("split");
  };

  return (
    <Layout>
      <div className="flex-1 flex flex-col" data-testid="app-root">
        {/* Header */}
        <header className="flex items-center justify-between p-4 bg-white border-b border-gray-200 dark:bg-gray-800 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              3D Asset Generator
            </h2>
            {currentSession && (
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Session: {currentSession.session_id.slice(-8)}
              </div>
            )}
          </div>

          {/* View mode controls + theme */}
          <div className="flex items-center gap-3">
            <div
              className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg"
              data-testid="view-mode-controls"
            >
              <Button
                variant={viewMode === "chat" ? "primary" : "ghost"}
                size="sm"
                onClick={() => setViewMode("chat")}
                leftIcon={<ChatBubbleLeftRightIcon className="w-4 h-4" />}
              >
                Chat
              </Button>
              <Button
                variant={viewMode === "split" ? "primary" : "ghost"}
                size="sm"
                onClick={() => setViewMode("split")}
              >
                Split
              </Button>
              <Button
                variant={viewMode === "assets" ? "primary" : "ghost"}
                size="sm"
                onClick={() => setViewMode("assets")}
                leftIcon={<FolderOpenIcon className="w-4 h-4" />}
              >
                Assets
              </Button>
              <Button
                variant={viewMode === "3d" ? "primary" : "ghost"}
                size="sm"
                onClick={() => setViewMode("3d")}
                leftIcon={<EyeIcon className="w-4 h-4" />}
                disabled={!latestAsset}
              >
                3D View
              </Button>
            </div>
            <BackendStatus onOpenDetails={() => setDiagOpen(true)} />
            <ThemeToggle />
          </div>
        </header>

        {/* Thin global progress bar under header */}
        {isGenerating && (
          <div className="h-0.5 bg-gray-200 dark:bg-gray-800">
            {(() => {
              const latest = progressMessages[progressMessages.length - 1];
              const pct = latest?.progress_percent ?? 5;
              return (
                <div
                  className="h-0.5 bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                />
              );
            })()}
          </div>
        )}

        <DiagnosticsDrawer open={diagOpen} onClose={() => setDiagOpen(false)} />

        {/* Main content area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Asset Manager */}
          {viewMode === "assets" && (
            <div className="flex-1">
              <AssetManager
                onAssetSelect={handleAssetSelect}
                onEditComplete={handleEditComplete}
              />
            </div>
          )}

          {/* Chat Interface */}
          {(viewMode === "chat" || viewMode === "split") && (
            <div
              className={`flex flex-col bg-white dark:bg-gray-900 ${
                viewMode === "split"
                  ? "w-1/2 border-r border-gray-200 dark:border-gray-700"
                  : "flex-1"
              }`}
            >
              <ChatInterface />
            </div>
          )}

          {/* 3D Viewer */}
          {(viewMode === "3d" || viewMode === "split") && (
            <div
              className={`bg-gray-100 dark:bg-gray-800 ${
                viewMode === "split" ? "w-1/2" : "flex-1"
              }`}
            >
              {latestAsset ? (
                <ThreeViewer
                  assetUrl={latestAsset.relative_path}
                  assetType={getAssetType(latestAsset.relative_path)}
                  showControls={true}
                  autoRotate={false}
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <EyeIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                      No 3D Asset Available
                    </h3>
                    <p className="text-gray-500 dark:text-gray-400 mb-4">
                      Generate a 3D asset to see it in the viewer
                    </p>
                    <ul className="text-sm text-gray-500 dark:text-gray-400 mb-4 space-y-1">
                      <li>• Live 3D preview with orbit controls</li>
                      <li>• Switch between solid and wireframe modes</li>
                      <li>• View and download exported formats</li>
                    </ul>
                    <Button
                      variant="primary"
                      onClick={() => setViewMode("chat")}
                      leftIcon={<ChatBubbleLeftRightIcon className="w-4 h-4" />}
                    >
                      Start Chatting
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Status bar */}
        <footer className="bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 px-4 py-2">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-4">
              <span>Ready to generate 3D assets</span>
              {currentSession && (
                <>
                  <span>•</span>
                  <span>
                    {currentSession.traces.filter((t) => t.end_time).length} of{" "}
                    {currentSession.traces.length} steps completed
                  </span>
                </>
              )}
            </div>
            <div className="flex items-center gap-4">
              <span>Supported formats: GLTF, OBJ, BLEND</span>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full" />
                <span>Online</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </Layout>
  );
}

// Helper function to determine asset type from file path
function getAssetType(filePath: string): string {
  const extension = filePath.split(".").pop()?.toLowerCase();
  switch (extension) {
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

export default App;

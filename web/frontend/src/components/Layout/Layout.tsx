import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  CubeIcon,
  FolderOpenIcon,
  Cog6ToothIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  PlusIcon,
} from "@heroicons/react/24/outline";
import { useSessionStore } from "@/stores/sessionStore";
import { SessionBrowser } from "@/components/Session/SessionBrowser";
import { WorkflowSteps } from "@/components/Workflow/WorkflowSteps";
import { Button } from "@/components/ui/Button";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import type { SessionDetail } from "@/types";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sessionBrowserOpen, setSessionBrowserOpen] = useState(false);
  const { currentSession, traces, startNewSession } = useSessionStore();

  const handleSelectSession = (session: SessionDetail) => {
    // This would typically update the current session state
    console.log("Selected session:", session.session_id);
    setSessionBrowserOpen(false);
  };

  const handleStartNewSession = () => {
    startNewSession();
    // Optional: show a confirmation or notification
    console.log("Started new session");
  };

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 dark:text-gray-100">
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: sidebarOpen ? 320 : 0 }}
        className="bg-white border-r border-gray-200 flex flex-col overflow-hidden dark:bg-gray-800 dark:border-gray-700"
      >
        <div className="flex-1 overflow-y-auto">
          {/* Header */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                  <CubeIcon className="w-5 h-5 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  LL3M
                </h1>
              </div>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(false)}
                className="p-1"
              >
                <ChevronLeftIcon className="w-4 h-4" />
              </Button>
            </div>

            {/* Actions */}
            <div className="space-y-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSessionBrowserOpen(true)}
                leftIcon={<FolderOpenIcon className="w-4 h-4" />}
                className="w-full justify-start"
              >
                Browse Sessions
              </Button>

              <Button
                variant="ghost"
                size="sm"
                leftIcon={<Cog6ToothIcon className="w-4 h-4" />}
                className="w-full justify-start"
              >
                Settings
              </Button>

              {/* Theme toggle */}
              <div className="pt-1">
                <ThemeToggle />
              </div>
            </div>
          </div>

          {/* Current Session Info */}
          {currentSession && (
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="mb-3">
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-1">
                  Current Session
                </h3>
                <p className="text-xs text-gray-600 dark:text-gray-300 line-clamp-2">
                  {currentSession.prompt}
                </p>
              </div>

              <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>ID: {currentSession.session_id.slice(-8)}</span>
                <span className="capitalize">
                  {currentSession.current_phase.replace("_", " ")}
                </span>
              </div>
            </div>
          )}

          {/* Workflow Steps */}
          {traces && traces.length > 0 && (
            <div className="p-4">
              <WorkflowSteps
                traces={traces}
                currentPhase={currentSession?.current_phase}
              />
            </div>
          )}

          {/* Placeholder when no session */}
          {!currentSession && (
            <div className="p-4 text-center">
              <CubeIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                No active session
              </p>
              <Button
                variant="primary"
                size="sm"
                onClick={handleStartNewSession}
                className="w-full"
                leftIcon={<PlusIcon className="w-4 h-4" />}
              >
                Start New Session
              </Button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900">
          <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
            <div className="flex items-center justify-between">
              <span>Version</span>
              <span className="font-mono">1.0.0</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Status</span>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full" />
                <span>Connected</span>
              </div>
            </div>
          </div>
        </div>
      </motion.aside>

      {/* Sidebar toggle button when collapsed */}
      {!sidebarOpen && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setSidebarOpen(true)}
          className="absolute top-4 left-4 z-10 p-2 bg-white dark:bg-gray-800 shadow-md"
        >
          <ChevronRightIcon className="w-4 h-4" />
        </Button>
      )}

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">{children}</main>

      {/* Session Browser Modal */}
      <SessionBrowser
        isOpen={sessionBrowserOpen}
        onClose={() => setSessionBrowserOpen(false)}
        onSelectSession={handleSelectSession}
      />
    </div>
  );
}

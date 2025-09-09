import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ClockIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  UserIcon,
  CogIcon,
  LightBulbIcon,
  QuestionMarkCircleIcon,
} from "@heroicons/react/24/outline";
import { cn } from "@/utils";

interface SubstepProgress {
  substep_id: string;
  name: string;
  status: "pending" | "in_progress" | "completed" | "failed" | "skipped";
  progress_percent: number;
  start_time?: string;
  end_time?: string;
  technical_details?: Record<string, any>;
}

interface InteractionPoint {
  interaction_id: string;
  phase: string;
  question: string;
  interaction_type: string;
  options: string[];
  created_at: string;
  timeout_seconds?: number;
  is_resolved: boolean;
  response?: string;
  resolved_at?: string;
}

interface AgentReflection {
  reflection_id: string;
  agent_name: string;
  phase: string;
  timestamp: string;
  reflection_text: string;
  confidence_score: number;
  decision_made: string;
  reasoning: string;
  alternative_approaches: string[];
}

interface ProgressMessage {
  timestamp: string;
  phase: string;
  title: string;
  description: string;
  progress_percent: number;
  estimated_remaining_seconds?: number;
  technical_details?: Record<string, any>;
  substeps?: SubstepProgress[];
  interaction_points?: InteractionPoint[];
  agent_reflections?: AgentReflection[];
}

interface ProgressDisplayProps {
  messages: ProgressMessage[];
  currentPhase: string;
  isGenerating: boolean;
  className?: string;
}

export function ProgressDisplay({
  messages,
  currentPhase,
  isGenerating,
  className,
}: ProgressDisplayProps) {
  if (!messages.length && !isGenerating) {
    return null;
  }

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case "completion":
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case "error":
        return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />;
      case "paused":
        return <ClockIcon className="w-5 h-5 text-yellow-500" />;
      default:
        return (
          <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        );
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case "completion":
        return "bg-green-50 border-green-200";
      case "error":
        return "bg-red-50 border-red-200";
      case "paused":
        return "bg-yellow-50 border-yellow-200";
      default:
        return "bg-blue-50 border-blue-200";
    }
  };

  const formatTimeRemaining = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const latestMessage = messages[messages.length - 1];
  const [expandedSubsteps, setExpandedSubsteps] = useState<string>("");
  const [expandedReflections, setExpandedReflections] = useState<string>("");

  const getSubstepIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircleIcon className="w-4 h-4 text-green-500" />;
      case "failed":
        return <ExclamationTriangleIcon className="w-4 h-4 text-red-500" />;
      case "in_progress":
        return (
          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        );
      case "skipped":
        return <div className="w-4 h-4 bg-gray-300 rounded-full" />;
      default:
        return <div className="w-4 h-4 bg-gray-200 rounded-full" />;
    }
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return "text-green-600 bg-green-50";
    if (score >= 0.5) return "text-yellow-600 bg-yellow-50";
    return "text-red-600 bg-red-50";
  };

  const renderSubsteps = (substeps: SubstepProgress[], messageId: string) => {
    if (!substeps?.length) return null;

    const isExpanded = expandedSubsteps === messageId;

    return (
      <div className="mt-3 border-t border-gray-200 pt-3">
        <button
          onClick={() => setExpandedSubsteps(isExpanded ? "" : messageId)}
          className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-800 transition-colors w-full text-left"
        >
          {isExpanded ? (
            <ChevronDownIcon className="w-4 h-4" />
          ) : (
            <ChevronRightIcon className="w-4 h-4" />
          )}
          <CogIcon className="w-4 h-4" />
          <span>
            Substeps ({substeps.length}) -{" "}
            {substeps.filter((s) => s.status === "completed").length} completed
          </span>
        </button>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-2 space-y-2"
            >
              {substeps.map((substep, index) => (
                <motion.div
                  key={substep.substep_id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center space-x-3 p-2 bg-gray-50 rounded-lg"
                >
                  <div className="flex-shrink-0">
                    {getSubstepIcon(substep.status)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-800 truncate">
                      {substep.name}
                    </div>
                    <div className="text-xs text-gray-500">
                      {substep.status} • {substep.progress_percent.toFixed(1)}%
                      complete
                    </div>
                  </div>
                  <div className="flex-shrink-0">
                    <div className="w-16 bg-gray-200 rounded-full h-1.5">
                      <div
                        className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${substep.progress_percent}%` }}
                      />
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  const renderInteractionPoints = (interactions: InteractionPoint[]) => {
    if (!interactions?.length) return null;

    const pendingInteractions = interactions.filter((i) => !i.is_resolved);

    return (
      <div className="mt-3 space-y-2">
        {pendingInteractions.map((interaction) => (
          <motion.div
            key={interaction.interaction_id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="p-3 bg-amber-50 border border-amber-200 rounded-lg"
          >
            <div className="flex items-start space-x-3">
              <QuestionMarkCircleIcon className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-amber-800 mb-2">
                  {interaction.question}
                </p>
                {interaction.options.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {interaction.options.map((option, index) => (
                      <button
                        key={index}
                        className="px-3 py-1 text-xs bg-amber-100 text-amber-700 rounded-full hover:bg-amber-200 transition-colors"
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    );
  };

  const renderAgentReflections = (
    reflections: AgentReflection[],
    messageId: string,
  ) => {
    if (!reflections?.length) return null;

    const isExpanded = expandedReflections === messageId;

    return (
      <div className="mt-3 border-t border-gray-200 pt-3">
        <button
          onClick={() => setExpandedReflections(isExpanded ? "" : messageId)}
          className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-800 transition-colors w-full text-left"
        >
          {isExpanded ? (
            <ChevronDownIcon className="w-4 h-4" />
          ) : (
            <ChevronRightIcon className="w-4 h-4" />
          )}
          <LightBulbIcon className="w-4 h-4" />
          <span>Agent Reflections ({reflections.length})</span>
        </button>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-2 space-y-2"
            >
              {reflections.map((reflection, index) => (
                <motion.div
                  key={reflection.reflection_id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="p-3 bg-slate-50 rounded-lg border border-slate-200"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-slate-200 rounded-full flex items-center justify-center">
                        <span className="text-xs font-medium text-slate-600">
                          {reflection.agent_name.charAt(0).toUpperCase()}
                        </span>
                      </div>
                      <span className="text-sm font-medium text-slate-800">
                        {reflection.agent_name}
                      </span>
                    </div>
                    <div
                      className={cn(
                        "px-2 py-1 rounded-full text-xs font-medium",
                        getConfidenceColor(reflection.confidence_score),
                      )}
                    >
                      {(reflection.confidence_score * 100).toFixed(0)}%
                      confident
                    </div>
                  </div>

                  <div className="text-sm text-slate-700 mb-2">
                    <strong>Decision:</strong> {reflection.decision_made}
                  </div>

                  <div className="text-sm text-slate-600">
                    {reflection.reflection_text}
                  </div>

                  {reflection.reasoning && (
                    <div className="mt-2 text-xs text-slate-500">
                      <strong>Reasoning:</strong> {reflection.reasoning}
                    </div>
                  )}
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("space-y-3", className)}
    >
      {/* Current Progress */}
      {latestMessage && (
        <motion.div
          key={latestMessage.timestamp}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className={cn(
            "p-4 rounded-xl border shadow-sm",
            getPhaseColor(latestMessage.phase),
          )}
        >
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-0.5">
              {getPhaseIcon(latestMessage.phase)}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-1">
                <h4 className="text-sm font-semibold text-gray-900">
                  {latestMessage.title}
                </h4>
                {latestMessage.estimated_remaining_seconds && (
                  <div className="flex items-center space-x-1 text-xs text-gray-500">
                    <ClockIcon className="w-3 h-3" />
                    <span>
                      ~
                      {formatTimeRemaining(
                        latestMessage.estimated_remaining_seconds,
                      )}
                    </span>
                  </div>
                )}
              </div>

              <p className="text-sm text-gray-600 mb-3">
                {latestMessage.description}
              </p>

              {/* Progress Bar */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-gray-500">
                  <span>Overall Progress</span>
                  <span>{latestMessage.progress_percent.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <motion.div
                    className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${latestMessage.progress_percent}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
              </div>

              {/* Enhanced Features */}
              {renderSubsteps(
                latestMessage.substeps || [],
                latestMessage.timestamp,
              )}
              {renderInteractionPoints(latestMessage.interaction_points || [])}
              {renderAgentReflections(
                latestMessage.agent_reflections || [],
                latestMessage.timestamp,
              )}

              {/* Technical Details (Expandable) */}
              {latestMessage.technical_details &&
                Object.keys(latestMessage.technical_details).length > 0 && (
                  <details className="mt-3">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                      Technical Details
                    </summary>
                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono text-gray-600 max-h-32 overflow-y-auto">
                      {Object.entries(latestMessage.technical_details).map(
                        ([key, value]) => (
                          <div key={key} className="truncate">
                            <span className="text-gray-500">{key}:</span>{" "}
                            {JSON.stringify(value)}
                          </div>
                        ),
                      )}
                    </div>
                  </details>
                )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Progress History (Collapsed by default) */}
      {messages.length > 1 && (
        <details className="group">
          <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700 flex items-center space-x-2">
            <InformationCircleIcon className="w-4 h-4" />
            <span>
              View Progress History ({messages.length - 1} previous steps)
            </span>
          </summary>

          <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
            <AnimatePresence>
              {messages
                .slice(0, -1)
                .reverse()
                .map((message, index) => (
                  <motion.div
                    key={message.timestamp}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center space-x-3 p-2 bg-gray-50 rounded-lg text-sm"
                  >
                    <div className="w-3 h-3 bg-gray-300 rounded-full flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-gray-700 truncate">
                        {message.title}
                      </div>
                      <div className="text-gray-500 text-xs truncate">
                        {message.description}
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 flex-shrink-0">
                      {message.progress_percent.toFixed(0)}%
                    </div>
                  </motion.div>
                ))}
            </AnimatePresence>
          </div>
        </details>
      )}
    </motion.div>
  );
}

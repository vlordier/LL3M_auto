import React from "react";
import { motion } from "framer-motion";
import {
  LightBulbIcon,
  CodeBracketIcon,
  CubeIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/outline";
import type { ExecutionTrace, SessionPhase } from "@/types";
import { formatRelativeTime } from "@/utils";

interface WorkflowStepsProps {
  traces: ExecutionTrace[];
  currentPhase?: SessionPhase;
  className?: string;
}

const phaseConfig = {
  prompt_analysis: {
    icon: LightBulbIcon,
    label: "Prompt Analysis",
    description: "Understanding your requirements",
    color: "blue",
  },
  code_generation: {
    icon: CodeBracketIcon,
    label: "Code Generation",
    description: "Writing Blender scripts",
    color: "purple",
  },
  asset_generation: {
    icon: CubeIcon,
    label: "Asset Generation",
    description: "Creating 3D models",
    color: "green",
  },
} as const;

function getStepStatus(
  trace: ExecutionTrace,
): "pending" | "running" | "completed" | "error" {
  if (trace.error_message) return "error";
  if (trace.end_time) return "completed";
  if (trace.start_time) return "running";
  return "pending";
}

function StepIcon({ phase, status }: { phase: SessionPhase; status: string }) {
  const config = phaseConfig[phase];
  const IconComponent = config.icon;

  if (status === "completed") {
    return <CheckCircleIcon className="w-5 h-5 text-green-600" />;
  }

  if (status === "error") {
    return <ExclamationTriangleIcon className="w-5 h-5 text-red-600" />;
  }

  if (status === "running") {
    return (
      <div className="relative">
        <IconComponent className={`w-5 h-5 text-${config.color}-600`} />
        <div className="absolute -inset-1 border-2 border-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  return <IconComponent className={`w-5 h-5 text-gray-400`} />;
}

export function WorkflowSteps({
  traces,
  currentPhase,
  className = "",
}: WorkflowStepsProps) {
  const sortedTraces = [...traces].sort(
    (a, b) =>
      new Date(a.start_time).getTime() - new Date(b.start_time).getTime(),
  );

  return (
    <div className={`space-y-4 ${className}`} data-testid="workflow-steps">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">
          Generation Pipeline
        </h3>
        {currentPhase && (
          <div className="flex items-center gap-2 text-sm text-blue-600">
            <ClockIcon className="w-4 h-4" />
            <span>Current: {phaseConfig[currentPhase]?.label}</span>
          </div>
        )}
      </div>

      <div className="relative">
        {/* Progress line */}
        <div className="absolute left-6 top-8 bottom-0 w-0.5 bg-gray-200" />

        <div className="space-y-6">
          {sortedTraces.map((trace, index) => {
            const status = getStepStatus(trace);
            const config = phaseConfig[trace.phase];
            const isLast = index === sortedTraces.length - 1;

            return (
              <motion.div
                key={trace.trace_id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="relative flex items-start gap-4"
              >
                {/* Step indicator */}
                <div
                  className={`relative z-10 w-12 h-12 rounded-full border-4 flex items-center justify-center ${
                    status === "completed"
                      ? "bg-green-50 border-green-200"
                      : status === "running"
                        ? "bg-blue-50 border-blue-200"
                        : status === "error"
                          ? "bg-red-50 border-red-200"
                          : "bg-gray-50 border-gray-200"
                  }`}
                >
                  <StepIcon phase={trace.phase} status={status} />
                </div>

                {/* Step content */}
                <div className="flex-1 min-w-0 pb-6">
                  <div className="flex items-center justify-between mb-2">
                    <h4
                      className={`font-medium ${
                        status === "error" ? "text-red-900" : "text-gray-900"
                      }`}
                    >
                      {config.label}
                    </h4>
                    <div className="flex items-center gap-3 text-sm text-gray-500">
                      {trace.execution_time && (
                        <span>{trace.execution_time.toFixed(1)}s</span>
                      )}
                      <span>{formatRelativeTime(trace.start_time)}</span>
                    </div>
                  </div>

                  <p className="text-sm text-gray-600 mb-3">
                    {config.description}
                  </p>

                  {/* Status and details */}
                  {status === "running" && (
                    <div className="flex items-center gap-2 text-sm text-blue-600">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                      <span>In progress...</span>
                    </div>
                  )}

                  {status === "completed" && trace.outputs && (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm text-green-600 mb-2">
                        <CheckCircleIcon className="w-4 h-4" />
                        <span>Completed successfully</span>
                      </div>

                      {/* Show key outputs */}
                      <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                        {Object.entries(trace.outputs).map(([key, value]) => (
                          <div key={key} className="text-xs">
                            <span className="font-mono text-gray-600">
                              {key}:
                            </span>
                            <span className="ml-2 text-gray-800">
                              {typeof value === "string"
                                ? value
                                : JSON.stringify(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {status === "error" && trace.error_message && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                      <div className="flex items-center gap-2 text-sm text-red-600 mb-2">
                        <ExclamationTriangleIcon className="w-4 h-4" />
                        <span>Error occurred</span>
                      </div>
                      <p className="text-sm text-red-800">
                        {trace.error_message}
                      </p>
                    </div>
                  )}

                  {/* Code version indicator */}
                  {trace.phase === "code_generation" && (
                    <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
                      <CodeBracketIcon className="w-3 h-3" />
                      <span>Code version generated</span>
                    </div>
                  )}

                  {/* Render preview placeholder */}
                  {trace.phase === "asset_generation" &&
                    status === "completed" && (
                      <div className="mt-3" data-testid="step-preview">
                        <div className="w-full h-32 bg-gradient-to-br from-gray-100 to-gray-200 rounded-lg flex items-center justify-center">
                          <div className="text-center">
                            <CubeIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                            <p className="text-sm text-gray-500">
                              Preview render
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                </div>

                {/* Progress line segment */}
                {!isLast && (
                  <div
                    className={`absolute left-6 top-12 w-0.5 h-6 ${
                      status === "completed"
                        ? "bg-green-300"
                        : status === "running"
                          ? "bg-blue-300"
                          : "bg-gray-200"
                    }`}
                  />
                )}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Overall progress indicator */}
      <div className="mt-6 bg-gray-100 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-900">
            Overall Progress
          </span>
          <span className="text-sm text-gray-600">
            {traces.filter((t) => getStepStatus(t) === "completed").length} /{" "}
            {traces.length} steps
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-500"
            style={{
              width: `${(traces.filter((t) => getStepStatus(t) === "completed").length / traces.length) * 100}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
}

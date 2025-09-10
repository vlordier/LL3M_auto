import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  CpuChipIcon,
  CheckCircleIcon,
  PencilIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { useSessionStore } from "@/stores/sessionStore";

interface Question {
  id: string;
  text: string;
  options?: string[];
  type?: 'choice' | 'text' | 'boolean';
}

interface ReformulationData {
  reformulated_request: string;
  improvements?: string;
  questions?: string | Question[];
  user_confirmation_required?: boolean;
  summary?: string;
  tags?: string[];
}

interface ReformulationMessageProps {
  data: ReformulationData;
  sessionId: string;
}

export function ReformulationMessage({ data, sessionId }: ReformulationMessageProps) {
  const { confirmReformulation } = useSessionStore();
  const [modifyInput, setModifyInput] = useState("");
  const [isModifyInputOpen, setIsModifyInputOpen] = useState(false);
  const [isHandled, setIsHandled] = useState(false);

  const handleConfirmation = async (
    action: "accept" | "modify" | "reject",
    modifiedPrompt?: string
  ) => {
    try {
      setIsHandled(true);
      await confirmReformulation(sessionId, action, modifiedPrompt);
      if (action === "modify") {
        setIsModifyInputOpen(false);
        setModifyInput("");
      }
    } catch (error) {
      console.error("Failed to confirm reformulation:", error);
      setIsHandled(false);
    }
  };

  const handleModifySubmit = () => {
    if (modifyInput.trim()) {
      handleConfirmation("modify", modifyInput.trim());
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-3 justify-start"
      data-testid="reformulation-message"
    >
      {/* Assistant Avatar */}
      <div className="flex-shrink-0">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-sm">
          <CpuChipIcon className="w-5 h-5 text-white" />
        </div>
      </div>

      {/* Message Content */}
      <div className="max-w-[85%] space-y-3">
        {/* Main reformulation bubble */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-4 border border-blue-100 dark:border-blue-800/50 shadow-lg hover:shadow-xl transition-shadow duration-200">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                ✨ I've enhanced your request
              </span>
            </div>
            <div className="text-xs text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-800/30 px-2 py-1 rounded-full">
              Enhanced
            </div>
          </div>
          
          <div className="space-y-3">
            {/* Reformulated request */}
            <div className="bg-white dark:bg-gray-800/50 rounded-lg p-3 border border-blue-200/50 dark:border-blue-700/50">
              <div className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2">
                Enhanced Request:
              </div>
              <div className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                {data.reformulated_request}
              </div>
            </div>

            {/* Improvements */}
            {data.improvements && (
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 border border-green-200 dark:border-green-700/50">
                <div className="text-sm text-green-800 dark:text-green-200">
                  <span className="font-medium flex items-center gap-2 mb-2">
                    <span className="text-green-500">✅</span>
                    Improvements Made:
                  </span>
                  <div className="text-sm leading-relaxed pl-6">
                    {data.improvements}
                  </div>
                </div>
              </div>
            )}

            {/* Summary */}
            {data.summary && (
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-700/50">
                <div className="text-sm text-blue-800 dark:text-blue-200">
                  <span className="font-medium flex items-center gap-2 mb-2">
                    <span className="text-blue-500">📝</span>
                    Summary:
                  </span>
                  <div className="text-sm leading-relaxed pl-6">
                    {data.summary}
                  </div>
                </div>
              </div>
            )}

            {/* Tags */}
            {data.tags && data.tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {data.tags.map((tag, index) => (
                  <span 
                    key={index}
                    className="px-2 py-1 text-xs bg-indigo-100 dark:bg-indigo-800/30 text-indigo-700 dark:text-indigo-300 rounded-full border border-indigo-200 dark:border-indigo-700 font-medium"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            )}

            {/* Questions - Structured or Legacy */}
            {data.questions && (
              <div className="space-y-2">
                {Array.isArray(data.questions) ? (
                  // Structured questions
                  <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3 border border-amber-200 dark:border-amber-700/50">
                    <div className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-2 flex items-center gap-2">
                      <span className="text-amber-500">❓</span>
                      Questions to refine your request:
                    </div>
                    <div className="space-y-3">
                      {data.questions.map((question, index) => (
                        <div key={question.id || index} className="bg-white dark:bg-gray-800/50 rounded-lg p-3 border border-amber-200/50 dark:border-amber-700/50">
                          <div className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2">
                            {index + 1}. {question.text}
                          </div>
                          {question.options && question.options.length > 0 && (
                            <div className="flex flex-wrap gap-2 mt-2">
                              {question.options.map((option, optIndex) => (
                                <span 
                                  key={optIndex}
                                  className="px-2 py-1 text-xs bg-amber-100 dark:bg-amber-800/30 text-amber-700 dark:text-amber-300 rounded-full border border-amber-200 dark:border-amber-700"
                                >
                                  {option}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  // Legacy text questions
                  <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3 border border-amber-200 dark:border-amber-700/50">
                    <div className="text-sm text-amber-800 dark:text-amber-200">
                      <span className="font-medium flex items-center gap-2 mb-2">
                        <span className="text-amber-500">❓</span>
                        Questions:
                      </span>
                      <div className="text-sm leading-relaxed pl-6">
                        {data.questions}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Action buttons */}
          {data.user_confirmation_required && !isHandled && (
            <div className="mt-4 space-y-3">
              <div className="text-xs text-blue-600 dark:text-blue-400 font-medium">
                How would you like to proceed?
              </div>
              
              <div className="flex flex-wrap gap-2">
                <button 
                  onClick={() => handleConfirmation("accept")}
                  className="inline-flex items-center gap-1.5 px-3 py-2 text-sm bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors shadow-sm font-medium"
                >
                  <CheckCircleIcon className="w-4 h-4" />
                  Use Enhanced Version
                </button>
                
                <button 
                  onClick={() => setIsModifyInputOpen(!isModifyInputOpen)}
                  className="inline-flex items-center gap-1.5 px-3 py-2 text-sm bg-amber-600 hover:bg-amber-700 text-white rounded-lg transition-colors shadow-sm font-medium"
                >
                  <PencilIcon className="w-4 h-4" />
                  Customize Further
                </button>
                
                <button 
                  onClick={() => handleConfirmation("reject")}
                  className="inline-flex items-center gap-1.5 px-3 py-2 text-sm bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors shadow-sm font-medium"
                >
                  <XMarkIcon className="w-4 h-4" />
                  Use Original
                </button>
              </div>

              {/* Custom input */}
              {isModifyInputOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="space-y-2"
                >
                  <textarea
                    value={modifyInput}
                    onChange={(e) => setModifyInput(e.target.value)}
                    placeholder="Describe your specific changes or additions..."
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={3}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                        e.preventDefault();
                        handleModifySubmit();
                      }
                    }}
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={handleModifySubmit}
                      disabled={!modifyInput.trim()}
                      className="px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors font-medium"
                    >
                      Submit Changes
                    </button>
                    <button
                      onClick={() => {
                        setIsModifyInputOpen(false);
                        setModifyInput("");
                      }}
                      className="px-3 py-1.5 text-sm bg-gray-300 hover:bg-gray-400 text-gray-700 rounded-lg transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Tip: Press Cmd/Ctrl + Enter to submit
                  </div>
                </motion.div>
              )}
            </div>
          )}

          {/* Handled state */}
          {isHandled && (
            <div className="mt-3 flex items-center gap-2 text-sm text-green-700 dark:text-green-300">
              <CheckCircleIcon className="w-4 h-4" />
              <span>Request confirmed and processing...</span>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <div className="text-xs text-gray-500 dark:text-gray-400">
          {new Date().toLocaleTimeString()}
        </div>
      </div>
    </motion.div>
  );
}
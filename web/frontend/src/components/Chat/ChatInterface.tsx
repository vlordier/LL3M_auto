import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PaperAirplaneIcon, StopIcon } from '@heroicons/react/24/outline';
import { useSessionStore } from '@/stores/sessionStore';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { EmptyState } from './EmptyState';
import { StepControls } from './StepControls';
import { ProgressDisplay } from './ProgressDisplay';
import { cn } from '@/utils';

interface ChatInterfaceProps {
  className?: string;
}

export function ChatInterface({ className }: ChatInterfaceProps) {
  const {
    messages,
    isGenerating,
    error,
    generateAsset,
    amendSession,
    currentSession,
    setIsGenerating,
    progressMessages,
    stepControl,
    pauseSession,
    resumeSession,
    stepSession,
    loadProgressMessages,
    loadStepControl,
  } = useSessionStore();
  
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Keyboard shortcut: Cmd/Ctrl+K focuses the prompt
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().includes('MAC');
      if ((isMac ? e.metaKey : e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Load progress data when session changes
  useEffect(() => {
    if (currentSession?.session_id) {
      loadProgressMessages(currentSession.session_id);
      loadStepControl(currentSession.session_id);
    }
  }, [currentSession?.session_id, loadProgressMessages, loadStepControl]);

  const handleSend = async () => {
    if (!input.trim() || isGenerating) return;

    const message = input.trim();
    setInput('');

    if (currentSession && !currentSession.success) {
      // If we have a current session that failed or is incomplete, treat as amendment
      await amendSession(currentSession.session_id, message);
    } else if (currentSession && currentSession.success) {
      // If we have a successful session, treat as amendment/refinement
      await amendSession(currentSession.session_id, message);
    } else {
      // New generation
      await generateAsset(message);
    }
  };

  const handleStop = async () => {
    try {
      if (currentSession?.session_id) {
        await pauseSession(currentSession.session_id);
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={cn('flex flex-col h-full bg-white dark:bg-gray-900', className)}>
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-gray-200 bg-gray-50 dark:bg-gray-900 dark:border-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">LL3M Assistant</h2>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Generate and refine 3D assets with natural language
            </p>
          </div>
          {isGenerating && (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-3 py-1.5 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors dark:bg-red-900/30 dark:text-red-300 dark:hover:bg-red-900/40"
            >
              <StopIcon className="w-4 h-4" />
              Cancel
            </button>
          )}
          {!isGenerating && stepControl?.can_resume && currentSession?.session_id && (
            <button
              onClick={async () => {
                await resumeSession(currentSession.session_id!);
                setIsGenerating(true);
              }}
              className="flex items-center gap-2 px-3 py-1.5 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors dark:bg-green-900/30 dark:text-green-300 dark:hover:bg-green-900/40"
            >
              ▶️ Resume
            </button>
          )}
        </div>
        {isGenerating && (
          <div className="mt-3" data-testid="progress-bar">
            {(() => {
              const latest = progressMessages[progressMessages.length - 1];
              const pct = latest?.progress_percent ?? 5;
              return (
                <div>
                  <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                    <span>{latest?.title || 'Working...'}</span>
                    <span>{pct.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-indigo-600 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                    />
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4" data-testid="messages-container">
        {messages.length === 0 ? (
          <EmptyState onExampleClick={setInput} />
        ) : (
          <AnimatePresence initial={false}>
            {messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <ChatMessage message={message} />
              </motion.div>
            ))}
          </AnimatePresence>
        )}
        
        {/* Progress Display - Enhanced loading with detailed progress */}
        <ProgressDisplay
          messages={progressMessages}
          currentPhase={currentSession?.current_phase || ''}
          isGenerating={isGenerating}
        />

        {/* Step Controls */}
        {stepControl && currentSession && (
          <StepControls
            isGenerating={isGenerating}
            executionMode={stepControl.execution_mode}
            currentStep={stepControl.current_step}
            totalSteps={stepControl.total_steps}
            canPause={stepControl.can_pause}
            canResume={stepControl.can_resume}
            canStep={stepControl.can_step}
            awaitingUserAction={stepControl.awaiting_user_action}
            onPause={() => pauseSession(currentSession.session_id)}
            onResume={() => resumeSession(currentSession.session_id)}
            onStep={() => stepSession(currentSession.session_id)}
          />
        )}

        {/* Error display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-red-50 border border-red-200 rounded-lg dark:bg-red-900/20 dark:border-red-800"
          >
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="flex-shrink-0 px-4 py-4 border-t border-gray-200 bg-white dark:bg-gray-900 dark:border-gray-800">
        <ChatInput
          ref={inputRef}
          value={input}
          onChange={setInput}
          onSend={handleSend}
          onKeyPress={handleKeyPress}
          disabled={isGenerating}
          placeholder={
            currentSession && currentSession.success
              ? "Describe how you'd like to refine this asset..."
              : currentSession && !currentSession.success
              ? "Let me know what went wrong or how to improve..."
              : "Describe the 3D asset you want to create..."
          }
        />
        
        {/* Suggestions */}
        {messages.length === 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {[
              "Create a wooden chair with carved details",
              "Design a futuristic spaceship",
              "Make a realistic tree with autumn leaves",
              "Build a modern coffee table",
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setInput(suggestion)}
                className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

import React from 'react';
import { motion } from 'framer-motion';
import { 
  PlayIcon, 
  PauseIcon, 
  ForwardIcon,
  ArrowPathIcon 
} from '@heroicons/react/24/outline';
import { cn } from '@/utils';

interface StepControlsProps {
  isGenerating: boolean;
  executionMode: 'continuous' | 'step_by_step' | 'paused' | 'stepping';
  currentStep: number;
  totalSteps: number;
  canPause: boolean;
  canResume: boolean;
  canStep: boolean;
  awaitingUserAction: boolean;
  onPause: () => void;
  onResume: () => void;
  onStep: () => void;
  className?: string;
}

export function StepControls({ 
  isGenerating,
  executionMode,
  currentStep,
  totalSteps,
  canPause,
  canResume,
  canStep,
  awaitingUserAction,
  onPause,
  onResume,
  onStep,
  className 
}: StepControlsProps) {
  if (!isGenerating && !awaitingUserAction) {
    return null;
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className={cn(
        "bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-4 shadow-sm",
        className
      )}
    >
      <div className="flex items-center justify-between">
        {/* Progress Info */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-gray-700">
              Step {currentStep} of {totalSteps}
            </span>
          </div>
          
          <div className="text-sm text-gray-500">
            {executionMode === 'continuous' && '🔄 Running automatically'}
            {executionMode === 'paused' && '⏸️ Paused'}
            {executionMode === 'stepping' && '👣 Step-by-step mode'}
            {executionMode === 'step_by_step' && '🎯 Manual stepping'}
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center space-x-2">
          {/* Pause Button */}
          {canPause && executionMode === 'continuous' && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onPause}
              className="flex items-center space-x-1 px-3 py-1.5 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
              disabled={!canPause}
            >
              <PauseIcon className="w-4 h-4" />
              <span>Pause</span>
            </motion.button>
          )}

          {/* Resume Button */}
          {canResume && (executionMode === 'paused' || awaitingUserAction) && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onResume}
              className="flex items-center space-x-1 px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
              disabled={!canResume}
            >
              <PlayIcon className="w-4 h-4" />
              <span>Resume</span>
            </motion.button>
          )}

          {/* Step Button */}
          {canStep && (executionMode === 'paused' || executionMode === 'stepping' || awaitingUserAction) && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onStep}
              className="flex items-center space-x-1 px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
              disabled={!canStep}
            >
              <ForwardIcon className="w-4 h-4" />
              <span>Next Step</span>
            </motion.button>
          )}

          {/* Running indicator */}
          {executionMode === 'continuous' && !canPause && (
            <div className="flex items-center space-x-1 px-3 py-1.5 bg-gray-100 text-gray-600 rounded-lg text-sm">
              <ArrowPathIcon className="w-4 h-4 animate-spin" />
              <span>Running...</span>
            </div>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mt-3">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Generation Progress</span>
          <span>{Math.round((currentStep / totalSteps) * 100)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div 
            className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${(currentStep / totalSteps) * 100}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Help Text */}
      {awaitingUserAction && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-3 p-2 bg-blue-50 rounded-lg border border-blue-200"
        >
          <p className="text-xs text-blue-700">
            💡 <strong>Generation is paused.</strong> Use the controls above to resume automatic generation or step through one phase at a time.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}
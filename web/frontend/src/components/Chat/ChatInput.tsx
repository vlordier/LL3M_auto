import React, { forwardRef } from 'react';
import { PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { cn } from '@/utils';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onKeyPress: (e: React.KeyboardEvent) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export const ChatInput = forwardRef<HTMLTextAreaElement, ChatInputProps>(
  ({ value, onChange, onSend, onKeyPress, disabled, placeholder, className }, ref) => {
    const canSend = value.trim().length > 0 && !disabled;

    return (
      <div className={cn('relative', className)}>
        <div className="flex items-end gap-3 p-3 border border-gray-300 rounded-2xl bg-white shadow-sm focus-within:border-primary-500 focus-within:ring-1 focus-within:ring-primary-500 transition-all dark:border-gray-700 dark:bg-gray-800">
          <textarea
            ref={ref}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyPress={onKeyPress}
            placeholder={placeholder || "Describe the 3D asset you want to create..."}
            disabled={disabled}
            rows={1}
            className={cn(
              'flex-1 resize-none border-none outline-none bg-transparent',
              'text-gray-900 placeholder-gray-500 dark:text-gray-100 dark:placeholder-gray-400',
              'min-h-[24px] max-h-32 overflow-y-auto',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
            style={{
              height: 'auto',
            }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = 'auto';
              target.style.height = Math.min(target.scrollHeight, 128) + 'px';
            }}
          />
          
          <button
            onClick={onSend}
            disabled={!canSend}
            className={cn(
              'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all',
              canSend
                ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-sm'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed dark:bg-gray-700 dark:text-gray-500'
            )}
            aria-label="Send"
          >
            <PaperAirplaneIcon className="w-4 h-4" />
          </button>
        </div>

        {/* Character count */}
        {value.length > 0 && (
          <div className="absolute -bottom-5 right-0 text-xs text-gray-500 dark:text-gray-400">
            {value.length}/2000
          </div>
        )}
      </div>
    );
  }
);

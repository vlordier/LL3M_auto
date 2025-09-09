import React from 'react';
import { cn } from '@/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'secondary' | 'white' | 'gray';
  className?: string;
  text?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  color = 'primary',
  className,
  text
}) => {
  const sizes = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-8 h-8 border-3',
    xl: 'w-12 h-12 border-4'
  };

  const colors = {
    primary: 'border-primary-600 border-t-transparent',
    secondary: 'border-gray-600 border-t-transparent',
    white: 'border-white border-t-transparent',
    gray: 'border-gray-400 border-t-transparent'
  };

  const textSizes = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
    xl: 'text-lg'
  };

  if (text) {
    return (
      <div className={cn('flex items-center gap-3', className)}>
        <div
          className={cn(
            'animate-spin rounded-full',
            sizes[size],
            colors[color]
          )}
        />
        <span className={cn('text-gray-600 font-medium', textSizes[size])}>
          {text}
        </span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'animate-spin rounded-full',
        sizes[size],
        colors[color],
        className
      )}
    />
  );
};

export { LoadingSpinner };
export type { LoadingSpinnerProps };
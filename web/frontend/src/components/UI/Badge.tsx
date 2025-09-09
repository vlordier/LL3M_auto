import React from 'react';
import { cn } from '@/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'sm' | 'md' | 'lg';
  pill?: boolean;
}

const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  size = 'md',
  pill = false,
  className,
  children,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium text-center transition-colors';
  
  const variants = {
    default: 'bg-gray-100 text-gray-800',
    primary: 'bg-primary-100 text-primary-800',
    secondary: 'bg-purple-100 text-purple-800',
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    danger: 'bg-red-100 text-red-800',
    info: 'bg-blue-100 text-blue-800'
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-sm'
  };

  return (
    <span
      className={cn(
        baseStyles,
        variants[variant],
        sizes[size],
        pill ? 'rounded-full' : 'rounded',
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};

export { Badge };
export type { BadgeProps };
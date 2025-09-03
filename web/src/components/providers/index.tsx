'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ThemeProvider } from 'next-themes';
import { useState } from 'react';

import { AuthProvider } from './auth-provider';
import { SocketProvider } from './socket-provider';

interface ProvidersProps {
  children: React.ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  // Create a client with sensible defaults
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 minute
            gcTime: 5 * 60 * 1000, // 5 minutes
            retry: (failureCount, error: any) => {
              // Don't retry on 4xx errors (except 408, 429)
              if (error?.response?.status >= 400 && error?.response?.status < 500) {
                if (error.response.status === 408 || error.response.status === 429) {
                  return failureCount < 3;
                }
                return false;
              }

              // Retry up to 3 times for other errors
              return failureCount < 3;
            },
            retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
          },
          mutations: {
            retry: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider
        attribute=\"data-theme\"
        defaultTheme=\"system\"
        enableSystem
        disableTransitionOnChange
        storageKey=\"ll3m-theme\"
      >
        <AuthProvider>
          <SocketProvider>
            {children}
          </SocketProvider>
        </AuthProvider>
      </ThemeProvider>

      <ReactQueryDevtools
        initialIsOpen={false}
        position=\"bottom-right\"
        buttonPosition=\"bottom-right\"
      />
    </QueryClientProvider>
  );
}

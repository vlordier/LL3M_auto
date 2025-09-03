import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Toaster } from 'react-hot-toast';

import { Providers } from '@/components/providers';
import { Header } from '@/components/layout/header';
import { Footer } from '@/components/layout/footer';

import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'LL3M - Large Language 3D Modelers',
  description: 'Generate stunning 3D assets from text prompts using AI',
  keywords: ['3D modeling', 'AI', 'Blender', 'asset generation', 'text to 3D'],
  authors: [{ name: 'LL3M Team' }],
  creator: 'LL3M',
  publisher: 'LL3M',
  robots: 'index, follow',
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#0ea5e9',

  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://ll3m.com',
    siteName: 'LL3M',
    title: 'LL3M - Large Language 3D Modelers',
    description: 'Generate stunning 3D assets from text prompts using AI',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'LL3M - Generate 3D Assets with AI',
      },
    ],
  },

  twitter: {
    card: 'summary_large_image',
    title: 'LL3M - Large Language 3D Modelers',
    description: 'Generate stunning 3D assets from text prompts using AI',
    images: ['/og-image.jpg'],
    creator: '@ll3m',
  },

  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang=\"en\" suppressHydrationWarning>
      <body className={`${inter.className} min-h-screen bg-gray-50 dark:bg-gray-900`}>
        <Providers>
          <div className=\"flex min-h-screen flex-col\">
            <Header />

            <main className=\"flex-1\">
              {children}
            </main>

            <Footer />
          </div>

          <Toaster
            position=\"top-right\"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'var(--toast-bg)',
                color: 'var(--toast-color)',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#ffffff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#ffffff',
                },
              },
            }}
          />
        </Providers>
      </body>
    </html>
  );
}

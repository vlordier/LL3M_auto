import Link from 'next/link';
import { ArrowRight, Zap, Shield, Globe, Sparkles } from 'lucide-react';

import { Hero } from '@/components/home/hero';
import { Features } from '@/components/home/features';
import { HowItWorks } from '@/components/home/how-it-works';
import { Examples } from '@/components/home/examples';
import { CTA } from '@/components/home/cta';

export default function HomePage() {
  return (
    <div className=\"min-h-screen\">
      {/* Hero Section */}
      <Hero />

      {/* Features Section */}
      <Features />

      {/* How It Works */}
      <HowItWorks />

      {/* Examples Gallery */}
      <Examples />

      {/* Call to Action */}
      <CTA />
    </div>
  );
}

// Hero component for homepage
function HeroSection() {
  return (
    <section className=\"relative overflow-hidden bg-gradient-to-br from-primary-50 via-white to-accent-50 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800\">
      {/* Background decoration */}
      <div className=\"absolute inset-0 bg-grid-pattern opacity-5\"></div>
      <div className=\"absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-primary-200 rounded-full blur-3xl opacity-20\"></div>

      <div className=\"relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-32\">
        <div className=\"text-center\">
          {/* Badge */}
          <div className=\"inline-flex items-center gap-2 bg-primary-100 dark:bg-primary-900/50 text-primary-800 dark:text-primary-200 px-4 py-2 rounded-full text-sm font-medium mb-8 animate-fade-in\">
            <Sparkles className=\"w-4 h-4\" />
            <span>Now with GPT-4 Vision and Advanced Blender Integration</span>
            <ArrowRight className=\"w-4 h-4\" />
          </div>

          {/* Headline */}
          <h1 className=\"text-4xl sm:text-6xl lg:text-7xl font-bold text-gray-900 dark:text-white mb-8 animate-slide-up\">
            Generate Stunning{' '}
            <span className=\"gradient-text\">3D Assets</span>
            <br />
            from Text Prompts
          </h1>

          {/* Subheadline */}
          <p className=\"text-xl sm:text-2xl text-gray-600 dark:text-gray-300 mb-12 max-w-3xl mx-auto animate-slide-up text-balance\">
            Transform your ideas into professional 3D models using the power of Large Language Models and Blender. No 3D modeling experience required.
          </p>

          {/* CTA Buttons */}
          <div className=\"flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up\">
            <Link
              href=\"/auth/register\"
              className=\"btn-primary px-8 py-4 text-lg interactive\"
            >
              <Zap className=\"w-5 h-5 mr-2\" />
              Start Creating for Free
            </Link>

            <Link
              href=\"/examples\"
              className=\"btn-secondary px-8 py-4 text-lg interactive\"
            >
              View Examples
            </Link>
          </div>

          {/* Trust indicators */}
          <div className=\"mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 items-center opacity-60\">
            <div className=\"center-col\">
              <Shield className=\"w-8 h-8 text-primary-600 mb-2\" />
              <span className=\"text-sm font-medium\">Enterprise Security</span>
            </div>
            <div className=\"center-col\">
              <Globe className=\"w-8 h-8 text-primary-600 mb-2\" />
              <span className=\"text-sm font-medium\">Global CDN</span>
            </div>
            <div className=\"center-col\">
              <Zap className=\"w-8 h-8 text-primary-600 mb-2\" />
              <span className=\"text-sm font-medium\">Lightning Fast</span>
            </div>
            <div className=\"center-col\">
              <Sparkles className=\"w-8 h-8 text-primary-600 mb-2\" />
              <span className=\"text-sm font-medium\">AI Powered</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

import React from 'react';
import { motion } from 'framer-motion';
import { CubeIcon, LightBulbIcon, SparklesIcon } from '@heroicons/react/24/outline';

interface EmptyStateProps {
  onExampleClick: (prompt: string) => void;
}

const examples = [
  {
    icon: CubeIcon,
    title: "Simple Objects",
    prompt: "Create a wooden chair with curved armrests",
    description: "Basic furniture and everyday items"
  },
  {
    icon: SparklesIcon,
    title: "Fantasy & Sci-Fi",
    prompt: "Design a glowing crystal sword with intricate details",
    description: "Magical items and futuristic objects"
  },
  {
    icon: LightBulbIcon,
    title: "Architectural",
    prompt: "Build a modern glass house with a rooftop garden",
    description: "Buildings and structures"
  },
];

export function EmptyState({ onExampleClick }: EmptyStateProps) {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-2xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-primary-500 to-primary-700 rounded-2xl flex items-center justify-center">
            <CubeIcon className="w-8 h-8 text-white" />
          </div>
          
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-3">
            Welcome to LL3M
          </h2>
          
          <p className="text-gray-600 dark:text-gray-300 mb-8 text-lg">
            Generate stunning 3D assets using natural language. 
            Describe what you want to create and watch it come to life.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {examples.map((example, index) => (
            <motion.button
              key={example.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 + index * 0.1 }}
              onClick={() => onExampleClick(example.prompt)}
              className="group p-4 text-left bg-white border border-gray-200 rounded-xl hover:border-primary-300 hover:shadow-lg transition-all duration-200 dark:bg-gray-800 dark:border-gray-700"
            >
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center group-hover:bg-primary-100 transition-colors dark:bg-gray-700">
                  <example.icon className="w-5 h-5 text-gray-600 group-hover:text-primary-600 dark:text-gray-200" />
                </div>
                
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                    {example.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                    {example.description}
                  </p>
                  <div className="text-xs text-primary-600 font-medium bg-primary-50 rounded-lg px-2 py-1 dark:bg-primary-900/20 dark:text-primary-300">
                    "{example.prompt}"
                  </div>
                </div>
              </div>
            </motion.button>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="text-sm text-gray-500 dark:text-gray-400 space-y-2"
        >
          <p>
            <strong>Tips:</strong> Be specific about materials, colors, and style.
            You can also refine and iterate on generated assets.
          </p>
          <p>
            Supported formats: BLEND, OBJ, GLTF • Real-time 3D preview • High-quality renders
          </p>
        </motion.div>
      </div>
    </div>
  );
}

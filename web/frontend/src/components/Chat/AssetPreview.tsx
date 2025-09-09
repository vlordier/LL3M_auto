import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { EyeIcon, CubeIcon } from '@heroicons/react/24/outline';
import { getAssetUrl } from '@/utils';

interface AssetPreviewProps {
  previewUrl: string;
  assetPath?: string;
  success: boolean;
  onView3D?: () => void;
}

export function AssetPreview({ previewUrl, assetPath, success, onView3D }: AssetPreviewProps) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  if (!success) {
    return (
      <div className="w-full h-32 bg-red-50 border-2 border-red-200 border-dashed rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 text-sm">Generation failed</div>
          <div className="text-red-400 text-xs mt-1">No preview available</div>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative w-full bg-gray-900 rounded-lg overflow-hidden"
      data-testid="asset-preview"
    >
      {/* Image preview */}
      <div className="relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900">
        {!imageError && (
          <img
            src={getAssetUrl(previewUrl)}
            alt="Generated 3D asset preview"
            className={`w-full h-full object-cover transition-opacity duration-300 ${
              imageLoaded ? 'opacity-100' : 'opacity-0'
            }`}
            onLoad={() => setImageLoaded(true)}
            onError={() => {
              setImageError(true);
              setImageLoaded(true);
            }}
          />
        )}
        
        {/* Loading/error state */}
        {!imageLoaded || imageError ? (
          <div className="absolute inset-0 flex items-center justify-center">
            {imageError ? (
              <div className="text-center">
                <CubeIcon className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                <div className="text-gray-400 text-sm">Preview not available</div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-gray-400">
                <div className="animate-spin w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full" />
                <span className="text-sm">Loading preview...</span>
              </div>
            )}
          </div>
        ) : null}

        {/* Overlay controls */}
        {imageLoaded && !imageError && (
          <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center opacity-0 hover:opacity-100">
            <div className="flex gap-2">
              <button
                onClick={() => window.open(getAssetUrl(previewUrl), '_blank')}
                className="px-3 py-1.5 bg-white text-gray-900 rounded-lg text-sm font-medium hover:bg-gray-100 flex items-center gap-1.5 transition-colors"
              >
                <EyeIcon className="w-4 h-4" />
                View Full Size
              </button>
              
              {assetPath && onView3D && (
                <button
                  onClick={onView3D}
                  className="px-3 py-1.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 flex items-center gap-1.5 transition-colors"
                >
                  <CubeIcon className="w-4 h-4" />
                  3D View
                </button>
              )}
            </div>
          </div>
        )}

        {/* Quality badge */}
        <div className="absolute top-3 left-3">
          <div className="px-2 py-1 bg-black bg-opacity-60 text-white text-xs rounded-full">
            HD Preview
          </div>
        </div>

        {/* Success indicator */}
        <div className="absolute top-3 right-3">
          <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
        </div>
      </div>

      {/* Asset info bar */}
      <div className="px-3 py-2 bg-gray-800 text-white text-xs flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse-subtle" />
          <span>Asset generated successfully</span>
        </div>
        {assetPath && (
          <span className="text-gray-400">
            {assetPath.split('.').pop()?.toUpperCase()} format
          </span>
        )}
      </div>
    </motion.div>
  );
}

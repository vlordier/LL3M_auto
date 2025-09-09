import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PlusIcon,
  FolderOpenIcon,
  Squares2X2Icon,
  ListBulletIcon
} from '@heroicons/react/24/outline';
import { AssetImporter } from './AssetImporter';
import { AssetBrowser } from './AssetBrowser';
import { AssetEditor } from './AssetEditor';
import { Button } from '@/components/UI/Button';
import { cn } from '@/utils';
import type { AssetMetadata } from '@/types';

interface AssetManagerProps {
  onAssetSelect?: (asset: AssetMetadata) => void;
  onEditComplete?: (sessionId: string) => void;
  className?: string;
}

type ViewMode = 'browser' | 'import' | 'edit';

export function AssetManager({ onAssetSelect, onEditComplete, className }: AssetManagerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('browser');
  const [selectedAsset, setSelectedAsset] = useState<AssetMetadata | null>(null);

  const handleImportComplete = (assetId: string) => {
    // Switch to browser view after successful import
    setViewMode('browser');
    // The asset will appear in the browser automatically
  };

  const handleAssetEdit = (asset: AssetMetadata) => {
    setSelectedAsset(asset);
    setViewMode('edit');
  };

  const handleEditComplete = (sessionId: string) => {
    setViewMode('browser');
    setSelectedAsset(null);
    if (onEditComplete) {
      onEditComplete(sessionId);
    }
  };

  const handleCloseEditor = () => {
    setViewMode('browser');
    setSelectedAsset(null);
  };

  return (
    <div className={cn('h-full flex flex-col bg-white dark:bg-gray-900', className)}>
      {/* Header with Navigation */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Asset Manager
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Import, preview, and edit 3D assets
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              onClick={() => setViewMode('browser')}
              variant={viewMode === 'browser' ? 'primary' : 'ghost'}
              size="sm"
              className="flex items-center gap-2"
            >
              <FolderOpenIcon className="w-4 h-4" />
              Browse
            </Button>
            
            <Button
              onClick={() => setViewMode('import')}
              variant={viewMode === 'import' ? 'primary' : 'ghost'}
              size="sm"
              className="flex items-center gap-2"
            >
              <PlusIcon className="w-4 h-4" />
              Import
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0">
        <AnimatePresence mode="wait">
          {viewMode === 'browser' && (
            <motion.div
              key="browser"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              <AssetBrowser
                onAssetSelect={onAssetSelect}
                onAssetEdit={handleAssetEdit}
              />
            </motion.div>
          )}

          {viewMode === 'import' && (
            <motion.div
              key="import"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="h-full overflow-y-auto"
            >
              <div className="max-w-2xl mx-auto p-6">
                <AssetImporter
                  onImportComplete={handleImportComplete}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Asset Editor Modal */}
      <AnimatePresence>
        {viewMode === 'edit' && selectedAsset && (
          <AssetEditor
            asset={selectedAsset}
            onClose={handleCloseEditor}
            onEditComplete={handleEditComplete}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
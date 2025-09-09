import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PaperAirplaneIcon,
  ArrowDownTrayIcon,
  EyeIcon,
  CogIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { useAssetStore } from '@/stores/assetStore';
import { useSessionStore } from '@/stores/sessionStore';
import { ThreeViewer } from '@/components/Viewer3D/ThreeViewer';
import { Button } from '@/components/UI/Button';
import { Badge } from '@/components/UI/Badge';
import { cn } from '@/utils';
import type { AssetMetadata, EditAssetRequest } from '@/types';

interface AssetEditorProps {
  asset: AssetMetadata;
  onClose: () => void;
  onEditComplete?: (sessionId: string) => void;
  className?: string;
}

export function AssetEditor({ asset, onClose, onEditComplete, className }: AssetEditorProps) {
  const [editInstruction, setEditInstruction] = useState('');
  const [preserveMaterials, setPreserveMaterials] = useState(true);
  const [exportFormats, setExportFormats] = useState(['blend', 'gltf']);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  const { editAsset, isEditing, error: assetError } = useAssetStore();
  const { 
    addMessage, 
    generateAsset, 
    isGenerating,
    error: sessionError 
  } = useSessionStore();

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const availableFormats = [
    { id: 'blend', label: 'Blender (.blend)', description: 'Native Blender format' },
    { id: 'gltf', label: 'glTF (.gltf/.glb)', description: 'Web-optimized 3D format' },
    { id: 'obj', label: 'Wavefront (.obj)', description: 'Universal 3D format' },
    { id: 'fbx', label: 'Autodesk FBX', description: 'Animation-friendly format' },
    { id: 'dae', label: 'COLLADA (.dae)', description: 'Open standard format' },
    { id: 'stl', label: 'STL', description: '3D printing format' }
  ];

  const formatAssetUrl = (filePath: string) => {
    // Convert absolute path to API URL
    if (filePath.includes('/outputs/')) {
      const relativePath = filePath.split('/outputs/')[1];
      return `/api/assets/${relativePath}`;
    }
    return `/api/assets/${filePath}`;
  };

  const handleFormatToggle = (formatId: string) => {
    setExportFormats(prev => 
      prev.includes(formatId)
        ? prev.filter(f => f !== formatId)
        : [...prev, formatId]
    );
  };

  const handleSubmit = async () => {
    if (!editInstruction.trim()) return;

    setIsSubmitting(true);

    try {
      // Add user message to chat
      addMessage({
        id: `edit-${Date.now()}`,
        type: 'user',
        content: `Edit "${asset.asset_name}": ${editInstruction}`,
        timestamp: new Date(),
        artifacts: [asset]
      });

      // Create edit request
      const editRequest: EditAssetRequest = {
        edit_instruction: editInstruction,
        preserve_materials: preserveMaterials,
        export_formats: exportFormats
      };

      // Start the editing process
      const result = await editAsset(asset.asset_id, editRequest);

      // Add assistant response
      addMessage({
        id: `edit-response-${Date.now()}`,
        type: 'assistant',
        content: `I'm editing "${asset.asset_name}" based on your instructions. This will create a new version with your requested changes.`,
        timestamp: new Date(),
        session_id: result.session_id,
        generation_result: result
      });

      // Clear the input
      setEditInstruction('');

      // Notify parent component
      if (onEditComplete && result.session_id) {
        onEditComplete(result.session_id);
      }

      // Close editor
      onClose();

    } catch (error) {
      console.error('Edit failed:', error);
      addMessage({
        id: `edit-error-${Date.now()}`,
        type: 'system',
        content: `Failed to edit asset: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const suggestions = [
    "Make it twice as large",
    "Change the color to blue",
    "Add more geometric detail",
    "Simplify the mesh topology",
    "Apply a metallic material",
    "Add procedural textures",
    "Create a low-poly version",
    "Improve the lighting setup"
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={cn('fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4', className)}
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="bg-white dark:bg-gray-900 rounded-lg shadow-xl w-full max-w-6xl h-5/6 flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex-shrink-0 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                  Edit Asset
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {asset.asset_name}
                </p>
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant="secondary">{asset.asset_type.toUpperCase()}</Badge>
                {asset.tags.map(tag => (
                  <Badge key={tag} variant="outline" size="sm">{tag}</Badge>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                onClick={() => setShowAdvanced(!showAdvanced)}
                variant="ghost"
                size="sm"
                className="flex items-center gap-2"
              >
                <CogIcon className="w-4 h-4" />
                {showAdvanced ? 'Hide' : 'Show'} Options
              </Button>
              
              <Button onClick={onClose} variant="ghost" size="sm">
                <XMarkIcon className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex min-h-0">
          {/* Left Panel: 3D Preview */}
          <div className="flex-1 min-w-0 bg-gray-50 dark:bg-gray-800">
            <ThreeViewer
              assetUrl={formatAssetUrl(asset.file_path)}
              assetType={asset.asset_type as any}
              showControls={true}
              autoRotate={false}
            />
          </div>

          {/* Right Panel: Edit Interface */}
          <div className="w-96 flex flex-col bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700">
            {/* Asset Info */}
            <div className="flex-shrink-0 p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Size:</span>
                  <span className="font-medium dark:text-gray-200">
                    {(asset.size_bytes / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Created:</span>
                  <span className="font-medium dark:text-gray-200">
                    {new Date(asset.created_at).toLocaleDateString()}
                  </span>
                </div>
                {asset.description && (
                  <div>
                    <p className="text-gray-600 dark:text-gray-400 mb-1">Description:</p>
                    <p className="text-sm dark:text-gray-200">{asset.description}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Advanced Options */}
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="border-b border-gray-200 dark:border-gray-700 overflow-hidden"
                >
                  <div className="p-4 space-y-4">
                    {/* Preserve Materials */}
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Preserve Materials
                      </label>
                      <input
                        type="checkbox"
                        checked={preserveMaterials}
                        onChange={(e) => setPreserveMaterials(e.target.checked)}
                        className="w-4 h-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500"
                      />
                    </div>

                    {/* Export Formats */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Export Formats
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        {availableFormats.map(format => (
                          <label
                            key={format.id}
                            className="flex items-center gap-2 p-2 border border-gray-200 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              checked={exportFormats.includes(format.id)}
                              onChange={() => handleFormatToggle(format.id)}
                              className="w-3 h-3 text-primary-600 rounded"
                            />
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-medium dark:text-gray-200">{format.label}</p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                                {format.description}
                              </p>
                            </div>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Edit Instructions */}
            <div className="flex-1 flex flex-col min-h-0">
              <div className="p-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Edit Instructions
                </label>
                <textarea
                  ref={inputRef}
                  value={editInstruction}
                  onChange={(e) => setEditInstruction(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Describe how you want to modify this asset..."
                  className="w-full h-32 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none dark:bg-gray-800 dark:text-gray-200 dark:placeholder-gray-400"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                  Press Cmd/Ctrl+Enter to submit
                </p>
              </div>

              {/* Quick Suggestions */}
              <div className="px-4 pb-4">
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Quick Suggestions:
                </p>
                <div className="flex flex-wrap gap-2">
                  {suggestions.slice(0, 4).map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => setEditInstruction(suggestion)}
                      className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>

              {/* Error Display */}
              {(assetError || sessionError) && (
                <div className="mx-4 mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                  <p className="text-red-700 dark:text-red-300 text-sm">
                    {assetError || sessionError}
                  </p>
                </div>
              )}

              {/* Submit Button */}
              <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                <Button
                  onClick={handleSubmit}
                  disabled={!editInstruction.trim() || isSubmitting || isEditing}
                  loading={isSubmitting || isEditing}
                  className="w-full flex items-center justify-center gap-2"
                >
                  <PaperAirplaneIcon className="w-4 h-4" />
                  {isSubmitting || isEditing ? 'Processing...' : 'Edit Asset'}
                </Button>
                
                <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-2">
                  This will create a new session for the edited asset
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
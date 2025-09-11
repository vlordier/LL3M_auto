import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MagnifyingGlassIcon,
  FunnelIcon,
  EyeIcon,
  PencilSquareIcon,
  TrashIcon,
  TagIcon,
  CalendarIcon,
  DocumentIcon,
} from "@heroicons/react/24/outline";
import { useAssetStore } from "@/stores/assetStore";
import { ThreeViewer } from "@/components/Viewer3D/ThreeViewer";
import { Button } from "@/components/UI/Button";
import { Badge } from "@/components/UI/Badge";
import { cn } from "@/utils";
import type { AssetMetadata } from "@/types";

interface AssetBrowserProps {
  onAssetSelect?: (asset: AssetMetadata) => void;
  onAssetEdit?: (asset: AssetMetadata) => void;
  className?: string;
}

export function AssetBrowser({
  onAssetSelect,
  onAssetEdit,
  className,
}: AssetBrowserProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAssetType, setSelectedAssetType] = useState<string>("all");
  const [selectedAsset, setSelectedAsset] = useState<AssetMetadata | null>(
    null,
  );
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showPreview, setShowPreview] = useState(false);

  const { assets, loading, error, loadAssets, deleteAsset } = useAssetStore();

  useEffect(() => {
    loadAssets();
  }, [loadAssets]);

  const filteredAssets = assets.filter((asset) => {
    const matchesSearch =
      asset.asset_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      asset.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      asset.tags.some((tag) =>
        tag.toLowerCase().includes(searchQuery.toLowerCase()),
      );

    const matchesType =
      selectedAssetType === "all" || asset.asset_type === selectedAssetType;

    return matchesSearch && matchesType;
  });

  const assetTypes = [
    "all",
    ...new Set(assets.map((asset) => asset.asset_type)),
  ];

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const handlePreview = (asset: AssetMetadata) => {
    setSelectedAsset(asset);
    setShowPreview(true);
  };

  const handleEdit = (asset: AssetMetadata) => {
    if (onAssetEdit) {
      onAssetEdit(asset);
    }
  };

  const handleDelete = async (asset: AssetMetadata) => {
    if (confirm(`Are you sure you want to delete "${asset.asset_name}"?`)) {
      await deleteAsset(asset.asset_id);
      if (selectedAsset?.asset_id === asset.asset_id) {
        setSelectedAsset(null);
        setShowPreview(false);
      }
    }
  };

  const handleSelect = (asset: AssetMetadata) => {
    if (onAssetSelect) {
      onAssetSelect(asset);
    }
  };

  return (
    <div className={cn("flex flex-col h-full bg-white", className)}>
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              Asset Browser
            </h2>
            <p className="text-sm text-gray-600">
              {filteredAssets.length} asset
              {filteredAssets.length !== 1 ? "s" : ""} found
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              onClick={() => setViewMode(viewMode === "grid" ? "list" : "grid")}
              variant="outline"
              size="sm"
            >
              {viewMode === "grid" ? "List" : "Grid"}
            </Button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search assets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          <div className="flex items-center gap-2">
            <FunnelIcon className="w-4 h-4 text-gray-400" />
            <select
              value={selectedAssetType}
              onChange={(e) => setSelectedAssetType(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              {assetTypes.map((type) => (
                <option key={type} value={type}>
                  {type === "all" ? "All Types" : type.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Asset List */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin w-8 h-8 border-2 border-primary-600 border-t-transparent rounded-full" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-red-600 font-medium">{error}</p>
              <Button
                onClick={() => loadAssets()}
                className="mt-2"
                variant="outline"
                size="sm"
              >
                Retry
              </Button>
            </div>
          </div>
        ) : filteredAssets.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <DocumentIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-600 font-medium">No assets found</p>
              <p className="text-gray-500 text-sm">
                {searchQuery || selectedAssetType !== "all"
                  ? "Try adjusting your search or filters"
                  : "Import your first 3D asset to get started"}
              </p>
            </div>
          </div>
        ) : (
          <div
            className={cn(
              "p-6",
              viewMode === "grid"
                ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
                : "space-y-3",
            )}
          >
            <AnimatePresence>
              {filteredAssets.map((asset) => (
                <motion.div
                  key={asset.asset_id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className={cn(
                    "bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer",
                    viewMode === "list" && "flex items-center gap-4 p-4",
                  )}
                  onClick={() => handleSelect(asset)}
                >
                  {viewMode === "grid" ? (
                    <div className="p-4">
                      {/* Asset Preview */}
                      <div className="aspect-square bg-gray-100 rounded-lg mb-3 flex items-center justify-center">
                        {asset.preview_path ? (
                          <img
                            src={`/api/assets/${asset.preview_path}`}
                            alt={asset.asset_name}
                            className="w-full h-full object-cover rounded-lg"
                          />
                        ) : (
                          <DocumentIcon className="w-8 h-8 text-gray-400" />
                        )}
                      </div>

                      {/* Asset Info */}
                      <div className="space-y-2">
                        <div>
                          <h3 className="font-medium text-gray-900 truncate">
                            {asset.asset_name}
                          </h3>
                          <p className="text-xs text-gray-500 uppercase tracking-wide">
                            {asset.asset_type}
                          </p>
                        </div>

                        {asset.description && (
                          <p className="text-sm text-gray-600 line-clamp-2">
                            {asset.description}
                          </p>
                        )}

                        {asset.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {asset.tags.slice(0, 2).map((tag) => (
                              <Badge key={tag} variant="secondary" size="sm">
                                {tag}
                              </Badge>
                            ))}
                            {asset.tags.length > 2 && (
                              <Badge variant="outline" size="sm">
                                +{asset.tags.length - 2}
                              </Badge>
                            )}
                          </div>
                        )}

                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span>{formatFileSize(asset.size_bytes)}</span>
                          <span>{formatDate(asset.created_at)}</span>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-1 mt-3 pt-3 border-t border-gray-100">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePreview(asset);
                          }}
                          variant="ghost"
                          size="sm"
                          className="flex-1"
                        >
                          <EyeIcon className="w-4 h-4 mr-1" />
                          Preview
                        </Button>
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleEdit(asset);
                          }}
                          variant="ghost"
                          size="sm"
                        >
                          <PencilSquareIcon className="w-4 h-4" />
                        </Button>
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(asset);
                          }}
                          variant="ghost"
                          size="sm"
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* List View */}
                      <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center flex-shrink-0">
                        {asset.preview_path ? (
                          <img
                            src={`/api/assets/${asset.preview_path}`}
                            alt={asset.asset_name}
                            className="w-full h-full object-cover rounded-lg"
                          />
                        ) : (
                          <DocumentIcon className="w-6 h-6 text-gray-400" />
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between">
                          <div className="min-w-0 flex-1">
                            <h3 className="font-medium text-gray-900 truncate">
                              {asset.asset_name}
                            </h3>
                            <p className="text-sm text-gray-600 truncate">
                              {asset.description}
                            </p>

                            <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
                              <span className="uppercase tracking-wide">
                                {asset.asset_type}
                              </span>
                              <span>{formatFileSize(asset.size_bytes)}</span>
                              <span className="flex items-center gap-1">
                                <CalendarIcon className="w-3 h-3" />
                                {formatDate(asset.created_at)}
                              </span>
                            </div>

                            {asset.tags.length > 0 && (
                              <div className="flex items-center gap-1 mt-2">
                                <TagIcon className="w-3 h-3 text-gray-400" />
                                <div className="flex flex-wrap gap-1">
                                  {asset.tags.slice(0, 3).map((tag) => (
                                    <Badge
                                      key={tag}
                                      variant="secondary"
                                      size="sm"
                                    >
                                      {tag}
                                    </Badge>
                                  ))}
                                  {asset.tags.length > 3 && (
                                    <Badge variant="outline" size="sm">
                                      +{asset.tags.length - 3}
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>

                          <div className="flex items-center gap-1 ml-4">
                            <Button
                              onClick={(e) => {
                                e.stopPropagation();
                                handlePreview(asset);
                              }}
                              variant="ghost"
                              size="sm"
                            >
                              <EyeIcon className="w-4 h-4" />
                            </Button>
                            <Button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleEdit(asset);
                              }}
                              variant="ghost"
                              size="sm"
                            >
                              <PencilSquareIcon className="w-4 h-4" />
                            </Button>
                            <Button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDelete(asset);
                              }}
                              variant="ghost"
                              size="sm"
                              className="text-red-600 hover:text-red-700 hover:bg-red-50"
                            >
                              <TrashIcon className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Preview Modal */}
      <AnimatePresence>
        {showPreview && selectedAsset && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowPreview(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-white rounded-lg max-w-4xl w-full h-3/4 flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-4 border-b border-gray-200">
                <div>
                  <h3 className="font-semibold text-gray-900">
                    {selectedAsset.asset_name}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {selectedAsset.description}
                  </p>
                </div>
                <Button
                  onClick={() => setShowPreview(false)}
                  variant="ghost"
                  size="sm"
                >
                  ×
                </Button>
              </div>

              <div className="flex-1 min-h-0">
                <ThreeViewer
                  assetUrl={selectedAsset.file_path}
                  assetType={selectedAsset.asset_type}
                  showControls={true}
                  autoRotate={false}
                />
              </div>

              <div className="p-4 border-t border-gray-200 flex justify-end gap-2">
                <Button
                  onClick={() => handleEdit(selectedAsset)}
                  className="flex items-center gap-2"
                >
                  <PencilSquareIcon className="w-4 h-4" />
                  Edit Asset
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

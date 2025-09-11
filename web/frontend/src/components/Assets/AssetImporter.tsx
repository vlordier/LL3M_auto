import React, { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  CloudArrowUpIcon,
  DocumentIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  FolderOpenIcon,
  LinkIcon,
  ComputerDesktopIcon,
} from "@heroicons/react/24/outline";
import { useAssetStore } from "@/stores/assetStore";
import { Button } from "@/components/ui/Button";
import { cn } from "@/utils";

interface AssetImporterProps {
  onImportComplete?: (assetId: string) => void;
  className?: string;
}

type ImportMode = "upload" | "path" | "url";

export function AssetImporter({
  onImportComplete,
  className,
}: AssetImporterProps) {
  const [importMode, setImportMode] = useState<ImportMode>("upload");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "success" | "error"
  >("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePath, setFilePath] = useState("");
  const [fileUrl, setFileUrl] = useState("");
  const [assetName, setAssetName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { importAsset, importAssetFromPath, importAssetFromUrl, error } =
    useAssetStore();

  const allowedTypes = {
    ".gltf": "model/gltf+json",
    ".glb": "application/octet-stream",
    ".vrm": "application/octet-stream",
    ".obj": "text/plain",
    ".blend": "application/octet-stream",
    ".fbx": "application/octet-stream",
    ".dae": "model/vnd.collada+xml",
    ".3ds": "application/octet-stream",
    ".ply": "application/octet-stream",
    ".stl": "application/octet-stream",
  };

  const isFileAllowed = (file: File) => {
    const extension = "." + file.name.split(".").pop()?.toLowerCase();
    return Object.keys(allowedTypes).includes(extension);
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (file: File) => {
    if (!isFileAllowed(file)) {
      setUploadStatus("error");
      return;
    }

    setSelectedFile(file);
    setAssetName(file.name.split(".")[0]);
    setUploadStatus("idle");
  };

  const addTag = () => {
    if (tagInput.trim() && !tags.includes(tagInput.trim())) {
      setTags([...tags, tagInput.trim()]);
      setTagInput("");
    }
  };

  const removeTag = (tagToRemove: string) => {
    setTags(tags.filter((tag) => tag !== tagToRemove));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addTag();
    }
  };

  const handleImport = async () => {
    if (!assetName.trim()) return;

    // Validate input based on import mode
    if (importMode === "upload" && !selectedFile) return;
    if (importMode === "path" && !filePath.trim()) return;
    if (importMode === "url" && !fileUrl.trim()) return;

    setIsUploading(true);
    setUploadStatus("idle");

    try {
      let result;
      const metadata = {
        asset_name: assetName,
        description: description.trim() || undefined,
        tags: tags.length > 0 ? tags : undefined,
      };

      if (importMode === "upload" && selectedFile) {
        const extension =
          "." + selectedFile.name.split(".").pop()?.toLowerCase();
        const assetType = extension.slice(1); // Remove the dot

        result = await importAsset({
          file: selectedFile,
          asset_name: assetName,
          asset_type: assetType,
          description: metadata.description,
          tags: metadata.tags,
        });
      } else if (importMode === "path") {
        const extension = filePath.split(".").pop()?.toLowerCase() || "unknown";
        result = await importAssetFromPath(filePath, {
          ...metadata,
          asset_type: extension,
        });
      } else if (importMode === "url") {
        const extension = fileUrl.split(".").pop()?.toLowerCase() || "unknown";
        result = await importAssetFromUrl(fileUrl, {
          ...metadata,
          asset_type: extension,
        });
      }

      setUploadStatus("success");
      reset();

      if (onImportComplete && result?.asset_id) {
        onImportComplete(result.asset_id);
      }
    } catch (error) {
      console.error("Import failed:", error);
      setUploadStatus("error");
    } finally {
      setIsUploading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setFilePath("");
    setFileUrl("");
    setAssetName("");
    setDescription("");
    setTags([]);
    setTagInput("");
    setUploadStatus("idle");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "bg-white border border-gray-200 rounded-lg shadow-sm",
        className,
      )}
    >
      <div className="p-6">
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Import 3D Asset
          </h3>
          <p className="text-sm text-gray-600">
            Upload, browse, or download a 3D model to preview, edit, and export
            in different formats
          </p>
        </div>

        {/* Import Mode Tabs */}
        <div className="mb-6">
          <nav className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setImportMode("upload")}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors",
                importMode === "upload"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900",
              )}
            >
              <CloudArrowUpIcon className="w-4 h-4" />
              Upload File
            </button>
            <button
              onClick={() => setImportMode("path")}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors",
                importMode === "path"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900",
              )}
            >
              <FolderOpenIcon className="w-4 h-4" />
              File Path
            </button>
            <button
              onClick={() => setImportMode("url")}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors",
                importMode === "url"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900",
              )}
            >
              <LinkIcon className="w-4 h-4" />
              URL Download
            </button>
          </nav>
        </div>

        {/* Import Content Based on Mode */}
        {importMode === "upload" && (
          <div
            className={cn(
              "relative border-2 border-dashed rounded-lg transition-colors mb-6",
              isDragging
                ? "border-primary-400 bg-primary-50"
                : selectedFile
                  ? "border-green-300 bg-green-50"
                  : "border-gray-300 hover:border-gray-400",
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept={Object.keys(allowedTypes).join(",")}
              onChange={handleFileInput}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              disabled={isUploading}
            />

            <div className="p-6 text-center">
              {selectedFile ? (
                <div className="flex items-center justify-center gap-3">
                  <DocumentIcon className="w-8 h-8 text-green-600" />
                  <div className="text-left">
                    <p className="font-medium text-gray-900">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={reset}
                    className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <>
                  <CloudArrowUpIcon className="w-10 h-10 text-gray-400 mx-auto mb-3" />
                  <p className="text-gray-600 font-medium mb-1">
                    Drop your 3D file here or click to browse
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports: GLTF, GLB, VRM, OBJ, BLEND, FBX, DAE, 3DS, PLY,
                    STL
                  </p>
                </>
              )}
            </div>
          </div>
        )}

        {/* File Path Import */}
        {importMode === "path" && (
          <div className="space-y-4 mb-6">
            <div className="p-6 border-2 border-dashed border-gray-300 rounded-lg">
              <div className="text-center">
                <FolderOpenIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-sm text-gray-600 mb-4">
                  Enter the absolute path to your 3D model file
                </p>
                <div className="max-w-md mx-auto">
                  <input
                    type="text"
                    value={filePath}
                    onChange={(e) => setFilePath(e.target.value)}
                    placeholder="/path/to/your/model.gltf"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                  <p className="text-xs text-gray-500 mt-2">
                    Supported formats: GLTF, GLB, VRM, OBJ, BLEND, FBX, DAE,
                    3DS, PLY, STL
                  </p>
                </div>
              </div>
            </div>

            {/* File Browser Button */}
            <div className="flex justify-center">
              <Button
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                leftIcon={<ComputerDesktopIcon className="w-4 h-4" />}
              >
                Browse Files
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept={Object.keys(allowedTypes).join(",")}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setFilePath(file.name);
                    setSelectedFile(file);
                  }
                }}
                className="hidden"
              />
            </div>
          </div>
        )}

        {/* URL Import */}
        {importMode === "url" && (
          <div className="space-y-4 mb-6">
            <div className="p-6 border-2 border-dashed border-gray-300 rounded-lg">
              <div className="text-center">
                <LinkIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-sm text-gray-600 mb-4">
                  Enter a direct URL to download the 3D model
                </p>
                <div className="max-w-md mx-auto">
                  <input
                    type="url"
                    value={fileUrl}
                    onChange={(e) => setFileUrl(e.target.value)}
                    placeholder="https://example.com/model.gltf"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                  <p className="text-xs text-gray-500 mt-2">
                    The URL must point directly to a 3D model file
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Asset Details Form */}
        <AnimatePresence>
          {(selectedFile ||
            (importMode === "path" && filePath) ||
            (importMode === "url" && fileUrl)) && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-4 mb-6"
            >
              {/* Asset Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Asset Name *
                </label>
                <input
                  type="text"
                  value={assetName}
                  onChange={(e) => setAssetName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  placeholder="Enter asset name"
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  placeholder="Describe the asset (optional)"
                />
              </div>

              {/* Tags */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tags
                </label>
                <div className="flex gap-2 mb-2">
                  <input
                    type="text"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    placeholder="Add tags (press Enter)"
                  />
                  <Button
                    onClick={addTag}
                    variant="outline"
                    size="sm"
                    disabled={!tagInput.trim()}
                  >
                    Add
                  </Button>
                </div>
                {tags.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {tags.map((tag) => (
                      <span
                        key={tag}
                        className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-sm text-gray-700 rounded-full"
                      >
                        {tag}
                        <button
                          onClick={() => removeTag(tag)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <XMarkIcon className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Import Button */}
              <div className="flex justify-end gap-3 pt-4">
                <Button onClick={reset} variant="outline">
                  Cancel
                </Button>
                <Button
                  onClick={handleImport}
                  disabled={
                    !assetName.trim() ||
                    isUploading ||
                    (importMode === "upload" && !selectedFile) ||
                    (importMode === "path" && !filePath.trim()) ||
                    (importMode === "url" && !fileUrl.trim())
                  }
                  loading={isUploading}
                >
                  Import Asset
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Status Messages */}
        <AnimatePresence>
          {uploadStatus === "success" && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 bg-green-50 border border-green-200 rounded-lg"
            >
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="w-5 h-5 text-green-600" />
                <p className="text-green-800 font-medium">
                  Asset imported successfully!
                </p>
              </div>
            </motion.div>
          )}

          {(uploadStatus === "error" || error) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 bg-red-50 border border-red-200 rounded-lg"
            >
              <div className="flex items-center gap-2">
                <ExclamationTriangleIcon className="w-5 h-5 text-red-600" />
                <div>
                  <p className="text-red-800 font-medium">Import failed</p>
                  <p className="text-red-700 text-sm mt-1">
                    {error || "Please check the file format and try again"}
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

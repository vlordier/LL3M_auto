import React from "react";
import { motion } from "framer-motion";
import {
  UserIcon,
  CpuChipIcon,
  ArrowDownTrayIcon,
  EyeIcon,
} from "@heroicons/react/24/outline";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { ChatMessage as ChatMessageType } from "@/types";
import {
  formatRelativeTime,
  getAssetUrl,
  downloadFile,
  getPreviewUrl,
} from "@/utils";
import { AssetPreview } from "./AssetPreview";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.type === "user";
  const isSystem = message.type === "system";

  const handleDownload = (filePath: string, filename: string) => {
    const url = getAssetUrl(filePath);
    downloadFile(url, filename);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && (
        <div className="flex-shrink-0">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              isSystem
                ? "bg-gray-100 dark:bg-gray-700"
                : "bg-primary-100 dark:bg-primary-900/30"
            }`}
          >
            {isSystem ? (
              <div className="w-2 h-2 bg-gray-500 dark:bg-gray-300 rounded-full" />
            ) : (
              <CpuChipIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
            )}
          </div>
        </div>
      )}

      <div className={`max-w-[80%] ${isUser ? "order-first" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "bg-primary-600 text-white"
              : isSystem
                ? "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100"
                : "bg-gray-50 text-gray-900 dark:bg-gray-800 dark:text-gray-100"
          }`}
        >
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || "");
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>

          {/* Asset preview */}
          {message.generation_result && message.preview_url && (
            <div className="mt-3 space-y-3">
              <AssetPreview
                previewUrl={getPreviewUrl(message.preview_url)}
                assetPath={message.generation_result.asset_path}
                success={message.generation_result.success}
              />

              {/* Generation details */}
              {message.generation_result.success && (
                <div className="flex items-center justify-between text-xs text-gray-500 bg-gray-100 rounded-lg px-3 py-2 dark:bg-gray-700 dark:text-gray-300">
                  <div>
                    {message.generation_result.generation_time && (
                      <span>
                        Generated in{" "}
                        {message.generation_result.generation_time.toFixed(1)}s
                      </span>
                    )}
                    {message.generation_result.cached && (
                      <span className="ml-2 px-1.5 py-0.5 bg-green-100 text-green-700 rounded text-xs dark:bg-green-900/30 dark:text-green-300">
                        Cached
                      </span>
                    )}
                  </div>

                  {message.generation_result.asset_path && (
                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          const url = getAssetUrl(
                            message.generation_result!.asset_path!,
                          );
                          window.open(url, "_blank");
                        }}
                        className="flex items-center gap-1 px-2 py-1 bg-white text-gray-600 rounded hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
                      >
                        <EyeIcon className="w-3 h-3" />
                        View
                      </button>
                      <button
                        onClick={() =>
                          handleDownload(
                            message.generation_result!.asset_path!,
                            message
                              .generation_result!.asset_path!.split("/")
                              .pop() || "asset",
                          )
                        }
                        className="flex items-center gap-1 px-2 py-1 bg-white text-gray-600 rounded hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
                      >
                        <ArrowDownTrayIcon className="w-3 h-3" />
                        Download
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Export Files */}
          {message.generation_result?.exported_files &&
            message.generation_result.exported_files.length > 0 && (
              <div className="mt-3 space-y-2">
                <div className="text-xs text-gray-500 font-medium dark:text-gray-300">
                  Exported Formats:
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {message.generation_result.exported_files.map(
                    (filePath, index) => {
                      const filename = filePath.split("/").pop() || "file";
                      const extension =
                        filename.split(".").pop()?.toUpperCase() || "";

                      return (
                        <div
                          key={`export-${index}`}
                          className="flex items-center justify-between p-2 bg-gray-100 rounded-lg text-xs dark:bg-gray-700"
                        >
                          <div className="flex items-center gap-2">
                            <div className="w-6 h-6 bg-blue-100 dark:bg-blue-900/30 rounded flex items-center justify-center">
                              <span className="text-blue-600 dark:text-blue-400 font-bold text-xs">
                                {extension.slice(0, 2)}
                              </span>
                            </div>
                            <div>
                              <div className="font-medium text-gray-900 dark:text-gray-100 truncate max-w-24">
                                {filename}
                              </div>
                            </div>
                          </div>
                          <button
                            onClick={() => handleDownload(filePath, filename)}
                            className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-300 dark:hover:text-gray-100"
                            title={`Download ${extension} file`}
                          >
                            <ArrowDownTrayIcon className="w-3 h-3" />
                          </button>
                        </div>
                      );
                    },
                  )}
                </div>
              </div>
            )}

          {/* Artifacts */}
          {message.artifacts && message.artifacts.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="text-xs text-gray-500 font-medium dark:text-gray-300">
                Generated Files:
              </div>
              <div className="grid grid-cols-1 gap-2">
                {message.artifacts.map((artifact) => (
                  <div
                    key={artifact.artifact_id}
                    className="flex items-center justify-between p-2 bg-gray-100 rounded-lg text-xs dark:bg-gray-700"
                  >
                    <div>
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {artifact.relative_path.split("/").pop()}
                      </div>
                      <div className="text-gray-500 dark:text-gray-300">
                        {artifact.artifact_type} •{" "}
                        {(artifact.size_bytes / 1024).toFixed(1)} KB
                      </div>
                    </div>
                    <button
                      onClick={() =>
                        handleDownload(
                          artifact.relative_path,
                          artifact.relative_path.split("/").pop() || "file",
                        )
                      }
                      className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-300 dark:hover:text-gray-100"
                    >
                      <ArrowDownTrayIcon className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Edit Information for imported assets */}
          {message.generation_result?.original_asset_path && (
            <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="text-xs text-blue-700 dark:text-blue-300 font-medium">
                Edited Asset
              </div>
              <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                Original:{" "}
                {message.generation_result.original_asset_path.split("/").pop()}
              </div>
              {message.generation_result.edit_instruction && (
                <div className="text-xs text-blue-600 dark:text-blue-400 mt-1 italic">
                  "{message.generation_result.edit_instruction}"
                </div>
              )}
            </div>
          )}
        </div>

        <div
          className={`mt-1 text-xs text-gray-500 ${isUser ? "text-right" : "text-left"} dark:text-gray-400`}
        >
          {formatRelativeTime(message.timestamp)}
          {message.session_id && (
            <span className="ml-2 opacity-50">
              #{message.session_id.slice(-8)}
            </span>
          )}
        </div>
      </div>

      {isUser && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center dark:bg-gray-700">
            <UserIcon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
          </div>
        </div>
      )}
    </motion.div>
  );
}

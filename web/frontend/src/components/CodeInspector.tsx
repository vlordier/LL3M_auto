import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  CodeBracketIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  ClipboardIcon,
  CheckIcon,
} from "@heroicons/react/24/outline";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface CodeInspectorProps {
  code: string;
  title?: string;
  language?: string;
  className?: string;
  isExpanded?: boolean;
  showLineNumbers?: boolean;
}

export function CodeInspector({
  code,
  title = "Generated Python Code",
  language = "python",
  className = "",
  isExpanded = false,
  showLineNumbers = true,
}: CodeInspectorProps) {
  const [expanded, setExpanded] = useState(isExpanded);
  const [copied, setCopied] = useState(false);

  if (!code?.trim()) {
    return null;
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy code:", err);
    }
  };

  const codeLines = code.split('\n').length;
  const codePreview = code.split('\n').slice(0, 3).join('\n');

  return (
    <div className={`border border-gray-200 rounded-lg bg-gray-50 dark:bg-gray-800 dark:border-gray-700 ${className}`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full p-3 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
      >
        <div className="flex items-center space-x-2">
          {expanded ? (
            <ChevronDownIcon className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronRightIcon className="w-4 h-4 text-gray-500" />
          )}
          <CodeBracketIcon className="w-4 h-4 text-blue-500" />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {title}
          </span>
          <span className="text-xs text-gray-500 bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded-full">
            {codeLines} lines
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">
            {expanded ? "Click to collapse" : "Click to inspect code"}
          </span>
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-gray-200 dark:border-gray-700"
          >
            <div className="relative">
              <div className="absolute top-2 right-2 z-10">
                <button
                  onClick={handleCopy}
                  className="flex items-center space-x-1 px-2 py-1 text-xs bg-gray-800 text-gray-200 rounded hover:bg-gray-700 transition-colors"
                  title="Copy code to clipboard"
                >
                  {copied ? (
                    <>
                      <CheckIcon className="w-3 h-3" />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <ClipboardIcon className="w-3 h-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <SyntaxHighlighter
                language={language}
                style={oneDark}
                showLineNumbers={showLineNumbers}
                customStyle={{
                  margin: 0,
                  borderRadius: 0,
                  fontSize: '12px',
                  lineHeight: '1.4',
                }}
                lineNumberStyle={{
                  minWidth: '3em',
                  paddingRight: '1em',
                  color: '#6b7280',
                  fontSize: '11px',
                }}
              >
                {code}
              </SyntaxHighlighter>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!expanded && codePreview && (
        <div className="px-3 pb-3">
          <div className="text-xs text-gray-500 mb-2">Preview (first 3 lines):</div>
          <div className="bg-gray-900 text-gray-300 p-2 rounded text-xs font-mono overflow-hidden">
            <pre className="whitespace-pre-wrap truncate">
              {codePreview}
              {codeLines > 3 && <span className="text-gray-500">...</span>}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ClockIcon,
  MagnifyingGlassIcon,
  TrashIcon,
  PlayIcon,
  DocumentTextIcon,
  CubeIcon,
  FolderOpenIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import type { SessionDetail } from '@/types';
import { formatRelativeTime, formatDuration } from '@/utils';
import { useSessionStore } from '@/stores/sessionStore';

interface SessionBrowserProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSession: (session: SessionDetail) => void;
  className?: string;
}

export function SessionBrowser({ isOpen, onClose, onSelectSession, className = '' }: SessionBrowserProps) {
  const [sessions, setSessions] = useState<SessionDetail[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(new Set());
  
  const { resumeSession, deleteSession } = useSessionStore();

  useEffect(() => {
    if (isOpen) {
      loadSessions();
    }
  }, [isOpen]);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/sessions');
      const data = await response.json();
      setSessions(data);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredSessions = sessions.filter(session =>
    session.prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
    session.session_id.includes(searchTerm)
  );

  const handleResumeSession = async (session: SessionDetail) => {
    await resumeSession(session.session_id);
    onSelectSession(session);
    onClose();
  };

  const handleDeleteSession = async (sessionId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
      await deleteSession(sessionId);
      setSessions(prev => prev.filter(s => s.session_id !== sessionId));
    }
  };

  const handleBulkDelete = async () => {
    if (selectedSessions.size === 0) return;
    
    if (confirm(`Delete ${selectedSessions.size} selected sessions? This action cannot be undone.`)) {
      for (const sessionId of selectedSessions) {
        await deleteSession(sessionId);
      }
      setSessions(prev => prev.filter(s => !selectedSessions.has(s.session_id)));
      setSelectedSessions(new Set());
    }
  };

  const toggleSessionSelection = (sessionId: string) => {
    setSelectedSessions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sessionId)) {
        newSet.delete(sessionId);
      } else {
        newSet.add(sessionId);
      }
      return newSet;
    });
  };

  const getStatusColor = (phase: string) => {
    switch (phase) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'asset_generation': return 'text-blue-600 bg-blue-100';
      case 'code_generation': return 'text-purple-600 bg-purple-100';
      case 'prompt_analysis': return 'text-orange-600 bg-orange-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          onClick={e => e.stopPropagation()}
          className={`bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[80vh] overflow-hidden ${className}`}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <FolderOpenIcon className="w-6 h-6 text-gray-600" />
              <h2 className="text-xl font-semibold text-gray-900">Session Browser</h2>
            </div>
            <div className="flex items-center gap-3">
              {selectedSessions.size > 0 && (
                <button
                  onClick={handleBulkDelete}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm text-red-600 bg-red-50 rounded-lg hover:bg-red-100 transition-colors"
                >
                  <TrashIcon className="w-4 h-4" />
                  Delete Selected ({selectedSessions.size})
                </button>
              )}
              <button
                onClick={onClose}
                className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Search */}
          <div className="p-6 border-b border-gray-200">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search sessions by prompt or ID..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>

          {/* Sessions List */}
          <div className="flex-1 overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center p-12">
                <div className="flex items-center gap-3">
                  <div className="animate-spin w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full" />
                  <span className="text-gray-600">Loading sessions...</span>
                </div>
              </div>
            ) : filteredSessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-12 text-center">
                <FolderOpenIcon className="w-12 h-12 text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {searchTerm ? 'No matching sessions' : 'No sessions found'}
                </h3>
                <p className="text-gray-500">
                  {searchTerm 
                    ? 'Try adjusting your search terms' 
                    : 'Start generating assets to see your session history here'
                  }
                </p>
              </div>
            ) : (
              <div className="p-6 space-y-4">
                {filteredSessions.map((session) => (
                  <motion.div
                    key={session.session_id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`group border rounded-lg p-4 hover:shadow-md transition-all cursor-pointer ${
                      selectedSessions.has(session.session_id) 
                        ? 'border-primary-300 bg-primary-50' 
                        : 'border-gray-200 bg-white'
                    }`}
                    onClick={() => handleResumeSession(session)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4 flex-1">
                        {/* Selection checkbox */}
                        <div onClick={e => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={selectedSessions.has(session.session_id)}
                            onChange={() => toggleSessionSelection(session.session_id)}
                            className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                          />
                        </div>

                        {/* Session info */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-medium text-gray-900 truncate">
                              {session.prompt.length > 80 
                                ? `${session.prompt.substring(0, 80)}...` 
                                : session.prompt
                              }
                            </h3>
                            <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(session.current_phase)}`}>
                              {session.current_phase.replace('_', ' ')}
                            </span>
                          </div>

                          <div className="flex items-center gap-4 text-sm text-gray-500 mb-3">
                            <div className="flex items-center gap-1">
                              <ClockIcon className="w-4 h-4" />
                              <span>{formatRelativeTime(session.created_at)}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <DocumentTextIcon className="w-4 h-4" />
                              <span>{session.traces.length} steps</span>
                            </div>
                            {session.artifacts.length > 0 && (
                              <div className="flex items-center gap-1">
                                <CubeIcon className="w-4 h-4" />
                                <span>{session.artifacts.length} assets</span>
                              </div>
                            )}
                            {session.total_duration && (
                              <span>Duration: {formatDuration(session.total_duration)}</span>
                            )}
                          </div>

                          <div className="flex items-center gap-2 text-xs text-gray-400">
                            <span>ID: {session.session_id}</span>
                          </div>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleResumeSession(session);
                          }}
                          className="p-2 text-primary-600 hover:text-primary-700 hover:bg-primary-50 rounded-lg transition-colors"
                          title="Resume session"
                        >
                          <PlayIcon className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => handleDeleteSession(session.session_id, e)}
                          className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                          title="Delete session"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {/* Recent artifacts preview */}
                    {session.artifacts.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <div className="flex items-center gap-2 mb-2">
                          <CubeIcon className="w-4 h-4 text-gray-500" />
                          <span className="text-sm font-medium text-gray-700">Generated Assets</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {session.artifacts.slice(0, 3).map((artifact, index) => (
                            <div
                              key={index}
                              className="px-2 py-1 bg-gray-100 rounded text-xs text-gray-600"
                            >
                              {artifact.artifact_type}: {artifact.relative_path.split('/').pop()}
                            </div>
                          ))}
                          {session.artifacts.length > 3 && (
                            <div className="px-2 py-1 bg-gray-100 rounded text-xs text-gray-500">
                              +{session.artifacts.length - 3} more
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
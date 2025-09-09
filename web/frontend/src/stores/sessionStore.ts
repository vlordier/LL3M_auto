import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { Session, SessionDetail, ChatMessage, GenerationProgress, ProgressMessage, StepControl } from '@/types';

interface SessionStore {
  // Current session data
  currentSession: SessionDetail | null;
  sessions: Session[];
  
  // Chat messages
  messages: ChatMessage[];
  
  // UI state
  isGenerating: boolean;
  generationProgress: GenerationProgress | null;
  isLoading: boolean;
  error: string | null;
  
  // Progress tracking
  progressMessages: ProgressMessage[];
  stepControl: StepControl | null;
  ws: WebSocket | null;

  // Actions
  setCurrentSession: (session: SessionDetail | null) => void;
  setSessions: (sessions: Session[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
  setIsGenerating: (generating: boolean) => void;
  setGenerationProgress: (progress: GenerationProgress | null) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Progress tracking actions
  setProgressMessages: (messages: ProgressMessage[]) => void;
  addProgressMessage: (message: ProgressMessage) => void;
  setStepControl: (control: StepControl | null) => void;
  connectWebSocket: (sessionId: string) => void;
  disconnectWebSocket: () => void;
  
  // Step control actions
  pauseSession: (sessionId: string) => Promise<void>;
  resumeSession: (sessionId: string) => Promise<void>;
  stepSession: (sessionId: string) => Promise<void>;
  
  // Async actions
  loadSession: (sessionId: string) => Promise<void>;
  loadSessions: () => Promise<void>;
  startNewSession: () => void;
  generateAsset: (prompt: string, assetName?: string) => Promise<void>;
  amendSession: (sessionId: string, feedback: string) => Promise<void>;
  loadProgressMessages: (sessionId: string) => Promise<void>;
  loadStepControl: (sessionId: string) => Promise<void>;
}

export const useSessionStore = create<SessionStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      currentSession: null,
      sessions: [],
      messages: [],
      isGenerating: false,
      generationProgress: null,
      isLoading: false,
      error: null,
      
      // Progress tracking state
      progressMessages: [],
      stepControl: null,
      ws: null,

      // Sync actions
      setCurrentSession: (session) => set({ currentSession: session }),
      setSessions: (sessions) => set({ sessions }),
      
      addMessage: (message) => 
        set((state) => ({ messages: [...state.messages, message] })),
      
      updateMessage: (id, updates) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, ...updates } : msg
          ),
        })),

      setIsGenerating: (generating) => set({ isGenerating: generating }),
      setGenerationProgress: (progress) => set({ generationProgress: progress }),
      setIsLoading: (loading) => set({ isLoading: loading }),
      setError: (error) => set({ error }),

      // Progress tracking sync actions
      setProgressMessages: (messages) => set({ progressMessages: messages }),
      addProgressMessage: (message) => 
        set((state) => ({ progressMessages: [...state.progressMessages, message] })),
      setStepControl: (control) => set({ stepControl: control }),

      connectWebSocket: (sessionId: string) => {
        try {
          const existing = get().ws;
          if (existing) {
            existing.close();
            set({ ws: null });
          }
          const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
          const host = window.location.hostname;
          const ws = new WebSocket(`${proto}://${host}:8000/ws/${sessionId}`);
          ws.onopen = () => {
            try { ws.send(JSON.stringify({ type: 'subscribe_updates' })); } catch {}
          };
          ws.onmessage = async (evt) => {
            try {
              const msg = JSON.parse(evt.data);
              const t = msg?.type as string | undefined;
              if (!t) return;
              if (t === 'progress' || t === 'substep_added' || t === 'substep_updated' || t === 'step_control_changed') {
                await get().loadProgressMessages(sessionId);
                await get().loadStepControl(sessionId);
              }
              if (t === 'generation_complete') {
                await get().loadSession(sessionId);
                get().disconnectWebSocket();
                set({ isGenerating: false });
              }
              if (t === 'error' || t === 'generation_failed') {
                set({ isGenerating: false });
              }
            } catch (e) {
              console.debug('WS message parse error', e);
            }
          };
          ws.onclose = () => set({ ws: null });
          set({ ws });
        } catch (e) {
          console.warn('WS connect failed', e);
        }
      },
      disconnectWebSocket: () => {
        try {
          const ws = get().ws;
          if (ws) ws.close();
        } finally {
          set({ ws: null });
        }
      },

      // Async actions
      loadSession: async (sessionId: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await fetch(`/api/sessions/${sessionId}`);
          if (!response.ok) throw new Error('Failed to load session');
          
          const session: SessionDetail = await response.json();
          set({ currentSession: session });
          
          // Convert session to chat messages
          const messages: ChatMessage[] = [
            {
              id: `user-${sessionId}`,
              type: 'user',
              content: session.prompt,
              timestamp: new Date(session.created_at),
              session_id: sessionId,
            },
          ];

          if (session.final_result) {
            messages.push({
              id: `assistant-${sessionId}`,
              type: 'assistant',
              content: session.success 
                ? `I've successfully generated "${session.asset_name}" for you! 🎉` 
                : 'Generation failed. Please check the details.',
              timestamp: new Date(session.updated_at),
              session_id: sessionId,
              generation_result: session.final_result,
              preview_url: session.final_result.screenshot_path,
              artifacts: session.artifacts,
            });
          }

          set({ messages });
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Unknown error' });
        } finally {
          set({ isLoading: false });
        }
      },

      loadSessions: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await fetch('/api/sessions?limit=50');
          if (!response.ok) throw new Error('Failed to load sessions');
          
          const sessions: Session[] = await response.json();
          set({ sessions });
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Unknown error' });
        } finally {
          set({ isLoading: false });
        }
      },

      startNewSession: () => {
        // Clear current session state to start fresh
        set({
          currentSession: null,
          messages: [],
          generationProgress: null,
          progressMessages: [],
          stepControl: null,
          error: null,
        });
      },

      generateAsset: async (prompt: string, assetName = 'asset') => {
        set({ isGenerating: true, error: null, generationProgress: null });

        // Add user message immediately
        const userMessage: ChatMessage = {
          id: `user-${Date.now()}`,
          type: 'user',
          content: prompt,
          timestamp: new Date(),
        };
        get().addMessage(userMessage);

        // Add loading assistant message
        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          type: 'assistant',
          content: 'Generating your 3D asset... This may take a few minutes.',
          timestamp: new Date(),
        };
        get().addMessage(assistantMessage);

        try {
          // Prefer async generation with granular progress; fallback to sync
          let sessionId: string | null = null;
          try {
            const startResp = await fetch('/api/generate-async', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                prompt,
                asset_name: assetName,
                export_formats: ['blend', 'obj', 'gltf'],
              }),
            });
            if (startResp.ok) {
              const started = await startResp.json();
              sessionId = started.session_id as string;
            }
          } catch {}

          if (!sessionId) {
            // Fallback to synchronous endpoint
            const response = await fetch('/api/generate', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                prompt,
                asset_name: assetName,
                export_formats: ['blend', 'obj', 'gltf'],
              }),
            });
            if (!response.ok) throw new Error('Generation failed');
            const result = await response.json();

            get().updateMessage(assistantMessage.id, {
              content: result.success
                ? `🎉 Successfully generated "${assetName}"! Generation took ${result.generation_time?.toFixed(1) || 'N/A'}s.`
                : `❌ Generation failed: ${result.error || 'Unknown error'}`,
              generation_result: result,
              preview_url: result.screenshot_path,
            });

            if (result.session_id) {
              await get().loadSession(result.session_id);
            }
            set({ isGenerating: false, generationProgress: null });
            return;
          }

          // With async session: attach to it and poll for updates
          // Attach WS and load session details
          get().connectWebSocket(sessionId);
          await get().loadSession(sessionId);

          let attempts = 0;
          const maxAttempts = 60 * 5; // ~5 minutes at 1s interval
          const poll = async () => {
            attempts += 1;
            try {
              await get().loadProgressMessages(sessionId!);

              // Update the assistant message with the latest progress title/percent
              const state = get();
              const latest = state.progressMessages[state.progressMessages.length - 1];
              if (latest) {
                get().updateMessage(assistantMessage.id, {
                  content: `${latest.title} — ${latest.progress_percent.toFixed(1)}%\n${latest.description}`,
                });
              }

              // Check if session has completed
              const detailResp = await fetch(`/api/sessions/${sessionId}`);
              if (detailResp.ok) {
                const detail = await detailResp.json();
                if (detail.success && detail.final_result) {
                  get().updateMessage(assistantMessage.id, {
                    content: `🎉 Successfully generated "${assetName}"! Generation took ${detail.final_result.generation_time?.toFixed?.(1) || detail.final_result.generation_time || 'N/A'}s.`,
                    generation_result: detail.final_result,
                    preview_url: detail.final_result.screenshot_path,
                  });
                  set({ isGenerating: false, generationProgress: null, currentSession: detail });
                  return true;
                }
              }
            } catch (e) {
              console.error('Polling error', e);
            }
            if (attempts >= maxAttempts) {
              set({ isGenerating: false });
              return true;
            }
            return false;
          };

          // Simple polling loop
          const interval = setInterval(async () => {
            const done = await poll();
            if (done) clearInterval(interval);
          }, 1000);
        } catch (error) {
          get().updateMessage(assistantMessage.id, {
            content: `❌ Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          });
          set({ error: error instanceof Error ? error.message : 'Unknown error' });
          set({ isGenerating: false });
        }
      },

      amendSession: async (sessionId: string, feedback: string) => {
        set({ isGenerating: true, error: null });
        
        // Add user feedback message
        const userMessage: ChatMessage = {
          id: `user-amend-${Date.now()}`,
          type: 'user',
          content: `Amend request: ${feedback}`,
          timestamp: new Date(),
          session_id: sessionId,
        };
        get().addMessage(userMessage);

        // Add loading message
        const assistantMessage: ChatMessage = {
          id: `assistant-amend-${Date.now()}`,
          type: 'assistant',
          content: 'Creating an amended version based on your feedback...',
          timestamp: new Date(),
          session_id: sessionId,
        };
        get().addMessage(assistantMessage);

        try {
          const response = await fetch(`/api/sessions/${sessionId}/amend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feedback }),
          });

          if (!response.ok) throw new Error('Amendment failed');
          
          const result = await response.json();
          
          // Update assistant message
          get().updateMessage(assistantMessage.id, {
            content: result.success 
              ? `✨ Amendment completed! I've refined the asset based on your feedback.`
              : `❌ Amendment failed: ${result.error || 'Unknown error'}`,
            generation_result: result,
            preview_url: result.screenshot_path,
          });

          // Reload the current session to get updated data
          if (get().currentSession) {
            await get().loadSession(get().currentSession.session_id);
          }
        } catch (error) {
          get().updateMessage(assistantMessage.id, {
            content: `❌ Amendment failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          });
          set({ error: error instanceof Error ? error.message : 'Unknown error' });
        } finally {
          set({ isGenerating: false });
        }
      },

      // Progress tracking async actions
      loadProgressMessages: async (sessionId: string) => {
        try {
          const response = await fetch(`/api/sessions/${sessionId}/progress?limit=50`);
          if (!response.ok) throw new Error('Failed to load progress messages');
          
          const messages: ProgressMessage[] = await response.json();
          set({ progressMessages: messages });
        } catch (error) {
          console.error('Failed to load progress messages:', error);
          set({ error: error instanceof Error ? error.message : 'Failed to load progress' });
        }
      },

      loadStepControl: async (sessionId: string) => {
        try {
          const response = await fetch(`/api/sessions/${sessionId}/step-control`);
          if (!response.ok) throw new Error('Failed to load step control');
          
          const control: StepControl = await response.json();
          set({ stepControl: control });
        } catch (error) {
          console.error('Failed to load step control:', error);
          set({ error: error instanceof Error ? error.message : 'Failed to load step control' });
        }
      },

      // Step control async actions
      pauseSession: async (sessionId: string) => {
        try {
          const response = await fetch(`/api/sessions/${sessionId}/step-control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'pause' })
          });
          
          if (!response.ok) throw new Error('Failed to pause session');
          
          // Reload step control to get updated state
          await get().loadStepControl(sessionId);
          await get().loadProgressMessages(sessionId);
        } catch (error) {
          console.error('Failed to pause session:', error);
          set({ error: error instanceof Error ? error.message : 'Failed to pause session' });
        }
      },

      resumeSession: async (sessionId: string) => {
        try {
          const response = await fetch(`/api/sessions/${sessionId}/step-control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'resume' })
          });
          
          if (!response.ok) throw new Error('Failed to resume session');
          
          // Reload step control to get updated state
          await get().loadStepControl(sessionId);
          await get().loadProgressMessages(sessionId);
        } catch (error) {
          console.error('Failed to resume session:', error);
          set({ error: error instanceof Error ? error.message : 'Failed to resume session' });
        }
      },

      stepSession: async (sessionId: string) => {
        try {
          const response = await fetch(`/api/sessions/${sessionId}/step-control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'step' })
          });
          
          if (!response.ok) throw new Error('Failed to step session');
          
          // Reload step control to get updated state
          await get().loadStepControl(sessionId);
          await get().loadProgressMessages(sessionId);
        } catch (error) {
          console.error('Failed to step session:', error);
          set({ error: error instanceof Error ? error.message : 'Failed to step session' });
        }
      },
    }),
    { name: 'session-store' }
  )
);

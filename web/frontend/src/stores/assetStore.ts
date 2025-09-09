import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { AssetMetadata, ImportAssetRequest, EditAssetRequest } from '@/types';

interface AssetState {
  // State
  assets: AssetMetadata[];
  loading: boolean;
  error: string | null;
  currentAsset: AssetMetadata | null;
  isEditing: boolean;

  // Actions
  loadAssets: (assetType?: string) => Promise<void>;
  importAsset: (request: ImportAssetRequest) => Promise<AssetMetadata>;
  importAssetFromPath: (filePath: string, metadata: Partial<ImportAssetRequest>) => Promise<AssetMetadata>;
  importAssetFromUrl: (url: string, metadata: Partial<ImportAssetRequest>) => Promise<AssetMetadata>;
  getAsset: (assetId: string) => Promise<AssetMetadata>;
  editAsset: (assetId: string, request: EditAssetRequest) => Promise<any>;
  deleteAsset: (assetId: string) => Promise<void>;
  setCurrentAsset: (asset: AssetMetadata | null) => void;
  clearError: () => void;
}

export const useAssetStore = create<AssetState>()(
  devtools(
    (set, get) => ({
      // Initial state
      assets: [],
      loading: false,
      error: null,
      currentAsset: null,
      isEditing: false,

      // Load all assets
      loadAssets: async (assetType?: string) => {
        set({ loading: true, error: null });

        try {
          const url = new URL('/api/assets', window.location.origin);
          if (assetType && assetType !== 'all') {
            url.searchParams.set('asset_type', assetType);
          }

          const response = await fetch(url.toString(), {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            throw new Error(`Failed to load assets: ${response.statusText}`);
          }

          const assets: AssetMetadata[] = await response.json();
          set({ assets, loading: false });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error occurred';
          set({ error: message, loading: false });
        }
      },

      // Import a new asset
      importAsset: async (request: ImportAssetRequest) => {
        set({ loading: true, error: null });

        try {
          const formData = new FormData();
          formData.append('file', request.file);
          formData.append('asset_name', request.asset_name);
          formData.append('asset_type', request.asset_type);
          
          if (request.description) {
            formData.append('description', request.description);
          }
          
          if (request.tags && request.tags.length > 0) {
            formData.append('tags', JSON.stringify(request.tags));
          }

          const response = await fetch('/api/assets/import', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Import failed: ${response.statusText}`);
          }

          const newAsset: AssetMetadata = await response.json();
          
          // Update the assets list
          set(state => ({
            assets: [newAsset, ...state.assets],
            loading: false,
            currentAsset: newAsset
          }));

          return newAsset;
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error occurred';
          set({ error: message, loading: false });
          throw error;
        }
      },

      // Import asset from file path
      importAssetFromPath: async (filePath: string, metadata: Partial<ImportAssetRequest>) => {
        set({ loading: true, error: null });

        try {
          const response = await fetch('/api/assets/import-path', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              file_path: filePath,
              asset_name: metadata.asset_name || 'Imported Asset',
              asset_type: metadata.asset_type || filePath.split('.').pop()?.toLowerCase() || 'unknown',
              description: metadata.description,
              tags: metadata.tags,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Path import failed: ${response.statusText}`);
          }

          const newAsset: AssetMetadata = await response.json();
          
          // Update the assets list
          set(state => ({
            assets: [newAsset, ...state.assets],
            loading: false,
            currentAsset: newAsset
          }));

          return newAsset;
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error occurred';
          set({ error: message, loading: false });
          throw error;
        }
      },

      // Import asset from URL
      importAssetFromUrl: async (url: string, metadata: Partial<ImportAssetRequest>) => {
        set({ loading: true, error: null });

        try {
          const response = await fetch('/api/assets/import-url', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              url: url,
              asset_name: metadata.asset_name || 'Imported Asset',
              asset_type: metadata.asset_type || url.split('.').pop()?.toLowerCase() || 'unknown',
              description: metadata.description,
              tags: metadata.tags,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `URL import failed: ${response.statusText}`);
          }

          const newAsset: AssetMetadata = await response.json();
          
          // Update the assets list
          set(state => ({
            assets: [newAsset, ...state.assets],
            loading: false,
            currentAsset: newAsset
          }));

          return newAsset;
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error occurred';
          set({ error: message, loading: false });
          throw error;
        }
      },

      // Get a specific asset
      getAsset: async (assetId: string) => {
        set({ error: null });

        try {
          const response = await fetch(`/api/assets/${assetId}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            throw new Error(`Failed to get asset: ${response.statusText}`);
          }

          const asset: AssetMetadata = await response.json();
          set({ currentAsset: asset });
          
          return asset;
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error occurred';
          set({ error: message });
          throw error;
        }
      },

      // Edit an asset
      editAsset: async (assetId: string, request: EditAssetRequest) => {
        set({ isEditing: true, error: null });

        try {
          const response = await fetch(`/api/assets/${assetId}/edit`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Edit failed: ${response.statusText}`);
          }

          const result = await response.json();
          set({ isEditing: false });
          
          // Optionally reload assets to get the updated version
          get().loadAssets();
          
          return result;
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error occurred';
          set({ error: message, isEditing: false });
          throw error;
        }
      },

      // Delete an asset
      deleteAsset: async (assetId: string) => {
        set({ error: null });

        try {
          const response = await fetch(`/api/assets/${assetId}`, {
            method: 'DELETE',
            headers: {
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Delete failed: ${response.statusText}`);
          }

          // Remove from the local state
          set(state => ({
            assets: state.assets.filter(asset => asset.asset_id !== assetId),
            currentAsset: state.currentAsset?.asset_id === assetId ? null : state.currentAsset
          }));
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error occurred';
          set({ error: message });
          throw error;
        }
      },

      // Set current asset
      setCurrentAsset: (asset: AssetMetadata | null) => {
        set({ currentAsset: asset });
      },

      // Clear error
      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'asset-store',
      partialize: (state) => ({
        // Don't persist loading states or errors
        assets: state.assets,
        currentAsset: state.currentAsset,
      }),
    }
  )
);
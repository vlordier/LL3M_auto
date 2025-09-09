export interface Session {
  session_id: string;
  prompt: string;
  asset_name: string;
  current_phase: string;
  completion_percentage: number;
  success: boolean;
  created_at: string;
  updated_at: string;
  total_duration?: number;
}

export interface SessionDetail extends Session {
  traces: WorkflowTrace[];
  code_versions: CodeVersion[];
  artifacts: Artifact[];
  final_result?: GenerationResult;
}

export interface WorkflowTrace {
  phase: string;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
  status: "running" | "completed" | "failed" | "skipped";
  errors: string[];
  llm_calls: number;
  agent_actions: number;
}

export interface CodeVersion {
  version: number;
  timestamp: string;
  source_phase: string;
  validation_status: "valid" | "invalid" | "healing" | "final";
  healing_applied: boolean;
  quality_score?: number;
  code_length: number;
  code_content?: string;
  ruff_issues?: string[];
  execution_errors?: string[];
}

export interface Artifact {
  artifact_id: string;
  artifact_type:
    | "blend_file"
    | "screenshot"
    | "code"
    | "logs"
    | "trace"
    | "exports";
  file_path: string;
  relative_path: string;
  created_at: string;
  phase: string;
  version: number;
  size_bytes: number;
  tags: string[];
}

export interface GenerationResult {
  success: boolean;
  cached: boolean;
  asset_path?: string;
  screenshot_path?: string;
  generation_time?: number;
  exported_files?: string[];
  session_id?: string;
  error?: string;
}

export interface GenerateAssetRequest {
  prompt: string;
  asset_name?: string;
  export_formats?: string[];
  output_directory?: string;
}

export interface AmendSessionRequest {
  feedback: string;
  step_id?: string;
}

export interface ChatMessage {
  id: string;
  type: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  session_id?: string;
  generation_result?: GenerationResult;
  preview_url?: string;
  artifacts?: Artifact[];
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
}

// UI State Types
export interface ViewMode {
  layout: "desktop" | "tablet" | "mobile";
  activePanel: "chat" | "viewer" | "steps";
  viewerMode: "solid" | "wireframe" | "material";
}

export interface GenerationProgress {
  session_id: string;
  current_phase: string;
  completion_percentage: number;
  estimated_remaining: number;
  current_step: string;
}

export interface ProgressMessage {
  timestamp: string;
  phase: string;
  title: string;
  description: string;
  progress_percent: number;
  estimated_remaining_seconds?: number;
  technical_details?: Record<string, any>;
}

export interface StepControl {
  execution_mode: "continuous" | "step_by_step" | "paused" | "stepping";
  current_step: number;
  total_steps: number;
  can_pause: boolean;
  can_resume: boolean;
  can_step: boolean;
  awaiting_user_action: boolean;
}

export interface ThreeViewerProps {
  assetUrl?: string;
  assetType?: "gltf" | "obj" | "blend";
  showControls?: boolean;
  showWireframe?: boolean;
  autoRotate?: boolean;
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

// Asset Management Types
export interface AssetMetadata {
  asset_id: string;
  asset_name: string;
  asset_type: string;
  file_path: string;
  preview_path?: string;
  size_bytes: number;
  created_at: string;
  description?: string;
  tags: string[];
  import_metadata?: {
    original_filename: string;
    file_extension: string;
    content_type: string;
    imported_at: string;
  };
}

export interface ImportAssetRequest {
  file: File;
  asset_name: string;
  asset_type: string;
  description?: string;
  tags?: string[];
}

export interface EditAssetRequest {
  edit_instruction: string;
  preserve_materials?: boolean;
  export_formats?: string[];
}

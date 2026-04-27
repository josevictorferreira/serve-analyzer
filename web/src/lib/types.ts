export type JobPhase = 'idle' | 'uploading' | 'analyzing' | 'clipping' | 'done' | 'error';

export interface ServeCandidate {
  candidate_index?: number;
  contact_frame?: number;
  contact_time_sec: number;
  post_contact_max_kmh?: number;
  score?: number;
  support_count?: number;
  selector_rank?: number;
  [key: string]: unknown;
}

export interface ClipMeta {
  filename: string;
  url_path: string;
  serve_index: number;
  contact_time_sec: number;
  duration: number;
}

export interface JobStatus {
  status: 'idle' | 'analyzing' | 'clipping' | 'done' | 'error' | 'busy';
  phase: JobPhase | null;
  error?: string;
  clips?: ClipMeta[];
  selected_serves?: ServeCandidate[];
  candidates?: ServeCandidate[];
  count_inferred?: boolean;
  inferred_count?: number;
  estimated_duration_sec?: number | null;
}

export interface ServeAttempt {
  frame: number;
  timestamp: number;
  velocity_mps: number;
  velocity_kmh: number;
  confidence: number;
}

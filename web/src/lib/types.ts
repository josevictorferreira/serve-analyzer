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
  fps?: number;
  width?: number;
  height?: number;
  start_frame?: number;
  end_frame?: number;
  contact_frame?: number;
  contact_clip_time_sec?: number;
  velocity_kmh?: number | null;
  mean_velocity_kmh?: number | null;
  ball_positions?: BallPosition[];
}

export interface BallPosition {
  frame_number: number;
  clip_time_sec: number;
  x: number;
  y: number;
}

export interface JobStatus {
  status: 'idle' | 'uploading' | 'analyzing' | 'clipping' | 'done' | 'error' | 'busy';
  phase: JobPhase | null;
  error?: string;
  clips?: ClipMeta[];
  selected_serves?: ServeCandidate[];
  candidates?: ServeCandidate[];
  count_inferred?: boolean;
  inferred_count?: number;
  detector?: string | null;
  detector_version?: string | null;
  detector_label?: string | null;
  estimated_duration_sec?: number | null;
}

export interface DetectorVersionInfo {
  version: string;
  label: string;
  description: string;
}

export interface DetectorVersionsResponse {
  detectors: DetectorVersionInfo[];
  default_version: string;
}

export interface ServeAttempt {
  frame: number;
  timestamp: number;
  velocity_mps: number;
  velocity_kmh: number;
  confidence: number;
}

export interface AnnotationBbox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface AnnotationPrediction {
  bbox: AnnotationBbox;
  confidence: number;
  model: string;
}

export interface AnnotationLabel {
  class_id: number;
  class_name: string;
  bbox: AnnotationBbox;
  source: string;
}

export type AnnotationStatus = 'pending' | 'accepted' | 'corrected' | 'absent' | 'skipped';

export interface AnnotationFrame {
  frame_id: string;
  frame_number: number;
  time_sec: number;
  image_filename: string;
  width: number;
  height: number;
  split: 'train' | 'val' | 'test';
  prediction: AnnotationPrediction | null;
  label: AnnotationLabel | null;
  status: AnnotationStatus;
  reviewed_at?: string;
}

export interface AnnotationProgress {
  total: number;
  pending: number;
  accepted: number;
  corrected: number;
  absent: number;
  skipped: number;
  reviewed: number;
  exportable: number;
}

export interface AnnotationSessionSummary {
  id: string;
  source_filename: string;
  created_at: string;
  updated_at: string;
  video: Record<string, unknown>;
  sampling: Record<string, unknown>;
  prelabel: Record<string, unknown>;
  progress: AnnotationProgress;
}

export interface AnnotationSession extends AnnotationSessionSummary {
  classes: Array<{ id: number; name: string }>;
  source_video: string;
  frames: AnnotationFrame[];
  exports: Array<Record<string, unknown>>;
  evaluations: Array<Record<string, unknown>>;
}

export interface AnnotationExport {
  export_id: string;
  dataset_dir: string;
  data_yaml: string;
  counts: Record<string, { images: number; positive: number; negative: number }>;
  [key: string]: unknown;
}

export interface AnnotationEvaluation {
  precision: number;
  recall: number;
  reviewed_frames: number;
  visible_frames: number;
  detected_visible_frames: number;
  false_positive: number;
  false_negative: number;
  true_positive: number;
  true_negative: number;
  [key: string]: unknown;
}

export interface WallVideoUploadResponse {
  video_id: string;
  video_url: string;
  filename: string;
  duration_sec: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
}

export interface WallVideoMetadataResponse {
  video_id: string;
  filename: string;
  duration_sec: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
}

export interface WallCalibrationPoint {
  name: string;
  pixel: number[];
  wall_m: number[];
}

export interface WallCalibrationSetup {
  serve_contact_distance_m: number;
  camera_wall_distance_m: number;
  serve_contact_height_m: number;
  wall_reference_points: WallCalibrationPoint[];
  hook_reference?: Record<string, unknown> | null;
  chair_references?: Array<Record<string, unknown>> | null;
}

export interface WallCalibrationRequest {
  video_id: string;
  calibration_frame: number;
  calibration_time_sec: number;
  setup: WallCalibrationSetup;
  video_override?: Record<string, unknown> | null;
  intrinsics?: Record<string, unknown> | null;
  manual_corrections?: Record<string, unknown> | null;
  trim_start_frame?: number | null;
  trim_end_frame?: number | null;
}

export interface WallCalibrationResponse {
  video_id: string;
  point_count: number;
  rms_m?: number | null;
}

export interface WallCalibrationGetResponse {
  video_id: string;
  calibration_frame: number;
  calibration_time_sec: number;
  calibration: Record<string, unknown>;
  point_count: number;
  rms_m?: number | null;
  trim_start_frame?: number | null;
  trim_end_frame?: number | null;
}

export interface WallJobStatus {
  status: string;
  phase?: string | null;
  error?: string | null;
  result?: Record<string, unknown> | null;
}

export interface WallAnalyzeResponse {
  status: string;
  message: string;
}

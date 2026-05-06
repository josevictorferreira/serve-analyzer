import type {
  WallCalibrationGetResponse,
  WallCalibrationRequest,
  WallCalibrationResponse,
  WallJobStatus,
  WallVideoMetadataResponse,
  WallVideoUploadResponse,
} from './wall-types';

export async function uploadWallVideo(file: File): Promise<WallVideoUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('video', file);

    xhr.upload.onprogress = (event) => {
      // Progress can be tracked by the caller if needed
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error('Invalid server response.'));
        }
      } else if (xhr.status === 409) {
        reject(new Error('Another analysis is already in progress. Please wait.'));
      } else if (xhr.status === 413) {
        reject(new Error('The video file is too large.'));
      } else if (xhr.status === 400) {
        try {
          const body = JSON.parse(xhr.responseText);
          reject(new Error(body.detail || 'Invalid video file or format.'));
        } catch {
          reject(new Error('Invalid video file or format.'));
        }
      } else {
        reject(new Error(`Upload failed (${xhr.status}).`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload.'));
    xhr.onabort = () => reject(new Error('Upload aborted.'));

    xhr.open('POST', '/api/wall/video');
    xhr.send(formData);
  });
}

export async function getWallVideoMetadata(videoId: string): Promise<WallVideoMetadataResponse> {
  const response = await fetch(`/api/wall/video/${videoId}/metadata`);
  if (!response.ok) {
    throw new Error('Failed to fetch video metadata');
  }
  return response.json();
}

export async function saveWallCalibration(
  calibration: WallCalibrationRequest
): Promise<WallCalibrationResponse> {
  const response = await fetch('/api/wall/calibration', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(calibration),
  });
  if (!response.ok) {
    throw new Error('Failed to save calibration');
  }
  return response.json();
}

export async function getWallCalibration(): Promise<WallCalibrationGetResponse> {
  const response = await fetch('/api/wall/calibration');
  if (!response.ok) {
    throw new Error('Failed to fetch calibration');
  }
  return response.json();
}

export async function deleteWallCalibration(): Promise<void> {
  const response = await fetch('/api/wall/calibration', { method: 'DELETE' });
  if (!response.ok) {
    throw new Error('Failed to delete calibration');
  }
}

export async function startWallAnalysis(): Promise<{ status: string; message: string }> {
  const response = await fetch('/api/wall/analyze', { method: 'POST' });
  if (response.status === 409) {
    throw new Error('Another analysis is already in progress. Please wait.');
  }
if (!response.ok) {
throw new Error('Failed to start analysis');
}
return response.json();
}

export async function getWallJob(): Promise<WallJobStatus> {
  const response = await fetch('/api/wall/job');
  if (!response.ok) {
    throw new Error('Failed to fetch wall job status');
  }
  return response.json();
}

export async function resetWallJob(): Promise<void> {
  const response = await fetch('/api/wall/job/reset', { method: 'POST' });
  if (!response.ok) {
    throw new Error('Failed to reset wall job');
  }
}

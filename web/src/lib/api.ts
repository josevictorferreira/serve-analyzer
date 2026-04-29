import type {
  AnnotationBbox,
  AnnotationEvaluation,
  AnnotationExport,
  AnnotationSession,
  AnnotationSessionSummary,
  DetectorVersionsResponse,
  JobStatus,
} from './types';

const VIDEO_EXTENSIONS = ['.mov', '.mp4', '.avi', '.webm', '.mkv'];

function isValidVideoFile(file: File): boolean {
  if (file.type.startsWith('video/')) {
    return true;
  }
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return VIDEO_EXTENSIONS.includes(ext);
}

export async function getJob(): Promise<JobStatus> {
  const response = await fetch('/api/job');
  if (!response.ok) {
    throw new Error('Failed to fetch job status');
  }
  return response.json();
}

export async function resetJob(): Promise<void> {
  const response = await fetch('/api/job/reset', {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error('Failed to reset job');
  }
}

export async function listDetectorVersions(): Promise<DetectorVersionsResponse> {
  const response = await fetch('/api/detectors');
  if (!response.ok) {
    throw new Error('Failed to fetch detector versions');
  }
  return response.json();
}

export async function analyzeVideoWithProgress(
  file: File,
  onProgress: (percent: number) => void,
  detectorVersion?: string
): Promise<void> {
  if (!isValidVideoFile(file)) {
    throw new Error('Please select a valid video file.');
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('video', file);
    if (detectorVersion) {
      formData.append('detector_version', detectorVersion);
    }

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        onProgress(percent);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else if (xhr.status === 409) {
        reject(new Error('Another analysis is already in progress. Please wait.'));
      } else if (xhr.status === 413) {
        reject(new Error('The video file is too large.'));
      } else if (xhr.status === 400) {
        reject(new Error('Invalid video file or format.'));
      } else {
        reject(new Error(`Upload failed (${xhr.status}).`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload.'));
    xhr.onabort = () => reject(new Error('Upload aborted.'));

    xhr.open('POST', '/api/analyze');
    xhr.send(formData);
  });
}

export async function listAnnotationSessions(): Promise<AnnotationSessionSummary[]> {
  const response = await fetch('/api/annotation/sessions');
  if (!response.ok) {
    throw new Error('Failed to fetch annotation sessions');
  }
  const data = await response.json();
  return data.sessions;
}

export async function getAnnotationSession(sessionId: string): Promise<AnnotationSession> {
  const response = await fetch(`/api/annotation/sessions/${sessionId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch annotation session');
  }
  const data = await response.json();
  return data.session;
}

export async function createAnnotationSessionWithProgress(
  file: File,
  options: { maxFrames: number; prelabel: boolean },
  onProgress: (percent: number) => void
): Promise<AnnotationSession> {
  if (!isValidVideoFile(file)) {
    throw new Error('Please select a valid video file.');
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    const query = new URLSearchParams({
      max_frames: String(options.maxFrames),
      prelabel: String(options.prelabel),
    });
    formData.append('video', file);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText).session);
        } catch (_err) {
          reject(new Error('Annotation session response was invalid.'));
        }
      } else if (xhr.status === 400) {
        reject(new Error('Invalid annotation video or options.'));
      } else {
        reject(new Error(`Annotation upload failed (${xhr.status}).`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during annotation upload.'));
    xhr.onabort = () => reject(new Error('Annotation upload aborted.'));
    xhr.open('POST', `/api/annotation/sessions?${query.toString()}`);
    xhr.send(formData);
  });
}

export async function reviewAnnotationFrame(
  sessionId: string,
  frameId: string,
  action: 'accept' | 'correct' | 'absent' | 'skip',
  bbox?: AnnotationBbox
): Promise<AnnotationSession> {
  const response = await fetch(
    `/api/annotation/sessions/${sessionId}/frames/${frameId}/review`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, bbox }),
    }
  );
  if (!response.ok) {
    throw new Error('Failed to save annotation review');
  }
  const data = await response.json();
  return data.session;
}

export async function undoAnnotationFrame(
  sessionId: string,
  frameId: string
): Promise<AnnotationSession> {
  const response = await fetch(
    `/api/annotation/sessions/${sessionId}/frames/${frameId}/undo`,
    { method: 'POST' }
  );
  if (!response.ok) {
    throw new Error('Failed to undo annotation review');
  }
  const data = await response.json();
  return data.session;
}

export async function exportAnnotationDataset(sessionId: string): Promise<AnnotationExport> {
  const response = await fetch(`/api/annotation/sessions/${sessionId}/export`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error('Export requires at least one accepted, corrected, or absent frame.');
  }
  const data = await response.json();
  return data.export;
}

export async function evaluateAnnotationBaseline(
  sessionId: string
): Promise<AnnotationEvaluation> {
  const response = await fetch(`/api/annotation/sessions/${sessionId}/baseline`);
  if (!response.ok) {
    throw new Error('Baseline evaluation requires reviewed labels.');
  }
  const data = await response.json();
  return data.evaluation;
}

export async function getTrainingEnvironment(): Promise<Record<string, unknown>> {
  const response = await fetch('/api/annotation/training-environment');
  if (!response.ok) {
    throw new Error('Failed to check training environment');
  }
  const data = await response.json();
  return data.environment;
}

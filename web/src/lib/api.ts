import type { JobStatus } from './types';

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

export async function analyzeVideoWithProgress(
  file: File,
  onProgress: (percent: number) => void
): Promise<void> {
  // Validate file type - check MIME or extension fallback
const VIDEO_EXTENSIONS = ['.mov', '.mp4', '.avi', '.webm', '.mkv'];
function isValidVideoFile(file: File): boolean {
  if (file.type.startsWith('video/')) {
    return true;
  }
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return VIDEO_EXTENSIONS.includes(ext);
}

if (!isValidVideoFile(file)) {
  throw new Error('Please select a valid video file.');
}

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('video', file);

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

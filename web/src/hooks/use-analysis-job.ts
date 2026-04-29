import { useState, useEffect, useCallback, useRef } from 'react';
import { getJob, analyzeVideoWithProgress, resetJob } from '@/lib/api';
import type { JobStatus, JobPhase } from '@/lib/types';

export function useAnalysisJob() {
  const [phase, setPhase] = useState<JobPhase>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | undefined>();
  const [jobStatus, setJobStatus] = useState<JobStatus | undefined>();
  const [isUploading, setIsUploading] = useState(false);
  const [estimatedDurationSec, setEstimatedDurationSec] = useState<number | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);
  const analysisStartRef = useRef<number | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    analysisStartRef.current = null;
  }, []);

  const startProgressTimer = useCallback((estimatedSec: number) => {
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    analysisStartRef.current = Date.now();
    setAnalysisProgress(0);
    progressTimerRef.current = setInterval(() => {
      if (!analysisStartRef.current) return;
      const elapsed = (Date.now() - analysisStartRef.current) / 1000;
      const pct = Math.min(90, (elapsed / estimatedSec) * 100);
      setAnalysisProgress(pct);
    }, 1000);
  }, []);

  const poll = useCallback(async () => {
    try {
      const status = await getJob();
      setJobStatus(status);
      if (status.estimated_duration_sec != null) {
        setEstimatedDurationSec(status.estimated_duration_sec);
      }

      // Check status.status for terminal/active states FIRST (authoritative), then phase (progress hint)
      if (status.status === 'error') {
        setPhase('error');
      } else if (status.status === 'done') {
        setPhase('done');
      } else if (status.status === 'clipping') {
        setPhase('clipping');
      } else if (status.status === 'idle') {
        setPhase('idle');
      } else if (status.phase) {
        setPhase(status.phase);
      } else if (status.status === 'analyzing') {
        setPhase('analyzing');
      }

      setError(status.error);

      if (status.status === 'done' || status.status === 'error') {
        stopPolling();
      }

      // Start progress timer on first analyzing poll with estimate
      if ((status.status === 'analyzing' || status.phase === 'analyzing')
          && status.estimated_duration_sec != null
          && status.estimated_duration_sec > 0
          && analysisStartRef.current === null) {
        startProgressTimer(status.estimated_duration_sec);
      }

      // Update progress for clipping/done phases
      if (status.status === 'clipping' || status.status === 'done') {
        setAnalysisProgress(status.status === 'done' ? 100 : 95);
        if (progressTimerRef.current) {
          clearInterval(progressTimerRef.current);
          progressTimerRef.current = null;
        }
      }
    } catch (err) {
      console.error('Polling error:', err);
      setError('Connection lost. Still trying to reach server...');
    }
  }, [stopPolling, startProgressTimer]);

  const startPolling = useCallback(() => {
    stopPolling();
    poll(); // Initial check
    pollIntervalRef.current = setInterval(poll, 2000);
  }, [poll, stopPolling]);

  const upload = useCallback(async (file: File, detectorVersion?: string) => {
    setIsUploading(true);
    setPhase('uploading');
    setProgress(0);
    setError(undefined);
    setJobStatus(undefined);
    setEstimatedDurationSec(null);
    setAnalysisProgress(0);

    try {
      if (detectorVersion) {
        await analyzeVideoWithProgress(file, (p) => setProgress(p), detectorVersion);
      } else {
        await analyzeVideoWithProgress(file, (p) => setProgress(p));
      }
      setIsUploading(false);
      setPhase('analyzing');
      startPolling();
    } catch (err) {
      setIsUploading(false);
      setPhase('error');
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  }, [startPolling]);

  const reset = useCallback(async () => {
    stopPolling();
    try {
      await resetJob();
    } catch (err) {
      console.error('Reset error:', err);
    }
    setPhase('idle');
    setProgress(0);
    setError(undefined);
    setJobStatus(undefined);
    setIsUploading(false);
    setEstimatedDurationSec(null);
    setAnalysisProgress(0);
    analysisStartRef.current = null;
  }, [stopPolling]);

  // Sync on mount
  useEffect(() => {
    const init = async () => {
      try {
        const status = await getJob();
        if (status.status !== 'idle') {
          startPolling();
        }
      } catch (err) {
        console.error('Initial sync error:', err);
      }
    };
    init();
  }, [startPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    phase,
    progress,
    error,
    jobStatus,
    upload,
    reset,
    isUploading,
    estimatedDurationSec,
    analysisProgress
  };
}

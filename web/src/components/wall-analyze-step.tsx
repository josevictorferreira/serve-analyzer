import { useCallback, useEffect, useRef, useState } from 'react';
import { startWallAnalysis, getWallJob, resetWallJob } from '@/lib/wall-api';
import type { WallJobStatus } from '@/lib/wall-types';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Play, RotateCcw, AlertCircle, Loader2, CheckCircle2 } from 'lucide-react';

const PHASE_LABELS: Record<string, string> = {
  idle: 'Preparing...',
  uploading: 'Uploading...',
  calibrating: 'Calibrating...',
  analyzing: 'Analyzing ball trajectory...',
  artifacting: 'Generating artifacts...',
  done: 'Analysis complete!',
  error: 'Analysis failed.',
};

const ACTIVE_PHASES = ['idle', 'uploading', 'calibrating', 'analyzing', 'artifacting'] as const;

interface WallAnalyzeStepProps {
  onDone: (result: Record<string, unknown>) => void;
  onError: (error: string) => void;
  isCalibrated: boolean;
}

type LocalState =
  | { kind: 'idle' }
  | { kind: 'busy' }
  | { kind: 'polling'; phase: string }
  | { kind: 'error'; message: string }
  | { kind: 'done' };

export function WallAnalyzeStep({ onDone, onError, isCalibrated }: WallAnalyzeStepProps) {
  const [local, setLocal] = useState<LocalState>({ kind: 'idle' });
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPolling = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Cleanup interval on unmount
  useEffect(() => clearPolling, [clearPolling]);

  const pollJob = useCallback(() => {
    getWallJob()
      .then((job: WallJobStatus) => {
        const phase = job.phase ?? 'idle';

        if (job.status === 'done' || phase === 'done') {
          clearPolling();
          setLocal({ kind: 'done' });
          if (job.result) {
            onDone(job.result as Record<string, unknown>);
          }
          return;
        }

        if (job.status === 'error' || phase === 'error') {
          clearPolling();
          const msg = job.error ?? 'Unknown error occurred.';
          setLocal({ kind: 'error', message: msg });
          onError(msg);
          return;
        }

        setLocal({ kind: 'polling', phase });
      })
      .catch(() => {
        // Transient polling error — keep polling, don't break the flow
      });
  }, [clearPolling, onDone, onError]);

  const handleStart = useCallback(async () => {
    try {
      await startWallAnalysis();
    } catch (err) {
      const message = err instanceof Error ? err.message : '';
      if (message.includes('already in progress') || message.includes('Conflict')) {
        setLocal({ kind: 'busy' });
        return;
      }
      setLocal({ kind: 'error', message });
      onError(message);
      return;
    }

    // 202 accepted — start polling
    setLocal({ kind: 'polling', phase: 'idle' });
    pollJob(); // immediate first poll
    intervalRef.current = setInterval(pollJob, 1000);
  }, [pollJob, onError]);

  const handleReset = useCallback(async () => {
    clearPolling();
    try {
      await resetWallJob();
    } catch {
      // Local state is cleared regardless
    }
    setLocal({ kind: 'idle' });
  }, [clearPolling]);

  const isAnalyzing = local.kind === 'polling';
  const isDone = local.kind === 'done';
  const isError = local.kind === 'error';
  const isBusy = local.kind === 'busy';

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle className="text-center">Analyze</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Idle state */}
        {local.kind === 'idle' && (
          <div className="flex flex-col items-center gap-4">
            <p className="text-sm text-muted-foreground text-center">
              Start the wall serve analysis pipeline and track progress.
            </p>
            <Button
              onClick={handleStart}
              disabled={!isCalibrated}
              className="gap-2"
              size="lg"
            >
              <Play className="w-4 h-4" />
              Start Analysis
            </Button>
            {!isCalibrated && (
              <p className="text-xs text-muted-foreground">
                Complete calibration before starting analysis.
              </p>
            )}
          </div>
        )}

        {/* Busy / 409 state */}
        {isBusy && (
          <div className="flex flex-col items-center gap-4">
            <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400">
              <AlertCircle className="w-5 h-5" />
              <p className="text-sm font-medium">Another analysis is running</p>
            </div>
            <p className="text-sm text-muted-foreground text-center">
              Reset to clear the current job and try again.
            </p>
            <Button onClick={handleReset} variant="outline" className="gap-2">
              <RotateCcw className="w-4 h-4" />
              Reset and Retry
            </Button>
          </div>
        )}

        {/* Polling / progress state */}
        {isAnalyzing && (
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <p className="text-sm font-medium">
              {PHASE_LABELS[local.phase] ?? 'Processing...'}
            </p>
            {/* Progress bar */}
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full animate-pulse"
                style={{ width: '60%' }}
              />
            </div>
          </div>
        )}

        {/* Error state */}
        {isError && (
          <div className="flex flex-col items-center gap-4">
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="w-5 h-5" />
              <p className="text-sm font-medium">Analysis failed</p>
            </div>
            <p className="text-sm text-muted-foreground text-center">
              {local.message}
            </p>
            <Button onClick={handleReset} variant="outline" className="gap-2">
              <RotateCcw className="w-4 h-4" />
              Reset
            </Button>
          </div>
        )}

        {/* Done state */}
        {isDone && (
          <div className="flex flex-col items-center gap-4">
            <CheckCircle2 className="w-8 h-8 text-green-600 dark:text-green-400" />
            <p className="text-sm font-medium text-green-600 dark:text-green-400">
              {PHASE_LABELS.done}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

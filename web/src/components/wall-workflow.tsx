import { useCallback, useState } from 'react';
import { WallUploadStep, WallMetadataDisplay } from './wall-upload-step';
import { WallCalibrationCanvas, type CalibrationPoint } from './wall-calibration-canvas';
import { WallAssumptionsForm } from './wall-assumptions-form';
import { WallAnalyzeStep } from './wall-analyze-step';
import { WallResultsDashboard } from './wall-results-dashboard';
import { resetWallJob } from '@/lib/wall-api';
import type { WallVideoUploadResponse } from '@/lib/wall-types';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { RotateCcw, Check, ChevronRight } from 'lucide-react';

const STEPS = [
  { key: 'upload', label: 'Upload' },
  { key: 'calibrate', label: 'Calibrate' },
  { key: 'configure', label: 'Configure' },
  { key: 'analyze', label: 'Analyze' },
  { key: 'results', label: 'Results' },
] as const;

type StepKey = (typeof STEPS)[number]['key'];
type WorkflowPhase = 'idle' | 'uploaded' | 'calibrated' | 'configured' | 'analyzing' | 'done';

function phaseToActiveStep(phase: WorkflowPhase): StepKey {
  switch (phase) {
    case 'idle': return 'upload';
    case 'uploaded': return 'calibrate';
    case 'calibrated': return 'configure';
    case 'configured': return 'analyze';
    case 'analyzing': return 'analyze';
    case 'done': return 'results';
  }
}

function getStepStatus(stepKey: StepKey, activeStep: StepKey, phase: WorkflowPhase): 'completed' | 'active' | 'upcoming' {
  const stepOrder = STEPS.map(s => s.key);
  const stepIndex = stepOrder.indexOf(stepKey);
  const activeIndex = stepOrder.indexOf(activeStep);

  if (stepIndex < activeIndex) return 'completed';
  if (stepIndex === activeIndex) return 'active';
  return 'upcoming';
}

export function WallWorkflow() {
  const [phase, setPhase] = useState<WorkflowPhase>('idle');
  const [videoData, setVideoData] = useState<WallVideoUploadResponse | null>(null);
  const [resetting, setResetting] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [calibrationPoints, setCalibrationPoints] = useState<CalibrationPoint[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);

  const activeStep = phaseToActiveStep(phase);

  const handleUploadComplete = useCallback((data: WallVideoUploadResponse) => {
    setVideoData(data);
    setPhase('uploaded');
    setCalibrationPoints([]);
    setCurrentFrame(0);
  }, []);

  const handleReset = useCallback(async () => {
    setResetting(true);
    try {
      await resetWallJob();
    } catch {
      // Silently fail — local state is cleared regardless
    } finally {
      setVideoData(null);
      setPhase('idle');
      setCalibrationPoints([]);
      setCurrentFrame(0);
      setAnalysisResult(null);
      setAnalysisError(null);
      setResetting(false);
    }
  }, []);

  return (
    <div className="space-y-8">
      {/* Stepper */}
      <nav className="flex items-center justify-center gap-0" aria-label="Wall analysis workflow">
        {STEPS.map((step, index) => {
          const status = getStepStatus(step.key, activeStep, phase);
          const isCompleted = status === 'completed';
          const isActive = status === 'active';

          return (
            <div key={step.key} className="flex items-center">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`
                    flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold transition-all
                    ${isCompleted ? 'bg-primary text-primary-foreground' : ''}
                    ${isActive ? 'bg-primary/20 text-primary ring-2 ring-primary' : ''}
                    ${status === 'upcoming' ? 'bg-muted text-muted-foreground' : ''}
                  `}
                >
                  {isCompleted ? <Check className="h-4 w-4" /> : index + 1}
                </div>
                <span className={`text-xs font-medium ${isActive ? 'text-foreground' : 'text-muted-foreground'}`}>
                  {step.label}
                </span>
              </div>
              {index < STEPS.length - 1 && (
                <ChevronRight className="mx-2 h-4 w-4 text-muted-foreground/50" />
              )}
            </div>
          );
        })}
      </nav>

      {/* Step Content */}
      <div className="flex flex-col items-center">
        {activeStep === 'upload' && (
          <WallUploadStep onUploadComplete={handleUploadComplete} disabled={resetting} />
        )}

        {activeStep === 'calibrate' && videoData && (
          <div className="w-full max-w-3xl space-y-6">
            <WallMetadataDisplay data={videoData} />
            <WallCalibrationCanvas
              videoUrl={videoData.video_url}
              videoMetadata={videoData}
              points={calibrationPoints}
              onPointsChange={setCalibrationPoints}
              currentFrame={currentFrame}
              onFrameChange={setCurrentFrame}
            />
            <WallAssumptionsForm
              calibrationPoints={calibrationPoints}
              videoId={videoData.video_id}
              calibrationFrame={currentFrame}
              fps={videoData.fps}
              onCalibrated={() => setPhase('calibrated')}
            />
          </div>
        )}

        {activeStep === 'configure' && (
          <PlaceholderCard title="Configure">
            <p>Configure analysis parameters — contact height, wall distance, and more.</p>
          </PlaceholderCard>
        )}

        {activeStep === 'analyze' && (
          <WallAnalyzeStep
            isCalibrated={phase === 'configured'}
            onDone={(result) => {
              setAnalysisResult(result);
              setPhase('done');
            }}
            onError={(error) => {
              setAnalysisError(error);
            }}
          />
        )}

        {activeStep === 'results' && analysisResult && (
          <div className="w-full space-y-6">
            {videoData && <WallMetadataDisplay data={videoData} />}
            <WallResultsDashboard result={analysisResult as Record<string, unknown>} />
          </div>
        )}
      </div>

      {/* Reset */}
      {phase !== 'idle' && (
        <div className="flex justify-center">
          <Button variant="outline" onClick={handleReset} disabled={resetting} className="gap-2">
            <RotateCcw className="w-4 h-4" />
            {resetting ? 'Resetting...' : 'Start Over'}
          </Button>
        </div>
      )}
    </div>
  );
}

function PlaceholderCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="w-full max-w-2xl border-dashed">
      <CardHeader>
        <CardTitle className="text-center">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex items-center justify-center h-48 text-muted-foreground text-center">
        {children}
      </CardContent>
    </Card>
  );
}

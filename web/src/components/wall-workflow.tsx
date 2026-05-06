import { useCallback, useState } from 'react';
import { WallUploadStep, WallMetadataDisplay } from './wall-upload-step';
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

  const activeStep = phaseToActiveStep(phase);

  const handleUploadComplete = useCallback((data: WallVideoUploadResponse) => {
    setVideoData(data);
    setPhase('uploaded');
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
          <div className="w-full max-w-2xl space-y-6">
            <WallMetadataDisplay data={videoData} />
            <Card className="border-dashed">
              <CardHeader>
                <CardTitle className="text-center">Calibration</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-center h-48 text-muted-foreground">
                <p>Calibration step coming next — mark wall reference points on the video.</p>
              </CardContent>
            </Card>
          </div>
        )}

        {activeStep === 'configure' && (
          <PlaceholderCard title="Configure">
            <p>Configure analysis parameters — contact height, wall distance, and more.</p>
          </PlaceholderCard>
        )}

        {activeStep === 'analyze' && (
          <PlaceholderCard title="Analyze">
            <p>Start the wall serve analysis pipeline and track progress.</p>
          </PlaceholderCard>
        )}

        {activeStep === 'results' && videoData && (
          <div className="w-full max-w-2xl space-y-6">
            <WallMetadataDisplay data={videoData} />
            <PlaceholderCard title="Results">
              <p>View wall impact analysis, velocity estimates, and landing projections.</p>
            </PlaceholderCard>
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

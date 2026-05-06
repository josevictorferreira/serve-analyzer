import { useState, useEffect } from 'react'
import { useAnalysisJob } from './hooks/use-analysis-job'
import { UploadDropzone } from './components/upload-dropzone'
import { AnnotationWorkspace } from './components/annotation-workspace'
import { Timeline } from './components/timeline'
import { VideoPlayer } from './components/video-player'
import { listDetectorVersions } from './lib/api'
import type { DetectorVersionInfo } from './lib/types'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { AlertCircle, Clock, Gauge, RefreshCw, Target, Trophy } from 'lucide-react'
import { WallWorkflow } from './components/wall-workflow'

const PHASE_LABELS: Record<string, string> = {
  idle: 'Idle',
  uploading: 'Uploading video...',
  analyzing: 'Detecting serves...',
  clipping: 'Generating clips...',
  done: 'Analysis complete',
  error: 'Error'
}

const FALLBACK_DETECTORS: DetectorVersionInfo[] = [
  {
    version: 'v1',
    label: 'V1 baseline',
    description: 'Existing candidate generator and selector.',
  },
  {
    version: 'v2',
    label: 'V2 continuity refinement',
    description: 'Continuity, history, and motion-cue refinement.',
  },
]

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m === 0) return `${s}s`;
  return `${m}m ${s}s`;
}

function App() {
  const { phase, progress, error, jobStatus, upload, reset, estimatedDurationSec, analysisProgress } = useAnalysisJob()
  const [activeClipIndex, setActiveClipIndex] = useState(0)
  const [mode, setMode] = useState<'analysis' | 'annotation' | 'wall'>('analysis')
  const [detectorVersions, setDetectorVersions] = useState(FALLBACK_DETECTORS)
  const [selectedDetectorVersion, setSelectedDetectorVersion] = useState('v1')

  const clips = jobStatus?.clips || []
  const candidates = jobStatus?.selected_serves || []

  // Reset to first clip when new results arrive
  useEffect(() => {
    if (clips.length > 0) {
      setActiveClipIndex(0)
    }
  }, [clips.length])

  const activeClip = clips[activeClipIndex]
  const activeCandidate = candidates[activeClipIndex]
  const activeVelocity = activeClip?.velocity_kmh ?? activeCandidate?.post_contact_max_kmh
  const detectorLabel = jobStatus?.detector_label
    || detectorVersions.find((detector) => detector.version === jobStatus?.detector_version)?.label
    || (jobStatus?.detector === 'tracknetv2' ? 'TrackNetV2' : 'YOLO/HSV')

  useEffect(() => {
    listDetectorVersions()
      .then((response) => {
        if (response.detectors.length > 0) {
          setDetectorVersions(response.detectors)
        }
        setSelectedDetectorVersion(response.default_version)
      })
      .catch(() => {
        setDetectorVersions(FALLBACK_DETECTORS)
      })
  }, [])

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto py-4 px-6 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Serve Analyzer</h1>
          <div className="flex items-center gap-2">
            <Button
              variant={mode === 'analysis' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setMode('analysis')}
            >
              Analyze Serves
            </Button>
            <Button
              variant={mode === 'annotation' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setMode('annotation')}
            >
              Annotate Ball
            </Button>
            <Button
              variant={mode === 'wall' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setMode('wall')}
            >
              Wall Analysis
            </Button>
            {mode === 'analysis' && (phase === 'done' || phase === 'error') && (
              <Button variant="ghost" size="sm" onClick={reset} className="gap-2">
                <RefreshCw className="w-4 h-4" />
                New Analysis
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto py-8 px-6">
        {mode === 'wall' ? (
          <WallWorkflow />
        ) : mode === 'annotation' ? (
          <AnnotationWorkspace />
        ) : phase === 'idle' ? (
          <div className="h-full flex flex-col items-center justify-center py-12">
            <UploadDropzone
              onFileSelect={upload}
              detectorVersions={detectorVersions}
              selectedDetectorVersion={selectedDetectorVersion}
              onDetectorVersionChange={setSelectedDetectorVersion}
            />
          </div>
        ) : phase === 'done' && clips.length > 0 ? (
          <div className="space-y-6">
            <section className="overflow-hidden rounded-[2rem] border border-slate-200 bg-[radial-gradient(circle_at_top_left,_#e0f2fe,_transparent_32%),linear-gradient(135deg,_#f8fafc,_#e2e8f0)] p-5 shadow-xl shadow-slate-200/80 sm:p-7">
              <div className="mb-6 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white/70 px-3 py-1 text-[11px] font-black uppercase tracking-[0.22em] text-slate-600 shadow-sm backdrop-blur">
                    <Target className="h-3.5 w-3.5 text-cyan-600" />
                    Serve Review
                  </div>
                  <h2 className="text-3xl font-black tracking-tight text-slate-950 sm:text-5xl">
                    Serve #{activeClipIndex + 1}
                    <span className="block text-cyan-700">ball flight inspection</span>
                  </h2>
                </div>

                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl border border-white/70 bg-white/80 p-4 shadow-sm backdrop-blur">
                    <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-wider text-slate-500">
                      <Gauge className="h-4 w-4 text-cyan-600" />
                      Velocity
                    </div>
                    <div className="mt-1 text-2xl font-black tabular-nums text-slate-950">
                      {activeVelocity ? Math.round(activeVelocity) : '--'}
                      <span className="ml-1 text-xs font-bold text-slate-500">km/h</span>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/70 bg-white/80 p-4 shadow-sm backdrop-blur">
                    <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-wider text-slate-500">
                      <Clock className="h-4 w-4 text-amber-600" />
                      Contact
                    </div>
                    <div className="mt-1 text-2xl font-black tabular-nums text-slate-950">
                      {activeClip?.contact_time_sec.toFixed(2)}
                      <span className="ml-1 text-xs font-bold text-slate-500">s</span>
                    </div>
                  </div>

                  <div className="col-span-2 rounded-2xl border border-white/70 bg-white/80 p-4 shadow-sm backdrop-blur sm:col-span-1">
                    <div className="text-[11px] font-black uppercase tracking-wider text-slate-500">Detector</div>
                    <div className="mt-1 text-lg font-black text-slate-950">
                      {detectorLabel}
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
                <div className="space-y-4">
              {activeClip ? (
                <VideoPlayer clip={activeClip} />
              ) : (
                <Card className="aspect-video bg-slate-100 flex items-center justify-center border-dashed">
                  <p className="text-slate-400 font-medium">No clip selected</p>
                </Card>
              )}

                  <div className="flex flex-col gap-3 rounded-3xl border border-white/70 bg-white/75 p-4 shadow-sm backdrop-blur sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <div className="text-sm font-black text-slate-950">Clean clip with live metadata overlay</div>
                      <p className="text-sm text-slate-600">
                        The ball marker appears only on frames where the detector produced a coordinate. Native video controls stay unobstructed.
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        onClick={() => setActiveClipIndex((prev) => Math.max(prev - 1, 0))}
                        disabled={activeClipIndex === 0}
                      >
                        Previous
                      </Button>
                      <Button
                        onClick={() => setActiveClipIndex((prev) => Math.min(prev + 1, clips.length - 1))}
                        disabled={activeClipIndex >= clips.length - 1}
                      >
                        Next Serve
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="min-h-[28rem] xl:h-[calc(100vh-13rem)]">
                  <Timeline
                    clips={clips}
                    candidates={candidates}
                    activeClipIndex={activeClipIndex}
                    onClipSelect={setActiveClipIndex}
                  />
                </div>
              </div>
            </section>
          </div>
        ) : phase === 'done' && clips.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center py-12 space-y-4">
            <Trophy className="w-16 h-12 text-slate-300" />
            <h2 className="text-xl font-bold">No serves detected</h2>
            <p className="text-slate-500 text-center max-w-md">
              The analyzer couldn't find any clear serve sequences in this video. 
              Try a video with better lateral visibility.
            </p>
            <Button variant="outline" onClick={reset}>Try another video</Button>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center py-12">
            <div className="w-full max-w-2xl space-y-8">
              <Card className={phase === 'error' ? 'border-destructive' : ''}>
                <CardHeader>
                  <CardTitle className="text-center">
                    {phase === 'error' ? 'Analysis Failed' : 'Processing Video'}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6 py-8">
                  {phase === 'error' ? (
                    <div className="flex flex-col items-center space-y-4 text-destructive">
                      <AlertCircle className="w-12 h-12" />
                      <p className="text-center font-medium">{error || 'An unknown error occurred'}</p>
                      <Button variant="outline" onClick={reset}>Try Again</Button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex justify-between text-sm font-medium">
                        <span>{PHASE_LABELS[phase] || phase}</span>
                        {phase === 'uploading' ? (
                          <span>{progress}%</span>
                        ) : phase === 'analyzing' && analysisProgress > 0 ? (
                          <span>{Math.round(analysisProgress)}%</span>
                        ) : null}
                      </div>
                      <Progress 
                        value={phase === 'uploading' ? progress : phase === 'analyzing' ? analysisProgress || null : phase === 'clipping' ? 95 : phase === 'done' ? 100 : null} 
                        className="w-full" 
                      />
                      {phase === 'analyzing' && estimatedDurationSec && estimatedDurationSec > 0 && (
                        <p className="text-center text-sm text-muted-foreground">
                          Estimated time remaining: {formatDuration(Math.max(0, estimatedDurationSec * ((90 - analysisProgress) / 90)))}
                        </p>
                      )}
                      {phase === 'analyzing' && (!estimatedDurationSec || estimatedDurationSec <= 0) && (
                        <p className="text-center text-sm text-muted-foreground italic">
                          This might take a few minutes for long videos...
                        </p>
                      )}
                      {phase === 'clipping' && (
                        <p className="text-center text-sm text-muted-foreground italic">
                          Almost done! Extracting serve clips...
                        </p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App

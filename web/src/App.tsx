import { useState, useEffect } from 'react'
import { useAnalysisJob } from './hooks/use-analysis-job'
import { UploadDropzone } from './components/upload-dropzone'
import { Timeline } from './components/timeline'
import { VideoPlayer } from './components/video-player'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { AlertCircle, RefreshCw, Trophy } from 'lucide-react'

const PHASE_LABELS: Record<string, string> = {
  idle: 'Idle',
  uploading: 'Uploading video...',
  analyzing: 'Detecting serves...',
  clipping: 'Generating clips...',
  done: 'Analysis complete',
  error: 'Error'
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m === 0) return `${s}s`;
  return `${m}m ${s}s`;
}

function App() {
  const { phase, progress, error, jobStatus, upload, reset, estimatedDurationSec, analysisProgress } = useAnalysisJob()
  const [activeClipIndex, setActiveClipIndex] = useState(0)

  const clips = jobStatus?.clips || []
  const candidates = jobStatus?.selected_serves || []

  // Reset to first clip when new results arrive
  useEffect(() => {
    if (clips.length > 0) {
      setActiveClipIndex(0)
    }
  }, [clips.length])

  const activeClip = clips[activeClipIndex]

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto py-4 px-6 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Serve Analyzer</h1>
          {(phase === 'done' || phase === 'error') && (
            <Button variant="ghost" size="sm" onClick={reset} className="gap-2">
              <RefreshCw className="w-4 h-4" />
              New Analysis
            </Button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto py-8 px-6">
        {phase === 'idle' ? (
          <div className="h-full flex flex-col items-center justify-center py-12">
            <UploadDropzone onFileSelect={upload} />
          </div>
        ) : phase === 'done' && clips.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full">
            {/* Video Player Section */}
            <div className="lg:col-span-2 space-y-4">
              {activeClip ? (
                <VideoPlayer 
                  clip={activeClip} 
                  onNext={() => setActiveClipIndex((prev) => Math.min(prev + 1, clips.length - 1))}
                  onPrev={() => setActiveClipIndex((prev) => Math.max(prev - 1, 0))}
                  hasNext={activeClipIndex < clips.length - 1}
                  hasPrev={activeClipIndex > 0}
                />
              ) : (
                <Card className="aspect-video bg-slate-100 flex items-center justify-center border-dashed">
                  <p className="text-slate-400 font-medium">No clip selected</p>
                </Card>
              )}

              <div className="flex items-center justify-between bg-white p-4 rounded-xl border shadow-sm">
                <div>
                  <h2 className="text-lg font-black text-slate-900">Serve #{activeClipIndex + 1} Analysis</h2>
                  <p className="text-sm text-slate-500 font-medium">Automatic detection via HSV tracking & kinematic profiling</p>
                </div>
                {candidates[activeClipIndex]?.post_contact_max_kmh && (
                  <div className="text-right">
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Estimated Speed</div>
                    <div className="text-3xl font-black text-indigo-600 tabular-nums">
                      {Math.round(candidates[activeClipIndex].post_contact_max_kmh)}
                      <span className="text-sm ml-1 text-indigo-400">KM/H</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Sidebar / Timeline */}
            <div className="lg:col-span-1 h-[calc(100vh-12rem)]">
              <Timeline 
                clips={clips} 
                candidates={candidates}
                activeClipIndex={activeClipIndex}
                onClipSelect={setActiveClipIndex}
              />
            </div>
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

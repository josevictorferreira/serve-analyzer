import { useEffect, useState } from 'react'
import { Check, ChevronLeft, ChevronRight, Download, RotateCcw, Search, X } from 'lucide-react'
import { UploadDropzone } from './upload-dropzone'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import {
  createAnnotationSessionWithProgress,
  evaluateAnnotationBaseline,
  exportAnnotationDataset,
  getAnnotationSession,
  getTrainingEnvironment,
  listAnnotationSessions,
  reviewAnnotationFrame,
  undoAnnotationFrame,
} from '@/lib/api'
import type {
  AnnotationBbox,
  AnnotationEvaluation,
  AnnotationExport,
  AnnotationFrame,
  AnnotationSession,
  AnnotationSessionSummary,
} from '@/lib/types'

const DEFAULT_CORRECTION_BOX_PX = 32

function frameImageUrl(session: AnnotationSession, frame: AnnotationFrame): string {
  return `/api/annotation/sessions/${session.id}/frames/${frame.frame_id}/image`
}

function firstPendingIndex(session: AnnotationSession): number {
  const index = session.frames.findIndex((frame) => frame.status === 'pending')
  return index >= 0 ? index : 0
}

function nextReviewIndex(session: AnnotationSession, currentIndex: number): number {
  const next = session.frames.findIndex((frame, index) => index > currentIndex && frame.status === 'pending')
  if (next >= 0) return next
  const wrap = session.frames.findIndex((frame) => frame.status === 'pending')
  return wrap >= 0 ? wrap : currentIndex
}

function progressPercent(session: AnnotationSession): number {
  if (session.progress.total === 0) return 0
  return Math.round((session.progress.reviewed / session.progress.total) * 100)
}

function statusClass(status: string): string {
  if (status === 'accepted') return 'bg-emerald-100 text-emerald-700'
  if (status === 'corrected') return 'bg-blue-100 text-blue-700'
  if (status === 'absent') return 'bg-slate-200 text-slate-700'
  if (status === 'skipped') return 'bg-amber-100 text-amber-700'
  return 'bg-zinc-100 text-zinc-600'
}

function clampCorrectionBox(x: number, y: number, frame: AnnotationFrame): AnnotationBbox {
  const size = Math.min(DEFAULT_CORRECTION_BOX_PX, frame.width, frame.height)
  return {
    x: Math.max(0, Math.min(frame.width - size, x - size / 2)),
    y: Math.max(0, Math.min(frame.height - size, y - size / 2)),
    width: size,
    height: size,
  }
}

function BboxOverlay({
  bbox,
  label,
  color,
  dashed,
}: {
  bbox: AnnotationBbox
  label: string
  color: string
  dashed?: boolean
}) {
  return (
    <g>
      <rect
        x={bbox.x}
        y={bbox.y}
        width={bbox.width}
        height={bbox.height}
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeDasharray={dashed ? '12 8' : undefined}
        vectorEffect="non-scaling-stroke"
      />
      <text
        x={bbox.x}
        y={Math.max(18, bbox.y - 8)}
        fill={color}
        fontSize="18"
        fontWeight="700"
        paintOrder="stroke"
        stroke="black"
        strokeWidth="4"
        vectorEffect="non-scaling-stroke"
      >
        {label}
      </text>
    </g>
  )
}

export function AnnotationWorkspace() {
  const [sessions, setSessions] = useState<AnnotationSessionSummary[]>([])
  const [session, setSession] = useState<AnnotationSession | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [maxFrames, setMaxFrames] = useState(240)
  const [prelabel, setPrelabel] = useState(true)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isCreating, setIsCreating] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exportResult, setExportResult] = useState<AnnotationExport | null>(null)
  const [evaluation, setEvaluation] = useState<AnnotationEvaluation | null>(null)
  const [environment, setEnvironment] = useState<Record<string, unknown> | null>(null)

  const currentFrame = session?.frames[currentIndex]

  useEffect(() => {
    listAnnotationSessions()
      .then(setSessions)
      .catch(() => setSessions([]))
  }, [])

  useEffect(() => {
    if (!session || !currentFrame) return
    const activeSession = session
    const activeFrame = currentFrame

    function onKeyDown(event: KeyboardEvent) {
      if (event.target instanceof HTMLInputElement) return
      if (event.key === 'ArrowLeft') {
        setCurrentIndex((index) => Math.max(0, index - 1))
      } else if (event.key === 'ArrowRight') {
        setCurrentIndex((index) => Math.min(activeSession.frames.length - 1, index + 1))
      } else if (event.key.toLowerCase() === 'a') {
        event.preventDefault()
        if (activeFrame.prediction) void saveReview('accept')
      } else if (event.key.toLowerCase() === 'n') {
        event.preventDefault()
        void saveReview('absent')
      } else if (event.key.toLowerCase() === 's') {
        event.preventDefault()
        void saveReview('skip')
      } else if (event.key.toLowerCase() === 'u') {
        event.preventDefault()
        void undoReview()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [session, currentFrame])

  async function refreshSessions() {
    try {
      setSessions(await listAnnotationSessions())
    } catch (_err) {
      setSessions([])
    }
  }

  async function createSession(file: File) {
    setIsCreating(true)
    setError(null)
    setExportResult(null)
    setEvaluation(null)
    setUploadProgress(0)
    try {
      const created = await createAnnotationSessionWithProgress(
        file,
        { maxFrames, prelabel },
        setUploadProgress
      )
      setSession(created)
      setCurrentIndex(firstPendingIndex(created))
      await refreshSessions()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create annotation session')
    } finally {
      setIsCreating(false)
    }
  }

  async function loadSession(sessionId: string) {
    setError(null)
    setExportResult(null)
    setEvaluation(null)
    try {
      const loaded = await getAnnotationSession(sessionId)
      setSession(loaded)
      setCurrentIndex(firstPendingIndex(loaded))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load annotation session')
    }
  }

  async function saveReview(
    action: 'accept' | 'correct' | 'absent' | 'skip',
    bbox?: AnnotationBbox
  ) {
    if (!session || !currentFrame) return
    setIsSaving(true)
    setError(null)
    try {
      const updated = await reviewAnnotationFrame(session.id, currentFrame.frame_id, action, bbox)
      setSession(updated)
      setCurrentIndex(nextReviewIndex(updated, currentIndex))
      await refreshSessions()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save review')
    } finally {
      setIsSaving(false)
    }
  }

  async function undoReview() {
    if (!session || !currentFrame) return
    setIsSaving(true)
    setError(null)
    try {
      const updated = await undoAnnotationFrame(session.id, currentFrame.frame_id)
      setSession(updated)
      await refreshSessions()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to undo review')
    } finally {
      setIsSaving(false)
    }
  }

  async function exportDataset() {
    if (!session) return
    setError(null)
    try {
      setExportResult(await exportAnnotationDataset(session.id))
      const updated = await getAnnotationSession(session.id)
      setSession(updated)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export dataset')
    }
  }

  async function evaluateBaseline() {
    if (!session) return
    setError(null)
    try {
      setEvaluation(await evaluateAnnotationBaseline(session.id))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to evaluate baseline')
    }
  }

  async function checkEnvironment() {
    setError(null)
    try {
      setEnvironment(await getTrainingEnvironment())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check training environment')
    }
  }

  function handleFrameClick(event: React.MouseEvent<SVGSVGElement>) {
    if (!currentFrame || isSaving) return
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * currentFrame.width
    const y = ((event.clientY - rect.top) / rect.height) * currentFrame.height
    void saveReview('correct', clampCorrectionBox(x, y, currentFrame))
  }

  if (!session) {
    return (
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(320px,1fr)]">
        <Card className="border-dashed">
          <CardHeader>
            <CardTitle>Build Tennis-Ball Dataset</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Upload a serve video to extract review frames, optionally seed them with RJTPP predictions, and export a YOLO dataset after review.
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="space-y-1 text-sm font-medium">
                Max frames
                <input
                  className="w-full rounded-md border px-3 py-2"
                  type="number"
                  min={1}
                  max={2000}
                  value={maxFrames}
                  onChange={(event) => setMaxFrames(Number(event.target.value))}
                />
              </label>
              <label className="flex items-end gap-2 rounded-md border px-3 py-2 text-sm font-medium">
                <input
                  type="checkbox"
                  checked={prelabel}
                  onChange={(event) => setPrelabel(event.target.checked)}
                />
                Run RJTPP pre-labeling
              </label>
            </div>
            <UploadDropzone onFileSelect={createSession} disabled={isCreating} />
            {isCreating && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Uploading and extracting frames</span>
                  <span>{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} />
              </div>
            )}
            {error && <p className="text-sm font-medium text-destructive">{error}</p>}
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Resume Session</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {sessions.length === 0 ? (
                <p className="text-sm text-muted-foreground">No annotation sessions found.</p>
              ) : (
                sessions.slice(0, 6).map((item) => (
                  <button
                    key={item.id}
                    className="w-full rounded-lg border p-3 text-left transition-colors hover:bg-muted"
                    onClick={() => void loadSession(item.id)}
                  >
                    <div className="font-medium">{item.source_filename}</div>
                    <div className="text-xs text-muted-foreground">
                      {item.progress.reviewed}/{item.progress.total} reviewed · {item.id}
                    </div>
                  </button>
                ))
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Training Environment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button variant="outline" onClick={() => void checkEnvironment()} className="gap-2">
                <Search className="h-4 w-4" />
                Check PyTorch / Ultralytics
              </Button>
              {environment && (
                <pre className="max-h-56 overflow-auto rounded-md bg-muted p-3 text-xs">
                  {JSON.stringify(environment, null, 2)}
                </pre>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-2xl font-bold">Tennis-Ball Annotation</h2>
          <p className="text-sm text-muted-foreground">
            {session.source_filename} · {session.progress.reviewed}/{session.progress.total} reviewed · {session.progress.exportable} exportable
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => setSession(null)}>Sessions</Button>
          <Button variant="outline" onClick={() => void evaluateBaseline()}>Evaluate RJTPP</Button>
          <Button onClick={() => void exportDataset()} className="gap-2">
            <Download className="h-4 w-4" />
            Export YOLO
          </Button>
        </div>
      </div>

      <Card>
        <CardContent className="space-y-2 pt-0">
          <div className="flex justify-between text-sm font-medium">
            <span>{progressPercent(session)}% reviewed</span>
            <span>
              Accepted {session.progress.accepted} · Corrected {session.progress.corrected} · Absent {session.progress.absent} · Skipped {session.progress.skipped}
            </span>
          </div>
          <Progress value={progressPercent(session)} />
        </CardContent>
      </Card>

      {error && <p className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm font-medium text-destructive">{error}</p>}

      {currentFrame && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center justify-between gap-3">
                <span>Frame {currentFrame.frame_number}</span>
                <span className={`rounded-full px-3 py-1 text-xs font-bold uppercase ${statusClass(currentFrame.status)}`}>
                  {currentFrame.status}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative overflow-hidden rounded-xl border bg-black">
                <img
                  src={frameImageUrl(session, currentFrame)}
                  alt={`Frame ${currentFrame.frame_number}`}
                  className="block w-full select-none"
                  draggable={false}
                />
                <svg
                  className="absolute inset-0 h-full w-full cursor-crosshair"
                  viewBox={`0 0 ${currentFrame.width} ${currentFrame.height}`}
                  onClick={handleFrameClick}
                  role="img"
                  aria-label="Click the tennis ball to correct the label"
                >
                  {currentFrame.prediction?.bbox && (
                    <BboxOverlay
                      bbox={currentFrame.prediction.bbox}
                      label={`RJTPP ${Math.round(currentFrame.prediction.confidence * 100)}%`}
                      color="#facc15"
                      dashed
                    />
                  )}
                  {currentFrame.label?.bbox && (
                    <BboxOverlay
                      bbox={currentFrame.label.bbox}
                      label={currentFrame.label.source === 'manual' ? 'Manual label' : 'Accepted label'}
                      color="#22c55e"
                    />
                  )}
                </svg>
              </div>
              <p className="text-sm text-muted-foreground">
                Click directly on the ball to save a corrected {DEFAULT_CORRECTION_BOX_PX}px box. The dashed yellow box is RJTPP; the solid green box is the saved label.
              </p>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Review Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button
                  className="w-full gap-2"
                  disabled={!currentFrame.prediction || isSaving}
                  onClick={() => void saveReview('accept')}
                >
                  <Check className="h-4 w-4" />
                  Accept Prediction (A)
                </Button>
                <Button
                  variant="outline"
                  className="w-full gap-2"
                  disabled={isSaving}
                  onClick={() => void saveReview('absent')}
                >
                  <X className="h-4 w-4" />
                  Mark No Ball (N)
                </Button>
                <Button
                  variant="outline"
                  className="w-full"
                  disabled={isSaving}
                  onClick={() => void saveReview('skip')}
                >
                  Skip (S)
                </Button>
                <Button
                  variant="ghost"
                  className="w-full gap-2"
                  disabled={isSaving}
                  onClick={() => void undoReview()}
                >
                  <RotateCcw className="h-4 w-4" />
                  Undo (U)
                </Button>
                <div className="grid grid-cols-2 gap-2 pt-2">
                  <Button
                    variant="outline"
                    disabled={currentIndex === 0}
                    onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))}
                  >
                    <ChevronLeft className="h-4 w-4" />
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    disabled={currentIndex >= session.frames.length - 1}
                    onClick={() => setCurrentIndex((index) => Math.min(session.frames.length - 1, index + 1))}
                  >
                    Next
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Frame Metadata</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div>Time: {currentFrame.time_sec.toFixed(2)}s</div>
                <div>Split: {currentFrame.split}</div>
                <div>Size: {currentFrame.width} x {currentFrame.height}</div>
                <div>
                  Prediction: {currentFrame.prediction ? `${Math.round(currentFrame.prediction.confidence * 100)}% confidence` : 'none'}
                </div>
              </CardContent>
            </Card>

            {(exportResult || evaluation) && (
              <Card>
                <CardHeader>
                  <CardTitle>Results</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  {evaluation && (
                    <div className="rounded-lg border p-3">
                      <div className="font-bold">RJTPP Baseline</div>
                      <div>Precision: {(evaluation.precision * 100).toFixed(1)}%</div>
                      <div>Recall: {(evaluation.recall * 100).toFixed(1)}%</div>
                      <div>Detected visible frames: {evaluation.detected_visible_frames}/{evaluation.visible_frames}</div>
                    </div>
                  )}
                  {exportResult && (
                    <div className="rounded-lg border p-3">
                      <div className="font-bold">YOLO Export</div>
                      <div className="break-all">{exportResult.data_yaml}</div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

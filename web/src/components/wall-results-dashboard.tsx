import { useEffect, useRef, useState } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  CheckCircle2,
  XCircle,
  Download,
  FileJson,
  FileSpreadsheet,
  Clock,
  Zap,
  MapPin,
  AlertTriangle,
  Info,
  Video,
  Play,
  Image,
} from "lucide-react"

// --- Type helpers for the six-section result ---

interface WallResult {
  measured?: Record<string, unknown>
  inferred?: Record<string, unknown>
  assumed?: Record<string, unknown>
  confidence?: number | Record<string, unknown>
  warnings?: Array<{ code?: string; message: string }>
  artifacts?: Record<string, unknown>
}

// --- Helpers ---

function fmt(n: unknown, decimals = 2): string {
  if (n == null) return "—"
  const v = typeof n === "number" ? n : Number(n)
  if (isNaN(v)) return "—"
  return v.toFixed(decimals)
}

function pct(n: unknown): string {
  if (n == null) return "—"
  const v = typeof n === "number" ? n : Number(n)
  return `${(v * 100).toFixed(1)}%`
}

function getArtifactUrl(
  artifacts: Record<string, unknown> | undefined,
  ...keys: string[]
): string | undefined {
  let curr: unknown = artifacts
  for (const k of keys) {
    if (curr == null || typeof curr !== "object") return undefined
    curr = (curr as Record<string, unknown>)[k]
    if (curr == null) return undefined
    if (typeof curr === "object" && "url" in curr) {
      return (curr as { url?: string }).url
    }
  }
  return typeof curr === "string" ? curr : undefined
}

function confidenceColor(confidence: number): string {
  if (confidence >= 0.8) return "text-green-600 dark:text-green-400"
  if (confidence >= 0.5) return "text-amber-600 dark:text-amber-400"
  return "text-red-600 dark:text-red-400"
}

function confidenceBadgeVariant(confidence: number): "default" | "secondary" | "destructive" | "outline" {
  if (confidence >= 0.8) return "default"
  if (confidence >= 0.5) return "secondary"
  return "destructive"
}

// --- Sub-components ---

function MeasuredImpactCard({ measured }: { measured: Record<string, unknown> }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          Measured Impact
        </CardTitle>
        <CardDescription>Direct measurements from video analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="space-y-3 text-sm">
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Impact Time</dt>
            <dd className="font-mono font-semibold">{fmt(measured.impact_time_sec)} s</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Impact Frame</dt>
            <dd className="font-mono">{fmt(measured.impact_frame, 0)}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Wall Position</dt>
            <dd className="font-mono font-semibold">
              {fmt(measured.wall_x_m)} × {fmt(measured.wall_y_m)} m
            </dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Autonomous Frame</dt>
            <dd className="font-mono">{fmt(measured.autonomous_frame, 0)}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Calibration RMS</dt>
            <dd className="font-mono">{fmt(measured.calibration_reprojection_rms_px)} px</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Raw Track Samples</dt>
            <dd className="font-mono">{fmt(measured.raw_track_samples, 0)}</dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  )
}

function VelocityCard({ inferred }: { inferred: Record<string, unknown> }) {
  const speedMs = inferred.speed_m_s
  const speedKmh = inferred.speed_km_h
  const speedMph = inferred.speed_mph
  const uncertainty = inferred.speed_uncertainty_m_s

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-amber-500" />
          Velocity
        </CardTitle>
        <CardDescription>Estimated ball speed at impact</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Peak speed highlighted */}
          <div className="flex items-baseline justify-between rounded-lg bg-muted/50 px-3 py-2">
            <span className="text-sm font-medium text-muted-foreground">Peak Speed</span>
            <span className="text-2xl font-bold font-mono">{fmt(speedMs)} m/s</span>
          </div>

          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-muted-foreground">km/h</dt>
              <dd className="font-mono font-semibold">{fmt(speedKmh)} km/h</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted-foreground">mph</dt>
              <dd className="font-mono font-semibold">{fmt(speedMph)} mph</dd>
            </div>
            {uncertainty != null && (
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Uncertainty (±)</dt>
                <dd className="font-mono">{fmt(uncertainty)} m/s</dd>
              </div>
            )}
          </dl>
        </div>
      </CardContent>
    </Card>
  )
}

function CourtProjectionCard({ inferred }: { inferred: Record<string, unknown> }) {
  const inBox = inferred.in_service_box
  const inBoxBool = inBox === true

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-muted-foreground" />
          Court Projection
        </CardTitle>
        <CardDescription>Gravity-only landing projection (no spin/drag)</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="space-y-3 text-sm">
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Landing X</dt>
            <dd className="font-mono">{fmt(inferred.landing_x_m)} m</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Landing Z</dt>
            <dd className="font-mono">{fmt(inferred.landing_z_m)} m</dd>
          </div>
          <div className="flex justify-between items-center">
            <dt className="text-muted-foreground">In Service Box</dt>
            <dd className="flex items-center gap-1.5">
              {inBoxBool ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              <span className={inBoxBool ? "text-green-600 dark:text-green-400 font-semibold" : "text-red-600 dark:text-red-400 font-semibold"}>
                {inBoxBool ? "IN" : "OUT"}
              </span>
            </dd>
          </div>
          {inferred.service_box_side != null && (
            <div className="flex justify-between">
              <dt className="text-muted-foreground">Service Box</dt>
              <dd className="font-mono">{String(inferred.service_box_side)}</dd>
            </div>
          )}
        </dl>
      </CardContent>
    </Card>
  )
}

function AnnotatedVideoCard({
  artifacts,
  impactTime: externalImpactTime,
}: {
  artifacts: Record<string, unknown>
  impactTime?: number | null
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [impactTime, setImpactTime] = useState<number | null>(externalImpactTime ?? null)

  // Sync external value if it changes
  useEffect(() => {
    if (externalImpactTime != null) {
      setImpactTime(externalImpactTime)
    }
  }, [externalImpactTime])

  const videoUrl = getArtifactUrl(artifacts, "annotated_video")

  const handleJumpToImpact = () => {
    if (videoRef.current && impactTime != null) {
      videoRef.current.currentTime = impactTime
      videoRef.current.play()
    }
  }

  if (!videoUrl) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-4 w-4 text-muted-foreground" />
          Annotated Video
        </CardTitle>
        <CardDescription>Ball tracking visualization with impact marker</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <video
          ref={videoRef}
          controls
          className="w-full max-w-xl rounded-lg"
          src={videoUrl}
        />
        <div className="flex items-center gap-2">
          <input
            type="number"
            step="0.01"
            min="0"
            placeholder="Impact time (s)"
            className="flex h-9 w-36 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            value={impactTime ?? ""}
            onChange={(e) => setImpactTime(e.target.value ? Number(e.target.value) : null)}
          />
          <Button
variant="outline"
size="sm"
className="gap-1.5"
disabled={impactTime == null}
onClick={handleJumpToImpact}
          >
            <Play className="h-3.5 w-3.5" />
            Jump to Impact
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function ReviewClipCard({
  artifacts,
}: {
  artifacts: Record<string, unknown>
}) {
  const clip = artifacts.review_clip as Record<string, unknown> | undefined
  if (!clip) return null

  const clipUrl = typeof clip.url === "string" ? clip.url : undefined
  if (!clipUrl) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-4 w-4 text-muted-foreground" />
          Review Clip
        </CardTitle>
        <CardDescription>Trimmed clip around the impact event</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <video controls className="w-full max-w-xl rounded-lg" src={clipUrl} />
        <dl className="flex gap-6 text-xs text-muted-foreground">
          {clip.start_time != null && (
            <div>
              <span className="font-medium">Start:</span> {fmt(clip.start_time)}s
            </div>
          )}
          {clip.impact_time != null && (
            <div>
              <span className="font-medium">Impact:</span> {fmt(clip.impact_time)}s
            </div>
          )}
          {clip.end_time != null && (
            <div>
              <span className="font-medium">End:</span> {fmt(clip.end_time)}s
            </div>
          )}
        </dl>
      </CardContent>
    </Card>
  )
}

function PlotsGallery({ artifacts }: { artifacts: Record<string, unknown> }) {
  const plots = artifacts.plots as Record<string, unknown> | undefined
  const [enlarged, setEnlarged] = useState<string | null>(null)

  if (!plots) return null

  const normalizedPlots = Object.entries(plots)
    .map(([name, value]) => ({
      name,
      url: typeof value === "string" ? value : (value as Record<string, unknown>)?.url as string | undefined,
    }))
    .filter((p): p is { name: string; url: string } => !!p.url)

  if (normalizedPlots.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Image className="h-4 w-4 text-muted-foreground" />
          Analysis Plots
        </CardTitle>
        <CardDescription>Speed profile, wall impact, and court landing visualizations</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {normalizedPlots.map(({ name, url }) => (
            <button
              key={name}
              type="button"
              className="group relative cursor-zoom-in overflow-hidden rounded-lg border bg-card"
              onClick={() => setEnlarged(enlarged === name ? null : name)}
            >
              <img
                src={url}
                alt={`${name} plot`}
                className="w-full transition-transform duration-200 group-hover:scale-105"
              />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent px-2 py-1 text-xs text-white">
                {name.replace(/_/g, " ")}
              </div>
            </button>
          ))}
        </div>
      </CardContent>
      {enlarged && normalizedPlots.find(p => p.name === enlarged) && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setEnlarged(null)}
        >
          <img
            src={normalizedPlots.find(p => p.name === enlarged)!.url}
            alt={`${enlarged} plot enlarged`}
            className="max-h-[90vh] max-w-[90vw] rounded-lg shadow-2xl"
          />
        </div>
      )}
    </Card>
  )
}

function ConfidenceWarningsCard({
  confidence,
  warnings,
}: {
  confidence?: number | Record<string, unknown>
  warnings?: Array<{ code?: string; message: string }>
}) {
  const scoreValue = typeof confidence === "number" ? confidence : ((confidence as Record<string, unknown>)?.score) as number | undefined
  const scoreNum = scoreValue ?? null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {scoreNum != null && scoreNum >= 0.8 ? (
            <CheckCircle2 className="h-4 w-4 text-green-500" />
          ) : (
            <AlertTriangle className="h-4 w-4 text-amber-500" />
          )}
          Confidence & Warnings
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Confidence score */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Confidence Score</span>
          <Badge variant={scoreNum != null ? confidenceBadgeVariant(scoreNum) : "outline"}>
            {pct(scoreNum ?? 0)}
          </Badge>
        </div>

        {/* Warnings */}
        {warnings && warnings.length > 0 ? (
          <div className="space-y-2">
            <span className="text-sm font-medium text-amber-600 dark:text-amber-400">
              {warnings.length} warning{warnings.length > 1 ? "s" : ""}
            </span>
            <ul className="space-y-1.5">
              {warnings.map((w, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                  <span>
                    {w.code && (
                      <Badge variant="secondary" className="mr-1 text-[10px]">
                        {w.code}
                      </Badge>
                    )}
                    {w.message}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="text-sm text-green-600 dark:text-green-400">No warnings — analysis looks clean.</p>
        )}
      </CardContent>
    </Card>
  )
}

function AssumptionsCard({ assumed }: { assumed: Record<string, unknown> }) {
  const entries = Object.entries(assumed).filter(
    ([, v]) => v !== null && v !== undefined
  )

  if (entries.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Info className="h-4 w-4 text-muted-foreground" />
          Assumptions
        </CardTitle>
        <CardDescription>Parameters used by the analysis pipeline</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="space-y-2 text-sm">
          {entries.map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <dt className="text-muted-foreground">
                {key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
              </dt>
              <dd className="font-mono">
                {typeof value === "boolean"
                  ? value
                    ? "Yes"
                    : "No"
                  : typeof value === "number"
                  ? fmt(value, 3)
                  : String(value)}
              </dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  )
}

function DownloadLinksCard({ artifacts }: { artifacts: Record<string, unknown> }) {
  const jsonUrl = getArtifactUrl(artifacts, "json")
  const csvUrl = getArtifactUrl(artifacts, "csv")

  if (!jsonUrl && !csvUrl) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="h-4 w-4 text-muted-foreground" />
          Export Data
        </CardTitle>
        <CardDescription>Download raw analysis results</CardDescription>
      </CardHeader>
      <CardFooter className="flex gap-3">
        {jsonUrl && (
          <a
            href={jsonUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex shrink-0 items-center justify-center rounded-lg border border-transparent bg-clip-padding text-sm font-medium whitespace-nowrap transition-all outline-none select-none h-7 gap-1.5 px-2.5 text-[0.8rem] border-input bg-background hover:bg-muted hover:text-foreground aria-expanded:bg-muted aria-expanded:text-foreground dark:border-input dark:bg-input/30 dark:hover:bg-input/50"
          >
            <FileJson className="h-3.5 w-3.5" />
            result.json
          </a>
        )}
        {csvUrl && (
          <a
            href={csvUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex shrink-0 items-center justify-center rounded-lg border border-transparent bg-clip-padding text-sm font-medium whitespace-nowrap transition-all outline-none select-none h-7 gap-1.5 px-2.5 text-[0.8rem] border-input bg-background hover:bg-muted hover:text-foreground aria-expanded:bg-muted aria-expanded:text-foreground dark:border-input dark:bg-input/30 dark:hover:bg-input/50"
          >
            <FileSpreadsheet className="h-3.5 w-3.5" />
            result.csv
          </a>
        )}
      </CardFooter>
    </Card>
  )
}

// --- Main Dashboard ---

interface WallResultsDashboardProps {
  result: WallResult
}

export function WallResultsDashboard({ result }: WallResultsDashboardProps) {
  const {
    measured,
    inferred,
    assumed = {},
    confidence,
    warnings,
    artifacts = {},
  } = result

  // Ensure measured/inferred are objects for safe access
  const measuredObj = measured ?? {}
  const inferredObj = inferred ?? {}
  const artifactsObj = artifacts as Record<string, unknown>

  return (
    <div className="w-full max-w-6xl space-y-6">
      {/* Row 1: Measured Impact + Velocity */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <MeasuredImpactCard measured={measuredObj} />
        <VelocityCard inferred={inferredObj} />
      </div>

      {/* Row 2: Court Projection + Annotated Video */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <CourtProjectionCard inferred={inferredObj} />
        <AnnotatedVideoCard
          artifacts={artifactsObj}
          impactTime={typeof measuredObj.impact_time_sec === "number" ? measuredObj.impact_time_sec : null}
        />
      </div>

      {/* Row 3: Review Clip (if available) */}
      {artifactsObj.review_clip != null && (
        <div className="grid grid-cols-1">
          <ReviewClipCard artifacts={artifactsObj} />
        </div>
      )}

      {/* Row 4: Confidence & Warnings */}
      <ConfidenceWarningsCard confidence={confidence} warnings={warnings} />

      {/* Row 5: Assumptions */}
      <AssumptionsCard assumed={assumed as Record<string, unknown>} />

      {/* Row 6: Plots Gallery */}
      <PlotsGallery artifacts={artifactsObj} />

      {/* Row 7: Download Links */}
      <div className="flex justify-end">
        <DownloadLinksCard artifacts={artifactsObj} />
      </div>
    </div>
  )
}

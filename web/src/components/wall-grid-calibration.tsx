import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { WallVideoMetadataResponse } from '@/lib/wall-types';
import { saveWallCalibration, getWallCalibration, deleteWallCalibration } from '@/lib/wall-api';
import type { WallCalibrationRequest } from '@/lib/wall-types';
import { CheckCircle, AlertCircle, RotateCcw } from 'lucide-react';

// --- Types ---

interface CornerHandle {
  name: string;
  pixelX: number; // original video pixel coords
  pixelY: number;
  wallMx: number;
  wallMy: number;
}

interface WallGridCalibrationProps {
  videoUrl: string;
  videoMetadata: WallVideoMetadataResponse;
  videoId: string;
  onCalibrated: () => void;
}

// --- Homography (DLT for 4 points) ---

/**
 * Solve 8x8 linear system Ax = b via Gaussian elimination with partial pivoting.
 * Returns x as array of 8 numbers.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = 8;
  // Build augmented matrix
  const M: number[][] = A.map((row, i) => [...row, b[i]]);

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(M[row][col]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    if (maxVal < 1e-12) continue;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) {
        M[row][j] -= factor * M[col][j];
      }
    }
  }

  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = M[i][i] !== 0 ? sum / M[i][i] : 0;
  }
  return x;
}

/**
 * Compute homography matrix H (3x3, flat [h0..h8]) mapping world->pixel.
 * worldPts: array of [wx, wy] (4 points)
 * pixelPts: array of [px, py] (4 points)
 * Returns H such that [px, py, 1]^T ~ H @ [wx, wy, 1]^T
 */
function computeHomography(
  worldPts: [number, number][],
  pixelPts: [number, number][]
): number[] {
  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const [wx, wy] = worldPts[i];
    const [px, py] = pixelPts[i];
    A.push([wx, wy, 1, 0, 0, 0, -wx * px, -wy * px]);
    b.push(px);
    A.push([0, 0, 0, wx, wy, 1, -wx * py, -wy * py]);
    b.push(py);
  }

  const h = solveLinearSystem(A, b);
  return [...h, 1]; // h8 = 1
}

/**
 * Project a world point through homography H to pixel coords.
 * H: flat 3x3 matrix [h0..h8]
 * world: [wx, wy]
 * Returns [px, py]
 */
function projectPoint(H: number[], world: [number, number]): [number, number] {
  const [wx, wy] = world;
  const d = H[6] * wx + H[7] * wy + H[8];
  if (Math.abs(d) < 1e-12) return [0, 0];
  return [(H[0] * wx + H[1] * wy + H[2]) / d, (H[3] * wx + H[4] * wy + H[5]) / d];
}

// --- Component ---

const CORNER_NAMES = ['BL', 'BR', 'TL', 'TR'] as const;

/**
 * Generate a composite localStorage key from video metadata.
 * Format: wall-cal-{filename}-{duration_sec}-{fps}-{frame_count}-{width}x{height}
 */
function getStorageKey(meta: WallVideoMetadataResponse): string {
  return `wall-cal-${meta.filename}-${meta.duration_sec.toFixed(2)}-${meta.fps}-${meta.frame_count}-${meta.width}x${meta.height}`;
}
export function WallGridCalibration({
  videoUrl,
  videoMetadata,
  videoId,
  onCalibrated,
}: WallGridCalibrationProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });
  const [currentFrame, setCurrentFrame] = useState(0);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);

  // Grid dimensions (world meters)
  const [gridWidth, setGridWidth] = useState('2.0');
  const [gridHeight, setGridHeight] = useState('3.0');
  const [bottomEdgeHeight, setBottomEdgeHeight] = useState('0.0');
  const [contactHeight, setContactHeight] = useState('2.80');
  const [contactDistance, setContactDistance] = useState('6.11');
  const [cameraDistance, setCameraDistance] = useState('1.57');

  // Corner handles (pixel coords in original video space)
  const [corners, setCorners] = useState<CornerHandle[]>(() => {
    const gw = 2.0;
    return [
      { name: 'BL', pixelX: 0, pixelY: 0, wallMx: -gw / 2, wallMy: 0 },
      { name: 'BR', pixelX: 0, pixelY: 0, wallMx: gw / 2, wallMy: 0 },
      { name: 'TL', pixelX: 0, pixelY: 0, wallMx: -gw / 2, wallMy: 3.0 },
      { name: 'TR', pixelX: 0, pixelY: 0, wallMx: gw / 2, wallMy: 3.0 },
    ];
  });

  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Seek to frame when currentFrame changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoLoaded || videoMetadata.fps <= 0) return;
    const targetTime = currentFrame / videoMetadata.fps;
    if (Math.abs(video.currentTime - targetTime) > 0.05) {
      video.currentTime = targetTime;
    }
  }, [currentFrame, videoLoaded, videoMetadata.fps]);

  // Track displayed video dimensions for coordinate mapping
  const updateDisplaySize = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    setDisplaySize({ width: video.clientWidth, height: video.clientHeight });
  }, []);

  const handleLoadedMetadata = useCallback(() => {
    updateDisplaySize();
    setVideoLoaded(true);
  }, [updateDisplaySize]);

  // Initialize default corner positions when video is loaded
  useEffect(() => {
    if (!videoLoaded) return;
    const { width, height } = videoMetadata;
    const margin = Math.min(width, height) * 0.15;
    setCorners([
      { name: 'BL', pixelX: margin, pixelY: height - margin, wallMx: -1.0, wallMy: 0 },
      { name: 'BR', pixelX: width - margin, pixelY: height - margin, wallMx: 1.0, wallMy: 0 },
      { name: 'TL', pixelX: margin, pixelY: margin, wallMx: -1.0, wallMy: 3.0 },
      { name: 'TR', pixelX: width - margin, pixelY: margin, wallMx: 1.0, wallMy: 3.0 },
    ]);
  }, [videoLoaded, videoMetadata.width, videoMetadata.height]);

  // Restore calibration from localStorage when video metadata is available
  useEffect(() => {
    try {
      const key = getStorageKey(videoMetadata);
      const raw = localStorage.getItem(key);
      if (raw) {
        const saved = JSON.parse(raw) as Record<string, unknown>;
        if (typeof saved.gridWidth === 'string') setGridWidth(saved.gridWidth);
        if (typeof saved.gridHeight === 'string') setGridHeight(saved.gridHeight);
        if (typeof saved.bottomEdgeHeight === 'string') setBottomEdgeHeight(saved.bottomEdgeHeight);
        if (typeof saved.contactHeight === 'string') setContactHeight(saved.contactHeight);
        if (typeof saved.contactDistance === 'string') setContactDistance(saved.contactDistance);
        if (typeof saved.cameraDistance === 'string') setCameraDistance(saved.cameraDistance);
        if (Array.isArray(saved.corners)) setCorners(saved.corners as CornerHandle[]);
        if (typeof saved.trimStart === 'number') setTrimStart(saved.trimStart);
        if (typeof saved.trimEnd === 'number') setTrimEnd(saved.trimEnd);
        if (typeof saved.currentFrame === 'number') setCurrentFrame(saved.currentFrame);
      }
    } catch {
      // Corrupted or unavailable — silent
    }
  }, [videoMetadata]);


  // Load existing calibration on mount
  useEffect(() => {
    let cancelled = false;
    const loadExisting = async () => {
      setLoading(true);
      try {
        const result = await getWallCalibration();
        if (cancelled) return;
        if (result.calibration_frame !== undefined && result.calibration) {
          const setup = result.calibration as Record<string, unknown>;
          if (typeof setup.serve_contact_height_m === 'number') {
            setContactHeight(String(setup.serve_contact_height_m));
          }
          if (typeof setup.serve_contact_distance_m === 'number') {
            setContactDistance(String(setup.serve_contact_distance_m));
          }
          if (typeof setup.camera_wall_distance_m === 'number') {
            setCameraDistance(String(setup.camera_wall_distance_m));
          }
          const rawPoints = setup.wall_reference_points as Array<Record<string, unknown>> | undefined;
          if (Array.isArray(rawPoints) && rawPoints.length >= 4) {
            const loadedCorners: CornerHandle[] = rawPoints.slice(0, 4).map((p, i) => {
              const wallM = Array.isArray(p.wall_m) ? p.wall_m : [0, 0];
              return {
                name: (p.name as string) || CORNER_NAMES[i],
                pixelX: Array.isArray(p.pixel) ? Math.round(p.pixel[0] as number) : 0,
                pixelY: Array.isArray(p.pixel) ? Math.round(p.pixel[1] as number) : 0,
                wallMx: Array.isArray(wallM) ? Number(wallM[0]) : 0,
                wallMy: Array.isArray(wallM) ? Number(wallM[1]) : 0,
              };
            });
            setCorners(loadedCorners);

            // Derive grid dimensions from wall_m values
            const xValues = loadedCorners.map((c) => c.wallMx);
            const yValues = loadedCorners.map((c) => c.wallMy);
            const minX = Math.min(...xValues);
            const maxX = Math.max(...xValues);
            const minY = Math.min(...yValues);
            const maxY = Math.max(...yValues);
            setGridWidth(String(maxX - minX));
            setGridHeight(String(maxY - minY));
          }
          setCurrentFrame(result.calibration_frame);
          setSuccess(true);
        }
      } catch {
        // No existing calibration — silent
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadExisting();
    return () => {
      cancelled = true;
    };
  }, [videoMetadata]);

  // Draw canvas: grid lines + corner handles + height markers
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoLoaded) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = displaySize.width;
    canvas.height = displaySize.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const gw = parseFloat(gridWidth) || 2.0;
    const gh = parseFloat(gridHeight) || 3.0;
    const beh = parseFloat(bottomEdgeHeight) || 0.0;
    const halfW = gw / 2;

    // Build world corner points (shifted by bottom edge height)
    const worldPts: [number, number][] = [
      [-halfW, beh],
      [halfW, beh],
      [-halfW, beh + gh],
      [halfW, beh + gh],
    ];

    // Build pixel corner points (original video coords → display coords)
    const pixelPts: [number, number][] = corners.map((c) => [
      (c.pixelX / videoMetadata.width) * displaySize.width,
      (c.pixelY / videoMetadata.height) * displaySize.height,
    ]);

    // Compute homography
    const H = computeHomography(worldPts, pixelPts);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.4)';
    ctx.lineWidth = 1;

    // Horizontal lines (every 0.5m)
    for (let y = 0; y <= gh; y += 0.5) {
      const start = projectPoint(H, [-halfW, y]);
      const end = projectPoint(H, [halfW, y]);
      ctx.beginPath();
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(end[0], end[1]);
      ctx.stroke();
    }

    // Vertical lines (every 0.5m)
    for (let x = -halfW; x <= halfW; x += 0.5) {
      const start = projectPoint(H, [x, 0]);
      const end = projectPoint(H, [x, gh]);
      ctx.beginPath();
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(end[0], end[1]);
      ctx.stroke();
    }

    // Height reference markers on left edge (only draw if >= bottom edge height)
    const heightRefs = [
      { y: 1.0, label: '1.0m (chair)', color: 'rgba(255, 200, 0, 0.7)' },
      { y: 2.45, label: '2.45m (hook)', color: 'rgba(255, 100, 100, 0.7)' },
      { y: 2.8, label: '2.80m (contact)', color: 'rgba(100, 255, 100, 0.7)' },
    ];

    ctx.font = '11px sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    for (const ref of heightRefs) {
      // Skip markers below the grid's bottom edge
      if (ref.y < beh) continue;

      const pt = projectPoint(H, [-halfW, ref.y]);
      const leftEdge = projectPoint(H, [-halfW - 0.1, ref.y]);
      ctx.strokeStyle = ref.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(leftEdge[0], leftEdge[1]);
      ctx.lineTo(pt[0] + 20, pt[1]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = ref.color;
      ctx.fillText(ref.label, pt[0] + 24, pt[1]);
    }

    // Draw corner handles
    const handleRadius = 12;
    corners.forEach((corner, i) => {
      const [dx, dy] = pixelPts[i];

      // Outer glow
      ctx.beginPath();
      ctx.arc(dx, dy, handleRadius + 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.fill();

      // Main circle
      ctx.beginPath();
      ctx.arc(dx, dy, handleRadius, 0, Math.PI * 2);
      ctx.fillStyle = draggingIndex === i ? 'rgba(0, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.7)';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(corner.name, dx, dy);
    });
  }, [
    corners,
    displaySize,
    videoLoaded,
    videoMetadata.width,
    videoMetadata.height,
    gridWidth,
    gridHeight,
    draggingIndex,
  ]);

  // Map display coords to original video pixel coords
  const displayToVideoPixel = useCallback(
    (displayX: number, displayY: number): [number, number] => {
      return [
        Math.round((displayX / displaySize.width) * videoMetadata.width),
        Math.round((displayY / displaySize.height) * videoMetadata.height),
      ];
    },
    [displaySize, videoMetadata]
  );

  // Get client coordinates from mouse or touch event
  const getClientCoords = useCallback(
    (e: MouseEvent | TouchEvent): { clientX: number; clientY: number } => {
      if ('touches' in e) {
        const touch = e.touches[0] || e.changedTouches[0];
        return { clientX: touch.clientX, clientY: touch.clientY };
      }
      return { clientX: e.clientX, clientY: e.clientY };
    },
    []
  );

  // Hit test: find corner handle under cursor
  const hitTestCorner = useCallback(
    (clientX: number, clientY: number): number | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const dx = clientX - rect.left;
      const dy = clientY - rect.top;

      for (let i = 0; i < corners.length; i++) {
        const cx = (corners[i].pixelX / videoMetadata.width) * displaySize.width;
        const cy = (corners[i].pixelY / videoMetadata.height) * displaySize.height;
        const dist = Math.sqrt((dx - cx) ** 2 + (dy - cy) ** 2);
        if (dist <= 16) return i; // slightly larger hit area
      }
      return null;
    },
    [corners, displaySize, videoMetadata]
  );

  // Drag handlers
  const handlePointerDown = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      const { clientX, clientY } = getClientCoords(e.nativeEvent as MouseEvent | TouchEvent);
      const hit = hitTestCorner(clientX, clientY);
      if (hit !== null) {
        setDraggingIndex(hit);
        e.preventDefault();
      }
    },
    [getClientCoords, hitTestCorner]
  );

  useEffect(() => {
    if (draggingIndex === null) return;

    const handlePointerMove = (e: MouseEvent | TouchEvent) => {
      e.preventDefault();
      const { clientX, clientY } = getClientCoords(e);
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const displayX = clientX - rect.left;
      const displayY = clientY - rect.top;

      // Clamp to canvas bounds
      const clampedX = Math.max(0, Math.min(displayX, displaySize.width));
      const clampedY = Math.max(0, Math.min(displayY, displaySize.height));

      const [videoX, videoY] = displayToVideoPixel(clampedX, clampedY);

      setCorners((prev) =>
        prev.map((c, i) =>
          i === draggingIndex ? { ...c, pixelX: videoX, pixelY: videoY } : c
        )
      );
    };

    const handlePointerUp = () => {
      setDraggingIndex(null);
    };

    window.addEventListener('mousemove', handlePointerMove, { passive: false });
    window.addEventListener('mouseup', handlePointerUp);
    window.addEventListener('touchmove', handlePointerMove, { passive: false });
    window.addEventListener('touchend', handlePointerUp);

    return () => {
      window.removeEventListener('mousemove', handlePointerMove);
      window.removeEventListener('mouseup', handlePointerUp);
      window.removeEventListener('touchmove', handlePointerMove);
      window.removeEventListener('touchend', handlePointerUp);
    };
  }, [draggingIndex, displaySize, displayToVideoPixel, getClientCoords]);

  // Frame scrubber
  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setCurrentFrame(parseInt(e.target.value, 10));
    },
    []
  );

  // Trim handlers — clamp currentFrame to new range
  const handleTrimStartChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newStart = parseInt(e.target.value, 10);
      setTrimStart(newStart);
      setCurrentFrame((prev) => Math.max(newStart, Math.min(prev, trimEnd)));
    },
    [trimEnd]
  );

  const handleTrimEndChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newEnd = parseInt(e.target.value, 10);
      setTrimEnd(newEnd);
      setCurrentFrame((prev) => Math.max(trimStart, Math.min(prev, newEnd)));
    },
    [trimStart]
  );

  // Initialize trim range when video metadata is available
  useEffect(() => {
    if (videoMetadata.frame_count > 0) {
      setTrimEnd(videoMetadata.frame_count - 1);
    }
  }, [videoMetadata.frame_count]);

  // Clear calibration
  const handleClearCalibration = useCallback(async () => {
    try {
      await deleteWallCalibration();
      // Also clear local storage entry
      const key = getStorageKey(videoMetadata);
      localStorage.removeItem(key);
    } catch {
      // Best effort
    }
    setSuccess(false);
    setError(null);
  }, []);

  // Save calibration
  const handleSave = useCallback(async () => {
    const gw = parseFloat(gridWidth) || 2.0;
    const gh = parseFloat(gridHeight) || 3.0;
    const beh = parseFloat(bottomEdgeHeight) || 0.0;
    const halfW = gw / 2;
    const ch = parseFloat(contactHeight);

    if (isNaN(ch) || ch <= 0) {
      setError('Contact height must be a positive number.');
      return;
    }

    // Validate corners: at least 4 with valid pixel positions
    const validCorners = corners.filter((c) => c.pixelX >= 0 && c.pixelY >= 0);
    if (validCorners.length < 4) {
      setError('At least 4 corner handles must be placed on the video.');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const calibrationTimeSec = currentFrame / videoMetadata.fps;
      const worldPts: [number, number][] = [
        [-halfW, beh],
        [halfW, beh],
        [-halfW, beh + gh],
        [halfW, beh + gh],
      ];

      const wallPoints = corners.map((c, i) => ({
        name: c.name,
        pixel: [c.pixelX, c.pixelY],
        wall_m: [worldPts[i][0], worldPts[i][1]],
      }));

      const request: WallCalibrationRequest = {
        video_id: videoId,
        calibration_frame: currentFrame,
        calibration_time_sec: calibrationTimeSec,
        setup: {
          serve_contact_distance_m: parseFloat(contactDistance) || 6.11,
          camera_wall_distance_m: parseFloat(cameraDistance) || 1.57,
          serve_contact_height_m: ch,
          wall_reference_points: wallPoints,
        },
        trim_start_frame: trimStart,
        trim_end_frame: trimEnd,
      };

      const response = await saveWallCalibration(request);
      if (response.point_count >= 4) {
        setSuccess(true);
        onCalibrated();

        // Persist to localStorage for quick restore
        try {
          const key = getStorageKey(videoMetadata);
          localStorage.setItem(
            key,
            JSON.stringify({
              gridWidth,
              gridHeight,
              bottomEdgeHeight,
              contactHeight,
              contactDistance,
              cameraDistance,
              corners,
              trimStart,
              trimEnd,
              currentFrame,
            }),
          );
        } catch {
          // Storage full or unavailable — non-fatal
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save calibration.');
    } finally {
      setSaving(false);
    }
  }, [
    gridWidth,
    gridHeight,
    bottomEdgeHeight,
    contactHeight,
    corners,
    currentFrame,
    videoMetadata.fps,
    videoId,
    contactDistance,
    cameraDistance,
    onCalibrated,
  ]);

  return (
    <Card className="w-full max-w-3xl">
      <CardHeader>
        <CardTitle className="text-center">Wall Calibration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Video + Canvas Overlay */}
        <div className="relative w-full rounded-lg overflow-hidden border border-border/50 bg-black">
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full block"
            onLoadedMetadata={handleLoadedMetadata}
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
            style={{ touchAction: 'none' }}
            onMouseDown={handlePointerDown}
            onTouchStart={handlePointerDown}
          />
        </div>

        {/* Frame Scrubber & Trim Controls */}
        <div className="space-y-3">
          {/* Trim Range Controls */}
          <div className="space-y-2 rounded-md border border-border/50 bg-muted/30 p-3">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span>Trim Range</span>
              <span>
                Trimmed: frame {trimStart} to {trimEnd} (
                {(trimStart / videoMetadata.fps).toFixed(1)}s – 
                {(trimEnd / videoMetadata.fps).toFixed(1)}s, 
                {((trimEnd - trimStart) / videoMetadata.fps).toFixed(1)}s)
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground" htmlFor="trim-start">
                  Start: frame {trimStart} ({(trimStart / videoMetadata.fps).toFixed(1)}s)
                </label>
                <input
                  id="trim-start"
                  type="range"
                  min={0}
                  max={Math.max(trimStart, trimEnd - 1)}
                  value={trimStart}
                  onChange={handleTrimStartChange}
                  className="w-full accent-primary"
                  aria-label="Trim start frame"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground" htmlFor="trim-end">
                  End: frame {trimEnd} ({(trimEnd / videoMetadata.fps).toFixed(1)}s)
                </label>
                <input
                  id="trim-end"
                  type="range"
                  min={Math.min(trimStart + 1, trimEnd)}
                  max={videoMetadata.frame_count - 1}
                  value={trimEnd}
                  onChange={handleTrimEndChange}
                  className="w-full accent-primary"
                  aria-label="Trim end frame"
                />
              </div>
            </div>
            {/* Visual trim region bar */}
            <div className="relative mt-1 h-2 rounded-full bg-muted">
              <div
                className="absolute top-0 h-2 rounded-full bg-primary/60"
                style={{
                  left: `${(trimStart / (videoMetadata.frame_count - 1)) * 100}%`,
                  width: `${((trimEnd - trimStart) / (videoMetadata.frame_count - 1)) * 100}%`,
                }}
              />
            </div>
          </div>

          {/* Frame Navigation Scrubber (clamped to trim range) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Frame: {currentFrame}</span>
              <span>Time: {(currentFrame / videoMetadata.fps).toFixed(2)}s</span>
              <span>Range: {trimStart}–{trimEnd}</span>
            </div>
            <input
              type="range"
              min={trimStart}
              max={trimEnd}
              value={currentFrame}
              onChange={handleSliderChange}
              className="w-full accent-primary"
              aria-label="Frame scrubber"
            />
          </div>
        </div>

        {/* Grid Settings */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium">Grid Dimensions</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="grid-width">
                Grid Width (m)
              </label>
              <input
                id="grid-width"
                type="number"
                step="0.1"
                value={gridWidth}
                onChange={(e) => setGridWidth(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="grid-height">
                Grid Height (m)
              </label>
              <input
                id="grid-height"
                type="number"
                step="0.1"
                value={gridHeight}
                onChange={(e) => setGridHeight(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="bottom-edge-height">
                Bottom Edge Height from Floor (m)
              </label>
              <input
                id="bottom-edge-height"
                type="number"
                step="0.1"
                value={bottomEdgeHeight}
                onChange={(e) => setBottomEdgeHeight(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
              <p className="text-xs text-muted-foreground">
                Height of the grid&#39;s bottom edge above the floor (0 if bottom edge is at floor level)
              </p>
            </div>
          </div>
        </div>

        {/* Calibration Settings */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium">Calibration Settings</h3>
          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="contact-height">
              Serve Contact Height (m)
            </label>
            <input
              id="contact-height"
              type="number"
              step="0.01"
              value={contactHeight}
              onChange={(e) => setContactHeight(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
            <p className="text-xs text-muted-foreground">
              Height of ball-racket contact above court surface (default: 2.80 m)
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="contact-distance">
                Serve Contact Distance (m)
              </label>
              <input
                id="contact-distance"
                type="number"
                step="0.01"
                value={contactDistance}
                onChange={(e) => setContactDistance(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
              <p className="text-xs text-muted-foreground">Distance from serve line to contact (default: 6.11)</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="camera-distance">
                Camera-Wall Distance (m)
              </label>
              <input
                id="camera-distance"
                type="number"
                step="0.01"
                value={cameraDistance}
                onChange={(e) => setCameraDistance(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
              <p className="text-xs text-muted-foreground">Distance from camera to wall (default: 1.57)</p>
            </div>
          </div>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="flex items-center gap-2 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}
        {success && (
          <div className="flex items-center gap-2 rounded-md border border-emerald-500/50 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-600">
            <CheckCircle className="w-4 h-4 shrink-0" />
            Calibration saved successfully.
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-between gap-3">
          <Button
            variant="outline"
            onClick={handleClearCalibration}
            disabled={loading}
            className="gap-1"
          >
            <RotateCcw className="w-3 h-3" />
            Clear Calibration
          </Button>
          <Button
            onClick={handleSave}
            disabled={saving || loading}
            className="gap-1"
          >
            {saving ? 'Saving...' : 'Save Calibration'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

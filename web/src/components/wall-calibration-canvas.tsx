import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { WallVideoMetadataResponse } from '@/lib/wall-types';
import { Trash2, X } from 'lucide-react';

function uuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export interface CalibrationPoint {
  id: string;
  pixelX: number;
  pixelY: number;
}

export interface WallCalibrationCanvasProps {
  videoUrl: string;
  videoMetadata: WallVideoMetadataResponse;
  points: CalibrationPoint[];
  onPointsChange: (points: CalibrationPoint[]) => void;
  currentFrame: number;
  onFrameChange: (frame: number) => void;
}

export function WallCalibrationCanvas({
  videoUrl,
  videoMetadata,
  points,
  onPointsChange,
  currentFrame,
  onFrameChange,
}: WallCalibrationCanvasProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });

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

  // Redraw canvas markers when points or display size change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoLoaded) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = displaySize.width;
    canvas.height = displaySize.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    points.forEach((point, index) => {
      const x = (point.pixelX / videoMetadata.width) * displaySize.width;
      const y = (point.pixelY / videoMetadata.height) * displaySize.height;

      // Circle background
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Number
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(index + 1), x, y);
    });
  }, [points, displaySize, videoLoaded, videoMetadata.width, videoMetadata.height]);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;

      // Map display coordinates back to original video pixels
      const pixelX = Math.round((clientX / displaySize.width) * videoMetadata.width);
      const pixelY = Math.round((clientY / displaySize.height) * videoMetadata.height);

      const newPoint: CalibrationPoint = {
        id: uuid(),
        pixelX: Math.max(0, Math.min(pixelX, videoMetadata.width - 1)),
        pixelY: Math.max(0, Math.min(pixelY, videoMetadata.height - 1)),
      };
      onPointsChange([...points, newPoint]);
    },
    [displaySize, videoMetadata, points, onPointsChange]
  );

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onFrameChange(parseInt(e.target.value, 10));
    },
    [onFrameChange]
  );

  const handleRemovePoint = useCallback(
    (id: string) => {
      onPointsChange(points.filter((p) => p.id !== id));
    },
    [points, onPointsChange]
  );

  const handleClearAll = useCallback(() => {
    onPointsChange([]);
  }, [onPointsChange]);

  return (
    <Card className="w-full max-w-3xl">
      <CardHeader>
        <CardTitle className="text-center">Calibration Points</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Video + Canvas */}
        <div ref={containerRef} className="relative w-full rounded-lg overflow-hidden border border-border/50 bg-black">
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
            className="absolute inset-0 w-full h-full cursor-crosshair"
            onClick={handleCanvasClick}
          />
        </div>

        {/* Frame Scrubber */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Frame: {currentFrame}</span>
            <span>Time: {(currentFrame / videoMetadata.fps).toFixed(2)}s</span>
            <span>Total: {videoMetadata.frame_count} frames</span>
          </div>
          <input
            type="range"
            min={0}
            max={videoMetadata.frame_count - 1}
            value={currentFrame}
            onChange={handleSliderChange}
            className="w-full accent-primary"
            aria-label="Frame scrubber"
          />
        </div>

        {/* Point List */}
        {points.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">
                Points placed: {points.length}
                {points.length < 4 && (
                  <span className="ml-2 text-xs text-amber-600">
                    (minimum 4 required)
                  </span>
                )}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClearAll}
                className="gap-1 text-destructive hover:text-destructive hover:bg-destructive/10"
              >
                <Trash2 className="w-3 h-3" />
                Clear All
              </Button>
            </div>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {points.map((point, index) => (
                <div
                  key={point.id}
                  className="flex items-center justify-between rounded-md border border-border/50 bg-muted/30 px-3 py-2 text-sm"
                >
                  <div className="flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                      {index + 1}
                    </span>
                    <span className="font-mono text-xs">
                      ({point.pixelX}, {point.pixelY})
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    onClick={() => handleRemovePoint(point.id)}
                    className="h-6 w-6 text-muted-foreground hover:text-destructive"
                    aria-label={`Remove point ${index + 1}`}
                  >
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

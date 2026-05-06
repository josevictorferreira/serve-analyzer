import React, { useCallback, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { uploadWallVideo } from '@/lib/wall-api';
import type { WallVideoUploadResponse } from '@/lib/wall-types';
import { AlertCircle, FileVideo, Upload } from 'lucide-react';

const VIDEO_EXTENSIONS = ['.mov', '.mp4', '.avi', '.webm', '.mkv'];

function isAcceptedVideoFile(file: File): boolean {
  if (file.type.startsWith('video/')) {
    return true;
  }
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return VIDEO_EXTENSIONS.includes(ext);
}

export interface WallUploadStepProps {
  onUploadComplete: (data: WallVideoUploadResponse) => void;
  disabled?: boolean;
}

export function WallUploadStep({ onUploadComplete, disabled }: WallUploadStepProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleUpload = useCallback(async (file: File) => {
    if (!isAcceptedVideoFile(file)) {
      setError('Please select a valid video file (.mov, .mp4, .avi, .webm, .mkv).');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const response = await uploadWallVideo(file);
      onUploadComplete(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred.';
      setError(message);
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (disabled || uploading) return;
    setIsDragging(true);
  }, [disabled, uploading]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled || uploading) return;

    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleUpload(file);
    }
  }, [disabled, uploading, handleUpload]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleUpload(file);
    }
  }, [handleUpload]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  if (uploading) {
    return (
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle className="text-center">Uploading Wall Video</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center h-48 text-muted-foreground">
          <div className="animate-pulse flex flex-col items-center space-y-4">
            <Upload className="w-12 h-12 text-primary" />
            <p className="font-medium">Uploading your video...</p>
            <p className="text-sm">This may take a moment for large files.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full max-w-2xl border-destructive">
        <CardHeader>
          <CardTitle className="text-center text-destructive">Upload Failed</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center h-48 space-y-4">
          <AlertCircle className="w-12 h-12 text-destructive" />
          <p className="text-center font-medium text-destructive">{error}</p>
          <Button variant="outline" onClick={handleClick} disabled={disabled}>
            Try Again
          </Button>
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="video/*,.mov,.mp4,.avi,.webm,.mkv"
            onChange={handleFileChange}
            disabled={disabled}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      className={cn(
        'w-full max-w-2xl border-dashed border-2 transition-colors cursor-pointer',
        isDragging ? 'border-primary bg-primary/5' : 'border-muted-foreground/25',
        (disabled || uploading) && 'opacity-50 cursor-not-allowed'
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={!disabled && !uploading ? handleClick : undefined}
    >
      <CardHeader>
        <CardTitle className="text-center">Upload Wall Video</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="video/*,.mov,.mp4,.avi,.webm,.mkv"
          onChange={handleFileChange}
          disabled={disabled || uploading}
        />
        <div className="flex flex-col items-center space-y-4">
          <div className={cn(
            'p-4 rounded-full bg-muted transition-colors',
            isDragging && 'bg-primary/10 text-primary'
          )}>
            <FileVideo className="w-12 h-12" />
          </div>
          <div className="text-center">
            <p className="font-medium">
              {isDragging ? 'Drop your video here' : 'Drag and drop your wall serve video'}
            </p>
            <p className="text-sm">or click to browse</p>
          </div>
          <Button variant="outline" disabled={disabled || uploading} onClick={(e) => {
            e.stopPropagation();
            handleClick();
          }}>
            Select File
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export function WallMetadataDisplay({ data }: { data: WallVideoUploadResponse }) {
  const formatDuration = (seconds: number): string => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    if (m === 0) return `${s}s`;
    return `${m}m ${s}s`;
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle className="text-center">Video Ready</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          <MetadataCard label="Filename" value={data.filename} />
          <MetadataCard label="Duration" value={formatDuration(data.duration_sec)} />
          <MetadataCard label="Frame Rate" value={`${data.fps.toFixed(1)} fps`} />
          <MetadataCard label="Resolution" value={`${data.width} × ${data.height}`} />
          <MetadataCard label="Frames" value={data.frame_count.toLocaleString()} />
        </div>
      </CardContent>
    </Card>
  );
}

function MetadataCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border/50 bg-muted/30 p-3">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{label}</div>
      <div className="mt-1 text-sm font-semibold text-foreground truncate">{value}</div>
    </div>
  );
}

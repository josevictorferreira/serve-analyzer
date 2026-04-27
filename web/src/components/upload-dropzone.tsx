import React, { useCallback, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

const VIDEO_EXTENSIONS = ['.mov', '.mp4', '.avi', '.webm', '.mkv'];

function isAcceptedVideoFile(file: File): boolean {
  // Accept if MIME type starts with video/ OR filename has known video extension
  if (file.type.startsWith('video/')) {
    return true;
  }
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return VIDEO_EXTENSIONS.includes(ext);
}

export function UploadDropzone({ onFileSelect, disabled }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file && isAcceptedVideoFile(file)) {
      onFileSelect(file);
    }
  }, [disabled, onFileSelect]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && isAcceptedVideoFile(file)) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card 
      className={cn(
        "w-full max-w-2xl border-dashed border-2 transition-colors cursor-pointer",
        isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25",
        disabled && "opacity-50 cursor-not-allowed"
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={!disabled ? handleClick : undefined}
    >
      <CardHeader>
        <CardTitle className="text-center">Upload Serve Video</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <input 
          type="file" 
          ref={fileInputRef} 
          className="hidden" 
          accept="video/*,.mov,.mp4,.avi,.webm,.mkv"
          onChange={handleFileChange}
          disabled={disabled}
        />
        <div className="flex flex-col items-center space-y-4">
          <div className={cn(
            "p-4 rounded-full bg-muted transition-colors",
            isDragging && "bg-primary/10 text-primary"
          )}>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-12 h-12"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" x2="12" y1="3" y2="15" />
            </svg>
          </div>
          <div className="text-center">
            <p className="font-medium">
              {isDragging ? "Drop your video here" : "Drag and drop your video file here"}
            </p>
            <p className="text-sm">or click to browse</p>
          </div>
          <Button variant="outline" disabled={disabled} onClick={(e) => { e.stopPropagation(); handleClick(); }}>
            Select File
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

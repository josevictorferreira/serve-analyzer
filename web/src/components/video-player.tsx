import React, { useRef, useEffect } from 'react';
import type { ClipMeta } from '../lib/types';
import { Card } from './ui/card';
import { Play, Pause, RotateCcw, SkipBack, SkipForward } from 'lucide-react';
import { Button } from './ui/button';

interface VideoPlayerProps {
  clip: ClipMeta;
  onNext?: () => void;
  onPrev?: () => void;
  hasPrev?: boolean;
  hasNext?: boolean;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  clip,
  onNext,
  onPrev,
  hasPrev,
  hasNext,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = React.useState(false);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load();
      videoRef.current.play().catch(() => {
        // Autoplay might be blocked
        setIsPlaying(false);
      });
      setIsPlaying(true);
    }
  }, [clip.url_path]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play();
        setIsPlaying(true);
      } else {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    }
  };

  const restart = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <Card className="overflow-hidden bg-black aspect-video relative group">
      <video
        ref={videoRef}
        src={clip.url_path}
        className="w-full h-full object-contain"
        loop
        playsInline
        controls
        preload="auto"
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />

      {/* Overlay Controls — pointer-events-none so native controls work */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-4 pointer-events-none">
        <div className="flex items-center justify-between pointer-events-auto">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/20"
              onClick={togglePlay}
            >
              {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/20"
              onClick={restart}
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/20"
              onClick={onPrev}
              disabled={!hasPrev}
            >
              <SkipBack className="w-5 h-5 fill-current" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/20"
              onClick={onNext}
              disabled={!hasNext}
            >
              <SkipForward className="w-5 h-5 fill-current" />
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

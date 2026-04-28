import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { BallPosition, ClipMeta } from '../lib/types';

interface VideoPlayerProps {
  clip: ClipMeta;
}

interface VideoRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

function formatSeconds(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '--';
  return `${value.toFixed(2)}s`;
}

function formatSpeed(value: number | null | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '--';
  return `${Math.round(value)} km/h`;
}

function useVideoRect(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [rect, setRect] = useState<VideoRect>({ left: 0, top: 0, width: 0, height: 0 });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateRect = () => {
      const bounds = video.getBoundingClientRect();
      const videoWidth = video.videoWidth || 16;
      const videoHeight = video.videoHeight || 9;
      const containerRatio = bounds.width / bounds.height;
      const videoRatio = videoWidth / videoHeight;

      if (containerRatio > videoRatio) {
        const width = bounds.height * videoRatio;
        setRect({ left: (bounds.width - width) / 2, top: 0, width, height: bounds.height });
      } else {
        const height = bounds.width / videoRatio;
        setRect({ left: 0, top: (bounds.height - height) / 2, width: bounds.width, height });
      }
    };

    updateRect();
    video.addEventListener('loadedmetadata', updateRect);
    window.addEventListener('resize', updateRect);

    return () => {
      video.removeEventListener('loadedmetadata', updateRect);
      window.removeEventListener('resize', updateRect);
    };
  }, [videoRef]);

  return rect;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ clip }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const videoRect = useVideoRect(videoRef);

  const pointsByFrame = useMemo(() => {
    const points = new Map<number, BallPosition>();
    for (const point of clip.ball_positions ?? []) {
      points.set(point.frame_number, point);
    }
    return points;
  }, [clip.ball_positions]);

  const activePoint = useMemo(() => {
    if (!clip.fps || clip.start_frame === undefined) return null;
    const frameNumber = Math.round(currentTime * clip.fps + clip.start_frame);
    return pointsByFrame.get(frameNumber) ?? null;
  }, [clip.fps, clip.start_frame, currentTime, pointsByFrame]);

  const marker = useMemo(() => {
    if (!activePoint || !clip.width || !clip.height || videoRect.width === 0 || videoRect.height === 0) {
      return null;
    }

    return {
      left: videoRect.left + (activePoint.x / clip.width) * videoRect.width,
      top: videoRect.top + (activePoint.y / clip.height) * videoRect.height,
      detectedAt: activePoint.clip_time_sec,
    };
  }, [activePoint, clip.width, clip.height, videoRect]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.load();
    video.play().catch(() => undefined);
  }, [clip.url_path]);

  useEffect(() => {
    let frameId = 0;

    const update = () => {
      if (videoRef.current) {
        setCurrentTime(videoRef.current.currentTime);
      }
      frameId = window.requestAnimationFrame(update);
    };

    frameId = window.requestAnimationFrame(update);
    return () => window.cancelAnimationFrame(frameId);
  }, []);

  return (
    <section className="relative overflow-hidden rounded-[2rem] border border-slate-800/80 bg-slate-950 shadow-2xl shadow-slate-950/40">
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-wrap items-start justify-between gap-3 p-4 sm:p-6">
        <div className="rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 text-white shadow-xl shadow-black/30 backdrop-blur-md">
          <div className="text-[10px] font-black uppercase tracking-[0.3em] text-cyan-300">Serve {clip.serve_index}</div>
          <div className="mt-1 text-2xl font-black tracking-tight sm:text-3xl">{formatSpeed(clip.velocity_kmh)}</div>
          <div className="text-xs font-semibold text-slate-300">max post-contact velocity</div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-right text-white shadow-xl shadow-black/30 backdrop-blur-md">
          <div className="text-[10px] font-black uppercase tracking-[0.3em] text-amber-200">Detected Contact</div>
          <div className="mt-1 text-xl font-black tabular-nums">{formatSeconds(clip.contact_time_sec)}</div>
          <div className="text-xs font-semibold text-slate-300">clip {formatSeconds(clip.contact_clip_time_sec)}</div>
        </div>
      </div>

      <div className="relative aspect-video bg-black">
        <video
          ref={videoRef}
          src={clip.url_path}
          className="h-full w-full object-contain"
          loop
          playsInline
          controls
          preload="auto"
        />

        {marker && (
          <div
            className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-1/2"
            style={{ left: marker.left, top: marker.top }}
            aria-hidden="true"
          >
            <div className="relative h-10 w-10 rounded-full border-2 border-cyan-300 shadow-[0_0_24px_rgba(34,211,238,0.9)]">
              <div className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-200 shadow-[0_0_16px_rgba(34,211,238,1)]" />
              <div className="absolute left-1/2 top-1/2 h-px w-12 -translate-x-1/2 bg-cyan-200/80" />
              <div className="absolute left-1/2 top-1/2 h-12 w-px -translate-y-1/2 bg-cyan-200/80" />
            </div>
            <div className="absolute left-7 top-7 whitespace-nowrap rounded-full border border-cyan-200/40 bg-slate-950/80 px-2.5 py-1 text-[11px] font-black text-cyan-100 shadow-lg backdrop-blur-md">
              {formatSpeed(clip.velocity_kmh)} · t={formatSeconds(marker.detectedAt)}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

import React from 'react';
import type { ClipMeta, ServeCandidate } from '../lib/types';
import { Card } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Clock, Gauge, Radar } from 'lucide-react';

interface TimelineProps {
  clips: ClipMeta[];
  candidates: ServeCandidate[];
  activeClipIndex: number;
  onClipSelect: (index: number) => void;
}

function speedFor(clip: ClipMeta, candidate?: ServeCandidate): number | undefined {
  return clip.velocity_kmh ?? candidate?.post_contact_max_kmh ?? undefined;
}

export const Timeline: React.FC<TimelineProps> = ({
  clips,
  candidates,
  activeClipIndex,
  onClipSelect,
}) => {
  return (
    <aside className="flex h-full flex-col overflow-hidden rounded-[1.75rem] border border-slate-800 bg-slate-950 text-white shadow-2xl shadow-slate-950/30">
      <div className="border-b border-white/10 bg-white/[0.03] p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.24em] text-cyan-300">
              <Radar className="h-4 w-4" />
              Timeline
            </div>
            <h3 className="mt-1 text-2xl font-black tracking-tight">Detected serves</h3>
          </div>
          <Badge className="rounded-full bg-cyan-300 px-3 py-1 text-[11px] font-black text-slate-950 hover:bg-cyan-300">
            {clips.length}
          </Badge>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-3 p-4">
          {clips.map((clip, idx) => {
            const candidate = candidates[idx];
            const isActive = activeClipIndex === idx;
            const speed = speedFor(clip, candidate);

            return (
              <button
                key={clip.filename}
                onClick={() => onClipSelect(idx)}
                className="group w-full text-left outline-none"
              >
                <Card
                  className={`overflow-hidden border p-0 transition-all duration-200 ${
                    isActive
                      ? 'border-cyan-300 bg-cyan-300 text-slate-950 shadow-lg shadow-cyan-500/25'
                      : 'border-white/10 bg-white/[0.06] text-white hover:border-white/25 hover:bg-white/[0.1]'
                  }`}
                >
                  <div className="flex gap-3 p-3">
                    <div
                      className={`relative flex h-16 w-24 shrink-0 items-center justify-center overflow-hidden rounded-2xl border ${
                        isActive ? 'border-slate-950/20 bg-slate-950/10' : 'border-white/10 bg-black/30'
                      }`}
                    >
                      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(34,211,238,0.35),transparent_32%),linear-gradient(135deg,rgba(15,23,42,0),rgba(15,23,42,0.45))]" />
                      <span className="relative text-xl font-black tabular-nums">#{idx + 1}</span>
                    </div>

                    <div className="min-w-0 flex-1 space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="text-sm font-black">Serve #{idx + 1}</div>
                          <div className={`mt-0.5 flex items-center gap-1 text-xs font-semibold ${isActive ? 'text-slate-700' : 'text-slate-400'}`}>
                            <Clock className="h-3.5 w-3.5" />
                            {clip.contact_time_sec.toFixed(2)}s contact
                          </div>
                        </div>
                        <div className={`rounded-full px-2 py-1 text-[10px] font-black uppercase tracking-wider ${isActive ? 'bg-slate-950 text-cyan-200' : 'bg-white/10 text-cyan-200'}`}>
                          {clip.duration?.toFixed(1)}s
                        </div>
                      </div>

                      <div className={`flex items-center gap-2 rounded-xl px-2.5 py-2 ${isActive ? 'bg-slate-950/10' : 'bg-black/20'}`}>
                        <Gauge className={`h-4 w-4 ${isActive ? 'text-slate-950' : 'text-cyan-300'}`} />
                        <span className="text-lg font-black tabular-nums">
                          {speed ? Math.round(speed) : '--'}
                          <span className={`ml-1 text-[10px] font-black uppercase tracking-wide ${isActive ? 'text-slate-700' : 'text-slate-400'}`}>km/h</span>
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              </button>
            );
          })}
        </div>
      </ScrollArea>
    </aside>
  );
};

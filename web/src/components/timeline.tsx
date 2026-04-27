import React from 'react';
import type { ClipMeta, ServeCandidate } from '../lib/types';
import { Card } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Clock, Gauge } from 'lucide-react';

interface TimelineProps {
  clips: ClipMeta[];
  candidates: ServeCandidate[];
  activeClipIndex: number;
  onClipSelect: (index: number) => void;
}

export const Timeline: React.FC<TimelineProps> = ({
  clips,
  candidates,
  activeClipIndex,
  onClipSelect,
}) => {
  return (
    <div className="flex flex-col h-full bg-slate-50/50 rounded-xl border p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          Serve Timeline
          <Badge variant="secondary" className="rounded-full px-2 py-0.5 text-[10px]">
            {clips.length} DETECTED
          </Badge>
        </h3>
      </div>

      <ScrollArea className="flex-1 -mx-2 px-2">
        <div className="space-y-2 pr-2">
          {clips.map((clip, idx) => {
            const candidate = candidates[idx];
            const isActive = activeClipIndex === idx;

            return (
              <button
                key={clip.filename}
                onClick={() => onClipSelect(idx)}
                className={`w-full text-left transition-all duration-200 group ${
                  isActive 
                    ? 'ring-2 ring-indigo-500 ring-offset-2 scale-[1.02]' 
                    : 'hover:bg-white/80'
                }`}
              >
                <Card className={`p-3 border-none shadow-sm ${isActive ? 'bg-indigo-50' : 'bg-white'}`}>
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-10 bg-slate-200 rounded overflow-hidden flex-shrink-0 relative group-hover:opacity-90">
                      {/* Thumbnail Placeholder */}
                      <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-slate-400">
                        #{idx + 1}
                      </div>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold text-slate-900">Serve #{idx + 1}</span>
                        <div className="flex items-center gap-1 text-[10px] text-slate-500 font-medium">
                          <Clock className="w-3 h-3" />
                          {clip.contact_time_sec.toFixed(1)}s ({clip.duration?.toFixed(1)}s clip)
                        </div>
                      </div>

                      {candidate?.post_contact_max_kmh && (
                        <div className="flex items-center gap-1.5">
                          <Gauge className="w-3.5 h-3.5 text-indigo-600" />
                          <span className="text-sm font-black text-indigo-600 tracking-tight">
                            {Math.round(candidate.post_contact_max_kmh)} <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide">km/h</span>
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              </button>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
};

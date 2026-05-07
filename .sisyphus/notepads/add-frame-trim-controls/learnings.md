## Frame trim controls implementation (2026-05-07)
- Used two separate `<input type="range">` sliders for trimStart/trimEnd rather than a custom dual-handle implementation
- Each trim handler clamps `currentFrame` via `setCurrentFrame((prev) => Math.max(..., Math.min(prev, ...)))` to keep the scrubber within bounds
- Visual trim region rendered as an absolutely-positioned `<div>` with percentage-based `left`/`width` — no CSS custom properties needed
- `trimEnd` initialized via `useEffect` watching `videoMetadata.frame_count` since the prop may not be available at initial render
- `min={Math.max(trimStart, trimEnd - 1)}` on trim-start and `min={Math.min(trimStart + 1, trimEnd)}` on trim-end prevents invalid ranges
- TypeScript passes `tsc --noEmit` with zero errors, no LSP diagnostics

import { useCallback, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { saveWallCalibration, getWallCalibration, deleteWallCalibration } from '@/lib/wall-api';
import type { CalibrationPoint } from './wall-calibration-canvas';
import type { WallCalibrationRequest } from '@/lib/wall-types';
import { CheckCircle, AlertCircle, RotateCcw } from 'lucide-react';

interface WallAssumptionsFormProps {
  calibrationPoints: CalibrationPoint[];
  videoId: string;
  calibrationFrame: number;
  fps: number;
  onCalibrated: () => void;
}

interface WallReferenceRow {
  name: string;
  pixelX: number;
  pixelY: number;
  wallMX: string;
  wallMY: string;
}

export function WallAssumptionsForm({
  calibrationPoints,
  videoId,
  calibrationFrame,
  fps,
  onCalibrated,
}: WallAssumptionsFormProps) {
  const [contactHeight, setContactHeight] = useState('2.80');
  const [contactDistance, setContactDistance] = useState('6.11');
  const [cameraDistance, setCameraDistance] = useState('1.57');
  const [wallRows, setWallRows] = useState<WallReferenceRow[]>([]);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Sync wall rows when calibration points change
  useEffect(() => {
    setWallRows((prev) => {
      const newRows: WallReferenceRow[] = calibrationPoints.map((p, i) => {
        const existing = prev[i];
        return {
          name: `P${i + 1}`,
          pixelX: p.pixelX,
          pixelY: p.pixelY,
          wallMX: existing?.wallMX ?? '',
          wallMY: existing?.wallMY ?? '',
        };
      });
      return newRows;
    });
  }, [calibrationPoints]);

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
          const rawPoints = setup.wall_reference_points as Array<Record<string, unknown>> | undefined;
          if (Array.isArray(rawPoints) && rawPoints.length > 0) {
            setWallRows(
              rawPoints.map((p, i) => ({
                name: (p.name as string) || `P${i + 1}`,
                pixelX: Array.isArray(p.pixel) ? Math.round(p.pixel[0] as number) : 0,
                pixelY: Array.isArray(p.pixel) ? Math.round(p.pixel[1] as number) : 0,
                wallMX: Array.isArray(p.wall_m) ? String(p.wall_m[0]) : '',
                wallMY: Array.isArray(p.wall_m) ? String(p.wall_m[1]) : '',
              }))
            );
          }
          setSuccess(true);
        }
      } catch {
        // No existing calibration — silent
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadExisting();
    return () => { cancelled = true; };
  }, []);

  const handleRowChange = useCallback(
    (index: number, field: 'wallMX' | 'wallMY', value: string) => {
      setWallRows((prev) =>
        prev.map((row, i) => (i === index ? { ...row, [field]: value } : row))
      );
    },
    []
  );

  const handleClearCalibration = useCallback(async () => {
    try {
      await deleteWallCalibration();
    } catch {
      // Best effort
    }
    setWallRows((prev) => prev.map((r) => ({ ...r, wallMX: '', wallMY: '' })));
    setContactHeight('2.80');
    setSuccess(false);
    setError(null);
  }, []);

  const validate = useCallback((): string | null => {
    if (calibrationPoints.length < 4) {
      return 'At least 4 calibration points are required.';
    }
    if (!contactHeight || isNaN(Number(contactHeight)) || Number(contactHeight) <= 0) {
      return 'Contact height must be a positive number.';
    }
    const filledRows = wallRows.filter((r) => r.wallMX !== '' && r.wallMY !== '');
    if (filledRows.length < 4) {
      return `At least 4 wall reference points must have wall coordinates filled (${filledRows.length}/4).`;
    }
    for (const row of filledRows) {
      if (isNaN(Number(row.wallMX)) || isNaN(Number(row.wallMY))) {
        return 'All wall coordinates must be valid numbers.';
      }
    }
    return null;
  }, [calibrationPoints.length, contactHeight, wallRows]);

  const handleSave = useCallback(async () => {
    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const calibrationTimeSec = calibrationFrame / fps;
      const wallPoints = wallRows
        .filter((r) => r.wallMX !== '' && r.wallMY !== '')
        .map((r) => ({
          name: r.name,
          pixel: [r.pixelX, r.pixelY],
          wall_m: [parseFloat(r.wallMX), parseFloat(r.wallMY)],
        }));

      const request: WallCalibrationRequest = {
        video_id: videoId,
        calibration_frame: calibrationFrame,
        calibration_time_sec: calibrationTimeSec,
        setup: {
          serve_contact_distance_m: parseFloat(contactDistance) || 6.11,
          camera_wall_distance_m: parseFloat(cameraDistance) || 1.57,
          serve_contact_height_m: parseFloat(contactHeight),
          wall_reference_points: wallPoints,
        },
      };

      const response = await saveWallCalibration(request);
      if (response.point_count >= 4) {
        setSuccess(true);
        onCalibrated();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save calibration.');
    } finally {
      setSaving(false);
    }
  }, [validate, calibrationFrame, fps, wallRows, videoId, contactDistance, cameraDistance, contactHeight, onCalibrated]);

  return (
    <Card className="w-full max-w-3xl">
      <CardHeader>
        <CardTitle className="text-center">Assumptions & Wall Coordinates</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Contact Height */}
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
            required
          />
          <p className="text-xs text-muted-foreground">
            Height of ball-racket contact above court surface (default: 2.80 m)
          </p>
        </div>

        {/* Optional distances */}
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

        {/* Wall Reference Points Table */}
        {wallRows.length > 0 && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Wall Reference Points</h3>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">#</th>
                    <th className="px-3 py-2 text-left font-medium">pixel_x</th>
                    <th className="px-3 py-2 text-left font-medium">pixel_y</th>
                    <th className="px-3 py-2 text-left font-medium">wall_m_x</th>
                    <th className="px-3 py-2 text-left font-medium">wall_m_y</th>
                  </tr>
                </thead>
                <tbody>
                  {wallRows.map((row, index) => (
                    <tr key={index} className="border-t border-border/50">
                      <td className="px-3 py-2 font-mono text-xs">{row.name}</td>
                      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{row.pixelX}</td>
                      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{row.pixelY}</td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          step="0.01"
                          value={row.wallMX}
                          onChange={(e) => handleRowChange(index, 'wallMX', e.target.value)}
                          placeholder="e.g. -4.0"
                          className="w-full rounded border border-input bg-background px-2 py-1 text-xs font-mono"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          step="0.01"
                          value={row.wallMY}
                          onChange={(e) => handleRowChange(index, 'wallMY', e.target.value)}
                          placeholder="e.g. 0.0"
                          className="w-full rounded border border-input bg-background px-2 py-1 text-xs font-mono"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground">
              Fill in the real-world wall coordinates (meters) for each point.
              At least 4 points required.
            </p>
          </div>
        )}

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

        {/* Actions */}
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
            disabled={saving || loading || calibrationPoints.length < 4}
            className="gap-1"
          >
            {saving ? 'Saving...' : 'Save Calibration'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAnalysisJob } from './use-analysis-job';
import * as api from '@/lib/api';

vi.mock('@/lib/api');

describe('useAnalysisJob', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    // Default mock for the initial sync in useEffect
    vi.mocked(api.getJob).mockResolvedValue({ status: 'idle', phase: 'idle' });
  });

  it('should start in idle state', async () => {
    const { result } = renderHook(() => useAnalysisJob());
    
    // Wait for the mount sync effect to finish
    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.phase).toBe('idle');
    expect(result.current.progress).toBe(0);
  });

  it('should handle successful upload and start polling', async () => {
    const mockFile = new File([''], 'test.mp4', { type: 'video/mp4' });
    const analyzeSpy = vi.mocked(api.analyzeVideoWithProgress).mockImplementation(
      async (_file, onProgress) => {
        onProgress(50);
        return Promise.resolve();
      }
    );

    // Initial mount sync returns idle, subsequent calls return analyzing
    vi.mocked(api.getJob)
      .mockResolvedValueOnce({ status: 'idle', phase: 'idle' }) // mount
      .mockResolvedValue({ status: 'analyzing', phase: 'analyzing' }); // polling

    const { result } = renderHook(() => useAnalysisJob());

    // Wait for mount sync
    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await result.current.upload(mockFile);
    });

    expect(analyzeSpy).toHaveBeenCalledWith(mockFile, expect.any(Function));
    expect(result.current.phase).toBe('analyzing');
    
    // getJob called once on mount, once immediately when startPolling() is called inside upload
    expect(api.getJob).toHaveBeenCalledTimes(2);

    // Advance timers for polling
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });

    // One more poll
    expect(api.getJob).toHaveBeenCalledTimes(3);
  });

  it('should stop polling when phase is done', async () => {
    const mockFile = new File([''], 'test.mp4', { type: 'video/mp4' });
    vi.mocked(api.analyzeVideoWithProgress).mockResolvedValue();
    
    vi.mocked(api.getJob)
      .mockResolvedValueOnce({ status: 'idle', phase: 'idle' }) // mount
      .mockResolvedValueOnce({ status: 'analyzing', phase: 'analyzing' }) // 1st poll
      .mockResolvedValueOnce({ status: 'done', phase: 'done' }); // 2nd poll

    const { result } = renderHook(() => useAnalysisJob());

    // Wait for mount sync
    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await result.current.upload(mockFile);
    });

    expect(result.current.phase).toBe('analyzing');

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });

    expect(result.current.phase).toBe('done');
    // 1 mount + 1st poll + 2nd poll
    expect(api.getJob).toHaveBeenCalledTimes(3);

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });

    // Should not have called getJob again
    expect(api.getJob).toHaveBeenCalledTimes(3);
  });

  it('should handle upload error', async () => {
    const mockFile = new File([''], 'test.mp4', { type: 'video/mp4' });
    vi.mocked(api.analyzeVideoWithProgress).mockRejectedValue(new Error('Upload failed'));

    const { result } = renderHook(() => useAnalysisJob());

    // Wait for mount sync
    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await result.current.upload(mockFile);
    });

    expect(result.current.phase).toBe('error');
    expect(result.current.error).toBe('Upload failed');
  });

  it('should reset state', async () => {
    vi.mocked(api.resetJob).mockResolvedValue();
    vi.mocked(api.getJob).mockResolvedValue({ status: 'idle', phase: 'idle' });

    const { result } = renderHook(() => useAnalysisJob());

    // Wait for mount sync
    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await result.current.reset();
    });

    expect(result.current.phase).toBe('idle');
    expect(api.resetJob).toHaveBeenCalled();
  });

  it('should accept .mov file with empty MIME type by extension', async () => {
    // Create a File with empty type but .mov extension
    const mockMovFile = new File([''], 'test.mov', { type: '' });
    const analyzeSpy = vi.mocked(api.analyzeVideoWithProgress).mockImplementation(
      async (_file, onProgress) => {
        onProgress(50);
        return Promise.resolve();
      }
    );

    vi.mocked(api.getJob)
      .mockResolvedValueOnce({ status: 'idle', phase: 'idle' })
      .mockResolvedValue({ status: 'analyzing', phase: 'analyzing' });

    const { result } = renderHook(() => useAnalysisJob());

    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await result.current.upload(mockMovFile);
    });

    // Should have called analyzeVideoWithProgress (file was accepted)
    expect(analyzeSpy).toHaveBeenCalledWith(mockMovFile, expect.any(Function));
    expect(result.current.phase).toBe('analyzing');
  });
});

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { WallCalibrationCanvas, type CalibrationPoint } from './wall-calibration-canvas'

const mockMetadata = {
  video_id: 'test-video-123',
  filename: 'test-serve.mov',
  duration_sec: 15.5,
  fps: 30,
  frame_count: 465,
  width: 1920,
  height: 1080,
}

describe('WallCalibrationCanvas', () => {
  it('renders video element and frame scrubber slider', () => {
    const points: CalibrationPoint[] = []
    const { container } = render(
      <WallCalibrationCanvas
        videoUrl="/api/wall/video/test-video-123"
        videoMetadata={mockMetadata}
        points={points}
        onPointsChange={vi.fn()}
        currentFrame={0}
        onFrameChange={vi.fn()}
      />
    )

    expect(screen.getByText('Calibration Points')).toBeInTheDocument()
    const video = container.querySelector('video')
    expect(video).toBeInTheDocument()
    expect(video).toHaveAttribute('src', '/api/wall/video/test-video-123')
  })

  it('calls onFrameChange when slider is moved', () => {
    const onFrameChange = vi.fn()
    const points: CalibrationPoint[] = []
    render(
      <WallCalibrationCanvas
        videoUrl="/api/wall/video/test-video-123"
        videoMetadata={mockMetadata}
        points={points}
        onPointsChange={vi.fn()}
        currentFrame={0}
        onFrameChange={onFrameChange}
      />
    )

    const slider = screen.getByRole('slider', { name: 'Frame scrubber' })
    fireEvent.change(slider, { target: { value: '100' } })
    expect(onFrameChange).toHaveBeenCalledWith(100)
  })

  it('shows point count and minimum warning when points are present but less than 4', () => {
    const points: CalibrationPoint[] = [
      { id: '1', pixelX: 100, pixelY: 200 },
      { id: '2', pixelX: 300, pixelY: 400 },
    ]
    render(
      <WallCalibrationCanvas
        videoUrl="/api/wall/video/test-video-123"
        videoMetadata={mockMetadata}
        points={points}
        onPointsChange={vi.fn()}
        currentFrame={0}
        onFrameChange={vi.fn()}
      />
    )

    expect(screen.getByText(/Points placed: 2/)).toBeInTheDocument()
    expect(screen.getByText('(minimum 4 required)')).toBeInTheDocument()
  })

  it('calls onPointsChange when Clear All is clicked', () => {
    const onPointsChange = vi.fn()
    const points: CalibrationPoint[] = [
      { id: '1', pixelX: 100, pixelY: 200 },
      { id: '2', pixelX: 300, pixelY: 400 },
    ]
    render(
      <WallCalibrationCanvas
        videoUrl="/api/wall/video/test-video-123"
        videoMetadata={mockMetadata}
        points={points}
        onPointsChange={onPointsChange}
        currentFrame={0}
        onFrameChange={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /Clear All/i }))
    expect(onPointsChange).toHaveBeenCalledWith([])
  })

  it('calls onPointsChange to remove individual point', () => {
    const onPointsChange = vi.fn()
    const points: CalibrationPoint[] = [
      { id: '1', pixelX: 100, pixelY: 200 },
      { id: '2', pixelX: 300, pixelY: 400 },
      { id: '3', pixelX: 500, pixelY: 600 },
    ]
    render(
      <WallCalibrationCanvas
        videoUrl="/api/wall/video/test-video-123"
        videoMetadata={mockMetadata}
        points={points}
        onPointsChange={onPointsChange}
        currentFrame={0}
        onFrameChange={vi.fn()}
      />
    )

    // Remove button for point 1 (first in list)
    const removeButtons = screen.getAllByRole('button', { name: /Remove point/i })
    fireEvent.click(removeButtons[0])
    expect(onPointsChange).toHaveBeenCalledWith([
      { id: '2', pixelX: 300, pixelY: 400 },
      { id: '3', pixelX: 500, pixelY: 600 },
    ])
  })
})

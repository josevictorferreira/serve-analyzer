import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { WallAssumptionsForm } from './wall-assumptions-form'
import * as wallApi from '@/lib/wall-api'

vi.mock('@/lib/wall-api')

const mockPoints = [
  { id: '1', pixelX: 100, pixelY: 200 },
  { id: '2', pixelX: 700, pixelY: 200 },
  { id: '3', pixelX: 100, pixelY: 500 },
  { id: '4', pixelX: 700, pixelY: 500 },
]

describe('WallAssumptionsForm', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(wallApi.getWallCalibration).mockRejectedValue(new Error('Not found'))
  })

  it('renders contact height input with default value', () => {
    render(
      <WallAssumptionsForm
        calibrationPoints={mockPoints}
        videoId="test-video-123"
        calibrationFrame={0}
        fps={30}
        onCalibrated={vi.fn()}
      />
    )

    expect(screen.getByText('Assumptions & Wall Coordinates')).toBeInTheDocument()
    const contactHeightInput = screen.getByLabelText('Serve Contact Height (m)')
    expect(contactHeightInput).toBeInTheDocument()
    expect(contactHeightInput).toHaveValue(2.8)
  })

  it('renders wall reference points table with pixel coordinates', () => {
    render(
      <WallAssumptionsForm
        calibrationPoints={mockPoints}
        videoId="test-video-123"
        calibrationFrame={0}
        fps={30}
        onCalibrated={vi.fn()}
      />
    )

    expect(screen.getByText('Wall Reference Points')).toBeInTheDocument()
    // Check pixel columns show correct values (use getAllByText since values repeat)
    const pixelCells = screen.getAllByText('100')
    expect(pixelCells.length).toBeGreaterThanOrEqual(2)
    const pixelCells700 = screen.getAllByText('700')
    expect(pixelCells700.length).toBeGreaterThanOrEqual(2)
  })

  it('renders wall coordinate inputs for each point', () => {
    render(
      <WallAssumptionsForm
        calibrationPoints={mockPoints}
        videoId="test-video-123"
        calibrationFrame={0}
        fps={30}
        onCalibrated={vi.fn()}
      />
    )

    const wallInputs = screen.getAllByRole('spinbutton')
    // Each point has 2 wall inputs (wall_m_x, wall_m_y), plus contact height, contact distance, camera distance
    expect(wallInputs.length).toBeGreaterThanOrEqual(4)
  })

  it('shows validation error when saving with fewer than 4 points', async () => {
    render(
      <WallAssumptionsForm
        calibrationPoints={mockPoints.slice(0, 2)}
        videoId="test-video-123"
        calibrationFrame={0}
        fps={30}
        onCalibrated={vi.fn()}
      />
    )

    const saveButton = screen.getByRole('button', { name: 'Save Calibration' })
    expect(saveButton).toBeDisabled()
  })

  it('populates form when existing calibration is loaded', async () => {
    const existingCalibration = {
      video_id: 'test-video-123',
      calibration_frame: 50,
      calibration_time_sec: 1.667,
      calibration: {
        serve_contact_height_m: 3.0,
        serve_contact_distance_m: 6.11,
        camera_wall_distance_m: 1.57,
        wall_reference_points: [
          { name: 'P1', pixel: [100, 200], wall_m: [-4.0, 0.0] },
          { name: 'P2', pixel: [700, 200], wall_m: [4.0, 0.0] },
          { name: 'P3', pixel: [100, 500], wall_m: [-4.0, 3.0] },
          { name: 'P4', pixel: [700, 500], wall_m: [4.0, 3.0] },
        ],
      },
      point_count: 4,
    }
    vi.mocked(wallApi.getWallCalibration).mockResolvedValue(existingCalibration)

    render(
      <WallAssumptionsForm
        calibrationPoints={mockPoints}
        videoId="test-video-123"
        calibrationFrame={0}
        fps={30}
        onCalibrated={vi.fn()}
      />
    )

    await waitFor(() => {
      const contactHeightInput = screen.getByLabelText('Serve Contact Height (m)')
      expect(contactHeightInput).toHaveValue(3)
    })
  })
})

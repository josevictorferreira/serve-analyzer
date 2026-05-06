import { act, fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { WallAnalyzeStep } from './wall-analyze-step'
import * as wallApi from '@/lib/wall-api'

vi.mock('@/lib/wall-api')

describe('WallAnalyzeStep', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  it('renders start analysis button', () => {
    render(
      <WallAnalyzeStep
        onDone={vi.fn()}
        onError={vi.fn()}
        isCalibrated={true}
      />
    )

    expect(screen.getByRole('button', { name: /Start Analysis/i })).toBeInTheDocument()
  })

  it('disables start button when not calibrated', () => {
    render(
      <WallAnalyzeStep
        onDone={vi.fn()}
        onError={vi.fn()}
        isCalibrated={false}
      />
    )

    const button = screen.getByRole('button', { name: /Start Analysis/i })
    expect(button).toBeDisabled()
  })

  it('shows calibration hint when not calibrated', () => {
    render(
      <WallAnalyzeStep
        onDone={vi.fn()}
        onError={vi.fn()}
        isCalibrated={false}
      />
    )

    expect(screen.getByText(/Complete calibration before starting analysis/i)).toBeInTheDocument()
  })

  it('calls startWallAnalysis and starts polling on click', async () => {
    vi.mocked(wallApi.startWallAnalysis).mockResolvedValue({ status: 'accepted', message: 'Started' })
    vi.mocked(wallApi.getWallJob).mockResolvedValue({ status: 'done', phase: 'done', result: { speed: 100 } })

    const onDone = vi.fn()
    render(
      <WallAnalyzeStep
        onDone={onDone}
        onError={vi.fn()}
        isCalibrated={true}
      />
    )

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }))
      await Promise.resolve()
    })

    expect(wallApi.startWallAnalysis).toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(1100)
      await Promise.resolve()
    })

    expect(wallApi.getWallJob).toHaveBeenCalled()
  })

  it('shows busy state on 409 response', async () => {
    vi.mocked(wallApi.startWallAnalysis).mockRejectedValue(
      new Error('Another analysis is already in progress. Please wait.')
    )

    render(
      <WallAnalyzeStep
        onDone={vi.fn()}
        onError={vi.fn()}
        isCalibrated={true}
      />
    )

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }))
      await Promise.resolve()
    })

    expect(screen.getByText(/Another analysis is running/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Reset and Retry/i })).toBeInTheDocument()
  })

  it('reset button calls resetWallJob and returns to idle', async () => {
    vi.mocked(wallApi.startWallAnalysis).mockRejectedValue(
      new Error('Another analysis is already in progress. Please wait.')
    )
    vi.mocked(wallApi.resetWallJob).mockResolvedValue()

    render(
      <WallAnalyzeStep
        onDone={vi.fn()}
        onError={vi.fn()}
        isCalibrated={true}
      />
    )

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }))
      await Promise.resolve()
    })

    expect(screen.getByText(/Another analysis is running/i)).toBeInTheDocument()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Reset and Retry/i }))
      await Promise.resolve()
    })

    expect(wallApi.resetWallJob).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /Start Analysis/i })).toBeInTheDocument()
  })

  it('shows error state when analysis fails', async () => {
    vi.mocked(wallApi.startWallAnalysis).mockResolvedValue({ status: 'accepted', message: 'Started' })
    vi.mocked(wallApi.getWallJob).mockResolvedValue({
      status: 'error',
      phase: 'error',
      error: 'Calibration data missing',
      result: null,
    })

    const onError = vi.fn()
    render(
      <WallAnalyzeStep
        onDone={vi.fn()}
        onError={onError}
        isCalibrated={true}
      />
    )

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }))
      await Promise.resolve()
    })

    expect(wallApi.startWallAnalysis).toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(1100)
      await Promise.resolve()
    })

    expect(screen.getByText(/Analysis failed/i)).toBeInTheDocument()
    expect(onError).toHaveBeenCalledWith('Calibration data missing')
  })
})

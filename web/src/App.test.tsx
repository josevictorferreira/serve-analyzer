import { fireEvent, render, screen } from '@testing-library/react'
import App from './App'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useAnalysisJob } from './hooks/use-analysis-job'
import { listAnnotationSessions, listDetectorVersions } from '@/lib/api'

vi.mock('./hooks/use-analysis-job', () => ({
  useAnalysisJob: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  listAnnotationSessions: vi.fn(),
  listDetectorVersions: vi.fn(),
  getAnnotationSession: vi.fn(),
  createAnnotationSessionWithProgress: vi.fn(),
  reviewAnnotationFrame: vi.fn(),
  undoAnnotationFrame: vi.fn(),
  exportAnnotationDataset: vi.fn(),
  evaluateAnnotationBaseline: vi.fn(),
  getTrainingEnvironment: vi.fn(),
}))

vi.mock('@/lib/wall-api', () => ({
  uploadWallVideo: vi.fn(),
  getWallVideoMetadata: vi.fn(),
  saveWallCalibration: vi.fn(),
  getWallCalibration: vi.fn(),
  deleteWallCalibration: vi.fn(),
  startWallAnalysis: vi.fn(),
  getWallJob: vi.fn(),
  resetWallJob: vi.fn(),
}))

describe('App', () => {
  beforeEach(() => {
    vi.mocked(useAnalysisJob).mockReturnValue({
      phase: 'idle',
      progress: 0,
      error: undefined,
      jobStatus: undefined,
      upload: vi.fn(),
      reset: vi.fn(),
      isUploading: false,
      estimatedDurationSec: null,
      analysisProgress: 0,
    })
    vi.mocked(listAnnotationSessions).mockResolvedValue([])
    vi.mocked(listDetectorVersions).mockResolvedValue({
      detectors: [
        {
          version: 'v1',
          label: 'V1 baseline',
          description: 'Existing candidate generator and selector.',
        },
        {
          version: 'v2',
          label: 'V2 continuity refinement',
          description: 'Continuity, history, and motion-cue refinement.',
        },
      ],
      default_version: 'v1',
    })
  })

  it('renders the header title', () => {
    render(<App />)
    const headerElement = screen.getByText(/Serve Analyzer/i)
    expect(headerElement).toBeInTheDocument()
  })

  it('renders the upload card', () => {
    render(<App />)
    const cardTitle = screen.getByText(/Upload Serve Video/i)
    expect(cardTitle).toBeInTheDocument()
  })

  it('switches to the annotation workspace', async () => {
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Annotate Ball/i }))

    expect(await screen.findByText(/Build Tennis-Ball Dataset/i)).toBeInTheDocument()
    expect(listAnnotationSessions).toHaveBeenCalled()
  })

  it('switches to wall analysis and renders the workflow stepper', () => {
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Wall Analysis/i }))

    expect(screen.getByText(/Upload Wall Video/i)).toBeInTheDocument()
    expect(screen.getByText('Upload')).toBeInTheDocument()
    expect(screen.getByText('Calibrate')).toBeInTheDocument()
    expect(screen.getByText('Analyze')).toBeInTheDocument()
    expect(screen.getByText('Results')).toBeInTheDocument()
  })

  it('shows the wall upload dropzone when wall mode is active', () => {
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Wall Analysis/i }))

    expect(screen.getByText(/Drag and drop your wall serve video/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Select File/i })).toBeInTheDocument()
  })

  it('preserves existing analysis mode when switching back from wall mode', () => {
    render(<App />)

    // Switch to wall
    fireEvent.click(screen.getByRole('button', { name: /Wall Analysis/i }))
    expect(screen.getByText(/Upload Wall Video/i)).toBeInTheDocument()

    // Switch back to analysis
    fireEvent.click(screen.getByRole('button', { name: /Analyze Serves/i }))
    expect(screen.getByText(/Upload Serve Video/i)).toBeInTheDocument()
  })
})

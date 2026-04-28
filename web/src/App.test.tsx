import { fireEvent, render, screen } from '@testing-library/react'
import App from './App'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useAnalysisJob } from './hooks/use-analysis-job'
import { listAnnotationSessions } from '@/lib/api'

vi.mock('./hooks/use-analysis-job', () => ({
  useAnalysisJob: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  listAnnotationSessions: vi.fn(),
  getAnnotationSession: vi.fn(),
  createAnnotationSessionWithProgress: vi.fn(),
  reviewAnnotationFrame: vi.fn(),
  undoAnnotationFrame: vi.fn(),
  exportAnnotationDataset: vi.fn(),
  evaluateAnnotationBaseline: vi.fn(),
  getTrainingEnvironment: vi.fn(),
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
})

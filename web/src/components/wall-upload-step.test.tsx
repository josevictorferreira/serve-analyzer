import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { WallUploadStep } from './wall-upload-step'
import * as wallApi from '@/lib/wall-api'

vi.mock('@/lib/wall-api')

const mockUploadResponse = {
  video_id: 'test-video-123',
  video_url: '/api/wall/video/test-video-123',
  filename: 'test-serve.mov',
  duration_sec: 15.5,
  fps: 30,
  frame_count: 465,
  width: 1920,
  height: 1080,
}

describe('WallUploadStep', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders file input and upload instructions', () => {
    render(<WallUploadStep onUploadComplete={vi.fn()} />)

    expect(screen.getByText(/Upload Wall Video/i)).toBeInTheDocument()
    expect(screen.getByText(/Drag and drop your wall serve video/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Select File/i })).toBeInTheDocument()
  })

  it('calls onUploadComplete with metadata on successful upload', async () => {
    const onComplete = vi.fn()
    vi.mocked(wallApi.uploadWallVideo).mockResolvedValue(mockUploadResponse)

    render(<WallUploadStep onUploadComplete={onComplete} />)

    const file = new File(['test-video-data'], 'test-serve.mov', { type: 'video/quicktime' })
    const input = screen.getByRole('button', { name: /Select File/i })
    // The file input is hidden but accessible via ref; simulate by triggering through button click
    // Since the actual file input is hidden, we need to find it
    const hiddenInput = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(hiddenInput).toBeInTheDocument()

    fireEvent.change(hiddenInput, { target: { files: [file] } })

    await waitFor(() => {
      expect(wallApi.uploadWallVideo).toHaveBeenCalledWith(file)
      expect(onComplete).toHaveBeenCalledWith(mockUploadResponse)
    })
  })

  it('shows error message on upload failure', async () => {
    vi.mocked(wallApi.uploadWallVideo).mockRejectedValue(new Error('Upload failed (413).'))

    render(<WallUploadStep onUploadComplete={vi.fn()} />)

    const file = new File(['test-video-data'], 'test-serve.mov', { type: 'video/quicktime' })
    const hiddenInput = document.querySelector('input[type="file"]') as HTMLInputElement

    fireEvent.change(hiddenInput, { target: { files: [file] } })

    await waitFor(() => {
      expect(screen.getByText('Upload failed (413).')).toBeInTheDocument()
    })
  })

  it('shows uploading state during upload', async () => {
    let resolveUpload: (value: typeof mockUploadResponse) => void
    vi.mocked(wallApi.uploadWallVideo).mockImplementation(
      () => new Promise((resolve) => { resolveUpload = resolve })
    )

    render(<WallUploadStep onUploadComplete={vi.fn()} />)

    const file = new File(['test-video-data'], 'test-serve.mov', { type: 'video/quicktime' })
    const hiddenInput = document.querySelector('input[type="file"]') as HTMLInputElement

    fireEvent.change(hiddenInput, { target: { files: [file] } })

    await waitFor(() => {
      expect(screen.getByText(/Uploading your video/i)).toBeInTheDocument()
    })

    // Resolve to clean up
    resolveUpload!(mockUploadResponse)
  })
})

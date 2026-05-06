import { describe, it, expect } from "vitest"
import { render, screen } from "@testing-library/react"
import { WallResultsDashboard } from "@/components/wall-results-dashboard"

function createMockResult() {
  return {
    measured: {
      video: "test_video.mp4",
      serve_index: 0,
      impact_time_sec: 3.45,
      impact_frame: 103,
      autonomous_frame: 98,
      autonomous_pixel: [320, 240],
      impact_pixel: [350, 250],
      wall_x_m: 2.34,
      wall_y_m: 1.56,
      calibration_reprojection_rms_px: 0.82,
      raw_track_samples: 45,
    },
    inferred: {
      speed_m_s: 42.5,
      speed_km_h: 153.0,
      speed_mph: 95.07,
      speed_uncertainty_m_s: 1.2,
      landing_x_m: 5.2,
      landing_z_m: 12.8,
      in_service_box: true,
      service_box_side: "deuce",
    },
    assumed: {
      gravity_m_s2: 9.81,
      serve_contact_height_m: 2.8,
      wall_distance_m: 1.57,
    },
    confidence: 0.87,
    warnings: [
      { code: "LOW_SAMPLES", message: "Fewer than 50 track samples detected" },
    ],
    artifacts: {
      annotated_video: { url: "/api/wall/artifacts/annotated.mp4" },
      review_clip: {
        url: "/api/wall/artifacts/review.mp4",
        start_time: 2.0,
        impact_time: 3.45,
        end_time: 5.0,
      },
      plots: {
        speed: { url: "/api/wall/artifacts/speed.png" },
        wall_impact: { url: "/api/wall/artifacts/wall_impact.png" },
        court_landing: { url: "/api/wall/artifacts/court_landing.png" },
      },
      json: { url: "/api/wall/artifacts/result.json" },
      csv: { url: "/api/wall/artifacts/result.csv" },
    },
  }
}

describe("WallResultsDashboard", () => {
  it("renders with mock result data", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    // Key section titles should be present
    expect(screen.getByText(/measured impact/i)).toBeInTheDocument()
    expect(screen.getByText(/velocity/i)).toBeInTheDocument()
    expect(screen.getByText(/court projection/i)).toBeInTheDocument()
    expect(screen.getByText(/annotated video/i)).toBeInTheDocument()
    expect(screen.getAllByText(/confidence/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/assumptions/i)).toBeInTheDocument()
    expect(screen.getByText(/analysis plots/i)).toBeInTheDocument()
    expect(screen.getByText(/export data/i)).toBeInTheDocument()
  })

  it("velocity card shows all 3 speed units", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    expect(screen.getByText(/42\.50 m\/s/i)).toBeInTheDocument()
    expect(screen.getByText(/153\.00 km\/h/i)).toBeInTheDocument()
    expect(screen.getByText(/95\.07 mph/i)).toBeInTheDocument()
    expect(screen.getByText(/1\.20 m\/s/i)).toBeInTheDocument() // uncertainty
  })

  it("video players render with artifact URLs", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    const videos = screen.getAllByRole("generic").filter(
      (el) => el.tagName === "VIDEO"
    )
    // At least the annotated video player should exist
    const annotatedVideo = document.querySelector('video[src="/api/wall/artifacts/annotated.mp4"]')
    expect(annotatedVideo).toBeInTheDocument()

    const reviewVideo = document.querySelector('video[src="/api/wall/artifacts/review.mp4"]')
    expect(reviewVideo).toBeInTheDocument()
  })

  it("plot images render for each plot URL", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    const images = screen.getAllByRole("img")
    const plotUrls = images.map((img) => (img as HTMLImageElement).src)

    // Check that all three plot URLs are rendered
    expect(plotUrls.some((url) => url.includes("speed.png"))).toBe(true)
    expect(plotUrls.some((url) => url.includes("wall_impact.png"))).toBe(true)
    expect(plotUrls.some((url) => url.includes("court_landing.png"))).toBe(true)
  })

  it("JSON/CSV links point to correct URLs", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    const jsonLink = screen.getByRole("link", { name: /result\.json/i })
    const csvLink = screen.getByRole("link", { name: /result\.csv/i })

    expect(jsonLink).toHaveAttribute("href", "/api/wall/artifacts/result.json")
    expect(csvLink).toHaveAttribute("href", "/api/wall/artifacts/result.csv")
  })

  it("displays measured impact values correctly", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    expect(screen.getByText("3.45 s")).toBeInTheDocument()
    expect(screen.getByText("103")).toBeInTheDocument() // impact frame
    expect(screen.getByText(/2\.34.*1\.56.*m/i)).toBeInTheDocument() // wall position
  })

  it("shows IN service box with green indicator when in_service_box is true", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    expect(screen.getByText("IN")).toBeInTheDocument()
  })

  it("shows OUT when in_service_box is false", () => {
    const result = {
      ...createMockResult(),
      inferred: {
        ...createMockResult().inferred,
        in_service_box: false,
      },
    }
    render(<WallResultsDashboard result={result} />)

    expect(screen.getByText("OUT")).toBeInTheDocument()
  })

  it("displays confidence score as percentage", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    expect(screen.getByText("87.0%")).toBeInTheDocument()
  })

  it("displays warnings when present", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    expect(screen.getByText(/fewer than 50 track samples/i)).toBeInTheDocument()
  })

  it("renders review clip metadata", () => {
    const result = createMockResult()
    render(<WallResultsDashboard result={result} />)

    expect(screen.getByText(/review clip/i)).toBeInTheDocument()
    expect(screen.getByText(/start:/i)).toBeInTheDocument()
    expect(screen.getByText(/impact:/i)).toBeInTheDocument()
    expect(screen.getByText(/end:/i)).toBeInTheDocument()
  })

  it("omits review clip card when review_clip is absent", () => {
    const result = createMockResult()
    delete (result.artifacts as Record<string, unknown>).review_clip
    render(<WallResultsDashboard result={result} />)

    expect(screen.queryByText(/review clip/i)).not.toBeInTheDocument()
  })
})

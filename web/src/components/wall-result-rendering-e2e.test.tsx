import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { WallResultsDashboard } from "@/components/wall-results-dashboard"

function createBackendResult() {
  return {
    measured: {
      video: "wall_e2e.mp4",
      serve_index: 0,
      impact_time_sec: 0.67,
      impact_frame: 20,
      autonomous_frame: 20,
      impact_pixel: [240, 240],
      wall_x_m: 1.27,
      wall_y_m: 1.0,
      calibration_reprojection_rms_px: 0,
      raw_track_samples: 21,
    },
    inferred: {
      speed_m_s: 12.4,
      speed_km_h: 44.64,
      speed_mph: 27.74,
      speed_uncertainty_m_s: 0.4,
      landing_x_m: 0.8,
      landing_z_m: 5.1,
      in_service_box: true,
      service_box_side: "deuce",
    },
    assumed: {
      gravity_m_s2: 9.81,
      serve_contact_height_m: 2.8,
      wall_distance_m: 1.57,
    },
    confidence: 0.91,
    warnings: [],
    artifacts: {
      annotated_video: "/api/wall/artifacts/wall_e2e_annotated.mp4",
      review_clip: { url: "/api/wall/artifacts/wall_e2e_impact_review.mp4" },
      plots: {
        speed_profile: "/api/wall/artifacts/plots/speed_profile.png",
        wall_impact: "/api/wall/artifacts/plots/wall_impact.png",
      },
      json: { url: "/api/wall/artifacts/result.json" },
      csv: { url: "/api/wall/artifacts/result.csv" },
    },
  }
}

describe("WallResultsDashboard backend result rendering", () => {
  it("renders the dashboard from normalized backend artifact URLs", () => {
    render(<WallResultsDashboard result={createBackendResult()} />)

    expect(screen.getByText(/measured impact/i)).toBeInTheDocument()
    expect(screen.getByText(/velocity/i)).toBeInTheDocument()
    expect(screen.getByText(/court projection/i)).toBeInTheDocument()
    expect(screen.getByText("44.64 km/h")).toBeInTheDocument()
    expect(screen.getByText("1.27 × 1.00 m")).toBeInTheDocument()
    expect(screen.getByText("91.0%")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /result\.json/i })).toHaveAttribute(
      "href",
      "/api/wall/artifacts/result.json"
    )
    expect(screen.getByRole("link", { name: /result\.csv/i })).toHaveAttribute(
      "href",
      "/api/wall/artifacts/result.csv"
    )
  })
})


## Known Limitations (T2)

### Single Camera View
- Only lateral view supported - no 3D reconstruction
- Cannot account for ball movement toward/away from camera
- Perspective distortion not corrected

### Tracking Limitations
- Template matching assumes smooth ball motion
- Fast or erratic ball movement may fail to track
- Low confidence matches (< 0.5) keep previous position
- Template not updated if ball deforms significantly

### Calibration Accuracy
- Manual 2-point calibration assumes known real-world distance
- User must estimate distance on screen accurately
- Only as good as the calibration plane
- Motion perpendicular to calibration plane may have inaccurate speed estimates

### Environmental Factors
- Lighting changes can affect tracking
- Ball may be occluded by player/racket
- Motion blur at high speeds
- Resolution limits at 4K (3840x2160 for iPhone 14 Pro Max)

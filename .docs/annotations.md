
# Improvements Ideas

## Ball Detection

- Consider that the ball may not move more than "X" pixels from frame to frame. Only if it wasn't detected by more thant "Y" frames. So ignore the "detection" if that case happens.
- Multiples detections can be done in parallel with another, so in a single frame we can detect for the ball, racket position, body position to identify the exact serve motion.
- Use past "Y" frames as information for the detection of the next frame.

## Ball Seed detection

- Only consider a certain amount of ball detections from frames that were actually happened one after the other, and only when the ball moved close to the previous frame(by "X" amount of pixel), that way we'll remove false positives.

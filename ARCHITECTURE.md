# Strider System Architecture

**Detailed design decisions, data flow, mathematical foundations, and implementation strategy.**

---

## Design Principles

1. **Distance-Based Spatial Quantification**  
   All metrics derived from world coordinates (meters), never frame indices or elapsed time. Quartile boundaries at 0m/3m/6m along walking axis, recomputed if patient turns before/after 6m.

2. **Two-Pass Processing**  
   Pass 1: Extract and store raw keypoints. Pass 2: Compute derived quantities. Enables calibration (which requires full trajectory) before metric computation.

3. **Pathological Gait Robustness**  
   No assumptions of periodic gait, symmetric walking, constant cadence, or normal movement. Algorithms adapt to shuffling, freezing, hesitation, asymmetric stepping.

4. **Extensibility Over Rigidity**  
   Core algorithms (foot strike, asymmetry) have clear hooks for enhancement without major refactors. Simple baseline → sophisticated pathological support.

5. **Modular Pipeline**  
   Each stage (pose → tracking → calibration → segmentation → events → metrics → clinical) is independently testable and replaceable.

---

## Data Flow Architecture

```
┌─────────────────────┐
│   Video Input       │
│  (30-60 fps)        │
└──────────┬──────────┘
           │
    ┌──────▼─────────────────────────────┐
    │   PASS 1: RAW EXTRACTION            │
    ├─────────────────────────────────────┤
    │ ▪ VideoReader (30–60 fps → 30 fps)  │
    │ ▪ PersonDetector (RTMDet-nano)      │
    │ ▪ ByteTracker (track_id per frame)  │
    │ ▪ PatientSelector (lock largest)    │
    │ ▪ PoseEstimator (RTMPose-l, 133 kpt)│
    │ ▪ KeypointSmoother (One-Euro)       │
    │ ▪ Store in memory (5.1 MB / 2 min)  │
    └──────────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────────-┐
        │   PASS 2: METRIC COMPUTATION         │
        ├──────────────────────────────────────┤
        │ ▪ SpatialMapper (homography)         │
        │ ▪ PathAxisFitter (SVD → meters)      │
        │ ▪ PhaseDetector (TOWARD/TURN/AWAY)   │
        │ ▪ QuartileEngine (Q1/Q2/Q3/Q4)       │
        │ ▪ FootStrikeDetector (heel strikes)  │
        │ ▪ StepValidator (plausibility)       │
        │ ▪ FOGDetector (spectral FI)          │
        │ ▪ MetricComputer (per-quartile)      │
        │ ▪ GlobalMetricAggregator             │
        │ ▪ ClinicalAnalyzer (flags)           │
        └──────────┬───────────────────────────┘
                   │
        ┌──────────▼──────────────┐
        │   OUTPUT ASSEMBLY       │
        ├─────────────────────────┤
        │ ▪ AnalysisResult JSON   │
        │ ▪ Annotated video       │
        │ ▪ CSV/JSON exports      │
        │ ▪ PDF report (Phase 2)  │
        └─────────────────────────┘
```

---

## Module Design Details

### 1. Pose Estimation (`strider/pose/rtmpose.py`)

**RTMPose-l WholeBody Model:**
- 133 keypoints: body (17) + hands (21 each) + face (468, reduced for efficiency)
- SimCC (Simulated Coordinate Classification) output format
- ONNX Runtime CPU inference: 15–25 fps on modern CPU

**SimCC Decoding:**
```python
# RTMPose outputs logits for each keypoint
# SimCC format: two separate logits arrays (x and y)
x_index = argmax(simcc_x_logits[kpt])
y_index = argmax(simcc_y_logits[kpt])
x_coord = x_index / simcc_split_ratio  # normalized
y_coord = y_index / simcc_split_ratio  # normalized

# Apply inverse affine transform to map back to original image space
keypoint_image = affine_inverse @ [x_coord, y_coord, 1]
```

**Keypoint Indices (RTMPose WholeBody):**
- Ankles: 15 (left), 16 (right) — used for foot strike detection
- Hips: 11 (left), 12 (right) — used for COM/sway computation
- Shoulders: 5 (left), 6 (right) — used for turning detection
- All 133 keypoints stored but only ~15 used for gait analysis

**Fallback to MediaPipe:**
If RTMPose unavailable, use MediaPipe Pose (33 keypoints, less accurate but functional).

### 2. Tracking (`strider/tracking/bytetrack.py`)

**ByteTrack Algorithm:**

```
For each new frame with detections D:
  1. Split D into high-confidence (>0.5) and low-confidence (≤0.5)
  2. Match high-conf detections to existing tracks using Hungarian algorithm
     - Cost matrix: IoU(detection, track) or 1 - IoU
  3. Match unmatched tracks to low-conf detections
     - Catches patient during turn (confidence drops)
  4. Create new tracks from unmatched high-conf detections
  5. Update Kalman state for all matched tracks
  6. Mark lost tracks; remove if lost for >30 frames (2 sec at 30fps)
```

**Kalman State Vector:**
```
x = [cx, cy, aspect_ratio, height, vx, vy, va, vh]
     ↑   ↑        ↑         ↑     ↑  ↑   ↑   ↑
    center position + velocity, for predicting next frame
```

**Patient Selection:**
- Frame 1: Lock patient_track_id = track with largest bbox area
- If track lost and patient hasn't completed trial: attempt re-ID by closest position to last-known location
- All downstream pose/metric computation uses only locked patient track

### 3. Spatial Calibration (`strider/calibration/`)

**Goal:** Map image coordinates to world space (meters along 6m walking path).

#### Mode 1: Manual Calibration

Clinician clicks 4 floor points in video frame, assigns world coordinates.

```python
# Frontend UI will provide these (Phase 4+)
image_pts = [click1, click2, click3, click4]  # user clicks
world_pts = [(0,0), (6,0), (0,0.5), (6,0.5)]   # 0-6m path, 0.5m width

# Compute homography
H, _ = cv2.findHomography(np.float32(image_pts), np.float32(world_pts), cv2.RANSAC)

# Apply to any point
world_pt = cv2.perspectiveTransform(np.float32([[image_pt]]), H)[0][0]
```

#### Mode 2: Auto Calibration (Default)

No user input required.

```python
# 1. Accumulate all ankle midpoint image positions across full video
ankle_positions_image = [midpoint(ankles[t]) for t in all_frames]

# 2. Fit line via SVD (principal component analysis)
centered = ankle_positions_image - mean(ankle_positions_image)
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
walking_axis_direction = Vt[0, :]  # first PC is walking direction

# 3. Check explained variance
variance_ratio = S[0]**2 / np.sum(S**2)
if variance_ratio < 0.85:
    warn("Low variance; consider manual calibration")

# 4. Project all points onto axis; compute range
projections = ankle_positions_image @ walking_axis_direction
pixel_range = max(projections) - min(projections)
meters_per_pixel = 6.0 / pixel_range

# 5. Construct pseudo-homography from derived points
# (Details omitted; results in matrix H)
```

**Validation (both modes):**
```python
# Roundtrip test
for pt in test_points:
    world = image_to_world(pt, H)
    image_back = world_to_image(world, H_inv)
    assert np.linalg.norm(pt - image_back) < 0.001  # < 1mm error
```

### 4. Phase Detection (`strider/segmentation/phase_detector.py`)

**Goal:** Label each frame as TOWARD, TURN, or AWAY based on spatial position.

```python
def detect_phases(world_x_positions, fps):
    """
    world_x_positions: (N,) array of x-coordinates in meters
    Returns: (N,) array of phase labels {0: TOWARD, 1: TURN, 2: AWAY}
    """
    # Compute velocity (distance traveled per frame)
    velocity = np.gradient(world_x_positions, 1/fps)
    
    # Smooth velocity (Gaussian, sigma=0.3s)
    velocity_smooth = gaussian_filter1d(velocity, sigma=0.3*fps)
    
    # Find sign change (from positive to negative)
    sign_change = np.where(velocity_smooth[:-1] * velocity_smooth[1:] < 0)[0]
    
    if len(sign_change) == 0:
        # Patient never reversed direction (unlikely for 6m walk test)
        return np.zeros_like(world_x_positions, dtype=int)
    
    turn_center = sign_change[0]
    
    # TURN phase: region around zero-crossing where |velocity| < threshold
    turn_start = turn_center
    while turn_start > 0 and abs(velocity_smooth[turn_start]) < 0.01:
        turn_start -= 1
    
    turn_end = turn_center
    while turn_end < len(velocity_smooth) and abs(velocity_smooth[turn_end]) < 0.01:
        turn_end += 1
    
    # Label phases
    phases = np.zeros_like(world_x_positions, dtype=int)
    phases[:turn_start] = 0  # TOWARD
    phases[turn_start:turn_end] = 1  # TURN
    phases[turn_end:] = 2  # AWAY
    
    return phases
```

**Edge Case: Turn before 6m**
- If velocity reversal occurs at world_x = 5.2m instead of 6m, the `turn_center` is still detected correctly
- Quartile boundaries are recomputed: Q1/Q2 boundary = 5.2/2 = 2.6m (not 3m)
- Clinical flag `NON_STANDARD_PATH` issued to clinician

### 5. Quartile Assignment (`strider/segmentation/quartile_engine.py`)

**CRITICAL MODULE: All spatial metrics depend on correct assignment.**

```python
class QuartileEngine:
    def __init__(self, path_length_m=6.0):
        self.half = path_length_m / 2  # 3.0m
        self.full = path_length_m      # 6.0m
    
    def assign_quartile(self, world_x, phase):
        """
        Assigns a step/event to its quartile based on SPATIAL POSITION.
        NOT based on timestamp or frame number.
        """
        if phase == Phase.TOWARD:
            # Toward phase: world_x goes from 0 to 6
            if world_x < self.half:
                return Quartile.Q1
            else:
                return Quartile.Q2
        
        elif phase == Phase.AWAY:
            # Away phase: world_x goes from 6 to 0 (decreasing)
            # Reframe as "distance away from turn point"
            distance_away = self.full - world_x
            if distance_away < self.half:
                return Quartile.Q3
            else:
                return Quartile.Q4
        
        else:  # TURN phase
            return Quartile.TURN
    
    def compute_quartile_time_windows(self, world_x, phases, timestamps):
        """
        For each quartile, find the time range when patient is in that zone.
        Returns {Quartile.Q1: (t_start, t_end), ...}
        """
        windows = {}
        for qk in [Quartile.Q1, Q2, Q3, Q4]:
            in_quartile = np.array([
                self.assign_quartile(world_x[t], phases[t]) == qk
                for t in range(len(timestamps))
            ])
            if np.any(in_quartile):
                indices = np.where(in_quartile)[0]
                windows[qk] = (timestamps[indices[0]], timestamps[indices[-1]])
            else:
                windows[qk] = None
        return windows

# INVARIANT: For any trial, steps_Q1 + steps_Q2 + steps_Q3 + steps_Q4 = total_steps
# This invariant MUST hold. If it doesn't, there's a bug in assignment logic.
```

### 6. Foot Strike Detection (`strider/gait_events/foot_strike.py`)

**MOST CRITICAL MODULE: Errors cascade to all metrics.**

**Algorithm (Simple but Extensible):**

```python
def detect_foot_strikes(ankle_y_normalized, fps, prominence_factor=0.1):
    """
    ankle_y_normalized: (N,) array, normalized ankle Y position [0,1]
    fps: frame rate
    
    Returns: list of (frame_idx, side, world_position) tuples
    """
    from scipy.signal import find_peaks
    
    # Find local minima (ankle closest to ground)
    # We look for minima of -ankle_y (i.e., maxima of -ankle_y)
    
    # Adaptive prominence: scale with local range
    local_range = max(ankle_y_normalized) - min(ankle_y_normalized)
    min_prominence = prominence_factor * local_range
    
    peaks, properties = find_peaks(
        -ankle_y_normalized,  # Find maxima of -y = minima of y
        prominence=min_prominence,
        distance=int(0.2 * fps)  # Min 200ms between strikes (0.2s = min step time)
    )
    
    strikes = []
    for idx in peaks:
        # Determine side (left vs right) via keypoint index
        # Could also use amplitude heuristics (e.g., position relative to image center)
        side = "L" if idx % 2 == 0 else "R"  # Placeholder
        
        strikes.append(FootStrikeEvent(
            frame_idx=idx,
            timestamp=idx / fps,
            side=side,
            world_x=world_positions[idx],  # From spatial mapper
            world_y=0.0,  # Lateral position (computed separately)
            confidence=ankle_confidence[idx]
        ))
    
    return strikes

# SHUFFLING FALLBACK (future enhancement)
# If max(ankle_y) - min(ankle_y) < 0.02 (flat signal):
#   - Fall back to mediolateral (ML) displacement-based step detection
#   - Detect lateral weight shifts of COM → step attempts
#   - This is hooked in for Phase 2 enhancement without refactoring core
```

**Step Validation:**

```python
def validate_steps(strikes):
    """
    Temporal plausibility: no step interval < 0.15s or > 3.0s (patient might pause)
    Spatial plausibility: alternating left/right
    Remove spurious detections
    """
    validated = []
    for i, strike in enumerate(strikes):
        if i > 0:
            dt = strike.timestamp - strikes[i-1].timestamp
            if dt < 0.15 or dt > 3.0:
                continue  # Skip implausible interval
            
            # Check alternation
            if strike.side == strikes[i-1].side:
                # Same side twice in a row (rare but possible with shuffling)
                # Log but don't skip; clinical interest
                pass
        
        validated.append(strike)
    
    return validated
```

### 7. FOG Detection (`strider/gait_events/fog_detector.py`)

**Freezing of Gait (FOG) = high-frequency stepping tremor or cessation.**

```python
def compute_freeze_index(ankle_velocity, fps, window_sec=2.0):
    """
    ankle_velocity: (N,) vertical ankle velocity signal (m/s or normalized)
    
    Sliding window PSD analysis (scipy.signal.welch)
    """
    from scipy.signal import welch
    
    window_frames = int(window_sec * fps)
    step_frames = int(window_sec * 0.5 * fps)  # 50% overlap
    
    fi_values = np.zeros(len(ankle_velocity))
    
    for start_idx in range(0, len(ankle_velocity) - window_frames, step_frames):
        window = ankle_velocity[start_idx : start_idx + window_frames]
        
        # Compute PSD
        freqs, psd = welch(window, fs=fps, nperseg=min(256, len(window)))
        
        # Power in locomotion band [0.5, 3.0] Hz (normal stepping)
        loco_mask = (freqs >= 0.5) & (freqs <= 3.0)
        P_loco = np.sum(psd[loco_mask])
        
        # Power in freeze band [3.0, 8.0] Hz (high-frequency tremor)
        freeze_mask = (freqs >= 3.0) & (freqs <= 8.0)
        P_freeze = np.sum(psd[freeze_mask])
        
        # Freeze Index
        fi = (P_freeze ** 2) / (P_loco ** 2 + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Broadcast to all frames in window
        fi_values[start_idx : start_idx + window_frames] = fi
    
    return fi_values

def detect_fog_episodes(fi_values, fps, threshold=2.5, min_duration_sec=0.5):
    """
    FI > threshold for >= min_duration → FOG episode
    """
    min_frames = int(min_duration_sec * fps)
    
    in_episode = fi_values > threshold
    
    episodes = []
    start_idx = None
    
    for idx in range(len(in_episode)):
        if in_episode[idx] and start_idx is None:
            start_idx = idx  # Episode beginning
        elif not in_episode[idx] and start_idx is not None:
            duration = idx - start_idx
            if duration >= min_frames:  # Only count if long enough
                episodes.append(FOGEpisode(
                    start_frame=start_idx,
                    end_frame=idx,
                    duration_sec=(idx - start_idx) / fps,
                    severity=np.mean(fi_values[start_idx:idx])
                ))
            start_idx = None
    
    return episodes
```

### 8. Metrics Computation

#### Per-Quartile Metrics (`strider/metrics/per_quartile.py`)

**Step Count:**
```python
steps_Qk = [s for s in all_steps if assign_quartile(s.world_x, s.phase) == Qk]
step_count_Qk = len(steps_Qk)
```

**Cadence:**
```python
duration_Qk = t_end_Qk - t_start_Qk  # seconds (from quartile_time_windows)
cadence_Qk = (step_count_Qk / duration_Qk) * 60  # steps/min
```

**Asymmetry (Robinson Index):**
```python
steps_L_Qk = [s for s in steps_Qk if s.side == 'L']
steps_R_Qk = [s for s in steps_Qk if s.side == 'R']

mean_length_L = np.mean([s.step_length for s in steps_L_Qk])
mean_length_R = np.mean([s.step_length for s in steps_R_Qk])

AI_length = abs(mean_length_L - mean_length_R) / (0.5 * (mean_length_L + mean_length_R)) * 100

# Similar for swing time asymmetry
AI_swing = ...

# Composite (clinical weighting)
asymmetry_Qk = 0.6 * AI_length + 0.4 * AI_swing
```

**Sway (RMS Mediolateral Displacement):**
```python
# COM proxy: weighted average of hips + shoulders
COM_proxy_y = [
    0.6 * midpoint(L_hip[t], R_hip[t]).y +
    0.4 * midpoint(L_shoulder[t], R_shoulder[t]).y
    for t in quartile_frames
]

# RMS deviation from mean
mean_COM_y = np.mean(COM_proxy_y)
sway_RMS_Qk = np.sqrt(np.mean((COM_proxy_y - mean_COM_y) ** 2))  # meters
```

#### Global Aggregation (`strider/metrics/global_metrics.py`)

```python
# Total steps (must equal sum of quartile counts)
total_steps = step_count_Q1 + step_count_Q2 + step_count_Q3 + step_count_Q4

# Overall cadence
total_walk_time = (t_end_Q4 - t_start_Q1)  # excludes turn phase
cadence_global = (total_steps / total_walk_time) * 60

# Overall asymmetry (mean of quartile values)
asymmetry_global = np.mean([asymmetry_Q1, asymmetry_Q2, asymmetry_Q3, asymmetry_Q4])

# Overall sway (mean)
sway_global = np.mean([sway_Q1, sway_Q2, sway_Q3, sway_Q4])

# Turning time
turning_time = t_end_turn - t_start_turn
```

### 9. Clinical Analysis (`strider/clinical/flags.py`)

**Evidence-Based Thresholds:**

```python
CLINICAL_THRESHOLDS = {
    'FOG': {
        'trigger': lambda metrics: any(ep.severity > 0 for ep in metrics.fog_episodes),
        'severity': 'CRITICAL',
        'description': 'Freezing of Gait episodes detected',
        'source': 'Nieuwboer et al. 2004'
    },
    'ASYMMETRY': {
        'trigger': lambda metrics: metrics.asymmetry_global > 10,
        'severity': 'WARNING',
        'description': 'Gait asymmetry above normal range (>10%)',
        'source': 'Patterson et al. 2010'
    },
    'REDUCED_CADENCE': {
        'trigger': lambda metrics: metrics.cadence_global < 80,
        'severity': 'WARNING',
        'description': 'Below-normal cadence (< 80 steps/min)',
        'source': 'Bohannon 1997'
    },
    # ... more thresholds
}
```

---

## Error Handling & Failure Cases

### Case 1: Patient Partially Leaves Frame

**Symptom:** Keypoint confidence drops below threshold  
**Detection:** `confidence < 0.3` for most body keypoints  
**Handling:** Kalman filter predicts position for up to 15 frames (0.5s); beyond that, issue `DATA_GAP` warning

### Case 2: Shuffling Gait (Flat Ankle Trajectory)

**Symptom:** Peak amplitude in foot strike detection < 0.02 normalized units  
**Detection:** `max(ankle_y) - min(ankle_y) < 0.02`  
**Handling:** Switch to mediolateral (ML) displacement-based step detection (Phase 2 enhancement)

### Case 3: FOG Mid-Quartile

**Symptom:** Spectral freeze index > threshold for ≥0.5s within quartile  
**Detection:** `FOGDetector.detect_fog_episodes()`  
**Handling:** 
- Step count excludes shuffling-in-place (validated via step validator)
- Cadence computed on non-FOG intervals only
- FOG duration reported separately
- Flag `FREEZING_OF_GAIT` with CRITICAL severity

### Case 4: Auto-Calibration Failure (Non-Straight Path)

**Symptom:** Patient walks at diagonal or camera view is oblique  
**Detection:** PCA explained variance < 85%  
**Handling:** Issue `CALIBRATION_WARNING`; prompt for manual calibration; still compute relative metrics but mark spatial ones as unreliable

### Case 5: Multiple People in Frame

**Symptom:** Multiple bounding boxes detected consistently  
**Detection:** `len(active_tracks) > 1 throughout trial`  
**Handling:** Lock to largest bbox at frame 1; if cross-track interference, flag `INTERFERENCE_WARNING`

### Case 6: Turn Before 6m

**Symptom:** Velocity reversal at world_x < 6.0  
**Detection:** `actual_turn_distance ≠ 6.0`  
**Handling:** Recompute quartile boundaries from actual turn point; issue `NON_STANDARD_PATH` flag

### Case 7: Very Fast or Very Slow Walkers

**Symptom:** Estimated cadence < 30 or > 200 steps/min  
**Detection:** After 10 steps, compute `estimated_cadence_hz`  
**Handling:** Adjust `min_step_interval = 0.4 / estimated_cadence_hz` dynamically to adapt foot strike detection

### Case 8: GPU Unavailable

**Symptom:** ONNX Runtime CPU or PyTorch not available  
**Detection:** Model load fails  
**Handling:** Tiered fallback:
   1. RTMPose ONNX (primary)
   2. RTMPose PyTorch CPU (if torch available)
   3. MediaPipe (last resort, lower accuracy)

---

## Testing Strategy

### Unit Tests

1. **Quartile Engine** (`test_quartile_engine.py`)
   - Test Q1/Q2/Q3/Q4 assignment at various world_x values
   - Test boundary conditions (exactly at 3m, 6m)
   - Test AWAY phase logic (decreasing world_x)
   - Test invalid phase (TURN)

2. **Homography** (`test_homography.py`)
   - SVD axis fitting on synthetic trajectory
   - Roundtrip error validation (< 1mm)
   - PC1 variance check

3. **Foot Strike** (`test_foot_strike.py`)
   - Synthetic sinusoidal ankle motion → verify peak detection
   - Flat signal (shuffling) → verify adaptive prominence
   - Step interval validation

4. **Phase Detection** (`test_phase_detector.py`)
   - Step through phase labels on synthetic position sequence
   - Turn point detection (velocity reversal)

### Integration Tests

1. **Full Pipeline with Synthetic Gait** (`test_pipeline_synthetic.py`)
   - Parametric gait generator: configurable cadence, asymmetry, sway
   - Run end-to-end pipeline
   - Assert: step count == ground truth ±1, metrics within expected ranges

### Manual Validation

1. Process your test videos (mix of patient + healthy)
2. Manually annotate 5–10 steps in each video (frame-accurate)
3. Compare automated vs manual: target ≤2-frame error on 90% of events

---

## Performance Targets

| Stage | Time Budget | Target FPS |
|-------|-----------|-----------|
| Pose Estimation | 40% of total | 15–25 fps |
| Tracking | 10% | — |
| Spatial Mapping | 5% | — |
| Gait Event Detection | 30% | — |
| Metric Computation | 10% | — |
| Rendering | 5% | — |

**Overall:** < 3× real-time (2-minute video in < 6 minutes on CPU)

---

## Future Enhancements

- **Pathological Gait Indicators:** Freezing severity scoring, hesitation detection, step regularity (Lyapunov exponents)
- **Advanced Metrics:** Movement smoothness (jerk), balance confidence, gait entropy
- **Multi-Modal Fusion:** Optional IMU/wearable data for acceleration-based metrics
- **Real-Time Processing:** Streaming inference on mobile devices
- **Comparison Across Sessions:** Track metrics over time for rehabilitation monitoring

---

## References

1. Nieuwboer A, et al. (2004). Characteristics of freezing of gait. Movement Disorders. 19(7):846-848.
2. Bohannon RW, et al. (1997). Comfortable and maximum walking speed of adults aged 20–79 years. Journal of Aging & Physical Activity. 5(4):259-269.
3. Robinson RO, et al. (1987). Walking test for neurological disorders. Clinical Biomechanics. 2(3):139-146.
4. Brach JS, et al. (2005). Stride variability in older adults. Journals of Gerontology. 60(12):1504-1510.
5. Moe-Nilssen R, et al. (2004). Trunk accelerometry as a measure of balance control during quiet standing. Gait & Posture. 16(1):60-68.
6. Salarian A, et al. (2010). Quantification of turning during walking in subjects with Parkinson's disease. Neuroscience Letters. 483(2):124-128.
7. Patterson KK, et al. (2010). Gait asymmetry in community-ambulating stroke survivors. Archives of Physical Medicine and Rehabilitation. 88(5):596-601.

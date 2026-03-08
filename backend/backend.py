
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
from collections import deque
from typing import Optional

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
API_KEY       = os.environ.get("API_KEY", "")
MODEL_ID      = "deepseek/deepseek-chat"
PROCESS_EVERY = 1          # process EVERY frame (no skipping)
MAX_WIDTH     = 640        # resize input to this width before pose inference
SMOOTH_WIN    = 5          # rolling-average window for angle smoothing
MIN_VIS       = 0.55       # minimum MediaPipe landmark visibility to trust
# ICC thresholds
ICC_ELBOW_LIMIT        = 15.0   # degrees (Law 24)
SCR_INJURY_LIMIT       = 30.0   # shoulder counter-rotation > 30° = injury risk
KNEE_MIN               = 130.0  # degrees – too bent at FFC
KNEE_MAX               = 175.0  # degrees – too rigid
HSS_MIN                = 15.0   # hip-shoulder separation minimum
HSS_MAX                = 45.0   # hip-shoulder separation maximum
STRIDE_MIN             = 1.0
STRIDE_MAX             = 1.55

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_ds      = mp.solutions.drawing_styles

PL = mp_pose.PoseLandmark   # shorthand


def make_pose():
    """Create a MediaPipe Pose instance configured for speed + accuracy balance."""
    return mp_pose.Pose(
        model_complexity=1,          # 1 = Full (best balance). 2 is too slow for CPU.
        smooth_landmarks=True,       # built-in temporal smoothing across frames
        enable_segmentation=False,   # we don't need segmentation mask
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        static_image_mode=False,     # video mode: tracker reused across frames
    )


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────

def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle at vertex B formed by rays B→A and B→C, in [0, 180] degrees.
    Works with 2-D or 3-D arrays.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos_val = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def axis_angle_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Return the bearing angle (degrees) of the vector from p1 to p2 in 2-D."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def hss_angle(hip_l, hip_r, sh_l, sh_r) -> float:
    """
    Hip-Shoulder Separation — 2-D image-plane approach.

    MediaPipe Z-depth is unreliable for monocular (single-camera) side-view
    footage. Instead we use the 2-D X-Y image plane:
      hip_bearing   = atan2(hip_R.y - hip_L.y,  hip_R.x - hip_L.x)
      shoulder_bearing = atan2(sh_R.y  - sh_L.y,  sh_R.x  - sh_L.x)
    HSS = |hip_bearing - shoulder_bearing| (unsigned, clamped to [0,90])

    For a true side-on delivery the hips rotate ahead of the shoulders;
    the angle captures that lead/lag. Optimal range  15–45° (per St Mary's
    University / Quintic Sports biomechanics literature).
    """
    hip_bearing = np.degrees(np.arctan2(
        hip_r[1] - hip_l[1], hip_r[0] - hip_l[0]
    ))
    sh_bearing  = np.degrees(np.arctan2(
        sh_r[1]  - sh_l[1],  sh_r[0]  - sh_l[0]
    ))
    diff = abs(hip_bearing - sh_bearing)
    # Normalise to [0, 90] — wrap-around artefact prevention
    if diff > 90:
        diff = 180.0 - diff
    return float(np.clip(diff, 0.0, 90.0))


def trunk_tilt(shoulder_l, shoulder_r, hip_l, hip_r) -> float:
    """
    Lateral trunk tilt: angle between the mid-shoulder→mid-hip spine vector
    and a perfect vertical (up direction).  0° = perfectly upright.
    """
    mid_sh  = (np.array(shoulder_l) + np.array(shoulder_r)) / 2.0
    mid_hip = (np.array(hip_l)      + np.array(hip_r))      / 2.0
    spine   = mid_sh - mid_hip
    vertical = np.array([0.0, -1.0, 0.0])   # MediaPipe: Y increases downward
    norm_sp = np.linalg.norm(spine)
    if norm_sp < 1e-9:
        return 0.0
    cos_val = np.dot(spine, vertical) / norm_sp
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# ─────────────────────────────────────────────
#  LANDMARK EXTRACTION
# ─────────────────────────────────────────────

def lm_xyz(lm_list, part: PL) -> tuple:
    """Return (xyz_array, visibility) for a landmark."""
    p = lm_list[part.value]
    return np.array([p.x, p.y, p.z]), p.visibility


def visible(vis: float) -> bool:
    return vis >= MIN_VIS


def pick_side_landmarks(lm, side: str):
    """
    Return a dict of np.arrays for the bowling-arm side and the front-leg side.
    side: "LEFT" or "RIGHT"
    For a right-arm bowler, bowling arm = RIGHT; front (non-dominant) leg = LEFT.
    """
    s = side  # "LEFT" or "RIGHT"
    f = "LEFT" if s == "RIGHT" else "RIGHT"  # front leg is opposite

    def get(name_prefix):
        part = PL[f"{name_prefix}"]
        p = lm[part.value]
        return np.array([p.x, p.y, p.z]), p.visibility

    bowl_sh, v1  = get(f"{s}_SHOULDER")
    bowl_el, v2  = get(f"{s}_ELBOW")
    bowl_wr, v3  = get(f"{s}_WRIST")
    front_hi, v4 = get(f"{f}_HIP")
    front_kn, v5 = get(f"{f}_KNEE")
    front_an, v6 = get(f"{f}_ANKLE")
    back_an,  v7 = get(f"{s}_ANKLE")
    right_sh, v8 = get("RIGHT_SHOULDER")
    left_sh,  v9 = get("LEFT_SHOULDER")
    right_hi, va = get("RIGHT_HIP")
    left_hi,  vb = get("LEFT_HIP")

    all_vis = [v1, v2, v3, v4, v5, v6, v7, v8, v9, va, vb]
    if min(all_vis) < MIN_VIS:
        return None

    return {
        "bowl_sh": bowl_sh, "bowl_el": bowl_el, "bowl_wr": bowl_wr,
        "front_hi": front_hi, "front_kn": front_kn, "front_an": front_an,
        "back_an": back_an,
        "right_sh": right_sh, "left_sh": left_sh,
        "right_hi": right_hi, "left_hi": left_hi,
    }


# ─────────────────────────────────────────────
#  BOWLING ARM AUTO-DETECTION
# ─────────────────────────────────────────────

def detect_bowling_arm(lm_history: list) -> str:
    """
    Analyse first few frames and pick the arm whose wrist is *higher*
    on average (i.e., lower Y value in image coordinates) — that is the
    bowling arm reaching over.  Falls back to "RIGHT" if inconclusive.
    """
    lw_ys, rw_ys = [], []
    for lm in lm_history:
        lw = lm[PL.LEFT_WRIST.value]
        rw = lm[PL.RIGHT_WRIST.value]
        if lw.visibility >= MIN_VIS:
            lw_ys.append(lw.y)
        if rw.visibility >= MIN_VIS:
            rw_ys.append(rw.y)
    if not lw_ys and not rw_ys:
        return "RIGHT"
    lw_mean = float(np.mean(lw_ys)) if lw_ys else 1.0
    rw_mean = float(np.mean(rw_ys)) if rw_ys else 1.0
    # Lower Y = higher in image = active bowling wrist
    return "LEFT" if lw_mean < rw_mean else "RIGHT"


# ─────────────────────────────────────────────
#  DELIVERY PHASE DETECTION
# ─────────────────────────────────────────────

class PhaseDetector:
    """
    Lightweight finite-state machine that tracks the bowling delivery phases:
      idle → loading → arm_horizontal → front_foot_contact → release → idle

    We detect:
      - arm_horizontal: bowling wrist Y ≈ bowling shoulder Y (arm at ~90°)
      - release:        bowling wrist Y reaches its minimum (highest point)

    Uses a small history of wrist Y values to find the peak.
    """
    def __init__(self):
        self.wrist_y_hist = deque(maxlen=30)
        self.phase = "idle"
        self.arm_horiz_frame  = None
        self.release_frame     = None
        self._frames_in_phase  = 0
        self._prev_wrist_y     = None

    def update(self, bowl_wr_y: float, bowl_sh_y: float, frame_idx: int) -> str:
        """
        bowl_wr_y, bowl_sh_y: normalised [0,1] Y coords (0 = top of image).
        Returns current phase string.
        """
        self.wrist_y_hist.append(bowl_wr_y)
        self._frames_in_phase += 1

        # arm_horizontal: wrist Y close to shoulder Y (within 5% of frame height)
        if self.phase == "idle":
            horiz_threshold = 0.08  # within 8% of shoulder height
            if abs(bowl_wr_y - bowl_sh_y) < horiz_threshold:
                self.phase = "arm_horizontal"
                self.arm_horiz_frame = frame_idx
                self._frames_in_phase = 0

        elif self.phase == "arm_horizontal":
            # Wait until wrist rises above shoulder (wrist Y < shoulder Y)
            if bowl_wr_y < bowl_sh_y - 0.05:
                self.phase = "pre_release"
                self._frames_in_phase = 0

        elif self.phase == "pre_release":
            # Look for the wrist Y minimum (highest point) — that marks release
            if (self._prev_wrist_y is not None and
                    bowl_wr_y > self._prev_wrist_y + 0.02 and
                    self._frames_in_phase >= 3):
                self.phase = "release"
                self.release_frame = frame_idx
                self._frames_in_phase = 0

        elif self.phase == "release":
            # Brief hold then reset
            if self._frames_in_phase > 10:
                self.phase = "idle"
                self._frames_in_phase = 0
                self.arm_horiz_frame = None
                self.release_frame   = None

        self._prev_wrist_y = bowl_wr_y
        return self.phase


# ─────────────────────────────────────────────
#  TEMPORAL SMOOTHER
# ─────────────────────────────────────────────

class AngleSmoother:
    """Per-metric rolling-window average smoother."""
    def __init__(self, window: int = SMOOTH_WIN):
        self._bufs = {}
        self._window = window

    def smooth(self, key: str, value: float) -> float:
        if key not in self._bufs:
            self._bufs[key] = deque(maxlen=self._window)
        self._bufs[key].append(value)
        return float(np.mean(self._bufs[key]))


# ─────────────────────────────────────────────
#  ANGLE CALCULATIONS (PER FRAME)
# ─────────────────────────────────────────────

def compute_angles(pts: dict) -> dict:
    """
    Given a dict of landmark arrays, compute all relevant biomechanical angles.
    Returns a dict of floats (all in degrees).
    """
    # 1. Elbow Flexion — angle at elbow joint (shoulder→elbow→wrist)
    elbow_angle = angle_3pt(pts["bowl_sh"], pts["bowl_el"], pts["bowl_wr"])

    # 2. Front Knee Angle — hip→knee→ankle of the front (non-dominant) leg
    front_knee = angle_3pt(pts["front_hi"], pts["front_kn"], pts["front_an"])

    # 3. Hip-Shoulder Separation (transverse plane) — using X and Z coords
    hss = hss_angle(pts["left_hi"], pts["right_hi"], pts["left_sh"], pts["right_sh"])

    # 4. Shoulder Alignment (bearing angle) — used to track SCR over phases
    sh_bearing = axis_angle_2d(
        np.array([pts["left_sh"][0], pts["left_sh"][2]]),
        np.array([pts["right_sh"][0], pts["right_sh"][2]])
    )

    # 5. Trunk Lateral Tilt
    tilt = trunk_tilt(pts["left_sh"], pts["right_sh"], pts["left_hi"], pts["right_hi"])

    # 6. Stride Length (normalised by shoulder width)
    shoulder_w = np.linalg.norm(pts["right_sh"][:2] - pts["left_sh"][:2])
    stride_raw = np.linalg.norm(pts["front_an"][:2] - pts["back_an"][:2])
    stride = (stride_raw / shoulder_w) if shoulder_w > 1e-9 else 0.0

    return {
        "elbow_angle":   round(elbow_angle, 1),
        "front_knee":    round(front_knee,  1),
        "hss":           round(hss,          1),
        "sh_bearing":    round(sh_bearing,   1),
        "trunk_tilt":    round(tilt,         1),
        "stride":        round(stride,       2),
    }


# ─────────────────────────────────────────────
#  ICC COMPLIANCE CHECKER
# ─────────────────────────────────────────────

def icc_compliance_check(
    elbow_at_horizontal: Optional[float],
    elbow_at_release:    Optional[float],
    front_knee:          Optional[float],
    scr:                 Optional[float],
    hss:                 Optional[float],
    stride:              Optional[float],
) -> dict:
    """
    Apply ICC Law 24 and biomechanical thresholds.
    Returns a dict with a verdict per metric and an overall pass/fail.
    """
    results = {}

    # ── ICC Law 24: Elbow Extension Delta ─────────────────
    if elbow_at_horizontal is not None and elbow_at_release is not None:
        # ICC measures HOW MUCH the arm STRAIGHTENS (extension):
        #   release_angle > horizontal_angle  →  arm opened up (positive delta)
        # A positive delta > 15° = ILLEGAL throw per Law 24.
        delta = elbow_at_release - elbow_at_horizontal  # correct: +ve = extension
        legal = delta <= ICC_ELBOW_LIMIT
        results["elbow_extension"] = {
            "delta_deg": round(delta, 1),
            "angle_at_horizontal": round(elbow_at_horizontal, 1),
            "angle_at_release":    round(elbow_at_release, 1),
            "limit": ICC_ELBOW_LIMIT,
            "legal": legal,
            "note":  "LEGAL" if legal else f"ILLEGAL — exceeds {ICC_ELBOW_LIMIT}° limit"
        }
    else:
        results["elbow_extension"] = {"legal": None, "note": "Phase not detected in video"}

    # ── Front Knee ─────────────────────────────────────────
    if front_knee is not None:
        if KNEE_MIN <= front_knee <= KNEE_MAX:
            knee_note = f"Good ({front_knee}°). Target {KNEE_MIN}–{KNEE_MAX}°."
            knee_ok = True
        elif front_knee < KNEE_MIN:
            knee_note = f"Too bent ({front_knee}°). Brace front leg more (>{KNEE_MIN}°)."
            knee_ok = False
        else:
            knee_note = f"Hyper-extended ({front_knee}°). Risk of knee injury; aim <{KNEE_MAX}°."
            knee_ok = False
        results["front_knee"] = {"angle_deg": front_knee, "ok": knee_ok, "note": knee_note}

    # ── Shoulder Counter-Rotation ──────────────────────────
    if scr is not None:
        scr_ok = scr <= SCR_INJURY_LIMIT
        results["shoulder_counter_rotation"] = {
            "angle_deg": round(scr, 1),
            "limit": SCR_INJURY_LIMIT,
            "ok": scr_ok,
            "note": ("Good SCR." if scr_ok
                     else f"High SCR ({scr:.1f}°) — elevated lower-back injury risk.")
        }

    # ── Hip-Shoulder Separation ────────────────────────────
    if hss is not None:
        hss_ok = HSS_MIN <= hss <= HSS_MAX
        results["hip_shoulder_separation"] = {
            "angle_deg": round(hss, 1),
            "ok": hss_ok,
            "note": (f"Good HSS ({hss:.1f}°)." if hss_ok
                     else f"HSS {hss:.1f}° outside optimal {HSS_MIN}–{HSS_MAX}°.")
        }

    # ── Stride Length ──────────────────────────────────────
    if stride is not None:
        stride_ok = STRIDE_MIN <= stride <= STRIDE_MAX
        results["stride_length"] = {
            "ratio": round(stride, 2),
            "ok": stride_ok,
            "note": (f"Good stride ({stride:.2f}×)." if stride_ok
                     else (f"Stride too short ({stride:.2f}×). Aim ≥{STRIDE_MIN}×." if stride < STRIDE_MIN
                           else f"Stride too long ({stride:.2f}×). Aim ≤{STRIDE_MAX}×."))
        }

    # ── Overall Verdict ────────────────────────────────────
    all_ok = all(
        v.get("legal", True) if "legal" in v else v.get("ok", True)
        for v in results.values()
        if v.get("legal") is not None or v.get("ok") is not None
    )
    results["overall"] = "LEGAL & OPTIMAL" if all_ok else "REVIEW NEEDED"
    return results


# ─────────────────────────────────────────────
#  AI COACHING via OpenRouter
# ─────────────────────────────────────────────

def ai_coaching(compliance: dict) -> str:
    """Send the compliance summary to DeepSeek for plain-English coaching tips."""
    if not API_KEY or "XXXXX" in API_KEY:
        return "⚠️  API key not configured. Rule-based feedback only."

    issues = [
        v["note"] for v in compliance.values()
        if isinstance(v, dict) and not v.get("legal", True) and not v.get("ok", True)
    ]
    if not issues:
        issues = ["✅ Action appears legal and biomechanically sound."]

    prompt = (
        "You are an elite ICC-certified cricket biomechanics coach.\n"
        "Based on these flagged issues from a bowler's pose analysis:\n"
        + "\n".join(f"- {i}" for i in issues) +
        "\n\nProvide exactly 3 concise coaching tips (≤ 25 words each), numbered 1-3. "
        "Be direct and actionable."
    )

    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}], "temperature": 0.5},
            timeout=25
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
        return f"⚠️  API error {res.status_code}"
    except Exception as e:
        return f"❌ API call failed: {e}"


# ─────────────────────────────────────────────
#  ON-FRAME OVERLAY RENDERER
# ─────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, angles: dict, phase: str, side: str) -> np.ndarray:
    """Draw a semi-transparent HUD with live metrics on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel background
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 340, 165
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Header
    cv2.putText(frame, f"BowlForm AI | {side}-ARM", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Phase: {phase.upper()}", (panel_x + 10, panel_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)

    # Metrics
    lines = [
        (f"Elbow Angle    : {angles.get('elbow_angle', '--')}°",   (200, 255, 200)),
        (f"Front Knee     : {angles.get('front_knee',  '--')}°",   (200, 230, 255)),
        (f"Hip-Sh Sep     : {angles.get('hss',         '--')}°",   (255, 230, 150)),
        (f"Trunk Tilt     : {angles.get('trunk_tilt',  '--')}°",   (240, 200, 255)),
        (f"Stride (norm)  : {angles.get('stride',      '--')}×",    (255, 200, 200)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (panel_x + 10, panel_y + 65 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze_bowling_video(
    video_path: str,
    output_path: str = "bowlform_output.webm",
    skel_path:   str = "bowlform_skeleton.webm",
    progress_callback=None
) -> dict:
    """
    Generates two output videos:
      1. output_path  — original + skeleton overlay + HUD
      2. skel_path    — black background stickfigure only
    Also produces per-frame metrics dict for live frontend sync.
    """
    if not os.path.exists(video_path):
        print(f"❌  Video not found: {video_path}")
        return {"error": f"Video not found: {video_path}"}

    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  Cannot open video file: {video_path}")
        return {"error": f"Cannot open video file: {video_path}"}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_w == 0 or orig_h == 0:
        return {"error": "Invalid video dimensions or corrupted file."}

    scale  = min(1.0, MAX_WIDTH / orig_w)
    proc_w = int(orig_w * scale)
    proc_h = int(orig_h * scale)

    fourcc      = cv2.VideoWriter_fourcc(*"vp80")
    writer_ann  = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    writer_skel = cv2.VideoWriter(skel_path,   fourcc, fps, (orig_w, orig_h))

    pose_model = make_pose()
    smoother   = AngleSmoother(SMOOTH_WIN)
    phase_det  = PhaseDetector()

    # ── Phase 1: arm detection ────────────────────────────────────
    arm_detection_lms = []
    side      = "RIGHT"
    frame_idx = 0

    if progress_callback:
        progress_callback(2)
    print("🔍 Detecting bowling arm...")
    while cap.isOpened() and len(arm_detection_lms) < 45:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        small = cv2.resize(frame, (proc_w, proc_h))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res   = pose_model.process(rgb)
        if res.pose_landmarks:
            arm_detection_lms.append(res.pose_landmarks.landmark)

    if arm_detection_lms:
        side = detect_bowling_arm(arm_detection_lms)
    print(f"✅ Bowling arm detected: {side}")

    # ── Phase 2: full analysis pass ─────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
    frame_idx = 0

    # Store phase-snapshot angles
    angles_at_horizontal: Optional[dict] = None
    angles_at_horizontal: Optional[dict] = None
    angles_at_release:    Optional[dict] = None
    current_angles: dict = {}
    current_phase:  str  = "idle"
    frame_metrics:  dict = {}   # {str(frame_idx): {angles + phase}}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if progress_callback:
        progress_callback(10)
    print("⚙️  Running full pose analysis...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Black canvas for skeleton-only video
        skel_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # Inference
        small = cv2.resize(frame, (proc_w, proc_h))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res   = pose_model.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # Draw on annotated video (original + skeleton)
            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(50, 220, 170), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(100, 200, 255), thickness=2))

            # Draw on skeleton-only video (black background)
            mp_drawing.draw_landmarks(
                skel_frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(50, 230, 180), thickness=2, circle_radius=5),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(120, 210, 255), thickness=3))

            pts = pick_side_landmarks(lm, side)
            if pts is not None:
                raw = compute_angles(pts)
                for k in raw:
                    raw[k] = smoother.smooth(k, raw[k])
                current_angles = {k: round(v, 1) for k, v in raw.items()}

                bowl_wr_y   = pts["bowl_wr"][1]
                bowl_sh_y   = pts["bowl_sh"][1]
                current_phase = phase_det.update(bowl_wr_y, bowl_sh_y, frame_idx)

                if current_phase == "arm_horizontal" and angles_at_horizontal is None:
                    angles_at_horizontal = dict(current_angles)
                    print(f"  📐 Arm-horizontal @ frame {frame_idx}: elbow={current_angles.get('elbow_angle')}°")
                if current_phase == "release" and angles_at_release is None:
                    angles_at_release = dict(current_angles)
                    print(f"  🎯 Release @ frame {frame_idx}: elbow={current_angles.get('elbow_angle')}°")

            # HUD overlay on both videos
            frame      = draw_overlay(frame,      current_angles, current_phase, side)
            skel_frame = draw_overlay(skel_frame, current_angles, current_phase, side)

        # Store this frame's metrics for live frontend sync
        if current_angles:
            frame_metrics[str(frame_idx)] = dict(current_angles)
            frame_metrics[str(frame_idx)]["phase"] = current_phase

        writer_ann.write(frame)
        writer_skel.write(skel_frame)

        if progress_callback and total_frames > 1:
            pct = 10 + int((frame_idx / total_frames) * 79)
            progress_callback(min(pct, 89))

    cap.release()
    writer_ann.release()
    writer_skel.release()
    pose_model.close()

    # ── SCR ──────────────────────────────────────────────────────
    scr = None
    if angles_at_horizontal and angles_at_release:
        scr = abs(angles_at_horizontal["sh_bearing"] - angles_at_release["sh_bearing"])

    # ── ICC Compliance ────────────────────────────────────────────
    # Make sure we don't crash if phase detection failed and we never got a release frame
    elbow_horiz   = angles_at_horizontal.get("elbow_angle") if angles_at_horizontal else None
    elbow_release = angles_at_release.get("elbow_angle")    if angles_at_release    else None

    fk     = angles_at_release.get("front_knee") if angles_at_release else None
    hss    = angles_at_release.get("hss")        if angles_at_release else None
    stride = angles_at_release.get("stride")     if angles_at_release else None

    compliance = icc_compliance_check(elbow_horiz, elbow_release, fk, scr, hss, stride)

    # ── AI Feedback ───────────────────────────────────────────────
    if progress_callback:
        progress_callback(91)
    print("🤖 Requesting AI coaching feedback...")
    coaching = ai_coaching(compliance)
    if progress_callback:
        progress_callback(99)

    # ── Summary ───────────────────────────────────────────────────
    summary = {
        "bowling_arm":   side,
        "fps":           fps,
        "total_frames":  total_frames,
        "phase_data": {
            "arm_horizontal": angles_at_horizontal,
            "release":        angles_at_release,
        },
        "icc_compliance":   compliance,
        "ai_coaching_tips": coaching,
        "output_video":     output_path,
        "skeleton_video":   skel_path,
        "frame_metrics":    frame_metrics,
    }

    printable = {k: v for k, v in summary.items() if k != "frame_metrics"}
    print("\n" + "=" * 60)
    print(json.dumps(printable, indent=2))
    print(f"frame_metrics: {len(frame_metrics)} frames")
    print(f"✅ Annotated  → {output_path}")
    print(f"✅ Skeleton   → {skel_path}")
    return summary


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    VIDEO_PATH  = "your_video.mp4"
    OUTPUT_PATH = "bowlform_analysis.mp4"
    SKEL_PATH   = "bowlform_skeleton.mp4"
    result = analyze_bowling_video(VIDEO_PATH, OUTPUT_PATH, SKEL_PATH)

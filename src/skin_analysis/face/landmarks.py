"""MediaPipe Face Landmarker wrapper (468 surface + 10 iris points).

MediaPipe is Apache-2.0 -- code AND the three bundled models (BlazeFace short-range,
FaceMesh V2, Blendshape V2), per their official model cards. It is one of the few face
components that clears this project's licensing gate (D3).

Two things are pinned and verified at load time, not assumed:

* **The bundle version and its sha256.** Landmark index semantics must not drift between
  releases. Every ROI polygon in ``config/rois.yaml`` is a list of indices into this exact
  model; a silent bundle swap would move every ROI without changing a line of code, and
  nothing downstream would look wrong.
* **Blendshapes and transformation matrices are off.** They are licensed but unused (D3).
  Computing them costs latency this pipeline has budgeted elsewhere (D10).

The landmarker object is expensive to construct and is NOT thread-safe, so it is cached
per (model path, options) and callers must not share one across threads. This module is
therefore not pure -- it is infrastructure, not a feature module.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Pinned artifact. Must match LICENSES/asset_manifest.csv.
MODEL_FILENAME = "face_landmarker.task"
MODEL_VARIANT = "float16/1"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

#: 468 surface vertices + 10 iris points. The iris points are what make the scale anchor
#: possible (see util/scale.py), so the 468-point variant is not a substitute.
EXPECTED_LANDMARK_COUNT = 478

#: Environment override for deployments that stage the bundle outside the repo.
MODEL_PATH_ENV = "SKIN_ANALYSIS_LANDMARKER_PATH"

_LOCK = threading.Lock()


class ModelIntegrityError(RuntimeError):
    """Raised when the landmarker bundle is missing or is not the audited artifact.

    Fails closed. A bundle whose hash does not match the manifest has not been through
    the licensing audit, and its landmark indices are not known to mean what
    ``config/rois.yaml`` says they mean.
    """


@dataclass(frozen=True)
class FaceGeometry:
    """Landmarks plus the pose that was derived from them.

    Bundled together because every caller that wants one wants the other, and deriving
    pose twice from the same landmarks is both wasteful and a chance to disagree.
    """

    landmarks: np.ndarray          # (478, 3) pixel coordinates
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def model_path() -> Path:
    """Resolve the landmarker bundle path, honouring the environment override."""
    override = os.environ.get(MODEL_PATH_ENV)
    return Path(override) if override else REPO_ROOT / "models" / MODEL_FILENAME


def verify_model(path: Path | None = None) -> str:
    """Verify the bundle exists and matches the pinned hash. Returns its sha256.

    Raises:
        ModelIntegrityError: if the file is missing or the hash differs.
    """
    path = path or model_path()
    if not path.exists():
        raise ModelIntegrityError(
            f"Face Landmarker bundle not found at {path}. Download the pinned artifact:\n"
            f"    {MODEL_URL}\n"
            f"Expected sha256 {MODEL_SHA256}. The bundle is gitignored on purpose -- "
            "weights are never vendored into this repository."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ModelIntegrityError(
            f"Face Landmarker bundle at {path} has sha256 {digest}, expected {MODEL_SHA256} "
            f"({MODEL_VARIANT}). This artifact has not been through the licensing audit, and "
            "its landmark indices are not known to match config/rois.yaml. Refusing to load."
        )
    return digest


@lru_cache(maxsize=4)
def _landmarker(num_faces: int):  # type: ignore[no-untyped-def]
    """Construct and cache the landmarker. Expensive; not thread-safe (see module docs).

    Cached on ``num_faces`` so a config change produces a new instance rather than
    silently reusing one built for a different limit.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    path = model_path()
    verify_model(path)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE,
        # One more than the QC limit: we must SEE the extra face to reject the capture.
        # Requesting exactly max_faces would make a two-person photo indistinguishable
        # from a one-person photo, and the second person would be silently cropped out.
        num_faces=num_faces,
        # D3: licensed but unused. Also saves latency against the D10 budget.
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options), mp


def detect(image: np.ndarray, config: dict) -> np.ndarray | None:
    """Return an (N, 3) array of landmark coordinates, or None if no usable face.

    Coordinates are in pixel units in the original image frame -- do not return
    normalized coordinates, since downstream ROI erosion is specified in pixels.

    Args:
        image: BGR uint8, as decoded. Not resized: the canonical QC crop (D11) is a QC
            concern, and ROI geometry must map back to original sensor pixels.
        config: the resolved capture profile.

    Returns:
        (478, 3) float64 array, or None when there is not exactly one face. Multiple faces
        are NOT silently resolved here -- ``detect_faces`` exposes the count so the QC gate
        can reject the capture. Picking "the biggest face" would analyze a stranger
        standing behind the user without telling anyone.
    """
    faces = detect_faces(image, config)
    if len(faces) != 1:
        return None
    return faces[0]


def detect_faces(image: np.ndarray, config: dict) -> list[np.ndarray]:
    """Return one (478, 3) pixel-coordinate array per detected face.

    The count matters to the caller: ``max_faces`` is a QC failure condition, not
    something this layer should quietly resolve.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")

    max_faces = int(config.get("face", {}).get("max_faces", 1))
    landmarker, mp = _landmarker(max_faces + 1)
    height, width = image.shape[:2]

    # MediaPipe expects RGB. Convert by slicing rather than via cv2 so this function has
    # no dependency on the OpenCV build being present.
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with _LOCK:
        result = landmarker.detect(mp_image)

    faces: list[np.ndarray] = []
    for face in result.face_landmarks:
        points = np.array(
            [(lm.x * width, lm.y * height, lm.z * width) for lm in face],
            dtype=np.float64,
        )
        if points.shape[0] != EXPECTED_LANDMARK_COUNT:
            raise ModelIntegrityError(
                f"landmarker returned {points.shape[0]} points, expected "
                f"{EXPECTED_LANDMARK_COUNT}. Every ROI polygon and the scale anchor index "
                "into the 478-point model; a different topology silently invalidates both."
            )
        faces.append(points)
    return faces


# ---------------------------------------------------------------------------- head pose

#: Canonical 3D model points, in the same landmark indices MediaPipe emits. Chosen to be
#: rigid: nose tip, chin, outer eye corners, mouth corners. Cheek and brow points move
#: with expression, which would turn a smile into a pose failure.
_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # 1   nose tip
        (0.0, -63.6, -12.5),      # 152 chin
        (-43.3, 32.7, -26.0),     # 33  left eye outer corner
        (43.3, 32.7, -26.0),      # 263 right eye outer corner
        (-28.9, -28.9, -24.1),    # 61  left mouth corner
        (28.9, -28.9, -24.1),     # 291 right mouth corner
    ],
    dtype=np.float64,
)


def head_pose(landmarks: np.ndarray, image_shape: tuple[int, int] | None = None) -> tuple[
    float, float, float
]:
    """Estimate (yaw, pitch, roll) in degrees for the pose gate.

    Solved with a PnP fit against a canonical rigid model rather than read off landmark
    ratios, because ratio heuristics conflate pose with face shape -- and this gate exists
    to keep ROI polygons corresponding to the same anatomy across scans (repeatability),
    not to describe anyone's face.

    Args:
        landmarks: (478, 3) pixel coordinates from :func:`detect`.
        image_shape: (height, width). Defaults to a frame inferred from the landmarks,
            which is adequate because only the ratio to focal length matters here.

    Returns:
        (yaw, pitch, roll) in degrees. Zero is facing the camera squarely.
    """
    import cv2

    if landmarks.shape[0] != EXPECTED_LANDMARK_COUNT:
        raise ValueError(
            f"head_pose needs the {EXPECTED_LANDMARK_COUNT}-point model, "
            f"got {landmarks.shape[0]}"
        )

    if image_shape is None:
        height = float(landmarks[:, 1].max() - landmarks[:, 1].min()) * 2.0
        width = float(landmarks[:, 0].max() - landmarks[:, 0].min()) * 2.0
    else:
        height, width = float(image_shape[0]), float(image_shape[1])

    image_points = np.ascontiguousarray(landmarks[list(_POSE_LANDMARKS), :2])

    # Pinhole approximation: focal length ~ image width. Selfie cameras vary, but the gate
    # is a coarse +/-15 degree threshold, not a measurement, and a per-device intrinsic
    # would have to be calibrated per phone to buy anything.
    focal = width
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    ok, rotation, _ = cv2.solvePnP(
        _POSE_MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),  # assume no lens distortion; phone JPEGs are already corrected
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("head pose solve failed; treat the capture as EXTREME_POSE")

    matrix, _ = cv2.Rodrigues(rotation)
    sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:  # gimbal lock; roll is undefined, report it as zero rather than as noise
        pitch = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = 0.0

    pitch_deg = float(np.degrees(pitch))
    # solvePnP returns pitch near +/-180 for a forward-facing head because the model's
    # y axis points down. Wrap to the signed offset from facing the camera.
    if pitch_deg > 90.0:
        pitch_deg -= 180.0
    elif pitch_deg < -90.0:
        pitch_deg += 180.0

    return float(np.degrees(yaw)), pitch_deg, float(np.degrees(roll))


def geometry(image: np.ndarray, config: dict) -> FaceGeometry | None:
    """Landmarks and pose in one pass. Returns None when there is not exactly one face."""
    points = detect(image, config)
    if points is None:
        return None
    yaw, pitch, roll = head_pose(points, image.shape[:2])
    return FaceGeometry(landmarks=points, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)

"""The one function that touches OD. See docs/argus_localization_spec.md section 8.

Everything upstream of this function stays fixed across retriever and matcher
swaps. Only this adapter changes at integration time.

Target format confirmed by reading FSW-Payload's actual Ceres-based batch
optimizer (include/navigation/{measurement_residuals,batch_optimization}.hpp,
not guessed): per landmark measurement row, `[timestamp, bearing unit vector
(3, BODY frame), landmark position (3, ECI)]`, grouped by source frame via a
parallel group-start boolean array. Neither lat/lon nor ECEF -- pixel needs a
camera model to become a body-frame bearing, and the tile's lat/lon needs a
time-dependent Earth-orientation rotation to become ECI, not a relabel.

Known limitation: this reaches into GNC-Payload (a sibling flight-software
repo at an absolute path on this machine) for `brahe`, the camera model, and
`lat_lon_to_ecef`, the same way retrievers/earthloc_retriever.py reaches into
third_party/EarthLoc -- except GNC-Payload isn't vendored under third_party/
here, so this only works on machines where it happens to be checked out at
GNC_PAYLOAD_ROOT. Flagging rather than hiding it: fine for this repo's
current single-machine research use, not fine to ship as-is.
"""

import os
import sys

import numpy as np

from core.types import LocalizationResult

GNC_PAYLOAD_ROOT = "/home/pvijayba/GNC-Payload"


def _load_gnc_payload_deps():
    added = GNC_PAYLOAD_ROOT not in sys.path
    if added:
        sys.path.insert(0, GNC_PAYLOAD_ROOT)
    try:
        import brahe
        from brahe.epoch import Epoch
        from sensors.camera_model import CameraModel
        from utils.earth_utils import lat_lon_to_ecef
    finally:
        if added:
            sys.path.remove(GNC_PAYLOAD_ROOT)
    return brahe, Epoch, CameraModel, lat_lon_to_ecef


# LandmarkMeasurementIdx column layout, matching
# FSW-Payload/include/navigation/measurement_residuals.hpp exactly.
LANDMARK_TIMESTAMP = 0
BEARING_VEC = slice(1, 4)
LANDMARK_POS = slice(4, 7)
LANDMARK_COUNT = 7


def to_batchopt_measurements(
    result: LocalizationResult,
    epoch,  # brahe.epoch.Epoch -- the query frame's capture time
    camera_model,  # sensors.camera_model.CameraModel -- which camera captured the frame
    query_image_shape: tuple,  # (height, width) of the frame actually passed to the pipeline
) -> tuple[np.ndarray, np.ndarray]:
    """Converts one LocalizationResult's tie_points into a batch-opt-ready
    (landmark_measurements, group_starts) pair. All tie_points come from the
    same frame, so group_starts marks only the first row True (this frame is
    one group); a caller assembling a multi-frame batch should np.vstack /
    np.concatenate these across frames and only vstack, never re-derive
    group_starts by hand for each chunk.

    Returns (np.zeros((0, LANDMARK_COUNT)), np.zeros((0,), dtype=bool)) for a
    "no_fix" result -- an empty, valid, zero-row contribution to a batch.
    """
    if result.status != "fix" or not result.tie_points:
        return np.zeros((0, LANDMARK_COUNT)), np.zeros((0,), dtype=bool)

    brahe, _, _, lat_lon_to_ecef = _load_gnc_payload_deps()

    n = len(result.tie_points)
    height, width = query_image_shape
    u = np.array([tp.u for tp in result.tie_points])
    v = np.array([tp.v for tp in result.tie_points])
    lat = np.array([tp.lat for tp in result.tie_points])
    lon = np.array([tp.lon for tp in result.tie_points])

    # Rescale from the pipeline's own query pixel space into CameraModel's
    # fixed (IMAGE_WIDTH, IMAGE_HEIGHT) pinhole frame -- see od_integration_test.py's
    # module docstring for the same caveat: correct mechanism, approximate
    # unless query_image_shape actually came from Argus's real camera.
    u_full = u * (camera_model.IMAGE_WIDTH / width)
    v_full = v * (camera_model.IMAGE_HEIGHT / height)
    bearing_body = camera_model.pixel_to_bearing_unit_vector(np.column_stack([u_full, v_full]))

    lat_lon = np.column_stack([lat, lon])
    ecef = lat_lon_to_ecef(lat_lon)  # (N,3) meters
    ecef_R_eci = brahe.frames.rECItoECEF(epoch)
    landmark_eci = (ecef_R_eci.T @ ecef.T).T  # (N,3) meters

    timestamp = epoch.jd()  # Julian date; caller's OD solver just needs a monotonic, consistent time base

    measurements = np.zeros((n, LANDMARK_COUNT))
    measurements[:, LANDMARK_TIMESTAMP] = timestamp
    measurements[:, BEARING_VEC] = bearing_body
    measurements[:, LANDMARK_POS] = landmark_eci

    group_starts = np.zeros(n, dtype=bool)
    group_starts[0] = True

    return measurements, group_starts

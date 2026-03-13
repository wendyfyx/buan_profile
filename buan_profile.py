#!/usr/bin/env python
"""
buan_profile.py

Computes a distance-weighted mean scalar profile along a bundle using the BUAN weighted-mean method.
Outputs a CSV with one row per along-tract segment. Multiple scalar maps can be processed in a single 
call, each writing its own output CSV, with a single optional mask applied to all maps.

REQUIRED PACKAGES
-----------
numpy, scipy, nibabel, dipy

PIPELINE STEPS
--------------
Step 1 — Load and resample bundle pairs. Three bundle files are required per subject: --rec_bundle 
(subject bundle registered to atlas/MNI space), --org_bundle (same bundle in native space), and 
--ref_bundle (the atlas reference bundle). Before any spatial computation, org_bundle and rec_bundle
are resampled via prepare_bundle_pairs() so that each corresponding streamline has the same number of 
evenly-spaced points. rec_bundle and org_bundle must have the same number of streamlines in matching order.

Step 2 — Compute the atlas centroid. compute_centroid() runs QuickBundles on ref_bundle to obtain an 
initial centroid, then optionally extends it beyond its natural endpoints via --robust_method. The 
standard QB centroid represents the mean trajectory of the bundle, and may cause terminal segments to span 
a disproportionately large anatomical extent. The robust linear extension (default) projects the terminal 
tangent vectors outward and trims to the 2nd-98th percentile of streamline endpoint projections. The spline
method fits a cubic parametric curve through the QB centroid and evaluates it outside [0, 1], less stable 
for short or strongly curved endpoints.

Step 3 — Assign segments. assign_segments() queries every point in rec_bundle against the centroid via a 
KD-tree, returning a segment index and a Euclidean distance for each point. Distances are later used as 
inverse weights in the BUAN mean. In segment-length mode (--s_len), terminal bins whose point count falls 
below 0.2 x median are merged into their neighbor, preventing near-empty edge segments from producing 
unreliable means. When --ns is set, no merging is performed and the output always has exactly ns rows.

Step 4 — Interpolate scalar values and apply mask. buan_profile() calls values_from_volume to trilinearly 
interpolate the scalar map at all org_bundle point coordinates. If --mask is provided, masking is applied 
after interpolation: each point's world coordinate is mapped to the nearest voxel in the mask volume and 
points outside the mask are excluded. Post-interpolation masking avoids the boundary artefacts that arise 
when zeros or NaNs are inserted into the scalar map before interpolation. A single mask is loaded once and 
applied to all scalar maps in the same call.

Step 5 — Compute per-segment mean and coverage volume. For each segment, valid (unmasked, non-NaN) points 
are used to compute the BUAN weighted mean, where each point's scalar value is weighted by the inverse of 
its distance to the nearest centroid point. point_volume() computes the convex hull of the valid points, 
and returns a volume estimate per segment (mm^3). Each scalar map writes an CSV with these columns: 
segment (1-based), mean, valid_point_count, volume_mm3. 

NOTES ON VALIDITY CHECKS
--------------
Three categories of checks are applied during profiling. 
1. Bundles with fewer than --min_lines streamlines (default 20) are skipped entirely.
2. Dring atlas parameterization, terminal segments whose point count falls below 0.2 x median are merged 
into their neighbor to prevent near-empty edge segments from producing unreliable means. This applies
only in --s_len mode and is suppressed when --ns is set. 
3. During profiling, individual streamline points are excluded if their interpolated scalar value is NaN, 
if they fall outside the --mask volume, or if they fall in a masked (zero) voxel.
4. When profiles are used in group analysis, we recommend users to employ additional group level validity
check to avoid making inferences in segments with poor coverage across subjects:
    1) thresholding the number of valid point per segment;
    2) thresholding the number of subjects with sufficients points or valid value per segment.

USAGE NOTES
-----------
1. SEGMENT COUNT vs SEGMENT LENGTH
   --ns overrides --s_len. When --ns is set the output always has exactly ns rows. When only --s_len is given,
   ns is derived from the atlas bundle arc length and small terminal bins may be merged.

2. MULTIPLE SCALAR MAPS
   Pass multiple maps and outputs in matching order:
     --scalar_maps fa.nii.gz md.nii.gz --outputs out_fa.csv out_md.csv
   Centroid and segment assignment are computed once and reused for all maps.

EXAMPLE
-------
  python tpa_profile_1d.py \\
      --rec_bundle sub01_AF_L_mni.trk \\
      --org_bundle sub01_AF_L_native.trk \\
      --ref_bundle atlas_AF_L.trk \\
      --scalar_maps sub01_fa.nii.gz sub01_md.nii.gz \\
      --outputs sub01_AF_L_fa.csv sub01_AF_L_md.csv \\
      --mask sub01_wm_mask.nii.gz \\
      --s_len 5.0 \\
      --robust_method linear
"""

import argparse
import csv
import gzip
import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.io as sio
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, QhullError

from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import length
from dipy.tracking.streamline import (
    set_number_of_points,
    Streamlines,
    transform_streamlines,
    values_from_volume,
)
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles


# ---------------------------------------------------------------------------
# Streamline I/O  
# adapted from MeTA, import it from MeTA if needed
# ---------------------------------------------------------------------------

def parse_tt(tinytrack: str):
    """
    Parse DSI-Studio TinyTrack format (.tt.gz) to extract streamlines and metadata.

    Parameters:
        tinytrack: Path to the TinyTrack file.

    Returns:
        streamlines: List of streamlines in LPS space.
        tt_affine: Affine transformation matrix.
        dimension: Dimensions of image.
        voxel_size: Voxel size of image.
        voxel_order: Voxel order of image.
    """

    if tinytrack.endswith('.tt.gz'):
        with gzip.open(tinytrack, 'rb') as f:
            data = f.read()
        mat = sio.loadmat(io.BytesIO(data), appendmat=False, spmatrix=False)
    tt_affine = mat['trans_to_mni'].reshape(4,4)
    dimension = tuple(mat['dimension'].ravel().astype(int))
    voxel_size = tuple(mat['voxel_size'].ravel().astype(float))
    voxel_order = "".join(nib.aff2axcodes(tt_affine))

    buf1 = mat.get('track').flatten()
    buf2 = buf1.view(np.int8)
    length = len(buf1)
    pos = []
    i = 0
    while i < length:
        pos.append(i)
        track_length = np.frombuffer(buf1[i:i+4].tobytes(), dtype=np.uint32)[0]
        i += int(track_length) + 13

    streamlines = []
    for p in pos:
        size_val = np.frombuffer(buf1[p:p+4].tobytes(), dtype=np.uint32)[0]
        num_points = int(size_val // 3)
        x = np.frombuffer(buf1[p+4:p+8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[p+8:p+12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[p+12:p+16].tobytes(), dtype=np.int32)[0]

        track_pts = np.empty((num_points, 3), dtype=np.float32)
        track_pts[0, :] = [x, y, z]
        p_offset = p + 16

        for j in range(1, num_points):
            dx = int(buf2[p_offset])
            dy = int(buf2[p_offset + 1])
            dz = int(buf2[p_offset + 2])
            x += dx; y += dy; z += dz
            track_pts[j, :] = [x, y, z]
            p_offset += 3
        track_pts /= 32.0
        streamlines.append(track_pts)

    logging.debug(f"Parsed {len(streamlines)} streamlines in total")
    logging.debug(f"Affine: \n {tt_affine},\n Dimension: {dimension},\n Voxel Size: {voxel_size},\n Voxel Order: {voxel_order}")
    return streamlines, tt_affine, dimension, voxel_size, voxel_order


def read_streamlines(
    bundle_path: str,
    reference=None,
    transform=None,
) -> Tuple[ArraySequence, dict, np.ndarray, Optional[tuple]]:
    """
    Read streamlines from .trk, .tck, or .tt.gz files

    Parameters
    ----------
    bundle_path : str
        Path to the bundle file.
    reference : str or nibabel image, optional
        Spatial reference for .tck files (passed to load_tractogram).
        Pass 'same' for .trk files (handled automatically).
    transform : np.ndarray, optional
        Optional (4, 4) affine applied after loading.

    Returns
    -------
    streamlines : ArraySequence
        Streamlines in RASMM space.
    groups : dict
    affine : np.ndarray, shape (4, 4)
    dimension : tuple of int or None
    """
    if bundle_path.endswith('.tck'):
        bundle = load_tractogram(bundle_path, reference, bbox_valid_check=False)
        streamlines = bundle.streamlines
        if transform is not None:
            streamlines = transform_streamlines(streamlines, transform)
        groups    = bundle.groups     if hasattr(bundle, 'groups')      else {}
        affine    = bundle.affine     if hasattr(bundle, 'affine')      else np.eye(4)
        dimension = bundle.dimensions if hasattr(bundle, 'dimensions')  else None
        return streamlines, groups, affine, dimension

    if bundle_path.endswith(('.trk', '.trx')):
        bundle      = load_tractogram(bundle_path, 'same', bbox_valid_check=False)
        streamlines = bundle.streamlines
        if transform is not None:
            streamlines = transform_streamlines(streamlines, transform)
        groups    = bundle.groups     if hasattr(bundle, 'groups')      else {}
        affine    = bundle.affine     if hasattr(bundle, 'affine')      else np.eye(4)
        dimension = bundle.dimensions if hasattr(bundle, 'dimensions')  else None
        return streamlines, groups, affine, dimension

    if bundle_path.endswith('.tt.gz'):
        streamlines_raw, tt_affine, dimension, _, _ = parse_tt(bundle_path)
        streamlines_lps = [s - 0.5 for s in streamlines_raw]
        streamlines     = transform_streamlines(streamlines_lps, tt_affine)
        if transform is not None:
            streamlines = transform_streamlines(streamlines, transform)
        return ArraySequence(streamlines), {}, tt_affine, dimension

    raise ValueError(
        f"Unsupported bundle format: '{bundle_path}'. "
        "Expected .trk, .tck, or .tt.gz"
    )

# ---------------------------------------------------------------------------
# Centroid utilities
# ---------------------------------------------------------------------------

def _get_initial_centroid(bundle, n_points: int = 100, thresh: float = 100.0) -> np.ndarray:
    """
    Compute a QuickBundles centroid for the atlas bundle.

    Parameters
    ----------
    bundle : ArraySequence
        Atlas bundle streamlines.
    n_points : int
        Number of points to resample to before clustering.
    thresh : float
        QuickBundles distance threshold (mm).

    Returns
    -------
    centroid : np.ndarray, shape (n_points, 3)
        The longest cluster centroid.
    """
    if isinstance(bundle, np.ndarray):
        bundle = ArraySequence(bundle[:, :, :3])
    resampled = set_number_of_points(bundle, nb_points=n_points)
    qb = QuickBundles(threshold=thresh, metric=AveragePointwiseEuclideanMetric())
    clusters  = qb.cluster(resampled)
    centroids = Streamlines(clusters.centroids)
    lens = np.array(list(length(centroids)))
    return np.array(centroids[np.argmax(lens)])

def compute_centroid(
    bundle,
    ns: Optional[int] = None,
    s_len: float = 5.0,
    robust_method: str = 'linear',
    thresh: float = 100.0,
    extrapolate_prop: float = 0.3,
) -> Tuple[np.ndarray, int, float]:
    """
    Compute the atlas bundle centroid with optional robust endpoint extension.

    When robust_method is 'linear' or 'spline', the QB centroid is extended
    beyond its natural endpoints to cover the full arc-length extent of the
    bundle's streamline endpoints (2nd-98th percentile projection). This
    avoids large terminal segments caused by the QB centroid being shorter
    than the longest streamlines in the bundle.

    Parameters
    ----------
    bundle : ArraySequence
        Atlas/reference bundle.
    ns : int, optional
        Target number of segments. When set, the centroid is always resampled
        to exactly ns points; s_len is derived from the resulting arc length.
        When None, ns is derived from mean bundle arc length / s_len.
    s_len : float
        Target segment length (mm). Used only when ns is None. Default 5.0.
    robust_method : str
        'none'   -- standard QB centroid resampled to ns points.
        'linear' -- extrapolate from terminal tangent vectors (default).
        'spline' -- fit cubic spline and evaluate beyond [0, 1].
    thresh : float
        QuickBundles clustering threshold (mm). Default 100.
    extrapolate_prop : float
        Proportion of QB centroid arc length to extend on each side.
        Default 0.3.

    Returns
    -------
    centroid : np.ndarray, shape (ns_actual, 3)
        Evenly arc-length-spaced centroid coordinates.
    ns_actual : int
        Actual number of segments (equals ns when ns is provided).
    s_len_actual : float
        Derived segment length (arc_length / (ns - 1)).
    """

    def _resample_centroid(initial: np.ndarray, ns: int) -> Tuple[np.ndarray, int, float]:
        """
        Resample a centroid curve to ns evenly arc-length-spaced points. Used as the fallback
        """
        centroid = np.array(set_number_of_points(Streamlines([initial]), nb_points=ns)[0])
        arc = float(np.linalg.norm(np.diff(centroid, axis=0), axis=1).sum())
        return centroid, ns, arc / max(ns - 1, 1)

    initial = _get_initial_centroid(bundle, n_points=100, thresh=thresh)

    if ns is None:
        avg_length = float(np.mean(list(length(bundle))))
        ns = max(int(np.round(avg_length / s_len)), 1)

    # Standard (non-robust) path
    if robust_method == 'none':
        return _resample_centroid(initial, ns)

    # Robust path: build extended curve
    centroid_arc = float(np.linalg.norm(np.diff(initial, axis=0), axis=1).sum())
    ext_mm = extrapolate_prop * centroid_arc
    tck = None   # only populated for spline path

    if robust_method == 'spline':
        if len(initial) < 4:
            logging.warning(
                "Initial centroid has fewer than 4 points; "
                "cannot fit cubic spline. Falling back to 'linear'."
            )
            robust_method = 'linear'
        else:
            tck, _ = splprep(initial.T, s=0, k=3)
            deriv_start = np.linalg.norm(np.array(splev(0.0, tck, der=1)))
            deriv_end = np.linalg.norm(np.array(splev(1.0, tck, der=1)))
            t_ext_start = ext_mm / deriv_start if deriv_start > 0 else 0.1
            t_ext_end = ext_mm / deriv_end   if deriv_end   > 0 else 0.1
            t_dense = np.linspace(0.0 - t_ext_start, 1.0 + t_ext_end, 1000)
            ext_pts = np.array(splev(t_dense, tck)).T   # (1000, 3)

    if robust_method == 'linear':
        tang_start = initial[0] - initial[1]
        tang_end = initial[-1] - initial[-2]
        tang_start /= np.linalg.norm(tang_start)
        tang_end /= np.linalg.norm(tang_end)
        n_ext = max(1, int(np.round(ext_mm / s_len)))
        steps = np.arange(1, n_ext + 1)[:, np.newaxis]
        ext_start = initial[0]  + steps * tang_start * s_len
        ext_end = initial[-1] + steps * tang_end   * s_len
        ext_pts = np.vstack([ext_start[::-1], initial, ext_end])

    # Project streamline endpoints onto the extended curve via KD-tree
    tree = cKDTree(ext_pts)
    i_starts, i_ends = [], []
    for sl in bundle:
        sl = np.array(sl)
        if len(sl) < 2:
            continue
        _, i_s = tree.query(sl[0])
        _, i_e = tree.query(sl[-1])
        i_starts.append(i_s)
        i_ends.append(i_e)

    if not i_starts:
        logging.warning("No valid endpoints found; falling back to standard centroid.")
        return _resample_centroid(initial, ns)

    i_min = np.minimum(np.array(i_starts), np.array(i_ends))
    i_max = np.maximum(np.array(i_starts), np.array(i_ends))
    i_start = int(np.round(np.percentile(i_min, 2.0)))
    i_end = int(np.round(np.percentile(i_max, 98.0)))

    if i_end <= i_start:
        logging.warning(
            f"Degenerate endpoint range (i_start={i_start} >= i_end={i_end}); "
            "falling back to standard centroid."
        )
        return _resample_centroid(initial, ns)

    ext_range = ext_pts[i_start:i_end + 1]
    seg_lens  = np.linalg.norm(np.diff(ext_range, axis=0), axis=1)
    arc_length = float(seg_lens.sum())
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    target_lens = np.linspace(0.0, arc_length, ns)

    if robust_method == 'spline' and tck is not None:
        t_range = t_dense[i_start:i_end + 1]
        t_resampled = np.interp(target_lens, cum_lens, t_range)
        centroid = np.array(splev(t_resampled, tck)).T
    else:
        centroid = np.array([
            np.interp(target_lens, cum_lens, ext_range[:, dim])
            for dim in range(3)
        ]).T

    s_len_actual = arc_length / max(ns - 1, 1)
    logging.info(
        f"compute_centroid ({robust_method}): ns={ns}, "
        f"arc_length={arc_length:.1f}mm, s_len={s_len_actual:.2f}mm"
    )
    return centroid, ns, s_len_actual


# ---------------------------------------------------------------------------
# Other utilities
# ---------------------------------------------------------------------------

def prepare_bundle_pairs(
    org_bundle: ArraySequence,
    rec_bundle: ArraySequence,
) -> Tuple[ArraySequence, ArraySequence]:
    """
    Resample org_bundle and rec_bundle so that each corresponding streamline
    has the same number of evenly-distributed points.

    Both bundles must have the same number of streamlines in matching order
    (i.e. the output of bundle registration). Point counts are taken from
    org_bundle; rec_bundle is resampled to match, preserving the one-to-one
    correspondence needed for segment assignment and scalar sampling.

    Parameters
    ----------
    org_bundle : ArraySequence
        Subject bundle in native space.
    rec_bundle : ArraySequence
        Subject bundle in atlas/MNI space.

    Returns
    -------
    org_bundle_rsp : ArraySequence
    rec_bundle_rsp : ArraySequence
    """
    if len(org_bundle) != len(rec_bundle):
        raise ValueError(
            f"org_bundle ({len(org_bundle)} streamlines) and "
            f"rec_bundle ({len(rec_bundle)} streamlines) must have the same count."
        )
    n_pts = [len(sl) for sl in org_bundle]
    org_rsp = ArraySequence([
        set_number_of_points(org_bundle[i], nb_points=n_pts[i])
        for i in range(len(org_bundle))
    ])
    rec_rsp = ArraySequence([
        set_number_of_points(rec_bundle[i], nb_points=n_pts[i])
        for i in range(len(rec_bundle))
    ])
    return org_rsp, rec_rsp


def point_volume(points: np.ndarray) -> float:
    """
    Estimate the spatial volume (mm^3) of a point cloud via convex hull.

    Returns 0.0 when the point cloud is degenerate (fewer than 4 points, or
    all points are coplanar/colinear), which can occur in terminal segments
    with poor coverage.

    Parameters
    ----------
    points : np.ndarray, shape (n, 3)
        3-D coordinates of the point cloud (e.g. valid streamline points
        within one along-tract segment, in mm).

    Returns
    -------
    vol : float
        Convex hull volume in mm^3, or 0.0 if the hull cannot be computed.
    """
    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def assign_segments(
    bundle: ArraySequence,
    centroid: np.ndarray,
    fixed_ns: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Assign each streamline point to the nearest along-tract segment via
    KD-tree query on the atlas centroid.

    When fixed_ns=False (segment-length mode), terminal bins whose point
    count falls below 0.2 x median are merged into their neighbor, preventing
    near-empty edge segments from producing unreliable means. When
    fixed_ns=True (--ns mode), all ns bins are preserved and the output
    always has exactly len(centroid) segments.

    Parameters
    ----------
    bundle : ArraySequence
        Subject bundle in atlas space (rec_bundle after resampling).
    centroid : np.ndarray, shape (ns, 3)
        Atlas centroid from compute_centroid().
    fixed_ns : bool
        If True, skip terminal-bin merging. Default False.

    Returns
    -------
    s_index : np.ndarray, shape (n_points,)
        Segment index for each streamline point.
    s_dist : np.ndarray, shape (n_points,)
        Distance to the nearest centroid point (inverse-distance weight
        denominator in buan_profile).
    ns_actual : int
        Number of segments after any merging.
    """
    points = bundle.get_data()
    ns = len(centroid)
    s_dist, s_index = cKDTree(centroid).query(points, k=1)

    if fixed_ns:
        return s_index, s_dist, ns

    # Terminal-bin merging (s_len mode only)
    point_count = np.bincount(s_index, minlength=ns)
    thresh = np.median(point_count) * 0.2
    merged = False

    if ns > 1 and point_count[0] < thresh:
        s_index[s_index == 0] = 1
        merged = True
    if ns > 1 and point_count[-1] < thresh:
        s_index[s_index == ns - 1] = ns - 2
        merged = True
    if merged:
        _, s_index = np.unique(s_index, return_inverse=True)

    ns_actual = int(s_index.max()) + 1
    if merged:
        logging.info(f"Terminal-bin merge: ns {ns} -> {ns_actual}")

    return s_index, s_dist, ns_actual


def buan_profile(
    bundle_native: ArraySequence,
    s_index: np.ndarray,
    s_dist: np.ndarray,
    ns: int,
    scalar_map: np.ndarray,
    scalar_affine: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    mask_affine: Optional[np.ndarray] = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a distance-weighted mean along-tract scalar profile.
    Scalar values are trilinearly interpolated at all streamline point
    coordinates via values_from_volume on the unmasked scalar map. If a mask
    is provided, points outside the mask are excluded.
 
    Parameters
    ----------
    bundle_native : ArraySequence
        Subject bundle in native/scalar-map space (org_bundle after resampling).
    s_index : np.ndarray, shape (n_points,)
        Segment index per point (from assign_segments).
    s_dist : np.ndarray, shape (n_points,)
        Distance to nearest centroid point; used as weight denominator.
    ns : int
        Number of segments.
    scalar_map : np.ndarray
        3-D scalar image array in native space.
    scalar_affine : np.ndarray, shape (4, 4)
        Affine of scalar_map.
    mask_data : np.ndarray, optional
        Binary 3-D mask (1=include, 0=exclude). Must share physical space
        with scalar_map; resolution may differ.
    mask_affine : np.ndarray, shape (4, 4), optional
        Affine of mask_data. Required when mask_data is not None.
    epsilon : float
        Added to distances to prevent division by zero. Default 1e-8.
 
    Returns
    -------
    profile : np.ndarray, shape (ns,)
        Distance-weighted mean per segment. NaN where valid_count == 0.
    valid_counts : np.ndarray, shape (ns,), dtype int
        Number of valid (unmasked, non-NaN) points per segment.
    volumes : np.ndarray, shape (ns,), dtype float
        Convex hull volume (mm^3) of valid points per segment. 0.0 for
        degenerate segments (fewer than 4 points or coplanar point cloud).
    """
    from nibabel.affines import apply_affine
    from scipy.ndimage import map_coordinates
 
    raw_values = values_from_volume(scalar_map, bundle_native, scalar_affine)
    values_flat = np.array([v for sl_vals in raw_values for v in sl_vals], dtype=float)
    valid = ~np.isnan(values_flat)
    points_flat = bundle_native.get_data().astype(float)
 
    if mask_data is not None and mask_affine is not None:
        vox = apply_affine(np.linalg.inv(mask_affine), points_flat) 
        # order=0: nearest-neighbour lookup
        # mode='constant', cval=0: points outside the mask volume are excluded.
        in_mask = map_coordinates(
            mask_data, vox.T, order=0, mode='constant', cval=0
        ) > 0
        valid &= in_mask
        logging.info(f"Mask filtered out {(~in_mask).sum()} / {len(valid)} points ({(~in_mask).mean()*100:.1f}%)")
 
    profile = np.full(ns, np.nan)
    valid_counts = np.zeros(ns, dtype=int)
    volumes = np.zeros(ns, dtype=float)
 
    for i in range(ns):
        seg_valid = (s_index == i) & valid
        n_valid = int(seg_valid.sum())
        valid_counts[i] = n_valid
        seg_pts = points_flat[seg_valid]
        volumes[i] = point_volume(seg_pts)
        if n_valid < 1:
            continue
        w  = 1.0 / (s_dist[seg_valid] + epsilon)
        w /= w.sum()
        profile[i] = float(np.sum(w * values_flat[seg_valid]))
 
    return profile, valid_counts, volumes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    """
    Main orchestration
    """
    rec_bundle, _, _, _ = read_streamlines(args.rec_bundle)
    org_bundle, _, _, _ = read_streamlines(args.org_bundle)
    ref_bundle, _, _, _ = read_streamlines(args.ref_bundle)

    if len(rec_bundle) < args.min_lines:
        logging.warning(
            f"Only {len(rec_bundle)} streamlines in rec_bundle "
            f"(min_lines={args.min_lines}). Skipping."
        )
        return

    # Resample org/rec to matching point counts
    org_bundle, rec_bundle = prepare_bundle_pairs(org_bundle, rec_bundle)

    # Compute atlas centroid once — reused for all scalar maps
    fixed_ns = args.ns is not None
    centroid, ns_actual, s_len_actual = compute_centroid(
        ref_bundle,
        ns=args.ns,
        s_len=args.s_len,
        robust_method=args.robust_method,
    )

    # Assign segments once — reused for all scalar maps
    s_index, s_dist, ns_final = assign_segments(
        rec_bundle, centroid, fixed_ns=fixed_ns
    )

    # Load mask once — applied to all scalar maps
    mask_data, mask_affine = None, None
    if args.mask:
        logging.info(f"Loading mask: {args.mask}")
        mask_data, mask_affine = load_nifti(args.mask)

    # Loop over scalar map / output path pairs
    fieldnames = ['segment', 'mean', 'valid_point_count', 'volume_mm3']
    for map_path, out_path in zip(args.scalar_maps, args.outputs):
        logging.info(f"Processing scalar map: {map_path}")
        scalar_map, scalar_affine = load_nifti(map_path)

        profile, valid_counts, volumes = buan_profile(
            bundle_native=org_bundle,
            s_index=s_index,
            s_dist=s_dist,
            ns=ns_final,
            scalar_map=scalar_map,
            scalar_affine=scalar_affine,
            mask_data=mask_data,
            mask_affine=mask_affine,
        )

        rows = [{
            'segment':           i + 1,
            'mean':              val,
            'valid_point_count': int(cnt),
            'volume_mm3':        vol,
            } for i, (val, cnt, vol) in enumerate(zip(profile, valid_counts, volumes))]

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        n_nan = int(np.isnan(profile).sum())
        logging.info(
            f"Written {len(rows)} rows to {out_path} "
            f"({n_nan} segment(s) with NaN mean)"
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s:%(module)s.%(funcName)s]: %(message)s",
        datefmt='%H:%M:%S',
    )
    parser = argparse.ArgumentParser(
        description='Along-tract BUAN profile extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs/Outputs
    parser.add_argument('--rec_bundle', '-rec', required=True,
                        help='Subject bundle in atlas/MNI space (.trk/.tck/.tt.gz)')
    parser.add_argument('--org_bundle', '-org', required=True,
                        help='Subject bundle in native space (.trk/.tck/.tt.gz)')
    parser.add_argument('--ref_bundle', '-ref', required=True,
                        help='Atlas/reference bundle (.trk/.tck/.tt.gz)')
    parser.add_argument('--scalar_maps', '-maps', nargs='+', required=True,
                        help='One or more scalar map NIfTIs in native space')
    parser.add_argument('--outputs', '-o', nargs='+', required=True,
                        help='Output CSV paths, one per scalar map (must match in count)')
    parser.add_argument('--mask', '-mask', default=None,
                        help=('Optional binary NIfTI mask applied to all scalar maps.'))

    # Parameterization
    parser.add_argument('--s_len', '-s', type=float, default=5.0,
                        help='Target segment length (mm). Ignored when --ns is set.')
    parser.add_argument('--ns', '-ns', type=int, default=None,
                        help=(
                            'Number of segments. Overrides --s_len; output always '
                            'has exactly this many rows with no terminal-bin merging.'
                        ))
    parser.add_argument('--robust_method', '-robust', default='linear',
                        choices=['linear', 'spline', 'none'],
                        help=(
                            'Centroid extension method. '
                            '"linear": extrapolate from terminal tangent vectors (default). '
                            '"spline": cubic spline extrapolation. '
                            '"none": standard QuickBundles centroid.'
                        ))

    # Guards
    parser.add_argument('--min_lines', '-min_lines', type=int, default=20,
                        help='Skip bundle if streamline count is below this threshold.')

    args = parser.parse_args()

    if args.ns is not None and args.ns < 1:
        parser.error("--ns must be a positive integer")
    if len(args.scalar_maps) != len(args.outputs):
        parser.error(
            f"--scalar_maps ({len(args.scalar_maps)}) and "
            f"--outputs ({len(args.outputs)}) must have the same number of entries"
        )

    run(args)


if __name__ == '__main__':
    main()
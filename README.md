# buan_profile

Along-tract scalar profile extraction for white matter tractometry using the BUndle ANalytics (BUAN) distance-weighted mean method.
 
## Requirements

- Python 3.10+
- numpy, scipy, nibabel, dipy
- uv (recommended for environment management)

## Installation

```bash
git clone https://github.com/wendyfyx/buan_profile.git
cd buan_profile
uv pip install -e .
```

To verify and run:

```bash
uv run buan_profile --help
```
or
```bash
source .venv/bin/activate
buan_profile --help
```

## Usage

```bash
buan-profile \
    --rec_bundle sub01_AF_L_mni.trk \
    --org_bundle sub01_AF_L_native.trk \
    --ref_bundle atlas_AF_L.trk \
    --scalar_maps sub01_fa.nii.gz sub01_md.nii.gz \
    --outputs sub01_AF_L_fa.csv sub01_AF_L_md.csv \
    --mask sub01_mask.nii.gz \
    --s_len 5.0 \
    --robust_method linear
```

Multiple scalar maps are processed in a single call — centroid computation and segment assignment are performed once and reused across all maps. `--scalar_maps` and `--outputs` must be provided in matching order with equal counts.

## Arguments

### Inputs / Outputs

| Argument | Required | Description |
|---|---|---|
| `--rec_bundle` / `-rec` | yes | Subject bundle in atlas/MNI space (`.trk`, `.tck`, `.tt.gz`) |
| `--org_bundle` / `-org` | yes | Same subject bundle in native space |
| `--ref_bundle` / `-ref` | yes | Atlas reference bundle |
| `--scalar_maps` / `-maps` | yes | One or more scalar map NIfTIs in native space |
| `--outputs` / `-o` | yes | Output CSV paths, one per scalar map |
| `--mask` / `-mask` | no | Binary NIfTI mask (1=include, 0=exclude), applied to all scalar maps |

### Parameterization and others

| Argument | Default | Description |
|---|---|---|
| `--s_len` / `-s` | `5.0` | Target segment length (mm). Ignored when `--ns` is set |
| `--ns` / `-ns` | `None` | Number of segments. Overrides `--s_len`; output always has exactly this many rows |
| `--robust_method` / `-robust` | `linear` | Centroid extension method: `linear`, `spline`, or `none` |
| `--min_lines` / `-min_lines` | `20` | Skip bundle if streamline count is below this threshold |

## Output format

One CSV file per scalar map with the following columns: 
- `segment` : segment index (1-based)
- `mean` : the mean scalar value for this segment
- `valid_point_count` : number or valid point for this segment
- `volume_mm3` : convex hull volume (mm3) of the valid streamline points within this segment. 

Segments with zero valid points receive `mean=NaN` and `volume_mm3=0.0`.

## Pipeline steps

**Step 1: Load and resample bundle pairs.** Three bundle files are required per subject: `--rec_bundle` (subject bundle registered to atlas/MNI space),  `--org_bundle` (same bundle in native space), and `--ref_bundle` (the atlas reference bundle). Before any spatial computation, org_bundle and rec_bundle are resampled via prepare_bundle_pairs() so that each corresponding streamline has the same number of evenly-spaced points. `rec_bundle` and `org_bundle` must have the same number of streamlines in matching order.

**Step 2: Compute the atlas centroid.** `compute_centroid()` runs QuickBundles on ref_bundle to obtain an initial centroid, then optionally extends it beyond its natural endpoints via --robust_method. The standard QB centroid represents the mean trajectory of the bundle, and may cause terminal segments to span a disproportionately large anatomical extent. The robust linear extension (default) projects the terminal tangent vectors outward and trims to the 2nd-98th percentile of streamline endpoint projections. The spline method fits a cubic parametric curve through the QB centroid and evaluates it outside [0, 1], less stable for short or strongly curved endpoints.

**Step 3: Assign segments.** `assign_segments()` queries every point in rec_bundle against the centroid via a KD-tree, returning a segment index and a Euclidean distance for each point. Distances are later used as inverse weights in the BUAN mean. In segment-length mode (`--s_len`), terminal bins whose point count falls  below 0.2 x median are merged into their neighbor, preventing near-empty edge segments from producing unreliable means. When `--ns` is set, no merging is performed and the output always has exactly ns rows.

**Step 4: Interpolate scalar values and apply mask.** `buan_profile()` calls values_from_volume to trilinearly interpolate the scalar map at all org_bundle point coordinates. If `--mask` is provided, masking is applied after interpolation: each point's world coordinate is mapped to the nearest voxel in the mask volume and points outside the mask are excluded. Post-interpolation masking avoids the boundary artefacts that arise when zeros or NaNs are inserted into the scalar map before interpolation. A single mask is loaded once and applied to all scalar maps in the same call.

**Step 5: Compute per-segment mean and coverage volume.** For each segment, valid (unmasked, non-NaN) points are used to compute the BUAN weighted mean, where each point's scalar value is weighted by the inverse of its distance to the nearest centroid point. `point_volume()` computes the convex hull of the valid points, 
and returns a volume estimate per segment (mm^3). Each scalar map writes an CSV with segment wise measures ([see details](#output-format)).

## Validity checks

1. **Minimum streamlines** — bundles with fewer than `--min_lines` streamlines (default 20) are skipped before any computation.
2. **Terminal segment merging** — in `--s_len` mode, terminal segments below 0.2 × median point count are merged into their neighbor. Suppressed when `--ns` is set.
3. **NaN exclusion** — points whose interpolated scalar value is NaN are excluded from the weighted mean and coverage volume.
4. **Mask exclusion** — when `--mask` is provided, points mapping to a zero voxel or outside the mask volume are excluded.
5. **Empty segment guard** — segments with no valid points receive `mean=NaN` and `volume_mm3=0.0`, and are identifiable via `valid_point_count=0`.

For group analysis we recommend additional subject-level checks before making inferences: (1) thresholding `valid_point_count` per segment to exclude segments with insufficient coverage in individual subjects; (2) thresholding the number of subjects with sufficient valid points per segment before including that segment in group-level statistics.

## Citation: 

If you use this repository in your research, please cite:

1. Chandio, B.Q., Risacher, S.L., Pestilli, F., Bullock, D., Yeh, F.C., Koudoro, S., Rokem, A., Harezlak, J. and Garyfallidis, E., 2020. Bundle analytics, a computational framework for investigating the shapes and profiles of brain pathways across populations. Scientific Reports, 10(1), p.17149. 

2. Chandio, B.Q., Villalon-Reina, J.E., Nir, T.M., Thomopoulos, S.I., Feng, Y., Benavidez, S., Jahanshad, N., Harezlak, J., Garyfallidis, E. and Thompson, P.M., 2024, July. Bundle analytics based data harmonization for multi-site diffusion MRI tractometry. In 2024, the 46th annual international conference of the IEEE Engineering in Medicine and Biology Society (EMBC) (pp. 1-7). IEEE.

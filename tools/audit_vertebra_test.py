import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt


ROOT = Path("data/vertebra/dataset/test")
OUT = Path("outputs/vertebra/dataset_audit")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "qc").mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def safe(v):
    if v is None:
        return None

    try:
        if isinstance(v, (str, int, float)):
            return v

        if hasattr(v, "__iter__"):
            return [safe(x) for x in v]

        return str(v)

    except Exception:
        return str(v)


def normalize_image(img):
    """
    Robust normalization for VISUAL QC only.
    Does not modify source data.
    """
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, [1, 99])

    if p99 <= p1:
        return np.zeros_like(img)

    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1)

    return img


def flatten_landmarks(d):
    """
    {
        "2": {
            "1": [x, y, z],
            ...
        }
    }

    ->
    list of:
        (vertebra, point_id, coordinate)
    """

    result = []

    if not isinstance(d, dict):
        return result

    for vertebra, points in d.items():

        if not isinstance(points, dict):
            continue

        for point_id, coord in points.items():

            if coord is None:
                continue

            result.append(
                (
                    str(vertebra),
                    str(point_id),
                    coord,
                )
            )

    return result


def make_key_map(d):
    result = {}

    for vertebra, point_id, coord in flatten_landmarks(d):
        result[(vertebra, point_id)] = np.asarray(
            coord,
            dtype=float,
        )

    return result


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 3:
        return np.nan

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def landmark_orientation_metrics(ct_lps, points_2d, view):

    ct = make_key_map(ct_lps)
    px = make_key_map(points_2d)

    keys = sorted(
        set(ct.keys()) & set(px.keys())
    )

    if len(keys) < 3:
        return {
            "n_common": len(keys),
            "corr_u_x": np.nan,
            "corr_u_y": np.nan,
            "corr_u_z": np.nan,
            "corr_v_x": np.nan,
            "corr_v_y": np.nan,
            "corr_v_z": np.nan,
        }

    xyz = np.stack([ct[k] for k in keys])
    uv = np.stack([px[k] for k in keys])

    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    U = uv[:, 0]
    V = uv[:, 1]

    return {
        "n_common": len(keys),

        "corr_u_x": pearson(U, X),
        "corr_u_y": pearson(U, Y),
        "corr_u_z": pearson(U, Z),

        "corr_v_x": pearson(V, X),
        "corr_v_y": pearson(V, Y),
        "corr_v_z": pearson(V, Z),
    }


def point_patch_intensity(img_norm, landmarks_px, radius=12):
    """
    Estimate image polarity.

    We sample small patches around annotated bone landmarks.
    Compare them with border/background intensity.

    Positive:
        landmarks brighter than background

    Negative:
        landmarks darker than background
    """

    pts = make_key_map(landmarks_px)

    samples = []

    H, W = img_norm.shape

    for _, p in pts.items():

        x, y = p[:2]

        x = int(round(x))
        y = int(round(y))

        x1 = max(0, x - radius)
        x2 = min(W, x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(H, y + radius + 1)

        if x2 <= x1 or y2 <= y1:
            continue

        patch = img_norm[y1:y2, x1:x2]

        samples.append(
            float(np.mean(patch))
        )

    landmark_mean = (
        float(np.mean(samples))
        if samples
        else np.nan
    )

    # Outer 8% border
    by = max(1, int(H * 0.08))
    bx = max(1, int(W * 0.08))

    border = np.concatenate(
        [
            img_norm[:by, :].ravel(),
            img_norm[-by:, :].ravel(),
            img_norm[:, :bx].ravel(),
            img_norm[:, -bx:].ravel(),
        ]
    )

    border_mean = float(np.mean(border))

    return (
        landmark_mean,
        border_mean,
        landmark_mean - border_mean,
    )


def inside_image(points, width, height):

    m = make_key_map(points)

    total = len(m)
    bad = []

    for key, p in m.items():

        x, y = p[:2]

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):
            bad.append(
                {
                    "landmark": key,
                    "xy": [float(x), float(y)],
                }
            )

    return total, bad


def find_dicom(case_dir, view):

    candidates = [
        case_dir / "xrays" / f"{view}.dcm",
        case_dir / "xrays" / f"{view.upper()}.dcm",
        case_dir / "xrays" / f"{view.lower()}.dcm",
        case_dir / f"{view}.dcm",
        case_dir / f"{view.upper()}.dcm",
        case_dir / f"{view.lower()}.dcm",
    ]

    for p in candidates:
        if p.exists():
            return p

    # fallback
    matches = list(
        case_dir.rglob(
            f"*{view}*.dcm"
        )
    )

    if not matches:
        matches = list(
            case_dir.rglob(
                f"*{view.lower()}*.dcm"
            )
        )

    if len(matches) == 1:
        return matches[0]

    return None


def plot_qc(
    case_name,
    view,
    img,
    landmarks,
    output,
    photometric,
    title_extra="",
):

    stored = normalize_image(img)

    # --------------------------------------------------------
    # DICOM display semantics
    # --------------------------------------------------------

    if str(photometric).upper() == "MONOCHROME1":
        dicom_display = 1.0 - stored
    else:
        dicom_display = stored.copy()

    # --------------------------------------------------------
    # Current xvr input
    #
    # Current xvr read_xray() ignores MONOCHROME1/MONOCHROME2
    # when --linearize is NOT used.
    # Therefore this is simply normalized stored PixelData.
    # --------------------------------------------------------

    xvr_input = stored.copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 10),
    )

    panels = [
        (
            axes[0],
            stored,
            "Stored PixelData\n(normalized)",
        ),
        (
            axes[1],
            dicom_display,
            "DICOM display\n(PhotometricInterpretation applied)",
        ),
        (
            axes[2],
            xvr_input,
            "Current xvr input\n(linearize=False)",
        ),
    ]

    for ax, shown, title in panels:

        ax.imshow(
            shown,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        for vertebra, point_id, p in flatten_landmarks(
            landmarks
        ):

            x, y = p[:2]

            ax.scatter(
                x,
                y,
                s=12,
                facecolors="none",
                edgecolors="red",
                linewidths=0.8,
            )

            ax.text(
                x + 4,
                y + 4,
                f"{vertebra}.{point_id}",
                fontsize=5,
            )

        ax.set_title(
            f"{case_name} {view}\n"
            f"{title}\n"
            f"{title_extra}",
            fontsize=9,
        )

        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Audit one view
# ============================================================

def audit_view(
    case_name,
    case_dir,
    view_name,
    view_json,
    ct_landmarks,
):

    dicom_path = find_dicom(
        case_dir,
        view_name.upper(),
    )

    row = {
        "case": case_name,
        "view": view_name.upper(),

        "dicom_exists": dicom_path is not None,
        "landmarks_json_exists": bool(view_json),

        "dicom_path": (
            str(dicom_path)
            if dicom_path
            else None
        ),
    }

    details = {
        "case": case_name,
        "view": view_name.upper(),
        "dicom_path": row["dicom_path"],
        "warnings": [],
    }

    if dicom_path is None:

        details["warnings"].append(
            "DICOM NOT FOUND"
        )

        return row, details

    ds = pydicom.dcmread(dicom_path)

    img = ds.pixel_array

    if img.ndim > 2:
        img = img[0]

    H, W = img.shape


    # --------------------------------------------------------
    # DICOM metadata
    # --------------------------------------------------------

    metadata_fields = [
        "Rows",
        "Columns",

        "PixelSpacing",
        "ImagerPixelSpacing",

        "DistanceSourceToDetector",
        "DistanceSourceToPatient",

        "DetectorActiveOrigin",

        "PositionerPrimaryAngle",
        "PositionerSecondaryAngle",

        "ViewPosition",
        "PatientOrientation",

        "PhotometricInterpretation",
        "PixelIntensityRelationship",

        "PresentationLUTShape",

        "RescaleSlope",
        "RescaleIntercept",

        "WindowCenter",
        "WindowWidth",
    ]

    metadata = {}

    for key in metadata_fields:
        metadata[key] = safe(
            getattr(ds, key, None)
        )

    details["dicom_metadata"] = metadata


    row.update(
        {
            "height": H,
            "width": W,

            "photometric": str(
                getattr(
                    ds,
                    "PhotometricInterpretation",
                    "",
                )
            ),

            "pixel_intensity_relationship": str(
                getattr(
                    ds,
                    "PixelIntensityRelationship",
                    "",
                )
            ),

            "view_position": str(
                getattr(ds, "ViewPosition", "")
            ),

            "patient_orientation": str(
                safe(
                    getattr(
                        ds,
                        "PatientOrientation",
                        None,
                    )
                )
            ),

            "primary_angle": (
                float(ds.PositionerPrimaryAngle)
                if hasattr(
                    ds,
                    "PositionerPrimaryAngle",
                )
                else np.nan
            ),

            "secondary_angle": (
                float(ds.PositionerSecondaryAngle)
                if hasattr(
                    ds,
                    "PositionerSecondaryAngle",
                )
                else np.nan
            ),

            "sdd": (
                float(ds.DistanceSourceToDetector)
                if hasattr(
                    ds,
                    "DistanceSourceToDetector",
                )
                else np.nan
            ),

            "sid_patient": (
                float(ds.DistanceSourceToPatient)
                if hasattr(
                    ds,
                    "DistanceSourceToPatient",
                )
                else np.nan
            ),

            "detector_active_origin_present":
                hasattr(
                    ds,
                    "DetectorActiveOrigin",
                ),
        }
    )


    # --------------------------------------------------------
    # Pixel spacing
    # --------------------------------------------------------

    pixel_spacing = getattr(
        ds,
        "PixelSpacing",
        None,
    )

    imager_spacing = getattr(
        ds,
        "ImagerPixelSpacing",
        None,
    )

    row["pixel_spacing"] = str(
        safe(pixel_spacing)
    )

    row["imager_pixel_spacing"] = str(
        safe(imager_spacing)
    )


    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    norm = normalize_image(img)

    row.update(
        {
            "raw_min": float(np.min(img)),
            "raw_max": float(np.max(img)),
            "raw_mean": float(np.mean(img)),
            "raw_median": float(np.median(img)),
        }
    )


    # --------------------------------------------------------
    # landmarks.json
    # --------------------------------------------------------

    landmarks_px = {}

    if isinstance(view_json, dict):
        landmarks_px = (
            view_json.get(
                "landmarks_px",
                {},
            )
            or {}
        )

    count, out_of_bounds = inside_image(
        landmarks_px,
        W,
        H,
    )

    row["landmark_count"] = count
    row["landmarks_out_of_bounds"] = len(
        out_of_bounds
    )

    details[
        "landmarks_out_of_bounds"
    ] = out_of_bounds


    # --------------------------------------------------------
    # Check dimensions recorded in JSON
    # --------------------------------------------------------

    json_size = (
        view_json.get(
            "image_size_px",
            {},
        )
        if isinstance(view_json, dict)
        else {}
    )

    json_H = json_size.get(
        "height",
        None,
    )

    json_W = json_size.get(
        "width",
        None,
    )

    size_match = (
        json_H is None
        or json_W is None
        or (
            int(json_H) == H
            and
            int(json_W) == W
        )
    )

    row["json_size_match"] = size_match

    if not size_match:
        details["warnings"].append(
            f"JSON image size "
            f"{json_W}x{json_H} != "
            f"DICOM {W}x{H}"
        )


    # --------------------------------------------------------
    # CT ↔ 2D landmark correspondence
    # --------------------------------------------------------

    metrics = landmark_orientation_metrics(
        ct_landmarks,
        landmarks_px,
        view_name,
    )

    row.update(metrics)


    # --------------------------------------------------------
    # Polarity heuristic
    # --------------------------------------------------------

    lm_mean, bg_mean, contrast = (
        point_patch_intensity(
            norm,
            landmarks_px,
        )
    )

    row["landmark_patch_mean"] = lm_mean
    row["border_mean"] = bg_mean
    row["landmark_minus_border"] = contrast

    if np.isnan(contrast):
        polarity_guess = "UNKNOWN"

    elif contrast > 0.05:
        polarity_guess = (
            "LANDMARKS_BRIGHTER"
        )

    elif contrast < -0.05:
        polarity_guess = (
            "LANDMARKS_DARKER"
        )

    else:
        polarity_guess = (
            "LOW_CONTRAST/UNCLEAR"
        )

    row["polarity_guess"] = polarity_guess
    row["xvr_polarity_ok"] = (
        polarity_guess == "LANDMARKS_BRIGHTER"
    )

    row["needs_intensity_inversion"] = (
        polarity_guess == "LANDMARKS_DARKER"
    )


    # --------------------------------------------------------
    # Initial warnings
    # --------------------------------------------------------

    if count == 0:
        details["warnings"].append(
            "NO 2D LANDMARKS"
        )

    if out_of_bounds:
        details["warnings"].append(
            f"{len(out_of_bounds)} landmarks "
            f"outside image"
        )

    if not row[
        "detector_active_origin_present"
    ]:
        details["warnings"].append(
            "DetectorActiveOrigin missing"
        )


    # --------------------------------------------------------
    # QC image
    # --------------------------------------------------------

    title_extra = (
        f"{row['photometric']} | "
        f"{polarity_guess}"
    )

    qc_path = (
        OUT
        / "qc"
        / f"{case_name}_{view_name.upper()}.png"
    )

    plot_qc(
        case_name,
        view_name.upper(),
        img,
        landmarks_px,
        qc_path,
        row["photometric"],
        title_extra,
    )

    row["qc_image"] = str(qc_path)

    return row, details


# ============================================================
# Main
# ============================================================

case_dirs = sorted(
    [
        p
        for p in ROOT.iterdir()
        if p.is_dir()
        and p.name.lower().startswith(
            "case"
        )
    ],
    key=lambda x: int(
        "".join(
            c
            for c in x.name
            if c.isdigit()
        )
    ),
)


print()
print("=" * 80)
print("VERTEBRA DATASET AUDIT")
print("=" * 80)
print("root :", ROOT)
print("cases:", len(case_dirs))
print()


rows = []
details_all = {}


for case_dir in case_dirs:

    case_name = case_dir.name

    print(
        f"[AUDIT] {case_name}"
    )

    landmark_file = (
        case_dir
        / "landmarks.json"
    )

    details_all[case_name] = {
        "case_dir": str(case_dir),
        "landmarks_json": str(
            landmark_file
        ),
        "warnings": [],
    }

    if not landmark_file.exists():

        print(
            "  ERROR: landmarks.json missing"
        )

        details_all[
            case_name
        ]["warnings"].append(
            "landmarks.json missing"
        )

        continue

    with open(
        landmark_file,
        "r",
        encoding="utf-8",
    ) as f:
        lm = json.load(f)


    ct_landmarks = (
        lm.get("ct", {})
        .get(
            "landmarks_lps_mm",
            {},
        )
    )

    views = lm.get(
        "views",
        {},
    )


    # CT landmark count
    ct_count = len(
        flatten_landmarks(
            ct_landmarks
        )
    )

    details_all[
        case_name
    ]["ct_landmark_count"] = (
        ct_count
    )


    # Find NIfTI files
    nii_files = sorted(
        str(p.relative_to(case_dir))
        for p in case_dir.rglob(
            "*.nii*"
        )
    )

    details_all[
        case_name
    ]["nifti_files"] = nii_files


    details_all[
        case_name
    ]["views"] = {}


    for view_name in [
        "ap",
        "lat",
    ]:

        row, details = audit_view(
            case_name,
            case_dir,
            view_name,
            views.get(
                view_name,
                {},
            ),
            ct_landmarks,
        )

        row[
            "ct_landmark_count"
        ] = ct_count

        rows.append(row)

        details_all[
            case_name
        ]["views"][
            view_name
        ] = details


# ============================================================
# Build dataframe
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# Dataset-level orientation outlier detection
#
# AP:
#   horizontal coordinate should strongly correlate with CT X.
#
# LAT:
#   horizontal coordinate should strongly correlate with CT Y.
#
# Both:
#   vertical coordinate should correlate with CT Z.
#
# We do NOT assume which sign is "correct" yet.
# We detect cases whose sign differs from the majority.
# ============================================================

def add_sign_outlier(
    df,
    view,
    metric,
    output_col,
    min_abs_corr=0.50,
):

    mask = (
        (df["view"] == view)
        &
        df[metric].notna()
        &
        (
            df[metric].abs()
            >= min_abs_corr
        )
    )

    values = df.loc[
        mask,
        metric,
    ]

    if len(values) == 0:
        df[output_col] = False
        return None

    majority_sign = (
        1
        if np.median(values) >= 0
        else -1
    )

    out = np.zeros(
        len(df),
        dtype=bool,
    )

    for idx in df.index:

        if (
            df.loc[idx, "view"]
            != view
        ):
            continue

        value = df.loc[
            idx,
            metric,
        ]

        if pd.isna(value):
            continue

        if abs(value) < min_abs_corr:
            continue

        sign = (
            1
            if value >= 0
            else -1
        )

        if sign != majority_sign:
            out[idx] = True

    df[output_col] = out

    return majority_sign


ap_lr_sign = add_sign_outlier(
    df,
    "AP",
    "corr_u_x",
    "lr_orientation_outlier",
)

lat_lr_sign = add_sign_outlier(
    df,
    "LAT",
    "corr_u_y",
    "lr_orientation_outlier_lat",
)

ap_ud_sign = add_sign_outlier(
    df,
    "AP",
    "corr_v_z",
    "ud_orientation_outlier",
)

lat_ud_sign = add_sign_outlier(
    df,
    "LAT",
    "corr_v_z",
    "ud_orientation_outlier_lat",
)


# ============================================================
# Polarity outlier detection by sign of landmark-background
# contrast, separately for AP and LAT.
# ============================================================

df["polarity_outlier"] = False

for view in [
    "AP",
    "LAT",
]:

    m = (
        (df["view"] == view)
        &
        df[
            "landmark_minus_border"
        ].notna()
        &
        (
            df[
                "landmark_minus_border"
            ].abs()
            > 0.03
        )
    )

    vals = df.loc[
        m,
        "landmark_minus_border",
    ]

    if len(vals) == 0:
        continue

    majority_sign = (
        1
        if np.median(vals) >= 0
        else -1
    )

    for idx in df.index:

        if (
            df.loc[idx, "view"]
            != view
        ):
            continue

        value = df.loc[
            idx,
            "landmark_minus_border",
        ]

        if pd.isna(value):
            continue

        if abs(value) <= 0.03:
            continue

        sign = (
            1
            if value >= 0
            else -1
        )

        if sign != majority_sign:
            df.loc[
                idx,
                "polarity_outlier",
            ] = True


# ============================================================
# Combined flag
# ============================================================

orientation_cols = [
    c
    for c in [
        "lr_orientation_outlier",
        "lr_orientation_outlier_lat",
        "ud_orientation_outlier",
        "ud_orientation_outlier_lat",
    ]
    if c in df.columns
]

df["needs_review"] = (
    df["polarity_outlier"]
    |
    (
        df[orientation_cols]
        .any(axis=1)
        if orientation_cols
        else False
    )
    |
    (df["landmarks_out_of_bounds"] > 0)
    |
    (~df["json_size_match"])
)


# ============================================================
# Save
# ============================================================

csv_path = OUT / "audit.csv"
json_path = OUT / "audit_details.json"

df.to_csv(
    csv_path,
    index=False,
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "root": str(ROOT),

            "n_cases": len(
                case_dirs
            ),

            "orientation_majority_signs": {
                "AP_corr_u_x":
                    ap_lr_sign,

                "LAT_corr_u_y":
                    lat_lr_sign,

                "AP_corr_v_z":
                    ap_ud_sign,

                "LAT_corr_v_z":
                    lat_ud_sign,
            },

            "cases":
                details_all,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Terminal summary
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

cols = [
    "case",
    "view",
    "photometric",
    "view_position",
    "primary_angle",
    "polarity_guess",
    "xvr_polarity_ok",
    "needs_intensity_inversion",
    "landmark_minus_border",
    "corr_u_x",
    "corr_u_y",
    "corr_v_z",
    "polarity_outlier",
    "needs_review",
]

existing = [
    c
    for c in cols
    if c in df.columns
]

with pd.option_context(
    "display.max_rows",
    100,
    "display.max_columns",
    None,
    "display.width",
    240,
):

    print(
        df[existing]
        .to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("FLAGGED VIEWS")
print("=" * 80)

flagged = df[
    df["needs_review"]
]

if len(flagged) == 0:

    print(
        "No automatic outliers detected."
    )

else:

    show = [
        c
        for c in [
            "case",
            "view",
            "polarity_outlier",
            "lr_orientation_outlier",
            "lr_orientation_outlier_lat",
            "ud_orientation_outlier",
            "ud_orientation_outlier_lat",
            "landmarks_out_of_bounds",
        ]
        if c in flagged.columns
    ]

    print(
        flagged[
            show
        ].to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("OUTPUT")
print("=" * 80)

print("CSV :", csv_path)
print("JSON:", json_path)
print("QC  :", OUT / "qc")
print()


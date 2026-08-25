import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt


ROOT = Path("data/vertebra/dataset/test")
OUT = Path("outputs/vertebra/ap_canonical_audit")

QC = OUT / "qc"

OUT.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def normalize_for_display(img):

    img = img.astype(np.float32)

    lo, hi = np.percentile(
        img,
        [1, 99],
    )

    if hi <= lo:
        return np.zeros_like(img)

    img = np.clip(
        img,
        lo,
        hi,
    )

    img = (
        (img - lo)
        /
        (hi - lo)
    )

    return img


def dicom_display(ds):

    img = ds.pixel_array

    if img.ndim > 2:
        img = img[0]

    img = normalize_for_display(img)

    photometric = str(
        getattr(
            ds,
            "PhotometricInterpretation",
            "",
        )
    ).upper()

    # DICOM viewing semantics
    if photometric == "MONOCHROME1":
        img = 1.0 - img

    return img


def make_map(d):

    out = {}

    for vertebra, points in d.items():

        for point_id, coord in points.items():

            out[
                (str(vertebra), str(point_id))
            ] = np.asarray(
                coord,
                dtype=float,
            )

    return out


def corr(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) < 3:
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def calculate_corr_u_x(
    ct_lps,
    ap_landmarks,
):

    ct = make_map(ct_lps)
    ap = make_map(ap_landmarks)

    keys = sorted(
        set(ct)
        & set(ap)
    )

    X = np.asarray(
        [ct[k][0] for k in keys]
    )

    U = np.asarray(
        [ap[k][0] for k in keys]
    )

    return corr(U, X)


def flip_landmarks_x(
    landmarks,
    width,
):

    result = {}

    for vertebra, points in landmarks.items():

        result[vertebra] = {}

        for point_id, xy in points.items():

            x = float(xy[0])
            y = float(xy[1])

            result[vertebra][point_id] = [
                (width - 1) - x,
                y,
            ]

    return result


def flatten_points(landmarks):

    rows = []

    for vertebra, points in landmarks.items():

        for point_id, xy in points.items():

            rows.append(
                (
                    str(vertebra),
                    str(point_id),
                    float(xy[0]),
                    float(xy[1]),
                )
            )

    return rows


# ============================================================
# Landmark / image edge alignment heuristic
#
# Anatomical landmarks are mostly placed on vertebral
# boundaries/corners. Correct image orientation should,
# statistically, put more landmarks near strong gradients.
#
# This is ONLY a ranking heuristic.
# Final decision still needs QC inspection.
# ============================================================

def gradient_image(img):

    gy, gx = np.gradient(
        img.astype(np.float32)
    )

    g = np.sqrt(
        gx * gx + gy * gy
    )

    return g


def landmark_edge_score(
    img,
    landmarks,
    radius=8,
):

    grad = gradient_image(img)

    H, W = img.shape

    scores = []

    for _, _, x, y in flatten_points(
        landmarks
    ):

        xi = int(round(x))
        yi = int(round(y))

        x1 = max(
            0,
            xi - radius,
        )

        x2 = min(
            W,
            xi + radius + 1,
        )

        y1 = max(
            0,
            yi - radius,
        )

        y2 = min(
            H,
            yi + radius + 1,
        )

        if x2 <= x1 or y2 <= y1:
            continue

        patch = grad[
            y1:y2,
            x1:x2,
        ]

        # Robust "there is an edge nearby" score
        scores.append(
            float(
                np.percentile(
                    patch,
                    90,
                )
            )
        )

    if not scores:
        return np.nan

    return float(
        np.mean(scores)
    )


def plot_candidate(
    ax,
    image,
    landmarks,
    title,
):

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    for vertebra, point_id, x, y in flatten_points(
        landmarks
    ):

        ax.scatter(
            x,
            y,
            s=15,
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
        )

        ax.text(
            x + 4,
            y + 4,
            f"{vertebra}.{point_id}",
            fontsize=4,
        )

    ax.set_title(
        title,
        fontsize=9,
    )

    ax.axis("off")


# ============================================================
# Main
# ============================================================

cases = sorted(
    [
        p
        for p in ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("case")
    ],
    key=lambda p: int(
        p.name.replace(
            "case",
            "",
        )
    ),
)


rows = []


for case_dir in cases:

    case = case_dir.name

    ap_path = (
        case_dir
        / "AP.dcm"
    )

    lm_path = (
        case_dir
        / "landmarks.json"
    )

    print(
        f"[AUDIT] {case}"
    )

    if not ap_path.exists():
        print(
            "  ERROR: AP.dcm missing"
        )
        continue

    if not lm_path.exists():
        print(
            "  ERROR: landmarks.json missing"
        )
        continue


    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    ds = pydicom.dcmread(
        ap_path
    )

    raw_pixels = ds.pixel_array

    if raw_pixels.ndim > 2:
        raw_pixels = raw_pixels[0]

    H, W = raw_pixels.shape

    display_raw = dicom_display(
        ds
    )

    display_flip = np.fliplr(
        display_raw
    )


    with open(
        lm_path,
        "r",
        encoding="utf-8",
    ) as f:
        lm = json.load(f)


    ct_lps = (
        lm["ct"]
        ["landmarks_lps_mm"]
    )

    ap_landmarks_original = (
        lm["views"]
        ["ap"]
        ["landmarks_px"]
    )


    # --------------------------------------------------------
    # Determine LANDMARK canonicalization
    #
    # Validated case01 canonical convention:
    #
    #     corr_u_x < 0
    #
    # Therefore:
    #
    #     negative -> keep landmark coordinates
    #     positive -> horizontal flip landmark coordinates
    #
    # This is independent of whether AP.dcm itself needs flip.
    # --------------------------------------------------------

    original_corr = (
        calculate_corr_u_x(
            ct_lps,
            ap_landmarks_original,
        )
    )


    if original_corr > 0:

        landmark_action = "FLIP"

        canonical_landmarks = (
            flip_landmarks_x(
                ap_landmarks_original,
                W,
            )
        )

    else:

        landmark_action = "KEEP"

        canonical_landmarks = (
            ap_landmarks_original
        )


    canonical_corr = (
        calculate_corr_u_x(
            ct_lps,
            canonical_landmarks,
        )
    )


    # --------------------------------------------------------
    # Compare RAW image vs FLIPPED image
    # using the SAME canonical landmark coordinates.
    # --------------------------------------------------------

    score_raw = landmark_edge_score(
        display_raw,
        canonical_landmarks,
    )

    score_flip = landmark_edge_score(
        display_flip,
        canonical_landmarks,
    )


    if np.isnan(score_raw) or np.isnan(
        score_flip
    ):

        image_guess = "UNKNOWN"
        ratio = np.nan
        confidence = "UNKNOWN"

    elif score_flip > score_raw:

        image_guess = "FLIP"

        ratio = (
            score_flip
            /
            max(
                score_raw,
                1e-12,
            )
        )

    else:

        image_guess = "KEEP"

        ratio = (
            score_raw
            /
            max(
                score_flip,
                1e-12,
            )
        )


    if not np.isnan(ratio):

        if ratio >= 1.25:
            confidence = "HIGH"

        elif ratio >= 1.10:
            confidence = "MEDIUM"

        else:
            confidence = "LOW"


    # --------------------------------------------------------
    # Check whether AP_flip.dcm already exists
    # --------------------------------------------------------

    existing_flip = (
        case_dir
        / "AP_flip.dcm"
    )

    has_ap_flip = (
        existing_flip.exists()
    )

    exact_flip_match = False


    if has_ap_flip:

        ds_flip = pydicom.dcmread(
            existing_flip
        )

        candidate = (
            ds_flip.pixel_array
        )

        if candidate.ndim > 2:
            candidate = candidate[0]

        if candidate.shape == raw_pixels.shape:

            exact_flip_match = (
                np.array_equal(
                    candidate,
                    raw_pixels[:, ::-1],
                )
            )


    # --------------------------------------------------------
    # QC figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 10),
    )


    plot_candidate(
        axes[0],
        display_raw,
        canonical_landmarks,
        (
            f"{case} AP — KEEP IMAGE\n"
            f"edge score={score_raw:.6f}"
        ),
    )


    plot_candidate(
        axes[1],
        display_flip,
        canonical_landmarks,
        (
            f"{case} AP — FLIP IMAGE\n"
            f"edge score={score_flip:.6f}"
        ),
    )


    fig.suptitle(
        (
            f"{case} | "
            f"landmark={landmark_action} | "
            f"corr {original_corr:+.4f} "
            f"→ {canonical_corr:+.4f} | "
            f"auto image guess={image_guess} "
            f"({confidence})"
        ),
        fontsize=11,
    )


    fig.tight_layout()


    qc_path = (
        QC
        / f"{case}_AP_orientation.png"
    )


    fig.savefig(
        qc_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    rows.append(
        {
            "case": case,

            "width": W,
            "height": H,

            "photometric":
                str(
                    getattr(
                        ds,
                        "PhotometricInterpretation",
                        "",
                    )
                ),

            "view_position":
                str(
                    getattr(
                        ds,
                        "ViewPosition",
                        "",
                    )
                ),

            "patient_orientation":
                str(
                    getattr(
                        ds,
                        "PatientOrientation",
                        "",
                    )
                ),

            "original_corr_u_x":
                original_corr,

            "landmark_action":
                landmark_action,

            "canonical_corr_u_x":
                canonical_corr,

            "edge_score_keep_image":
                score_raw,

            "edge_score_flip_image":
                score_flip,

            "image_action_guess":
                image_guess,

            "guess_ratio":
                ratio,

            "guess_confidence":
                confidence,

            "has_AP_flip_dcm":
                has_ap_flip,

            "AP_flip_exact_horizontal_flip":
                exact_flip_match,

            "qc":
                str(qc_path),
        }
    )


# ============================================================
# Save
# ============================================================

df = pd.DataFrame(
    rows
)

csv_path = (
    OUT
    / "ap_canonical_audit.csv"
)

df.to_csv(
    csv_path,
    index=False,
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 120)
print("AP CANONICAL AUDIT")
print("=" * 120)

cols = [
    "case",
    "original_corr_u_x",
    "landmark_action",
    "canonical_corr_u_x",
    "edge_score_keep_image",
    "edge_score_flip_image",
    "image_action_guess",
    "guess_ratio",
    "guess_confidence",
    "has_AP_flip_dcm",
    "AP_flip_exact_horizontal_flip",
]

with pd.option_context(
    "display.max_rows",
    100,
    "display.max_columns",
    None,
    "display.width",
    220,
):

    print(
        df[cols].to_string(
            index=False
        )
    )


print()
print("=" * 120)
print("LANDMARK ACTION")
print("=" * 120)

print()
print("KEEP:")

print(
    ", ".join(
        df.loc[
            df["landmark_action"]
            == "KEEP",
            "case",
        ]
    )
)


print()
print("FLIP:")

print(
    ", ".join(
        df.loc[
            df["landmark_action"]
            == "FLIP",
            "case",
        ]
    )
)


print()
print("=" * 120)
print("IMAGE ACTION — AUTOMATIC HEURISTIC ONLY")
print("=" * 120)

for _, r in df.iterrows():

    print(
        f"{r['case']}: "
        f"{r['image_action_guess']:4s} "
        f"confidence={r['guess_confidence']:6s} "
        f"ratio={r['guess_ratio']:.3f}"
    )


print()
print("CSV:", csv_path)
print("QC :", QC)


#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from diffdrr.metrics import (
    MultiscaleNormalizedCrossCorrelation2d,
    GradientNormalizedCrossCorrelation2d,
)

from xvr.utils import XrayTransforms

"""
python tools/vaild_image.py \
  --case case01 \
  --registration-dir outputs/vertebra/1001/ap_registration_again_train1500_2000_100
"""


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "First-layer AP registration validation: "
            "image similarity + visual overlay."
        )
    )

    parser.add_argument(
        "--case",
        required=True,
        help="Case ID, e.g. case01, 1, or 1001",
    )

    parser.add_argument(
        "--registration-dir",
        required=True,
        type=Path,
        help=(
            "Registration output directory, e.g. "
            "outputs/vertebra/1001/"
            "ap_registration_again_train1500_2000_100"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output image path. "
            "Default: <registration-dir>/AP/image_validation.png"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device, e.g. cuda or cpu. "
            "Default: cuda if available, otherwise cpu."
        ),
    )

    return parser.parse_args()


def normalize_case(case: str):
    """
    Supported forms:

        case01 -> case01
        1      -> case01
        1001   -> case01
    """

    value = str(case).strip().lower()

    if value.startswith("case"):
        index = int(value[4:])
    else:
        number = int(value)

        if number >= 1000:
            index = number - 1000
        else:
            index = number

    if index <= 0:
        raise ValueError(
            f"Invalid case: {case}"
        )

    return f"case{index:02d}"


# ============================================================
# ZNCC
# ============================================================

def zncc(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    a = a - a.mean()
    b = b - b.mean()

    den = np.sqrt(
        np.sum(a * a)
        *
        np.sum(b * b)
    )

    if den < 1e-12:
        return float("nan")

    return float(
        np.sum(a * b) / den
    )


# ============================================================
# Visualization normalization
# ============================================================

def norm(x):
    x = x.astype(np.float32)

    x = x - x.min()

    max_value = x.max()

    if max_value > 0:
        x = x / max_value

    return x


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    case_id = normalize_case(
        args.case
    )

    registration_dir = (
        args.registration_dir.resolve()
    )

    base = (
        registration_dir
        / "AP"
    )

    gt_path = base / "gt.png"
    init_path = base / "init_img.png"
    final_path = base / "final_img.png"

    for path in [
        gt_path,
        init_path,
        final_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required image not found: {path}"
            )

    if args.output is None:
        out_path = (
            base
            / "image_validation.png"
        )
    else:
        out_path = args.output.resolve()

    device = torch.device(
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    # ========================================================
    # 1. Load images
    # ========================================================

    gt_pil = Image.open(
        gt_path
    ).convert("L")

    init_pil = Image.open(
        init_path
    ).convert("L")

    final_pil = Image.open(
        final_path
    ).convert("L")

    print("=" * 80)
    print("FIRST-LAYER IMAGE VALIDATION")
    print("=" * 80)

    print("case             =", case_id)
    print(
        "registration_dir =",
        registration_dir,
    )
    print("device           =", device)

    print()
    print(
        "===== ORIGINAL IMAGE SIZES ====="
    )

    print(
        "GT    =",
        gt_pil.size[::-1],
    )

    print(
        "INIT  =",
        init_pil.size[::-1],
    )

    print(
        "FINAL =",
        final_pil.size[::-1],
    )

    # DRR actual size
    W, H = init_pil.size

    if final_pil.size != init_pil.size:
        raise ValueError(
            "INIT and FINAL DRR sizes differ: "
            f"INIT={init_pil.size}, "
            f"FINAL={final_pil.size}"
        )

    # Resize real X-ray to DRR resolution
    gt_small_pil = gt_pil.resize(
        (W, H),
        Image.Resampling.BILINEAR,
    )

    gt = np.asarray(
        gt_small_pil,
        dtype=np.float32,
    )

    init_img = np.asarray(
        init_pil,
        dtype=np.float32,
    )

    final_img = np.asarray(
        final_pil,
        dtype=np.float32,
    )

    print()
    print(
        "===== COMPARISON SIZE ====="
    )

    print("GT    =", gt.shape)
    print("INIT  =", init_img.shape)
    print("FINAL =", final_img.shape)

    # ========================================================
    # 2. Ordinary ZNCC
    # ========================================================

    init_zncc = zncc(
        gt,
        init_img,
    )

    final_zncc = zncc(
        gt,
        final_img,
    )

    # ========================================================
    # 3. XVR / DiffDRR MNCC + GNCC
    # ========================================================

    def to_tensor(x):
        return (
            torch.from_numpy(
                x.copy()
            )[None, None]
            .float()
            .to(device)
            / 255.0
        )

    gt_t = to_tensor(gt)

    init_t = to_tensor(
        init_img
    )

    final_t = to_tensor(
        final_img
    )

    # Same preprocessing idea used by XVR registration
    transform = XrayTransforms(
        H,
        W,
        equalize=False,
    )

    gt_t = transform(gt_t)
    init_t = transform(init_t)
    final_t = transform(final_t)

    mncc_metric = (
        MultiscaleNormalizedCrossCorrelation2d(
            [None, 9],
            [0.5, 0.5],
        )
    )

    gncc_metric = (
        GradientNormalizedCrossCorrelation2d(
            11,
            0.0,
        ).to(device)
    )

    with torch.no_grad():

        init_mncc = (
            mncc_metric(
                gt_t,
                init_t,
            ).item()
        )

        final_mncc = (
            mncc_metric(
                gt_t,
                final_t,
            ).item()
        )

        init_gncc = (
            gncc_metric(
                gt_t,
                init_t,
            ).item()
        )

        final_gncc = (
            gncc_metric(
                gt_t,
                final_t,
            ).item()
        )

        init_combined = (
            0.5 * init_mncc
            +
            0.5 * init_gncc
        )

        final_combined = (
            0.5 * final_mncc
            +
            0.5 * final_gncc
        )

    # ========================================================
    # Print metrics
    # ========================================================

    print()
    print("=" * 80)
    print("IMAGE SIMILARITY")
    print("=" * 80)

    print()
    print("===== INIT =====")

    print(
        f"ZNCC     = "
        f"{init_zncc:.6f}"
    )

    print(
        f"MNCC     = "
        f"{init_mncc:.6f}"
    )

    print(
        f"GNCC     = "
        f"{init_gncc:.6f}"
    )

    print(
        f"COMBINED = "
        f"{init_combined:.6f}"
    )

    print()
    print("===== FINAL =====")

    print(
        f"ZNCC     = "
        f"{final_zncc:.6f}"
    )

    print(
        f"MNCC     = "
        f"{final_mncc:.6f}"
    )

    print(
        f"GNCC     = "
        f"{final_gncc:.6f}"
    )

    print(
        f"COMBINED = "
        f"{final_combined:.6f}"
    )

    print()
    print("===== IMPROVEMENT =====")

    print(
        f"dZNCC     = "
        f"{final_zncc - init_zncc:+.6f}"
    )

    print(
        f"dMNCC     = "
        f"{final_mncc - init_mncc:+.6f}"
    )

    print(
        f"dGNCC     = "
        f"{final_gncc - init_gncc:+.6f}"
    )

    print(
        f"dCOMBINED = "
        f"{final_combined - init_combined:+.6f}"
    )

    # ========================================================
    # 4. Visualization
    # ========================================================

    g = norm(gt)
    i = norm(init_img)
    f = norm(final_img)

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(12, 16),
    )

    # --------------------------------------------------------
    # Top row
    # --------------------------------------------------------

    ax[0, 0].imshow(
        g,
        cmap="gray",
    )

    ax[0, 0].set_title(
        "GT AP (resized)"
    )

    ax[0, 0].axis("off")

    ax[0, 1].imshow(
        i,
        cmap="gray",
    )

    ax[0, 1].set_title(
        "INIT DRR"
    )

    ax[0, 1].axis("off")

    ax[0, 2].imshow(
        f,
        cmap="gray",
    )

    ax[0, 2].set_title(
        "FINAL DRR"
    )

    ax[0, 2].axis("off")

    # --------------------------------------------------------
    # Bottom row overlays
    # --------------------------------------------------------

    ax[1, 0].imshow(
        g,
        cmap="gray",
    )

    ax[1, 0].imshow(
        i,
        cmap="jet",
        alpha=0.35,
    )

    ax[1, 0].set_title(
        "Overlay: GT + INIT"
    )

    ax[1, 0].axis("off")

    ax[1, 1].imshow(
        g,
        cmap="gray",
    )

    ax[1, 1].imshow(
        f,
        cmap="jet",
        alpha=0.35,
    )

    ax[1, 1].set_title(
        "Overlay: GT + FINAL"
    )

    ax[1, 1].axis("off")

    # --------------------------------------------------------
    # Metrics panel
    # --------------------------------------------------------

    ax[1, 2].axis("off")

    ax[1, 2].text(
        0.02,
        0.95,
        (
            "IMAGE SIMILARITY\n\n"

            "INIT\n"
            f"ZNCC = {init_zncc:.4f}\n"
            f"MNCC = {init_mncc:.4f}\n"
            f"GNCC = {init_gncc:.4f}\n"
            f"Combined = {init_combined:.4f}\n\n"

            "FINAL\n"
            f"ZNCC = {final_zncc:.4f}\n"
            f"MNCC = {final_mncc:.4f}\n"
            f"GNCC = {final_gncc:.4f}\n"
            f"Combined = {final_combined:.4f}\n\n"

            "IMPROVEMENT\n"
            f"dZNCC = "
            f"{final_zncc - init_zncc:+.4f}\n"
            f"dMNCC = "
            f"{final_mncc - init_mncc:+.4f}\n"
            f"dGNCC = "
            f"{final_gncc - init_gncc:+.4f}\n"
            f"dCombined = "
            f"{final_combined - init_combined:+.4f}"
        ),
        va="top",
        fontsize=13,
    )

    fig.suptitle(
        (
            "First-layer registration validation\n"
            f"{case_id} | {registration_dir}"
        ),
        fontsize=14,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        "WROTE:",
        out_path,
    )


if __name__ == "__main__":
    main()

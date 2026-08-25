#!/usr/bin/env python3
"""First-layer visual validation for an XVR registration result.

The script creates ``image_validation.png`` from the images saved by
``xvr register ... --saveimg``.  It is intentionally a visual QC step only;
quantitative landmark error belongs to ``validate_registration_mtre.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def normalize_case_id(value: str) -> str:
    value = str(value).strip()
    lower = value.lower()
    if lower.startswith("case"):
        suffix = value[4:]
    else:
        suffix = value
    if suffix.isdigit():
        return f"case{int(suffix):02d}"
    return value


def resolve_result_dir(registration_dir: Path, view: str) -> Path:
    registration_dir = registration_dir.resolve()
    direct = registration_dir / "parameters.pt"
    nested = registration_dir / view / "parameters.pt"
    if direct.is_file():
        return registration_dir
    if nested.is_file():
        return registration_dir / view
    raise FileNotFoundError(
        "Could not find parameters.pt. Expected either "
        f"{direct} or {nested}."
    )


def load_gray(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Re-run registration with --saveimg before first-layer QC."
        )
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return image


def overlay(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Red=reference X-ray, green=DRR; agreement appears yellow."""
    rgb = np.zeros((*reference.shape, 3), dtype=np.float32)
    rgb[..., 0] = reference
    rgb[..., 1] = moving
    return np.clip(rgb, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-layer XVR visual QC: create image_validation.png."
    )
    parser.add_argument(
        "--registration-dir",
        required=True,
        type=Path,
        help="Registration output root, e.g. outputs/vertebra/case01_ap_registration_auto_xy0",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Dataset root containing caseXX folders, e.g. data/vertebra/dataset/test_corrected",
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Case identifier, e.g. case01 or 1",
    )
    parser.add_argument(
        "--view",
        default="AP",
        choices=["AP"],
        help="Validated view. Only AP is currently supported.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Default: <result>/<view>/image_validation.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_id = normalize_case_id(args.case)
    case_dir = (args.dataset_root / case_id).resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    dicom_path = case_dir / f"{args.view}.dcm"
    landmarks_path = case_dir / "landmarks.json"
    if not dicom_path.is_file():
        raise FileNotFoundError(f"Expected DICOM not found: {dicom_path}")
    if not landmarks_path.is_file():
        raise FileNotFoundError(f"Expected landmarks.json not found: {landmarks_path}")

    result_dir = resolve_result_dir(args.registration_dir, args.view)
    params_path = result_dir / "parameters.pt"
    params = torch.load(params_path, map_location="cpu", weights_only=False)

    orientation = str(params.get("drr", {}).get("orientation", "")).upper()
    if orientation and orientation != args.view:
        raise ValueError(
            f"Registration orientation is {orientation!r}, but --view is {args.view!r}."
        )

    saved_xray = Path(str(params.get("xray", {}).get("filename", "")))
    if saved_xray.name and saved_xray.name != dicom_path.name:
        raise ValueError(
            f"Registration used {saved_xray.name!r}, but selected case/view expects "
            f"{dicom_path.name!r}."
        )

    gt = load_gray(result_dir / "gt.png")
    init = load_gray(result_dir / "init_img.png")
    final = load_gray(result_dir / "final_img.png")
    if gt.shape != init.shape or gt.shape != final.shape:
        raise ValueError(
            f"Saved image shapes differ: gt={gt.shape}, init={init.shape}, final={final.shape}"
        )

    diff = np.abs(gt - final)
    output = args.output.resolve() if args.output is not None else result_dir / "image_validation.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    panels = [
        (gt, "Registered X-ray", "gray"),
        (init, "Initial DRR", "gray"),
        (final, "Final DRR", "gray"),
        (overlay(gt, init), "X-ray / Initial overlay\nred=X-ray, green=DRR", None),
        (overlay(gt, final), "X-ray / Final overlay\nred=X-ray, green=DRR", None),
        (diff, "|X-ray - Final DRR|", "gray"),
    ]
    for ax, (image, title, cmap) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"First-layer registration validation | {case_id} {args.view}\n"
        f"{args.registration_dir}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=" * 80)
    print("FIRST-LAYER VISUAL VALIDATION")
    print("=" * 80)
    print("case             =", case_id)
    print("view             =", args.view)
    print("registration     =", args.registration_dir)
    print("result directory =", result_dir)
    print("DICOM            =", dicom_path)
    print("output           =", output)
    print("image shape      =", gt.shape)


if __name__ == "__main__":
    main()

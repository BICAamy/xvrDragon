#!/usr/bin/env python3
"""First-layer visual validation for an XVR registration result.

The script creates ``image_validation.png`` from the images saved by
``xvr register ... --saveimg``. XVR saves ``gt.png`` at the original X-ray
resolution, while ``init_img.png`` and ``final_img.png`` are generated from the
DRR detector left at the final multiscale registration resolution. Therefore,
for overlay/difference QC, the X-ray is resized to the saved DRR resolution.

This is intentionally a visual QC step only; quantitative landmark error belongs
to ``validate_registration_mtre.py``.

example:

python tools/validate_registration_image.py \
  --registration-dir outputs/vertebra/case01_ap_registration_auto_xy0 \
  --case case01

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def resolve_case_dir(
    params: dict,
    dataset_root: Path | None,
    case_id: str,
    view: str,
) -> tuple[Path, Path]:
    """Resolve the selected case and DICOM, optionally from saved registration metadata."""
    saved_value = params.get("xray", {}).get("filename")
    saved_xray = Path(str(saved_value)).expanduser() if saved_value else None

    if dataset_root is None:
        if saved_xray is None or not saved_xray.is_file():
            raise FileNotFoundError(
                "--dataset-root was omitted, but the X-ray path stored in parameters.pt "
                f"cannot be resolved: {saved_xray}. Pass --dataset-root explicitly."
            )
        case_dir = saved_xray.resolve().parent
        dicom_path = saved_xray.resolve()
    else:
        case_dir = (dataset_root / case_id).resolve()
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        dicom_path = case_dir / f"{view}.dcm"
        if not dicom_path.is_file():
            raise FileNotFoundError(f"Expected DICOM not found: {dicom_path}")

        # If the original registration X-ray still exists, require an exact dataset match.
        # This prevents accidentally validating a test_corrected result against test_reverse,
        # or vice versa, when both contain a file named AP.dcm.
        if saved_xray is not None and saved_xray.is_file():
            if saved_xray.resolve() != dicom_path.resolve():
                raise ValueError(
                    "Selected dataset/case does not match the X-ray stored in parameters.pt:\n"
                    f"  registered: {saved_xray.resolve()}\n"
                    f"  selected:   {dicom_path.resolve()}"
                )

    if normalize_case_id(case_dir.name) != case_id:
        raise ValueError(
            f"Registration resolves to case directory {case_dir.name!r}, "
            f"but --case requested {case_id!r}."
        )
    if dicom_path.name != f"{view}.dcm":
        raise ValueError(
            f"Registration resolves to DICOM {dicom_path.name!r}, "
            f"but --view expects {view}.dcm."
        )

    return case_dir, dicom_path


def load_gray(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Re-run registration with --saveimg before first-layer QC."
        )
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def resize_gray(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D grayscale image to (height, width) for visual comparison."""
    if image.shape == target_shape:
        return image
    tensor = torch.from_numpy(image)[None, None]
    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


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
        type=Path,
        default=None,
        help=(
            "Optional dataset root containing caseXX folders. If omitted, the script "
            "uses the X-ray path stored in parameters.pt."
        ),
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

    result_dir = resolve_result_dir(args.registration_dir, args.view)
    params_path = result_dir / "parameters.pt"
    params = torch.load(params_path, map_location="cpu", weights_only=False)

    orientation = str(params.get("drr", {}).get("orientation", "")).upper()
    if orientation and orientation != args.view:
        raise ValueError(
            f"Registration orientation is {orientation!r}, but --view is {args.view!r}."
        )

    case_dir, dicom_path = resolve_case_dir(
        params,
        args.dataset_root,
        case_id,
        args.view,
    )
    landmarks_path = case_dir / "landmarks.json"
    if not landmarks_path.is_file():
        raise FileNotFoundError(f"Expected landmarks.json not found: {landmarks_path}")

    gt = load_gray(result_dir / "gt.png")
    init = load_gray(result_dir / "init_img.png")
    final = load_gray(result_dir / "final_img.png")

    # init/final DRRs must share one detector grid. gt.png is expected to be larger
    # when multiscale registration ends at a downsampled detector resolution.
    if init.shape != final.shape:
        raise ValueError(
            f"Saved DRR shapes differ: init={init.shape}, final={final.shape}."
        )

    comparison_shape = final.shape
    gt_compare = resize_gray(gt, comparison_shape)
    diff = np.abs(gt_compare - final)

    output = args.output.resolve() if args.output is not None else result_dir / "image_validation.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    panels = [
        (gt, f"Registered X-ray (saved full resolution)\n{gt.shape[1]} x {gt.shape[0]}", "gray"),
        (init, f"Initial DRR\n{init.shape[1]} x {init.shape[0]}", "gray"),
        (final, f"Final DRR\n{final.shape[1]} x {final.shape[0]}", "gray"),
        (
            overlay(gt_compare, init),
            "X-ray / Initial overlay at DRR resolution\nred=X-ray, green=DRR",
            None,
        ),
        (
            overlay(gt_compare, final),
            "X-ray / Final overlay at DRR resolution\nred=X-ray, green=DRR",
            None,
        ),
        (diff, "|resized X-ray - Final DRR|", "gray"),
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
    print("case                =", case_id)
    print("view                =", args.view)
    print("registration        =", args.registration_dir)
    print("result directory    =", result_dir)
    print("DICOM               =", dicom_path)
    print("output              =", output)
    print("saved X-ray shape    =", gt.shape)
    print("saved DRR shape      =", final.shape)
    if gt.shape != final.shape:
        print("comparison X-ray     = resized to", final.shape)
    else:
        print("comparison X-ray     = no resize needed")


if __name__ == "__main__":
    main()

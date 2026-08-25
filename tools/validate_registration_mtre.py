#!/usr/bin/env python3
"""Second-layer validation for an XVR AP registration result.

Reports final-pose landmark mTRE together with image-similarity metrics on the
same last multiscale detector grid used by XVR: ZNCC, MNCC, GNCC and Combined.
Supports both AP detector conventions, including ``reverse_x_axis=True``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
    NormalizedCrossCorrelation2d,
)
from diffdrr.pose import RigidTransform

from xvr.io import read_xray
from xvr.metrics import project_fiducials_to_image_pixels
from xvr.utils import XrayTransforms


MNCC_PATCH_SIZE = 9
GNCC_PATCH_SIZE = 11
GNCC_SIGMA = 0.0
COMBINED_BETA = 0.5


def normalize_case_id(value: str) -> str:
    value = str(value).strip()
    suffix = value[4:] if value.lower().startswith("case") else value
    if suffix.isdigit():
        return f"case{int(suffix):02d}"
    return value


def numeric_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


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


def resolve_case_file(saved_path, case_dir: Path, label: str) -> Path | None:
    if saved_path is None:
        return None
    saved = Path(str(saved_path)).expanduser()
    if saved.is_file():
        return saved.resolve()
    candidate = case_dir / saved.name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve {label}. Saved path={saved}; fallback={candidate}."
    )


def load_landmarks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        ct_lps = data["ct"]["landmarks_lps_mm"]
        ap_pixels = data["views"]["ap"]["landmarks_px"]
    except KeyError as exc:
        raise KeyError(
            f"{path} must contain ct.landmarks_lps_mm and views.ap.landmarks_px"
        ) from exc

    rows = []
    for vertebra in sorted(set(ct_lps) & set(ap_pixels), key=numeric_key):
        for point_id in sorted(
            set(ct_lps[vertebra]) & set(ap_pixels[vertebra]), key=numeric_key
        ):
            xyz = ct_lps[vertebra][point_id]
            uv = ap_pixels[vertebra][point_id]
            if len(xyz) != 3 or len(uv) != 2:
                raise ValueError(
                    f"Invalid landmark {vertebra}.{point_id}: 3D={xyz!r}, 2D={uv!r}"
                )
            rows.append(
                {
                    "name": f"{vertebra}.{point_id}",
                    "vertebra": int(vertebra) if str(vertebra).isdigit() else vertebra,
                    "point": int(point_id) if str(point_id).isdigit() else point_id,
                    "lps": [float(v) for v in xyz],
                    "gt": [float(v) for v in uv],
                }
            )
    if not rows:
        raise ValueError(f"No paired CT/AP landmarks found in {path}")
    return rows


def relative_registration_scales(scale_values, crop: int, height: int) -> list[float]:
    """Reproduce xvr.registrar.base._parse_scales exactly."""
    if isinstance(scale_values, str):
        scale_values = [x for x in scale_values.split(",") if x.strip()]
    pyramid = [1.0] + [
        float(x) * (height / (height + crop)) for x in scale_values
    ]
    return [
        pyramid[idx] / pyramid[idx + 1]
        for idx in range(len(pyramid) - 1)
    ]


def scalar_metric(metric, a: torch.Tensor, b: torch.Tensor) -> float:
    return float(metric(a, b).detach().double().mean().item())


def extract_stored_final_combined(params: dict) -> float | None:
    trajectory = params.get("trajectory")
    if trajectory is None:
        return None
    try:
        if hasattr(trajectory, "columns") and "ncc" in trajectory.columns:
            return float(trajectory["ncc"].iloc[-1])
    except Exception:
        return None
    if isinstance(trajectory, dict):
        values = trajectory.get("ncc")
        try:
            if values is not None and len(values):
                return float(values[-1])
        except Exception:
            return None
    return None


def compute_final_similarity(
    drr: DRR,
    pose: RigidTransform,
    params: dict,
    dicom_path: Path,
    full_height: int,
    full_width: int,
    device: torch.device,
) -> dict:
    """Recompute final similarity on the last detector grid used by XVR."""
    xray_cfg = params.get("xray", {})
    optimization = params.get("optimization", {})

    xray, *_ = read_xray(
        dicom_path,
        int(xray_cfg.get("crop", 0)),
        bool(xray_cfg.get("subtract_background", False)),
        bool(xray_cfg.get("linearize", False)),
        xray_cfg.get("reducefn", "max"),
    )
    if tuple(xray.shape[-2:]) != (full_height, full_width):
        raise ValueError(
            "Preprocessed X-ray shape does not match saved full detector size: "
            f"xray={tuple(xray.shape[-2:])}, saved={(full_height, full_width)}"
        )

    scales = relative_registration_scales(
        optimization.get("scales", ["8"]),
        int(xray_cfg.get("crop", 0)),
        full_height,
    )
    if not scales:
        raise ValueError("No registration scales found in parameters.pt")

    for scale in scales:
        drr.rescale_detector_(scale)

    sim_height = int(drr.detector.height)
    sim_width = int(drr.detector.width)
    transform = XrayTransforms(
        sim_height,
        sim_width,
        equalize=bool(optimization.get("equalize", False)),
    )

    with torch.no_grad():
        xray_sim = transform(xray).to(device)
        drr_sim = transform(drr(pose))

        zncc = scalar_metric(
            NormalizedCrossCorrelation2d(None).to(device), xray_sim, drr_sim
        )
        mncc = scalar_metric(
            MultiscaleNormalizedCrossCorrelation2d(
                [None, MNCC_PATCH_SIZE], [0.5, 0.5]
            ).to(device),
            xray_sim,
            drr_sim,
        )
        gncc = scalar_metric(
            GradientNormalizedCrossCorrelation2d(
                GNCC_PATCH_SIZE, GNCC_SIGMA
            ).to(device),
            xray_sim,
            drr_sim,
        )
        combined = COMBINED_BETA * mncc + (1.0 - COMBINED_BETA) * gncc

    stored_combined = extract_stored_final_combined(params)
    return {
        "grid_height": sim_height,
        "grid_width": sim_width,
        "zncc": float(zncc),
        "mncc": float(mncc),
        "gncc": float(gncc),
        "combined": float(combined),
        "combined_formula": "0.5 * MNCC + 0.5 * GNCC",
        "mncc_patch_sizes": [None, MNCC_PATCH_SIZE],
        "mncc_patch_weights": [0.5, 0.5],
        "gncc_patch_size": GNCC_PATCH_SIZE,
        "sigma": GNCC_SIGMA,
        "equalize": bool(optimization.get("equalize", False)),
        "stored_final_combined": stored_combined,
        "combined_minus_stored": (
            None if stored_combined is None else float(combined - stored_combined)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-layer XVR validation: final 2D projected mTRE plus "
            "final ZNCC/MNCC/GNCC/Combined image similarity."
        )
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
            "Optional dataset root containing caseXX folders. If omitted, use the "
            "X-ray path stored in parameters.pt."
        ),
    )
    parser.add_argument("--case", required=True, help="Case identifier, e.g. case01 or 1")
    parser.add_argument(
        "--view",
        default="AP",
        choices=["AP"],
        help="Validated view. Only AP is currently supported.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda or cpu. Default: cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Detailed landmark CSV output. Default: <result>/<view>/mTRE_2d.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Summary JSON output. Default: <result>/<view>/mTRE_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_id = normalize_case_id(args.case)

    result_dir = resolve_result_dir(args.registration_dir, args.view)
    params_path = result_dir / "parameters.pt"
    params = torch.load(params_path, map_location="cpu", weights_only=False)
    cfg = params["drr"]

    case_dir, dicom_path = resolve_case_dir(
        params, args.dataset_root, case_id, args.view
    )
    landmarks_path = case_dir / "landmarks.json"
    if not landmarks_path.is_file():
        raise FileNotFoundError(f"landmarks.json not found: {landmarks_path}")

    orientation = str(cfg["orientation"]).upper()
    if orientation != "AP" or args.view != "AP":
        raise NotImplementedError(
            f"This validator is currently validated only for AP; got orientation={orientation!r}."
        )
    if int(params.get("xray", {}).get("crop", 0)) != 0:
        raise ValueError(
            "This validator requires crop=0 because landmarks.json stores full-raster coordinates."
        )
    if bool(params.get("pf_to_af", False)):
        raise ValueError(
            "The registration applied a PF->AF horizontal flip; this AP validator refuses "
            "to silently mix that convention."
        )
    if params.get("final_pose") is None:
        raise RuntimeError("final_pose is None; no final registration pose is available.")

    reverse_x_axis = bool(cfg["reverse_x_axis"])
    rows = load_landmarks(landmarks_path)
    lps = np.asarray([r["lps"] for r in rows], dtype=np.float32)
    gt = np.asarray([r["gt"] for r in rows], dtype=np.float32)

    width = int(cfg["width"])
    height = int(cfg["height"])
    if (
        (gt[:, 0] < 0).any()
        or (gt[:, 0] >= width).any()
        or (gt[:, 1] < 0).any()
        or (gt[:, 1] >= height).any()
    ):
        raise ValueError(
            f"At least one AP landmark lies outside the saved {width}x{height} raster."
        )

    ras = lps.copy()
    ras[:, 0] *= -1
    ras[:, 1] *= -1
    fiducials_ras = torch.tensor(ras, dtype=torch.float32)[None]

    volume = resolve_case_file(cfg.get("volume"), case_dir, "volume")
    mask = resolve_case_file(cfg.get("mask"), case_dir, "mask")

    labels = cfg.get("labels")
    if isinstance(labels, str):
        labels = [int(x) for x in labels.split(",") if x.strip()]

    subject = read(
        str(volume),
        str(mask) if mask is not None else None,
        labels,
        orientation,
        fiducials=fiducials_ras,
        **cfg.get("read_kwargs", {}),
    )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    drr = DRR(
        subject,
        float(cfg["sdd"]),
        height,
        float(cfg["delx"]),
        width,
        float(cfg["dely"]),
        float(cfg["x0"]),
        float(cfg["y0"]),
        reverse_x_axis=reverse_x_axis,
        renderer=cfg["renderer"],
        **cfg.get("drr_kwargs", {}),
    ).to(device)

    pose = RigidTransform(params["final_pose"].float().to(device))
    fiducials = subject.fiducials.to(device)

    with torch.no_grad():
        pred = (
            project_fiducials_to_image_pixels(
                drr,
                pose,
                fiducials,
                orientation="AP",
                reverse_x_axis=reverse_x_axis,
            )[0]
            .cpu()
            .numpy()
        )

    delta = pred - gt
    du = delta[:, 0]
    dv = delta[:, 1]
    error_px = np.sqrt(du**2 + dv**2)
    delx = float(cfg["delx"])
    dely = float(cfg["dely"])
    error_mm = np.sqrt((du * delx) ** 2 + (dv * dely) ** 2)

    for i, row in enumerate(rows):
        row.update(
            {
                "gt_x": float(gt[i, 0]),
                "gt_y": float(gt[i, 1]),
                "pred_x": float(pred[i, 0]),
                "pred_y": float(pred[i, 1]),
                "du_px": float(du[i]),
                "dv_px": float(dv[i]),
                "error_px": float(error_px[i]),
                "error_mm": float(error_mm[i]),
            }
        )

    similarity = compute_final_similarity(
        drr,
        pose,
        params,
        dicom_path,
        height,
        width,
        device,
    )

    summary = {
        "case": case_id,
        "view": "AP",
        "reverse_x_axis": reverse_x_axis,
        "registration_dir": str(args.registration_dir),
        "result_dir": str(result_dir),
        "parameters": str(params_path),
        "dicom": str(dicom_path),
        "landmarks": str(landmarks_path),
        "n_landmarks": int(len(rows)),
        "mtre_px": float(error_px.mean()),
        "mtre_mm": float(error_mm.mean()),
        "median_mm": float(np.median(error_mm)),
        "max_mm": float(error_mm.max()),
        "mean_du_px": float(du.mean()),
        "mean_dv_px": float(dv.mean()),
        "delx_mm_per_px": delx,
        "dely_mm_per_px": dely,
        "similarity": similarity,
    }

    print("=" * 80)
    print("SECOND-LAYER FINAL 2D PROJECTED mTRE")
    print("=" * 80)
    print(f"case          = {case_id}")
    print(f"reverse_x_axis= {reverse_x_axis}")
    print(f"N             = {len(rows)}")
    print(f"mTRE          = {summary['mtre_px']:.3f} px")
    print(f"mTRE          = {summary['mtre_mm']:.3f} mm")
    print(f"median        = {summary['median_mm']:.3f} mm")
    print(f"max           = {summary['max_mm']:.3f} mm")
    print(f"mean du       = {summary['mean_du_px']:+.3f} px")
    print(f"mean dv       = {summary['mean_dv_px']:+.3f} px")

    print("\n" + "=" * 80)
    print("FINAL IMAGE SIMILARITY  (higher is better)")
    print("=" * 80)
    print(
        f"grid          = {similarity['grid_width']} x "
        f"{similarity['grid_height']} px"
    )
    print(f"ZNCC          = {similarity['zncc']:.6f}")
    print(f"MNCC          = {similarity['mncc']:.6f}")
    print(f"GNCC          = {similarity['gncc']:.6f}")
    print(f"Combined      = {similarity['combined']:.6f}")
    print("formula       = 0.5 * MNCC + 0.5 * GNCC")
    print("MNCC config   = patches=[None, 9], weights=[0.5, 0.5]")
    print("GNCC config   = patch=11, sigma=0.0")
    if similarity["stored_final_combined"] is not None:
        print(
            f"stored final  = {similarity['stored_final_combined']:.6f} "
            "(parameters.pt trajectory[-1].ncc)"
        )
        print(f"recalc-stored = {similarity['combined_minus_stored']:+.6e}")

    print("\n" + "=" * 80)
    print("PER VERTEBRA")
    print("=" * 80)
    vertebra_values = sorted({r["vertebra"] for r in rows}, key=numeric_key)
    per_vertebra = {}
    for vertebra in vertebra_values:
        idx = np.array([r["vertebra"] == vertebra for r in rows], dtype=bool)
        v_mm = float(error_mm[idx].mean())
        v_px = float(error_px[idx].mean())
        per_vertebra[str(vertebra)] = {
            "mtre_mm": v_mm,
            "mtre_px": v_px,
            "n": int(idx.sum()),
        }
        print(f"V{vertebra}: {v_mm:.3f} mm ({v_px:.3f} px), N={int(idx.sum())}")
    summary["per_vertebra"] = per_vertebra

    csv_path = args.csv.resolve() if args.csv is not None else result_dir / "mTRE_2d.csv"
    json_path = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else result_dir / "mTRE_summary.json"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "vertebra",
        "point",
        "gt_x",
        "gt_y",
        "pred_x",
        "pred_y",
        "du_px",
        "dv_px",
        "error_px",
        "error_mm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\nCSV saved      =", csv_path)
    print("summary saved  =", json_path)


if __name__ == "__main__":
    main()

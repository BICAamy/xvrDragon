#!/usr/bin/env python3
"""Second-layer validation: final 2D projected landmark mTRE for XVR AP registration.

All paired CT/AP landmarks from ``landmarks.json`` are used.  DiffDRR internal
projection coordinates are converted to raster coordinates through the formal
``xvr.metrics.project_fiducials_to_image_pixels`` adapter; no manual W-x flip
is performed here.
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
from diffdrr.pose import RigidTransform

from xvr.metrics import project_fiducials_to_image_pixels


def normalize_case_id(value: str) -> str:
    value = str(value).strip()
    lower = value.lower()
    suffix = value[4:] if lower.startswith("case") else value
    if suffix.isdigit():
        return f"case{int(suffix):02d}"
    return value


def numeric_key(value: str):
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


def resolve_case_file(saved_path, case_dir: Path, label: str) -> Path | None:
    if saved_path is None:
        return None
    saved = Path(str(saved_path))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Second-layer XVR validation: compute final 2D projected mTRE."
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
        help="Detailed CSV output. Default: <result>/<view>/mTRE_2d.csv",
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
    case_dir = (args.dataset_root / case_id).resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    landmarks_path = case_dir / "landmarks.json"
    if not landmarks_path.is_file():
        raise FileNotFoundError(f"landmarks.json not found: {landmarks_path}")

    result_dir = resolve_result_dir(args.registration_dir, args.view)
    params_path = result_dir / "parameters.pt"
    params = torch.load(params_path, map_location="cpu", weights_only=False)
    cfg = params["drr"]

    orientation = str(cfg["orientation"]).upper()
    if orientation != "AP" or args.view != "AP":
        raise NotImplementedError(
            f"This validator is currently validated only for AP; got orientation={orientation!r}."
        )
    if bool(cfg["reverse_x_axis"]):
        raise ValueError(
            "Canonical AP mTRE validation requires reverse_x_axis=False. "
            "Refusing to silently mix detector conventions."
        )
    if int(params.get("xray", {}).get("crop", 0)) != 0:
        raise ValueError(
            "This validator requires crop=0 because landmarks.json stores full-raster coordinates."
        )
    if bool(params.get("pf_to_af", False)):
        raise ValueError(
            "The registration applied a PF->AF horizontal flip; this AP validator refuses "
            "to silently mix that raster convention."
        )
    if params.get("final_pose") is None:
        raise RuntimeError("final_pose is None; no final registration pose is available.")

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

    # landmarks.json is LPS; DiffDRR/NIfTI use RAS.
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
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
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
        reverse_x_axis=False,
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
                reverse_x_axis=False,
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

    summary = {
        "case": case_id,
        "view": "AP",
        "registration_dir": str(args.registration_dir),
        "result_dir": str(result_dir),
        "parameters": str(params_path),
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
    }

    print("=" * 80)
    print("SECOND-LAYER FINAL 2D PROJECTED mTRE")
    print("=" * 80)
    print(f"case          = {case_id}")
    print(f"N             = {len(rows)}")
    print(f"mTRE          = {summary['mtre_px']:.3f} px")
    print(f"mTRE          = {summary['mtre_mm']:.3f} mm")
    print(f"median        = {summary['median_mm']:.3f} mm")
    print(f"max           = {summary['max_mm']:.3f} mm")
    print(f"mean du       = {summary['mean_du_px']:+.3f} px")
    print(f"mean dv       = {summary['mean_dv_px']:+.3f} px")

    print("\n" + "=" * 80)
    print("PER VERTEBRA")
    print("=" * 80)
    vertebra_values = sorted({r["vertebra"] for r in rows}, key=numeric_key)
    per_vertebra = {}
    for vertebra in vertebra_values:
        idx = np.array([r["vertebra"] == vertebra for r in rows], dtype=bool)
        v_mm = float(error_mm[idx].mean())
        v_px = float(error_px[idx].mean())
        per_vertebra[str(vertebra)] = {"mtre_mm": v_mm, "mtre_px": v_px, "n": int(idx.sum())}
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

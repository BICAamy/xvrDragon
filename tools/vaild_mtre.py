#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import torch
from diffdrr.pose import RigidTransform

from xvr.metrics import project_fiducials_to_image_pixels
from xvr.renderer import initialize_drr


"""
example:

python tools/vaild_mtre.py \
  --case case01 \
  --registration-dir outputs/vertebra/1001/ap_registration_again_train1500_2000_100
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute final AP landmark projection mTRE."
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
            "outputs/vertebra/1001/ap_registration_again_train1500_2000_100"
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Optional case data directory containing landmarks.json. "
            "If omitted, it is inferred from --case."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV path. Default: "
            "<registration-dir>/AP/landmark_error.csv"
        ),
    )

    return parser.parse_args()


def normalize_case(case: str):
    """
    Supported:
        case01 -> case01, subject 1001
        1      -> case01, subject 1001
        1001   -> case01, subject 1001
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

    case_id = f"case{index:02d}"
    subject_id = str(1000 + index)

    return case_id, subject_id


def resolve_data_dir(
    case_id: str,
    subject_id: str,
    explicit_data_dir: Path | None,
):
    """
    Prefer explicit path.

    Otherwise support both repository layouts:

        data/vertebra/1001/
        data/vertebra/dataset/test_reverse/case01/
        data/vertebra/dataset/test_corrected/case01/
        data/vertebra/dataset/test/case01/
    """

    if explicit_data_dir is not None:
        path = explicit_data_dir.resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"--data-dir does not exist: {path}"
            )

        return path

    candidates = [
        Path("data/vertebra") / subject_id,
        Path("data/vertebra/dataset/test_reverse") / case_id,
        Path("data/vertebra/dataset/test_corrected") / case_id,
        Path("data/vertebra/dataset/test") / case_id,
    ]

    for path in candidates:
        if (
            path.is_dir()
            and (path / "landmarks.json").is_file()
        ):
            return path.resolve()

    msg = "\n".join(
        f"  - {p}"
        for p in candidates
    )

    raise FileNotFoundError(
        "Could not infer case data directory.\n"
        "Tried:\n"
        f"{msg}\n"
        "Use --data-dir to specify it explicitly."
    )


def stats(x: torch.Tensor):
    return {
        "mean": x.mean().item(),
        "median": x.median().item(),
        "rmse": torch.sqrt(
            torch.mean(x ** 2)
        ).item(),
        "max": x.max().item(),
    }


def main():
    args = parse_args()

    case_id, subject_id = normalize_case(
        args.case
    )

    registration_dir = (
        args.registration_dir.resolve()
    )

    params_path = (
        registration_dir
        / "AP"
        / "parameters.pt"
    )

    if not params_path.is_file():
        raise FileNotFoundError(
            f"parameters.pt not found: "
            f"{params_path}"
        )

    data_dir = resolve_data_dir(
        case_id,
        subject_id,
        args.data_dir,
    )

    landmarks_path = (
        data_dir
        / "landmarks.json"
    )

    if args.output_csv is None:
        out_csv = (
            registration_dir
            / "AP"
            / "landmark_error.csv"
        )
    else:
        out_csv = (
            args.output_csv.resolve()
        )

    # ========================================================
    # Load registration
    # ========================================================

    p = torch.load(
        params_path,
        weights_only=False,
    )

    if p["final_pose"] is None:
        raise RuntimeError(
            "parameters.pt does not contain "
            "final_pose"
        )

    cfg = p["drr"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pose = RigidTransform(
        p["final_pose"].to(device)
    )

    print("=" * 80)
    print("REGISTRATION")
    print("=" * 80)

    print("case         =", case_id)
    print("subject      =", subject_id)
    print("data_dir     =", data_dir)
    print("parameters   =", params_path)
    print("orientation  =", cfg["orientation"])
    print("sdd          =", cfg["sdd"])
    print("delx         =", cfg["delx"])
    print("dely         =", cfg["dely"])
    print("x0           =", cfg["x0"])
    print("y0           =", cfg["y0"])
    print(
        "reverse_x    =",
        cfg["reverse_x_axis"],
    )

    # ========================================================
    # Reconstruct detector geometry exactly as registration
    # ========================================================

    drr = initialize_drr(
        volume=str(cfg["volume"]),
        mask=(
            str(cfg["mask"])
            if cfg["mask"] is not None
            else None
        ),
        labels=cfg["labels"],
        orientation=cfg["orientation"],

        height=cfg["height"],
        width=cfg["width"],

        sdd=cfg["sdd"],
        delx=cfg["delx"],
        dely=cfg["dely"],

        x0=cfg["x0"],
        y0=cfg["y0"],

        reverse_x_axis=(
            cfg["reverse_x_axis"]
        ),
        renderer=cfg["renderer"],

        read_kwargs=cfg.get(
            "read_kwargs",
            {},
        ),
        drr_kwargs=cfg.get(
            "drr_kwargs",
            {},
        ),
    )

    drr = drr.to(device)

    # ========================================================
    # Load landmarks
    # ========================================================

    with landmarks_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        lm = json.load(f)

    ct_vox = (
        lm["ct"]
        ["landmarks_voxel_ijk"]
    )

    ap_px = (
        lm["views"]
        ["ap"]
        ["landmarks_px"]
    )

    names = []
    voxels = []
    gt_pixels = []

    for vertebra in sorted(
        set(ct_vox.keys())
        & set(ap_px.keys()),
        key=int,
    ):
        point_ids = sorted(
            set(ct_vox[vertebra].keys())
            & set(ap_px[vertebra].keys()),
            key=int,
        )

        for point_id in point_ids:
            names.append(
                (vertebra, point_id)
            )

            voxels.append(
                ct_vox[vertebra][point_id]
            )

            gt_pixels.append(
                ap_px[vertebra][point_id]
            )

    voxels = torch.tensor(
        voxels,
        dtype=torch.float32,
        device=device,
    )

    gt_pixels = torch.tensor(
        gt_pixels,
        dtype=torch.float32,
        device=device,
    )

    n_landmarks = len(voxels)

    print()
    print(
        "Landmark count =",
        n_landmarks,
    )

    # ========================================================
    # CT voxel -> DiffDRR centered world coordinates
    # ========================================================

    ones = torch.ones(
        (n_landmarks, 1),
        dtype=torch.float32,
        device=device,
    )

    vox_h = torch.cat(
        [voxels, ones],
        dim=1,
    )

    A_centered = drr._affine[0]

    world_h = (
        A_centered
        @ vox_h.T
    ).T

    world = world_h[:, :3]

    # ========================================================
    # FINAL pose projection
    #
    # IMPORTANT:
    # use the formal DiffDRR projection -> raster adapter.
    #
    # Do NOT manually write:
    #
    #     pred[:, 0] = W - pred[:, 0]
    #
    # ========================================================

    with torch.no_grad():

        pred_pixels = (
            project_fiducials_to_image_pixels(
                drr,
                pose,
                world.unsqueeze(0),
                orientation="AP",
            )[0]
        )

    # ========================================================
    # Residuals
    # ========================================================

    delta = (
        pred_pixels
        - gt_pixels
    )

    du = delta[:, 0]
    dv = delta[:, 1]

    err_px = torch.linalg.norm(
        delta,
        dim=1,
    )

    delx = float(cfg["delx"])
    dely = float(cfg["dely"])

    err_mm = torch.sqrt(
        (du * delx) ** 2
        +
        (dv * dely) ** 2
    )

    s_px = stats(err_px)
    s_mm = stats(err_mm)

    # ========================================================
    # Overall
    # ========================================================

    print()
    print("=" * 80)
    print(
        "AP FINAL LANDMARK "
        "PROJECTION ERROR"
    )
    print("=" * 80)

    print()
    print("Pixel domain:")

    print(
        f"mean   = "
        f"{s_px['mean']:.3f} px"
    )
    print(
        f"median = "
        f"{s_px['median']:.3f} px"
    )
    print(
        f"RMSE   = "
        f"{s_px['rmse']:.3f} px"
    )
    print(
        f"max    = "
        f"{s_px['max']:.3f} px"
    )

    print()
    print(
        "Detector-plane physical distance:"
    )

    print(
        f"mTRE   = "
        f"{s_mm['mean']:.3f} mm"
    )
    print(
        f"median = "
        f"{s_mm['median']:.3f} mm"
    )
    print(
        f"RMSE   = "
        f"{s_mm['rmse']:.3f} mm"
    )
    print(
        f"max    = "
        f"{s_mm['max']:.3f} mm"
    )

    print()
    print(
        "Mean signed residual:"
    )

    print(
        f"du = "
        f"{du.mean().item():+.3f} px"
    )

    print(
        f"dv = "
        f"{dv.mean().item():+.3f} px"
    )

    # ========================================================
    # Per vertebra
    # ========================================================

    print()
    print("=" * 80)
    print("PER VERTEBRA")
    print("=" * 80)

    vertebrae = sorted(
        set(
            v
            for v, _
            in names
        ),
        key=int,
    )

    for v in vertebrae:

        idx = [
            i
            for i, (vv, _)
            in enumerate(names)
            if vv == v
        ]

        epx = err_px[idx]
        emm = err_mm[idx]

        print(
            f"vertebra {v}: "
            f"mean="
            f"{epx.mean().item():8.3f} px | "
            f"{emm.mean().item():7.3f} mm"
        )

    # ========================================================
    # Save CSV
    # ========================================================

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "vertebra",
            "point",
            "gt_x",
            "gt_y",
            "pred_x",
            "pred_y",
            "du",
            "dv",
            "error_px",
            "error_mm",
        ])

        for i, (v, pt) in enumerate(
            names
        ):

            writer.writerow([
                v,
                pt,

                f"{gt_pixels[i,0].item():.6f}",
                f"{gt_pixels[i,1].item():.6f}",

                f"{pred_pixels[i,0].item():.6f}",
                f"{pred_pixels[i,1].item():.6f}",

                f"{du[i].item():.6f}",
                f"{dv[i].item():.6f}",

                f"{err_px[i].item():.6f}",
                f"{err_mm[i].item():.6f}",
            ])

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        "CSV =",
        out_csv,
    )


if __name__ == "__main__":
    main()

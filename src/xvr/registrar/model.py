import json
import math
from pathlib import Path

import torch
from diffdrr.data import read
from diffdrr.pose import RigidTransform
from pydicom import dcmread

from ..io import read_xray
from ..metrics import project_fiducials_to_image_pixels
from ..model.inference import _construct_antipode, _correct_pose, predict_pose
from ..model.network import load_model
from .base import _RegistrarBase


class RegistrarModel(_RegistrarBase):
    def __init__(
        self,
        volume,
        mask,
        ckptpath,
        labels=None,
        crop=0,
        subtract_background=False,
        linearize=True,
        equalize=False,
        reducefn="max",
        warp=None,
        invert=False,
        antipodal=False,
        scales="8",
        n_itrs="100",
        reverse_x_axis=True,
        renderer="trilinear",
        parameterization="euler_angles",
        convention="ZXY",
        voxel_shift=0.0,
        lr_rot=1e-2,
        lr_xyz=1e0,
        patience=10,
        threshold=1e-4,
        max_n_plateaus=3,
        init_only=False,
        saveimg=False,
        verbose=1,
        read_kwargs={},
        drr_kwargs={},
        estimate_missing_y0=True,
        y0_search_min=-50.0,
        y0_search_max=50.0,
        y0_coarse_step=2.0,
        y0_fine_radius=3.0,
        y0_fine_step=0.25,
        y0_search_scale=6.0,
        y0_max_rounds=10,
        y0_convergence_tol=0.5,
        estimate_missing_x0=True,
        x0_search_min=-50.0,
        x0_search_max=50.0,
        x0_coarse_step=2.0,
        x0_fine_radius=3.0,
        x0_fine_step=0.25,
        landmarks=None,
    ):
        self.ckptpath = ckptpath
        self.model, self.config, self.date = load_model(self.ckptpath, meta=True)
        self.warp = warp
        self.invert = invert
        self.antipodal = antipodal
        self.landmarks = Path(landmarks) if landmarks is not None else None

        self.estimate_missing_x0 = estimate_missing_x0
        self.estimate_missing_y0 = estimate_missing_y0
        self.x0_search_min = float(x0_search_min)
        self.x0_search_max = float(x0_search_max)
        self.x0_coarse_step = float(x0_coarse_step)
        self.x0_fine_radius = float(x0_fine_radius)
        self.x0_fine_step = float(x0_fine_step)
        self.y0_search_min = float(y0_search_min)
        self.y0_search_max = float(y0_search_max)
        self.y0_coarse_step = float(y0_coarse_step)
        self.y0_fine_radius = float(y0_fine_radius)
        self.y0_fine_step = float(y0_fine_step)
        # Kept for backward compatibility. Landmark mTRE scoring does not render DRRs.
        self.y0_search_scale = float(y0_search_scale)
        self.y0_max_rounds = int(y0_max_rounds)
        self.y0_convergence_tol = float(y0_convergence_tol)
        self._validate_origin_search_config()

        super().__init__(
            volume,
            mask,
            self.config["orientation"],
            labels,
            crop,
            subtract_background,
            linearize,
            equalize,
            reducefn,
            scales,
            n_itrs,
            reverse_x_axis,
            renderer,
            parameterization,
            convention,
            voxel_shift,
            lr_rot,
            lr_xyz,
            patience,
            threshold,
            max_n_plateaus,
            init_only,
            saveimg,
            verbose,
            read_kwargs,
            drr_kwargs,
            save_kwargs={
                "type": "model",
                "ckptpath": self.ckptpath,
                "date": self.date,
                "warp": self.warp,
                "invert": self.invert,
                "y0_estimation": {
                    "enabled": self.estimate_missing_x0 or self.estimate_missing_y0,
                    "mode": "alternating_coordinate_x0_y0_mtre",
                    "objective": "2d_projected_landmark_mTRE_mm",
                    "performed": False,
                    "reason": None,
                    "landmarks": str(self.landmarks) if self.landmarks is not None else None,
                    "x0_enabled": self.estimate_missing_x0,
                    "y0_enabled": self.estimate_missing_y0,
                    "x0_search_min_mm": self.x0_search_min,
                    "x0_search_max_mm": self.x0_search_max,
                    "x0_coarse_step_mm": self.x0_coarse_step,
                    "x0_fine_radius_mm": self.x0_fine_radius,
                    "x0_fine_step_mm": self.x0_fine_step,
                    "search_min_mm": self.y0_search_min,
                    "search_max_mm": self.y0_search_max,
                    "coarse_step_mm": self.y0_coarse_step,
                    "fine_radius_mm": self.y0_fine_radius,
                    "fine_step_mm": self.y0_fine_step,
                    "max_rounds": self.y0_max_rounds,
                    "convergence_tol_mm": self.y0_convergence_tol,
                },
            },
        )

    @staticmethod
    def _validate_axis(name, lo, hi, coarse, radius, fine):
        if lo >= hi:
            raise ValueError(f"{name}_search_min must be smaller than {name}_search_max")
        if coarse <= 0 or radius <= 0 or fine <= 0:
            raise ValueError(f"{name} search steps/radius must be positive")

    def _validate_origin_search_config(self):
        self._validate_axis(
            "x0",
            self.x0_search_min,
            self.x0_search_max,
            self.x0_coarse_step,
            self.x0_fine_radius,
            self.x0_fine_step,
        )
        self._validate_axis(
            "y0",
            self.y0_search_min,
            self.y0_search_max,
            self.y0_coarse_step,
            self.y0_fine_radius,
            self.y0_fine_step,
        )
        if self.y0_max_rounds <= 0:
            raise ValueError("y0_max_rounds must be positive")
        if self.y0_convergence_tol <= 0:
            raise ValueError("y0_convergence_tol must be positive")

    @staticmethod
    def _detector_active_origin_present(filename):
        ds = dcmread(filename, stop_before_pixels=True)
        origin = getattr(ds, "DetectorActiveOrigin", None)
        try:
            return origin is not None and len(origin) == 2 and all(v is not None for v in origin)
        except TypeError:
            return False

    @staticmethod
    def _scan_values(start, stop, step, include=None):
        n = int(round((stop - start) / step))
        values = [start + i * step for i in range(n + 1)]
        if include is not None and start <= include <= stop:
            include = float(include)
            if all(abs(x - include) > 1e-9 for x in values):
                values.append(include)
                values.sort()
        return values

    @staticmethod
    def _sort_numeric_key(value):
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    def _resolve_landmarks_path(self, i2d):
        path = self.landmarks if self.landmarks is not None else Path(i2d).parent / "landmarks.json"
        if not path.exists():
            raise FileNotFoundError(
                "mTRE-based principal-point search requires 3D/2D landmarks. "
                f"No landmarks file found at {path}. The CLI expects landmarks.json next to the X-ray."
            )
        return path

    def _load_mtre_targets(self, i2d, height, width, pf_to_af):
        orientation = str(self.config["orientation"]).upper()
        if orientation != "AP":
            raise NotImplementedError(
                "mTRE-based automatic principal-point estimation is currently "
                f"validated only for AP, got {orientation!r}."
            )
        if self.reverse_x_axis:
            raise ValueError(
                "mTRE-based AP origin search requires the canonical reverse_x_axis=False convention."
            )
        if self.crop != 0:
            raise ValueError(
                "mTRE-based AP origin search currently requires --crop 0 because "
                "landmarks.json stores full-raster pixel coordinates."
            )
        if pf_to_af:
            raise ValueError(
                "The X-ray reader applied a hidden PF->AF horizontal flip. "
                "Landmark-assisted origin search refuses to mix that convention."
            )

        path = self._resolve_landmarks_path(i2d)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            ct_lps = data["ct"]["landmarks_lps_mm"]
            ap_pixels = data["views"]["ap"]["landmarks_px"]
        except KeyError as exc:
            raise KeyError(
                f"{path} must contain ct.landmarks_lps_mm and views.ap.landmarks_px"
            ) from exc

        points_lps, points_px, names = [], [], []
        for vertebra in sorted(set(ct_lps) & set(ap_pixels), key=self._sort_numeric_key):
            ct_points = ct_lps[vertebra]
            ap_points = ap_pixels[vertebra]
            for point_id in sorted(set(ct_points) & set(ap_points), key=self._sort_numeric_key):
                xyz = ct_points[point_id]
                uv = ap_points[point_id]
                if len(xyz) != 3 or len(uv) != 2:
                    raise ValueError(
                        f"Invalid landmark {vertebra}.{point_id}: 3D={xyz!r}, 2D={uv!r}"
                    )
                points_lps.append([float(v) for v in xyz])
                points_px.append([float(v) for v in uv])
                names.append(f"{vertebra}.{point_id}")

        if not points_lps:
            raise ValueError(f"No paired CT/AP landmarks found in {path}")

        target_pixels = torch.tensor(points_px, dtype=torch.float32)[None].cuda()
        if (
            (target_pixels[..., 0] < 0).any()
            or (target_pixels[..., 0] >= width).any()
            or (target_pixels[..., 1] < 0).any()
            or (target_pixels[..., 1] >= height).any()
        ):
            raise ValueError(
                f"At least one AP landmark in {path} lies outside the {width}x{height} raster."
            )

        fiducials_ras = torch.tensor(points_lps, dtype=torch.float32)
        fiducials_ras[:, 0] *= -1
        fiducials_ras[:, 1] *= -1
        fiducials_ras = fiducials_ras[None]

        labels = self.labels
        if isinstance(labels, str):
            labels = [int(x) for x in labels.split(",") if x.strip()]

        subject = read(
            self.volume,
            self.mask,
            labels,
            self.config["orientation"],
            fiducials=fiducials_ras,
            **self.read_kwargs,
        )
        fiducials = subject.fiducials.cuda()
        if fiducials.shape[-2] != target_pixels.shape[-2]:
            raise RuntimeError(
                "DiffDRR fiducial count changed unexpectedly: "
                f"3D={fiducials.shape[-2]}, 2D={target_pixels.shape[-2]}"
            )

        return {
            "path": path,
            "names": names,
            "fiducials": fiducials,
            "target_pixels": target_pixels,
            "count": len(names),
        }

    def _score_axis(
        self,
        pose,
        sdd,
        delx,
        dely,
        height,
        width,
        x0,
        y0,
        axis,
        candidates,
        fiducials,
        target_pixels,
    ):
        scores = []
        with torch.no_grad():
            for candidate in candidates:
                candidate_x0 = float(candidate) if axis == "x0" else float(x0)
                candidate_y0 = float(candidate) if axis == "y0" else float(y0)
                self.drr.set_intrinsics_(
                    sdd=sdd,
                    height=height,
                    width=width,
                    delx=delx,
                    dely=dely,
                    # CRITICAL SIGN CONVENTION:
                    # DICOM/XVR candidate x0 -> DiffDRR detector x0 = -candidate_x0.
                    x0=-candidate_x0,
                    y0=candidate_y0,
                )
                projected = project_fiducials_to_image_pixels(
                    self.drr,
                    pose,
                    fiducials,
                    orientation="AP",
                    reverse_x_axis=False,
                )
                delta = projected - target_pixels
                error_mm = torch.sqrt(
                    (delta[..., 0] * float(delx)) ** 2
                    + (delta[..., 1] * float(dely)) ** 2
                )
                mtre_mm = error_mm.mean().item()
                scores.append((float(candidate), float(mtre_mm)))
                if self.verbose > 1:
                    print(f"  {axis}={candidate:+7.2f} mm | mTRE={mtre_mm:.6f} mm")
        return scores

    def _search_axis(
        self,
        pose,
        sdd,
        delx,
        dely,
        height,
        width,
        x0,
        y0,
        axis,
        fiducials,
        target_pixels,
    ):
        if axis == "x0":
            lo, hi = self.x0_search_min, self.x0_search_max
            coarse, radius, fine = (
                self.x0_coarse_step,
                self.x0_fine_radius,
                self.x0_fine_step,
            )
            current = x0
        elif axis == "y0":
            lo, hi = self.y0_search_min, self.y0_search_max
            coarse, radius, fine = (
                self.y0_coarse_step,
                self.y0_fine_radius,
                self.y0_fine_step,
            )
            current = y0
        else:
            raise ValueError(f"Unknown principal-point axis: {axis}")

        coarse_values = self._scan_values(lo, hi, coarse, include=current)
        coarse_scores = self._score_axis(
            pose,
            sdd,
            delx,
            dely,
            height,
            width,
            x0,
            y0,
            axis,
            coarse_values,
            fiducials,
            target_pixels,
        )
        coarse_best, coarse_mtre = min(coarse_scores, key=lambda z: z[1])

        fine_lo = max(lo, coarse_best - radius)
        fine_hi = min(hi, coarse_best + radius)
        fine_values = self._scan_values(fine_lo, fine_hi, fine, include=coarse_best)
        fine_scores = self._score_axis(
            pose,
            sdd,
            delx,
            dely,
            height,
            width,
            x0,
            y0,
            axis,
            fine_values,
            fiducials,
            target_pixels,
        )
        best, mtre = min(fine_scores, key=lambda z: z[1])
        return {
            "best": float(best),
            "mtre_mm": float(mtre),
            "coarse_best": float(coarse_best),
            "coarse_mtre_mm": float(coarse_mtre),
            "coarse_scores": coarse_scores,
            "fine_scores": fine_scores,
            "boundary_hit": abs(best - fine_lo) <= fine / 2
            or abs(best - fine_hi) <= fine / 2,
        }

    def _correct_initial_pose(self, pose):
        pose = _correct_pose(pose, self.warp, self.volume, self.invert)
        if self.antipodal:
            pose = _construct_antipode(pose)
        return pose

    def initialize_pose(self, i2d, return_resampled=False):
        gt, sdd, delx, dely, x0, y0, pf_to_af = read_xray(
            i2d,
            self.crop,
            self.subtract_background,
            self.linearize,
            self.reducefn,
        )
        *_, height, width = gt.shape
        origin_present = self._detector_active_origin_present(i2d)
        metadata = self.save_kwargs["y0_estimation"]
        metadata.update(
            {
                "detector_active_origin_present": origin_present,
                "dicom_x0_mm": float(x0),
                "dicom_y0_mm": float(y0),
            }
        )

        auto_enabled = self.estimate_missing_x0 or self.estimate_missing_y0
        if origin_present or not auto_enabled:
            pose, resampled_gt = predict_pose(
                self.model, self.config, gt, sdd, delx, dely, x0, y0
            )
            pose = self._correct_initial_pose(pose)
            metadata["reason"] = (
                "DetectorActiveOrigin present in DICOM"
                if origin_present
                else "automatic principal-point estimation disabled"
            )
            result = (gt, sdd, delx, dely, x0, y0, pf_to_af, pose)
            return (*result, resampled_gt) if return_resampled else result

        targets = self._load_mtre_targets(i2d, height, width, pf_to_af)
        metadata.update(
            {
                "performed": True,
                "reason": "DetectorActiveOrigin missing",
                "landmarks": str(targets["path"]),
                "n_landmarks": targets["count"],
                "landmark_names": targets["names"],
            }
        )

        current_x0, current_y0 = float(x0), float(y0)
        rounds = []
        best = None
        converged = False

        for round_idx in range(self.y0_max_rounds):
            input_x0, input_y0 = current_x0, current_y0
            pose, resampled_gt = predict_pose(
                self.model,
                self.config,
                gt,
                sdd,
                delx,
                dely,
                input_x0,
                input_y0,
            )
            pose = self._correct_initial_pose(pose)

            if self.verbose > 0:
                print(
                    f"\n[Auto origin mTRE] Round {round_idx + 1}/{self.y0_max_rounds}: "
                    f"network input x0={input_x0:+.3f}, y0={input_y0:+.3f} mm; "
                    f"landmarks={targets['count']}"
                )

            x_result = None
            new_x0 = input_x0
            if self.estimate_missing_x0:
                x_result = self._search_axis(
                    pose,
                    sdd,
                    delx,
                    dely,
                    height,
                    width,
                    input_x0,
                    input_y0,
                    "x0",
                    targets["fiducials"],
                    targets["target_pixels"],
                )
                new_x0 = x_result["best"]

            y_result = None
            new_y0 = input_y0
            if self.estimate_missing_y0:
                y_result = self._search_axis(
                    pose,
                    sdd,
                    delx,
                    dely,
                    height,
                    width,
                    new_x0,
                    input_y0,
                    "y0",
                    targets["fiducials"],
                    targets["target_pixels"],
                )
                new_y0 = y_result["best"]

            final_result = y_result if y_result is not None else x_result
            if final_result is None:
                raise RuntimeError("No principal-point coordinate enabled for search")
            round_mtre = final_result["mtre_mm"]
            dx, dy = new_x0 - input_x0, new_y0 - input_y0
            delta = math.hypot(dx, dy)

            if best is None or round_mtre < best["mtre_mm"]:
                best = {
                    "mtre_mm": float(round_mtre),
                    "x0": float(new_x0),
                    "y0": float(new_y0),
                    "pose": RigidTransform(pose.matrix.detach().clone()),
                    "resampled_gt": resampled_gt.detach().clone(),
                    "round": round_idx + 1,
                    "input_x0": float(input_x0),
                    "input_y0": float(input_y0),
                }

            def trace(result, fallback):
                if result is None:
                    return {
                        "coarse_best": fallback,
                        "coarse_mtre_mm": None,
                        "best": fallback,
                        "mtre_mm": None,
                        "boundary_hit": False,
                        "coarse_scores": [],
                        "fine_scores": [],
                    }
                return result

            xr, yr = trace(x_result, input_x0), trace(y_result, input_y0)
            rounds.append(
                {
                    "round": round_idx + 1,
                    "input_x0_mm": float(input_x0),
                    "input_y0_mm": float(input_y0),
                    "coarse_best_x0_mm": xr["coarse_best"],
                    "coarse_best_x0_mtre_mm": xr["coarse_mtre_mm"],
                    "best_x0_mm": float(new_x0),
                    "x0_best_mtre_mm": xr["mtre_mm"],
                    "delta_x0_mm": abs(float(dx)),
                    "x0_boundary_hit": xr["boundary_hit"],
                    "x0_coarse_scores": xr["coarse_scores"],
                    "x0_fine_scores": xr["fine_scores"],
                    "coarse_best_y0_mm": yr["coarse_best"],
                    "coarse_best_y0_mtre_mm": yr["coarse_mtre_mm"],
                    "best_y0_mm": float(new_y0),
                    "y0_best_mtre_mm": yr["mtre_mm"],
                    "round_mtre_mm": float(round_mtre),
                    "delta_y0_mm": abs(float(dy)),
                    "delta_origin_mm": float(delta),
                    "y0_boundary_hit": yr["boundary_hit"],
                    "y0_coarse_scores": yr["coarse_scores"],
                    "y0_fine_scores": yr["fine_scores"],
                }
            )

            if self.verbose > 0:
                if x_result is not None:
                    print(
                        f"[Auto origin mTRE] x0 -> {new_x0:+.3f} mm "
                        f"(mTRE={x_result['mtre_mm']:.6f} mm)"
                    )
                if y_result is not None:
                    print(
                        f"[Auto origin mTRE] y0 -> {new_y0:+.3f} mm "
                        f"(mTRE={y_result['mtre_mm']:.6f} mm)"
                    )
                print(
                    f"[Auto origin mTRE] delta: dx0={abs(dx):.3f}, "
                    f"dy0={abs(dy):.3f}, norm={delta:.3f} mm"
                )

            current_x0, current_y0 = float(new_x0), float(new_y0)
            if delta <= self.y0_convergence_tol:
                converged = True
                if self.verbose > 0:
                    print(
                        f"[Auto origin mTRE] Converged: origin delta <= "
                        f"{self.y0_convergence_tol:.3f} mm"
                    )
                break

        x0, y0 = best["x0"], best["y0"]
        pose, resampled_gt = best["pose"], best["resampled_gt"]
        metadata.update(
            {
                "rounds": rounds,
                "n_rounds": len(rounds),
                "converged": converged,
                "estimated_x0_mm": x0,
                "estimated_y0_mm": y0,
                "estimated_origin_mm": [x0, y0],
                "selected_round": best["round"],
                "selected_pose_input_x0_mm": best["input_x0"],
                "selected_pose_input_y0_mm": best["input_y0"],
                "best_mtre_mm": best["mtre_mm"],
                "last_round_x0_mm": current_x0,
                "last_round_y0_mm": current_y0,
            }
        )
        if self.verbose > 0:
            print(
                f"\n[Auto origin mTRE] Historical best: round={best['round']}, "
                f"x0={x0:+.3f}, y0={y0:+.3f} mm, "
                f"mTRE={best['mtre_mm']:.6f} mm"
            )

        result = (gt, sdd, delx, dely, x0, y0, pf_to_af, pose)
        return (*result, resampled_gt) if return_resampled else result

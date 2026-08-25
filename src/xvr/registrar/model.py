import math

import torch
from diffdrr.pose import RigidTransform
from pydicom import dcmread

from ..io import read_xray
from ..model.inference import _construct_antipode, _correct_pose, predict_pose
from ..model.network import load_model
from ..utils import XrayTransforms
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
    ):
        self.ckptpath = ckptpath
        self.model, self.config, self.date = load_model(self.ckptpath, meta=True)
        self.warp = warp
        self.invert = invert
        self.antipodal = antipodal

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
        # Backward-compatible names: these now control the joint origin search.
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
                # Keep the historical key to avoid breaking result readers.
                "y0_estimation": {
                    "enabled": self.estimate_missing_x0 or self.estimate_missing_y0,
                    "mode": "alternating_coordinate_x0_y0",
                    "performed": False,
                    "reason": None,
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
                    "search_scale": self.y0_search_scale,
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
        if self.y0_search_scale <= 0:
            raise ValueError("y0_search_scale must be positive")
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
    def _zncc(a, b, eps=1e-8):
        a = a.flatten().double() - a.double().mean()
        b = b.flatten().double() - b.double().mean()
        denom = torch.sqrt((a * a).sum() * (b * b).sum())
        return ((a * b).sum() / (denom + eps)).item()

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

    def _score_axis(self, gt, pose, sdd, delx, dely, x0, y0, axis, candidates):
        """Score candidates in DICOM/XVR x0,y0 convention with pose fixed."""
        *_, height, width = gt.shape

        def set_intrinsics(candidate):
            candidate_x0 = float(candidate) if axis == "x0" else float(x0)
            candidate_y0 = float(candidate) if axis == "y0" else float(y0)
            self.drr.set_intrinsics_(
                sdd=sdd,
                height=height,
                width=width,
                delx=delx,
                dely=dely,
                # CRITICAL SIGN CONVENTION:
                # read_xray/predict_pose use DICOM x0; DiffDRR uses -x0.
                x0=-candidate_x0,
                y0=candidate_y0,
            )
            self.drr.rescale_detector_(1.0 / self.y0_search_scale)

        set_intrinsics(candidates[0])
        transform = XrayTransforms(self.drr.detector.height, self.drr.detector.width)
        gt_search = transform(gt).cuda()
        scores = []
        with torch.no_grad():
            for candidate in candidates:
                set_intrinsics(candidate)
                score = self._zncc(gt_search, transform(self.drr(pose)))
                scores.append((float(candidate), float(score)))
                if self.verbose > 1:
                    print(f"  {axis}={candidate:+7.2f} mm | ZNCC={score:.6f}")
        return scores

    def _search_axis(self, gt, pose, sdd, delx, dely, x0, y0, axis):
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
            gt, pose, sdd, delx, dely, x0, y0, axis, coarse_values
        )
        coarse_best, coarse_score = max(coarse_scores, key=lambda z: z[1])
        fine_lo = max(lo, coarse_best - radius)
        fine_hi = min(hi, coarse_best + radius)
        fine_values = self._scan_values(fine_lo, fine_hi, fine, include=coarse_best)
        fine_scores = self._score_axis(
            gt, pose, sdd, delx, dely, x0, y0, axis, fine_values
        )
        best, score = max(fine_scores, key=lambda z: z[1])
        return {
            "best": float(best),
            "score": float(score),
            "coarse_best": float(coarse_best),
            "coarse_score": float(coarse_score),
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

        metadata["performed"] = True
        metadata["reason"] = "DetectorActiveOrigin missing"
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
                    f"\n[Auto origin] Round {round_idx + 1}/{self.y0_max_rounds}: "
                    f"network input x0={input_x0:+.3f}, y0={input_y0:+.3f} mm"
                )

            # Alternating coordinate search with one fixed network pose:
            # x0 first (y0 fixed), then y0 (new x0 fixed).
            x_result = None
            new_x0 = input_x0
            if self.estimate_missing_x0:
                x_result = self._search_axis(
                    gt, pose, sdd, delx, dely, input_x0, input_y0, "x0"
                )
                new_x0 = x_result["best"]

            y_result = None
            new_y0 = input_y0
            if self.estimate_missing_y0:
                y_result = self._search_axis(
                    gt, pose, sdd, delx, dely, new_x0, input_y0, "y0"
                )
                new_y0 = y_result["best"]

            final_result = y_result if y_result is not None else x_result
            if final_result is None:
                raise RuntimeError("No principal-point coordinate enabled for search")
            score = final_result["score"]
            dx, dy = new_x0 - input_x0, new_y0 - input_y0
            delta = math.hypot(dx, dy)

            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
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
                        "coarse_score": None,
                        "best": fallback,
                        "score": None,
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
                    "coarse_best_x0_zncc": xr["coarse_score"],
                    "best_x0_mm": float(new_x0),
                    "x0_best_zncc": xr["score"],
                    "delta_x0_mm": abs(float(dx)),
                    "x0_boundary_hit": xr["boundary_hit"],
                    "x0_coarse_scores": xr["coarse_scores"],
                    "x0_fine_scores": xr["fine_scores"],
                    # Legacy y0-only fields remain available.
                    "coarse_best_y0_mm": yr["coarse_best"],
                    "coarse_best_zncc": yr["coarse_score"],
                    "best_y0_mm": float(new_y0),
                    "best_zncc": float(score),
                    "delta_y0_mm": abs(float(dy)),
                    "delta_origin_mm": float(delta),
                    "boundary_hit": yr["boundary_hit"],
                    "coarse_scores": yr["coarse_scores"],
                    "fine_scores": yr["fine_scores"],
                }
            )

            if self.verbose > 0:
                if x_result is not None:
                    print(
                        f"[Auto origin] x0 -> {new_x0:+.3f} mm "
                        f"(ZNCC={x_result['score']:.6f})"
                    )
                if y_result is not None:
                    print(
                        f"[Auto origin] y0 -> {new_y0:+.3f} mm "
                        f"(ZNCC={y_result['score']:.6f})"
                    )
                print(
                    f"[Auto origin] delta: dx0={abs(dx):.3f}, "
                    f"dy0={abs(dy):.3f}, norm={delta:.3f} mm"
                )

            current_x0, current_y0 = float(new_x0), float(new_y0)
            if delta <= self.y0_convergence_tol:
                converged = True
                if self.verbose > 0:
                    print(
                        f"[Auto origin] Converged: origin delta <= "
                        f"{self.y0_convergence_tol:.3f} mm"
                    )
                break

        # Restore the historical-best pose/origin tuple exactly as scored.
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
                "best_zncc": best["score"],
                "last_round_x0_mm": current_x0,
                "last_round_y0_mm": current_y0,
            }
        )
        if self.verbose > 0:
            print(
                f"\n[Auto origin] Historical best: round={best['round']}, "
                f"x0={x0:+.3f}, y0={y0:+.3f} mm, ZNCC={best['score']:.6f}"
            )

        result = (gt, sdd, delx, dely, x0, y0, pf_to_af, pose)
        return (*result, resampled_gt) if return_resampled else result

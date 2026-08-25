import torch
from pydicom import dcmread

from ..io import read_xray
from ..model.inference import _construct_antipode, _correct_pose, predict_pose
from ..model.network import load_model
from ..utils import XrayTransforms
from .base import _RegistrarBase
from diffdrr.pose import RigidTransform


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
    ):
        # Initialize the model and its config
        self.ckptpath = ckptpath
        self.model, self.config, self.date = load_model(self.ckptpath, meta=True)

        # Initial pose correction
        self.warp = warp
        self.invert = invert
        self.antipodal = antipodal

        # Missing principal-point estimation
        self.estimate_missing_y0 = estimate_missing_y0
        self.y0_search_min = float(y0_search_min)
        self.y0_search_max = float(y0_search_max)
        self.y0_coarse_step = float(y0_coarse_step)
        self.y0_fine_radius = float(y0_fine_radius)
        self.y0_fine_step = float(y0_fine_step)
        self.y0_search_scale = float(y0_search_scale)
        self.y0_max_rounds = int(y0_max_rounds)
        self.y0_convergence_tol = float(y0_convergence_tol)

        self._validate_y0_search_config()

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
                "enabled": self.estimate_missing_y0,
                "performed": False,
                "reason": None,

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

    def _validate_y0_search_config(self):
        if self.y0_search_min >= self.y0_search_max:
            raise ValueError("y0_search_min must be smaller than y0_search_max")
        if self.y0_coarse_step <= 0:
            raise ValueError("y0_coarse_step must be positive")
        if self.y0_fine_radius <= 0:
            raise ValueError("y0_fine_radius must be positive")
        if self.y0_fine_step <= 0:
            raise ValueError("y0_fine_step must be positive")
        if self.y0_search_scale <= 0:
            raise ValueError("y0_search_scale must be positive")
        if self.y0_max_rounds <= 0:
            raise ValueError("y0_max_rounds must be positive")
        if self.y0_convergence_tol <= 0:
            raise ValueError("y0_convergence_tol must be positive")

    @staticmethod
    def _detector_active_origin_present(filename):
        ds = dcmread(filename, stop_before_pixels=True)
        return getattr(ds, "DetectorActiveOrigin", None) is not None

    @staticmethod
    def _zncc(a, b, eps=1e-8):
        a = a.flatten().double()
        b = b.flatten().double()
        a = a - a.mean()
        b = b - b.mean()
        denominator = torch.sqrt((a * a).sum() * (b * b).sum())
        return ((a * b).sum() / (denominator + eps)).item()

    @staticmethod
    def _scan_values(start, stop, step):
        n = int(round((stop - start) / step))
        return [start + idx * step for idx in range(n + 1)]

    def _score_y0_candidates(self, gt, init_pose, sdd, delx, dely, x0, candidates):
        *_, height, width = gt.shape

        # Search at a reduced detector resolution to avoid full-resolution DRR rendering.
        self.drr.set_intrinsics_(
            sdd=sdd,
            height=height,
            width=width,
            delx=delx,
            dely=dely,
            x0=-x0,
            y0=float(candidates[0]),
        )
        self.drr.rescale_detector_(1.0 / self.y0_search_scale)
        transform = XrayTransforms(self.drr.detector.height, self.drr.detector.width)
        gt_search = transform(gt).cuda()

        scores = []
        with torch.no_grad():
            for candidate in candidates:
                self.drr.set_intrinsics_(
                    sdd=sdd,
                    height=height,
                    width=width,
                    delx=delx,
                    dely=dely,
                    x0=-x0,
                    y0=float(candidate),
                )
                self.drr.rescale_detector_(1.0 / self.y0_search_scale)
                pred = transform(self.drr(init_pose))
                score = self._zncc(gt_search, pred)
                scores.append((float(candidate), float(score)))

                if self.verbose > 1:
                    print(f"  y0={candidate:+7.2f} mm | ZNCC={score:.6f}")

        return scores
    def _search_best_y0(
        self,
        gt,
        pose,
        sdd,
        delx,
        dely,
        x0,
    ):
        """
        Search y0 while keeping the supplied pose fixed.

        Each call performs a completely new:
            global coarse search
                ->
            local fine search

        Therefore, whenever the network predicts a new pose,
        y0 is searched globally again.
        """

        # --------------------------------------------------------
        # Global coarse search
        # --------------------------------------------------------

        coarse_candidates = self._scan_values(
            self.y0_search_min,
            self.y0_search_max,
            self.y0_coarse_step,
        )

        coarse_scores = self._score_y0_candidates(
            gt,
            pose,
            sdd,
            delx,
            dely,
            x0,
            coarse_candidates,
        )

        coarse_best_y0, coarse_best_score = max(
            coarse_scores,
            key=lambda item: item[1],
        )

        # --------------------------------------------------------
        # Local fine search around NEW coarse optimum
        # --------------------------------------------------------

        fine_start = max(
            self.y0_search_min,
            coarse_best_y0 - self.y0_fine_radius,
        )

        fine_stop = min(
            self.y0_search_max,
            coarse_best_y0 + self.y0_fine_radius,
        )

        fine_candidates = self._scan_values(
            fine_start,
            fine_stop,
            self.y0_fine_step,
        )

        fine_scores = self._score_y0_candidates(
            gt,
            pose,
            sdd,
            delx,
            dely,
            x0,
            fine_candidates,
        )

        best_y0, best_score = max(
            fine_scores,
            key=lambda item: item[1],
        )

        boundary_hit = (
            abs(best_y0 - fine_start)
            <= self.y0_fine_step / 2
            or
            abs(best_y0 - fine_stop)
            <= self.y0_fine_step / 2
        )

        return {
            "best_y0_mm": float(best_y0),
            "best_zncc": float(best_score),

            "coarse_best_y0_mm": float(coarse_best_y0),
            "coarse_best_zncc": float(coarse_best_score),

            "coarse_scores": coarse_scores,
            "fine_scores": fine_scores,

            "boundary_hit": boundary_hit,
        }

    def _correct_initial_pose(self, pose):
        pose = _correct_pose(pose, self.warp, self.volume, self.invert)
        if self.antipodal:
            pose = _construct_antipode(pose)
        return pose

    def initialize_pose(self, i2d, return_resampled=False):

        # ========================================================
        # 1. Read X-ray and DICOM intrinsics
        # ========================================================

        gt, sdd, delx, dely, x0, y0, pf_to_af = read_xray(
            i2d,
            self.crop,
            self.subtract_background,
            self.linearize,
            self.reducefn,
        )

        detector_origin_present = (
            self._detector_active_origin_present(i2d)
        )

        metadata = self.save_kwargs["y0_estimation"]

        metadata["detector_active_origin_present"] = (
            detector_origin_present
        )

        metadata["dicom_y0_mm"] = float(y0)


        # ========================================================
        # 2. Normal path:
        #    DICOM already contains DetectorActiveOrigin
        # ========================================================

        if detector_origin_present or not self.estimate_missing_y0:

            init_pose, resampled_gt = predict_pose(
                self.model,
                self.config,
                gt,
                sdd,
                delx,
                dely,
                x0,
                y0,
            )

            init_pose = self._correct_initial_pose(init_pose)

            if detector_origin_present:
                metadata["reason"] = (
                    "DetectorActiveOrigin present in DICOM"
                )
            else:
                metadata["reason"] = (
                    "automatic y0 estimation disabled"
                )

            if return_resampled:
                return (
                    gt,
                    sdd,
                    delx,
                    dely,
                    x0,
                    y0,
                    pf_to_af,
                    init_pose,
                    resampled_gt,
                )

            return (
                gt,
                sdd,
                delx,
                dely,
                x0,
                y0,
                pf_to_af,
                init_pose,
            )


        # ========================================================
        # 3. DetectorActiveOrigin missing:
        #    alternating network pose / global y0 search
        # ========================================================

        metadata["performed"] = True
        metadata["reason"] = "DetectorActiveOrigin missing"

        rounds = []
        best_global_score = float("-inf")
        best_global_y0 = None
        best_global_pose = None
        best_global_resampled_gt = None
        best_global_round = None
        best_global_input_y0 = None

        current_y0 = float(y0)  # normally 0.0

        converged = False


        for round_idx in range(self.y0_max_rounds):

            # ----------------------------------------------------
            # A. Predict pose using CURRENT y0
            # ----------------------------------------------------

            init_pose, resampled_gt = predict_pose(
                self.model,
                self.config,
                gt,
                sdd,
                delx,
                dely,
                x0,
                current_y0,
            )

            init_pose = self._correct_initial_pose(init_pose)


            if self.verbose > 0:
                print()
                print(
                    f"[Auto y0] Round "
                    f"{round_idx + 1}/{self.y0_max_rounds}"
                )
                print(
                    f"[Auto y0] Network input y0 = "
                    f"{current_y0:+.3f} mm"
                )


            # ----------------------------------------------------
            # B. FIX this pose and perform a NEW global y0 search
            # ----------------------------------------------------

            result = self._search_best_y0(
                gt,
                init_pose,
                sdd,
                delx,
                dely,
                x0,
            )

            new_y0 = result["best_y0_mm"]

            delta_y0 = abs(new_y0 - current_y0)
            if result["best_zncc"] > best_global_score:
                best_global_score = float(result["best_zncc"])
                best_global_y0 = float(result["best_y0_mm"])
                best_global_pose = RigidTransform(
                    init_pose.matrix.detach().clone()
                )
                best_global_resampled_gt = resampled_gt.detach().clone()
                best_global_round = round_idx + 1
                best_global_input_y0 = float(current_y0)
            if self.verbose > 0:
                print(
                    f"[Auto y0] Coarse best = "
                    f"{result['coarse_best_y0_mm']:+.3f} mm "
                    f"(ZNCC={result['coarse_best_zncc']:.6f})"
                )

                print(
                    f"[Auto y0] Fine best   = "
                    f"{new_y0:+.3f} mm "
                    f"(ZNCC={result['best_zncc']:.6f})"
                )

                print(
                    f"[Auto y0] Delta       = "
                    f"{delta_y0:.3f} mm"
                )


            # ----------------------------------------------------
            # C. Save complete trace for this round
            # ----------------------------------------------------

            rounds.append(
                {
                    "round": round_idx + 1,
                    "input_y0_mm": float(current_y0),

                    "coarse_best_y0_mm":
                        result["coarse_best_y0_mm"],

                    "coarse_best_zncc":
                        result["coarse_best_zncc"],

                    "best_y0_mm":
                        result["best_y0_mm"],

                    "best_zncc":
                        result["best_zncc"],

                    "delta_y0_mm":
                        float(delta_y0),

                    "boundary_hit":
                        result["boundary_hit"],

                    "coarse_scores":
                        result["coarse_scores"],

                    "fine_scores":
                        result["fine_scores"],
                }
            )


            # ----------------------------------------------------
            # D. Update y0
            # ----------------------------------------------------

            current_y0 = float(new_y0)


            # ----------------------------------------------------
            # E. Convergence check
            # ----------------------------------------------------

            if delta_y0 <= self.y0_convergence_tol:

                converged = True

                if self.verbose > 0:
                    print(
                        f"[Auto y0] Converged: "
                        f"|delta y0| <= "
                        f"{self.y0_convergence_tol:.3f} mm"
                    )

                break


        # ========================================================
        # 4. Restore the HISTORICAL-BEST (pose, y0) pair.
        #
        #    Do NOT run the network again with best_global_y0 here.
        #    The selected score was measured with the pose saved
        #    from best_global_round and the swept best_global_y0.
        #    Re-predicting the pose would create a different state
        #    and could destroy the historical-best ZNCC.
        # ========================================================

        y0 = float(best_global_y0)
        init_pose = best_global_pose
        resampled_gt = best_global_resampled_gt

        # ========================================================
        # 5. Save final metadata
        # ========================================================

        metadata.update(
            {
                "rounds": rounds,
                "n_rounds": len(rounds),
                "converged": converged,

                "estimated_y0_mm": y0,

                "selected_round": best_global_round,
                "selected_pose_input_y0_mm": best_global_input_y0,
                "best_zncc": best_global_score,

                "last_round_y0_mm": float(current_y0),
            }
        )


        if self.verbose > 0:
            print()
            print(
                f"[Auto y0] Selected historical best: "
                f"round={best_global_round}, "
                f"y0={y0:+.3f} mm, "
                f"ZNCC={best_global_score:.6f}"
            )

            print(
                f"[Auto y0] rounds = {len(rounds)}, "
                f"converged = {converged}"
            )


        # ========================================================
        # 6. Return FINAL y0 and FINAL network pose
        # ========================================================

        if return_resampled:
            return (
                gt,
                sdd,
                delx,
                dely,
                x0,
                y0,
                pf_to_af,
                init_pose,
                resampled_gt,
            )

        return (
            gt,
            sdd,
            delx,
            dely,
            x0,
            y0,
            pf_to_af,
            init_pose,
        )
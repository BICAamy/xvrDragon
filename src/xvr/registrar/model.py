import torch
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

    def _estimate_missing_y0(self, gt, init_pose, sdd, delx, dely, x0):
        if self.verbose > 0:
            print(
                "DetectorActiveOrigin is missing; estimating detector y0 "
                "from the initial pose..."
            )

        coarse_candidates = self._scan_values(
            self.y0_search_min,
            self.y0_search_max,
            self.y0_coarse_step,
        )
        coarse_scores = self._score_y0_candidates(
            gt, init_pose, sdd, delx, dely, x0, coarse_candidates
        )
        coarse_best_y0, coarse_best_score = max(coarse_scores, key=lambda item: item[1])

        fine_start = max(self.y0_search_min, coarse_best_y0 - self.y0_fine_radius)
        fine_stop = min(self.y0_search_max, coarse_best_y0 + self.y0_fine_radius)
        fine_candidates = self._scan_values(
            fine_start,
            fine_stop,
            self.y0_fine_step,
        )
        fine_scores = self._score_y0_candidates(
            gt, init_pose, sdd, delx, dely, x0, fine_candidates
        )
        best_y0, best_score = max(fine_scores, key=lambda item: item[1])

        if self.verbose > 0:
            print(
                f"Estimated detector y0 = {best_y0:+.3f} mm "
                f"(coarse best {coarse_best_y0:+.3f} mm; ZNCC={best_score:.6f})"
            )

        boundary_hit = best_y0 in (self.y0_search_min, self.y0_search_max)
        self.save_kwargs["y0_estimation"].update(
            {
                "performed": True,
                "reason": "DetectorActiveOrigin missing",
                "estimated_y0_mm": best_y0,
                "coarse_best_y0_mm": coarse_best_y0,
                "coarse_best_zncc": coarse_best_score,
                "best_zncc": best_score,
                "boundary_hit": boundary_hit,
                "coarse_scores": coarse_scores,
                "fine_scores": fine_scores,
            }
        )
        if boundary_hit and self.verbose > 0:
            print(
                "Warning: best y0 is on the configured search boundary; "
                "consider widening the search range."
            )
        return best_y0

    def _refine_estimated_y0(self, gt, init_pose, sdd, delx, dely, x0, center_y0):
        fine_start = max(self.y0_search_min, center_y0 - self.y0_fine_radius)
        fine_stop = min(self.y0_search_max, center_y0 + self.y0_fine_radius)
        candidates = self._scan_values(fine_start, fine_stop, self.y0_fine_step)
        scores = self._score_y0_candidates(
            gt, init_pose, sdd, delx, dely, x0, candidates
        )
        best_y0, best_score = max(scores, key=lambda item: item[1])

        self.save_kwargs["y0_estimation"].update(
            {
                "pre_reprediction_y0_mm": float(center_y0),
                "estimated_y0_mm": best_y0,
                "best_zncc": best_score,
                "post_reprediction_scores": scores,
            }
        )
        if self.verbose > 0:
            print(
                f"Refined detector y0 after pose re-prediction = {best_y0:+.3f} mm "
                f"(ZNCC={best_score:.6f})"
            )
        return best_y0

    def _correct_initial_pose(self, pose):
        pose = _correct_pose(pose, self.warp, self.volume, self.invert)
        if self.antipodal:
            pose = _construct_antipode(pose)
        return pose

    def initialize_pose(self, i2d, return_resampled=False):
        # Preprocess X-ray image and get imaging system intrinsics.
        gt, sdd, delx, dely, x0, y0, pf_to_af = read_xray(
            i2d, self.crop, self.subtract_background, self.linearize, self.reducefn
        )

        # First network prediction. If DetectorActiveOrigin is absent, read_xray uses
        # y0=0 as a temporary value; the value is estimated immediately afterwards.
        init_pose, resampled_gt = predict_pose(
            self.model, self.config, gt, sdd, delx, dely, x0, y0
        )
        init_pose = self._correct_initial_pose(init_pose)

        detector_origin_present = self._detector_active_origin_present(i2d)
        y0_metadata = self.save_kwargs["y0_estimation"]
        y0_metadata["detector_active_origin_present"] = detector_origin_present
        y0_metadata["dicom_y0_mm"] = float(y0)

        if self.estimate_missing_y0 and not detector_origin_present:
            y0 = self._estimate_missing_y0(gt, init_pose, sdd, delx, dely, x0)

            # The model's resampling depends on y0, so re-run the pose regressor using
            # the estimated principal point before iterative registration begins.
            init_pose, resampled_gt = predict_pose(
                self.model, self.config, gt, sdd, delx, dely, x0, y0
            )
            init_pose = self._correct_initial_pose(init_pose)

            # Pose and principal point are coupled. After correcting the model input
            # with the first y0 estimate, perform one local y0 refinement and then
            # make the final network prediction used to start iterative registration.
            y0 = self._refine_estimated_y0(
                gt, init_pose, sdd, delx, dely, x0, y0
            )
            init_pose, resampled_gt = predict_pose(
                self.model, self.config, gt, sdd, delx, dely, x0, y0
            )
            init_pose = self._correct_initial_pose(init_pose)
        elif detector_origin_present:
            y0_metadata["reason"] = "DetectorActiveOrigin present in DICOM"
        else:
            y0_metadata["reason"] = "automatic y0 estimation disabled"

        # For debugging, let the user return the resampled ground truth for comparison.
        if return_resampled:
            return gt, sdd, delx, dely, x0, y0, pf_to_af, init_pose, resampled_gt

        return gt, sdd, delx, dely, x0, y0, pf_to_af, init_pose

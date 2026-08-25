from __future__ import annotations

import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform


def projection_to_image_pixels(
    points: torch.Tensor,
    width: int | float,
    *,
    orientation: str = "AP",
    reverse_x_axis: bool = False,
) -> torch.Tensor:
    """
    Convert ``DRR.perspective_projection`` output to the AP raster convention
    used by the vertebra datasets.

    Important: the external raster convention and DiffDRR's detector convention
    are two different things.

    ``test_corrected`` is paired with ``reverse_x_axis=False`` and
    ``test_reverse`` is its horizontal mirror paired with
    ``reverse_x_axis=True``. DiffDRR itself already changes the returned
    perspective-projection x coordinate when reverse-x is enabled. Because the
    corresponding real X-ray/landmark dataset is mirrored at the same time, the
    final projection-to-raster adapter is the SAME in both paired pipelines::

        u_image = width - u_diffdrr
        v_image = v_diffdrr

    For the same physical pose (ignoring the one-pixel discrete/continuous
    distinction), DiffDRR gives approximately::

        u_proj(reverse=True) = width - u_proj(reverse=False)

    Therefore applying ``width - u`` to both modes produces two predictions
    that are horizontal mirrors of each other, exactly matching the relationship
    between ``test_corrected`` and ``test_reverse``.

    ``reverse_x_axis`` is still required so callers explicitly state which DRR
    convention produced ``points``; ``project_fiducials_to_image_pixels`` checks
    that it matches ``drr.detector.reverse_x_axis``.

    Note that this continuous projection conversion uses ``width - u``. A true
    zero-based raster-image flip uses ``width - 1 - u``; those operations must
    not be mixed.
    """
    if points.shape[-1] != 2:
        raise ValueError(
            "Expected projected 2D points with final dimension 2, "
            f"got shape {tuple(points.shape)}"
        )

    if float(width) <= 0:
        raise ValueError(f"width must be positive, got {width}")

    orientation = str(orientation).upper()
    if orientation != "AP":
        raise NotImplementedError(
            "Projection-to-raster conversion has only been validated for AP. "
            f"Got orientation={orientation!r}."
        )

    # The vertebra AP raster adapter is deliberately identical for the two
    # paired dataset/detector conventions. DiffDRR has already encoded the
    # reverse-x choice in `points`; the external test_reverse dataset is also
    # horizontally mirrored relative to test_corrected.
    image_points = points.clone()
    image_points[..., 0] = float(width) - image_points[..., 0]
    return image_points


def project_fiducials_to_image_pixels(
    drr: DRR,
    pose: RigidTransform,
    fiducials: torch.Tensor,
    *,
    orientation: str = "AP",
    reverse_x_axis: bool = False,
) -> torch.Tensor:
    """Project 3D fiducials and return AP raster-image pixel coordinates."""
    detector_reverse = bool(drr.detector.reverse_x_axis)
    requested_reverse = bool(reverse_x_axis)
    if detector_reverse != requested_reverse:
        raise ValueError(
            "Projection adapter convention does not match the DRR detector: "
            f"detector.reverse_x_axis={detector_reverse}, "
            f"adapter reverse_x_axis={requested_reverse}."
        )

    projected = drr.perspective_projection(pose, fiducials)
    return projection_to_image_pixels(
        projected,
        drr.detector.width,
        orientation=orientation,
        reverse_x_axis=requested_reverse,
    )

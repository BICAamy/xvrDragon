from __future__ import annotations

import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform


def projection_to_image_pixels(
    points: torch.Tensor,
    width: int | float,
    *,
    orientation: str = "AP",
) -> torch.Tensor:
    """
    Convert DiffDRR perspective-projection coordinates to real AP
    raster-image pixel coordinates.

    AP convention:

        u_raster = width - u_diffdrr
        v_raster = v_diffdrr

    Important:
    This is a continuous detector-coordinate conversion, so it is
    `width - u`, NOT `width - 1 - u`.

    `width - 1 - u` is only used when horizontally flipping an actual
    zero-based raster image / raster annotation.
    """

    if points.shape[-1] != 2:
        raise ValueError(
            "Expected projected points with shape [..., 2], "
            f"got {tuple(points.shape)}"
        )

    width = float(width)

    if width <= 0:
        raise ValueError(
            f"width must be positive, got {width}"
        )

    orientation = str(orientation).upper()

    if orientation != "AP":
        raise NotImplementedError(
            "Projection-to-raster adapter is currently "
            f"validated only for AP, got {orientation!r}."
        )

    image_points = points.clone()

    image_points[..., 0] = (
        width - image_points[..., 0]
    )

    return image_points


def project_fiducials_to_image_pixels(
    drr: DRR,
    pose: RigidTransform,
    fiducials: torch.Tensor,
    *,
    orientation: str = "AP",
) -> torch.Tensor:
    """
    Project 3D fiducials with DiffDRR and convert the result
    directly into real raster-image pixel coordinates.
    """

    projected = drr.perspective_projection(
        pose,
        fiducials,
    )

    return projection_to_image_pixels(
        projected,
        drr.detector.width,
        orientation=orientation,
    )
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
    Convert ``DRR.perspective_projection`` output to raster-image pixels.

    This adapter is for comparing DiffDRR projected 3D fiducials against
    real 2D image annotations stored in raster/DICOM pixel coordinates.

    The AP convention below has been validated for the vertebra pipeline
    using canonical AP images and ``reverse_x_axis=False``::

        u_image = width - u_diffdrr
        v_image = v_diffdrr

    Important
    ---------
    ``width - u`` is intentional here. DiffDRR's projected points use a
    continuous detector coordinate convention. This is different from
    horizontally mirroring a zero-based raster annotation, where the
    correct transform is ``width - 1 - u``.

    Parameters
    ----------
    points:
        Tensor with final dimension ``[..., 2]`` returned by
        ``DRR.perspective_projection``.
    width:
        Detector/image width in pixels.
    orientation:
        X-ray orientation. Only ``"AP"`` has been validated so far.
    reverse_x_axis:
        DiffDRR detector setting used to produce the projection. The
        canonical vertebra AP pipeline requires ``False``.
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

    if reverse_x_axis:
        raise ValueError(
            "The canonical vertebra AP adapter is defined for "
            "reverse_x_axis=False. Do not silently mix detector conventions."
        )

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
    """
    Project 3D fiducials and return coordinates in raster-image pixel space.

    Use this helper for real-image 2D landmark error instead of comparing
    ``DRR.perspective_projection`` output directly with DICOM/raster
    annotations.
    """
    projected = drr.perspective_projection(pose, fiducials)
    return projection_to_image_pixels(
        projected,
        drr.detector.width,
        orientation=orientation,
        reverse_x_axis=reverse_x_axis,
    )

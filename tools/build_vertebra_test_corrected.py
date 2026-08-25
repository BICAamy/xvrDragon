import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


SRC_ROOT = Path("data/vertebra/dataset/test")
DST_ROOT = Path("data/vertebra/dataset/test_corrected")

# ------------------------------------------------------------
# Canonical AP convention:
# same raster direction as DiffDRR reverse_x_axis=False
# ------------------------------------------------------------

AP_FLIP_CASES = {
    "case01",
    "case02",
    "case03",
    "case04",
    "case05",
    "case06",
    "case11",
    "case12",
    "case13",
    "case14",
    "case15",
    "case16",
    "case17",
    "case18",
    "case19",
    "case20",
}

AP_KEEP_CASES = {
    "case07",
    "case08",
    "case09",
    "case10",
}

ALL_CASES = [
    f"case{i:02d}"
    for i in range(1, 21)
]


def horizontal_flip_dicom(src, dst):
    """
    Create a derived DICOM whose PixelData is horizontally flipped.

    This is a true raster-image transformation:
        output[..., x] = input[..., W-1-x]
    """

    ds = pydicom.dcmread(src)

    arr = ds.pixel_array

    # Supports ordinary 2D and possible multiframe data:
    # last dimension is image x / column direction.
    flipped = np.ascontiguousarray(
        arr[..., ::-1]
    )

    # If source was compressed, write corrected copy uncompressed.
    if getattr(
        ds.file_meta.TransferSyntaxUID,
        "is_compressed",
        False,
    ):
        ds.decompress()

    ds.file_meta.TransferSyntaxUID = (
        ExplicitVRLittleEndian
    )

    # Make corrected image a separate derived DICOM instance.
    new_uid = generate_uid()

    ds.SOPInstanceUID = new_uid

    if hasattr(
        ds.file_meta,
        "MediaStorageSOPInstanceUID",
    ):
        ds.file_meta.MediaStorageSOPInstanceUID = (
            new_uid
        )

    ds.PixelData = flipped.tobytes()

    # Make the derived nature explicit without touching
    # geometric acquisition metadata.
    try:
        image_type = list(ds.ImageType)

        if image_type:
            image_type[0] = "DERIVED"
            ds.ImageType = image_type

    except Exception:
        pass

    ds.save_as(
        dst,
        write_like_original=False,
    )


def flip_ap_landmarks(landmarks, width):
    """
    Horizontal flip in ordinary zero-based raster coordinates:

        x_new = W - 1 - x_old

    y is unchanged.
    """

    result = copy.deepcopy(
        landmarks
    )

    points = (
        result["views"]
        ["ap"]
        ["landmarks_px"]
    )

    count = 0

    for vertebra, vertebra_points in points.items():

        for point_id, xy in vertebra_points.items():

            x = float(xy[0])
            y = float(xy[1])

            xy[0] = (
                (width - 1)
                - x
            )

            xy[1] = y

            count += 1

    return result, count


def corr_u_x(landmarks):
    """
    corr(
        AP pixel u,
        CT LPS X
    )
    """

    ct = (
        landmarks["ct"]
        ["landmarks_lps_mm"]
    )

    ap = (
        landmarks["views"]
        ["ap"]
        ["landmarks_px"]
    )

    X = []
    U = []

    for vertebra in sorted(
        set(ct) & set(ap),
        key=int,
    ):

        ids = sorted(
            set(ct[vertebra])
            & set(ap[vertebra]),
            key=int,
        )

        for point_id in ids:

            X.append(
                float(
                    ct[vertebra]
                    [point_id][0]
                )
            )

            U.append(
                float(
                    ap[vertebra]
                    [point_id][0]
                )
            )

    return float(
        np.corrcoef(
            np.asarray(U),
            np.asarray(X),
        )[0, 1]
    )


def check_landmarks_in_bounds(
    landmarks,
    width,
    height,
):

    bad = []

    points = (
        landmarks["views"]
        ["ap"]
        ["landmarks_px"]
    )

    for vertebra, vertebra_points in points.items():

        for point_id, xy in vertebra_points.items():

            x = float(xy[0])
            y = float(xy[1])

            if not (
                0 <= x < width
                and
                0 <= y < height
            ):

                bad.append(
                    (
                        vertebra,
                        point_id,
                        x,
                        y,
                    )
                )

    return bad


def copy_other_files(
    src_case,
    dst_case,
):

    """
    Copy all original case data except:
      AP.dcm
      AP_flip.dcm
      landmarks.json

    LAT, CT, mask and other auxiliary files are preserved.
    """

    for src in src_case.iterdir():

        if src.name in {
            "AP.dcm",
            "AP_flip.dcm",
            "landmarks.json",
        }:
            continue

        dst = (
            dst_case
            / src.name
        )

        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(
                src,
                dst,
            )


def main():

    if DST_ROOT.exists():
        # User said this directory is already empty,
        # but fail if something unexpectedly remains.
        leftovers = list(
            DST_ROOT.iterdir()
        )

        if leftovers:
            raise RuntimeError(
                f"{DST_ROOT} is not empty. "
                "Refusing to overwrite existing corrected data."
            )

    DST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "source_root":
            str(SRC_ROOT),

        "corrected_root":
            str(DST_ROOT),

        "canonical_ap": {
            "diffdrr_reverse_x_axis":
                False,

            "image_flip_rule":
                "horizontal raster flip",

            "landmark_flip_rule":
                "x_new = width - 1 - x_old",

            "projection_adapter_for_2d_mtre":
                "u_raster = width - u_diffdrr_internal",

            "note":
                (
                    "projection adapter is NOT the same "
                    "operation as dataset landmark flipping"
                ),
        },

        "intensity": {
            "pixel_polarity_modified":
                False,

            "policy":
                (
                    "MONOCHROME1/MONOCHROME2 polarity "
                    "is handled by xvr read_xray code, "
                    "not by rewriting source DICOM intensities."
                ),
        },

        "lat": {
            "modified":
                False,

            "note":
                (
                    "LAT is copied unchanged. "
                    "LAT convention will be audited separately."
                ),
        },

        "cases": {},
    }

    rows = []

    for case in ALL_CASES:

        print()
        print("=" * 70)
        print(case)
        print("=" * 70)

        src_case = (
            SRC_ROOT
            / case
        )

        dst_case = (
            DST_ROOT
            / case
        )

        if not src_case.exists():
            raise FileNotFoundError(
                src_case
            )

        src_ap = (
            src_case
            / "AP.dcm"
        )

        src_json = (
            src_case
            / "landmarks.json"
        )

        if not src_ap.exists():
            raise FileNotFoundError(
                src_ap
            )

        if not src_json.exists():
            raise FileNotFoundError(
                src_json
            )

        dst_case.mkdir(
            parents=True,
            exist_ok=True,
        )

        copy_other_files(
            src_case,
            dst_case,
        )

        # ----------------------------------------------------
        # Read original AP geometry
        # ----------------------------------------------------

        ds = pydicom.dcmread(
            src_ap,
            stop_before_pixels=True,
        )

        width = int(
            ds.Columns
        )

        height = int(
            ds.Rows
        )

        # ----------------------------------------------------
        # Read original landmarks
        # ----------------------------------------------------

        with open(
            src_json,
            "r",
            encoding="utf-8",
        ) as f:

            original_landmarks = (
                json.load(f)
            )

        corr_before = corr_u_x(
            original_landmarks
        )

        # ----------------------------------------------------
        # Canonicalize AP
        # ----------------------------------------------------

        if case in AP_FLIP_CASES:

            action = "FLIP"

            horizontal_flip_dicom(
                src_ap,
                dst_case / "AP.dcm",
            )

            corrected_landmarks, n_flipped = (
                flip_ap_landmarks(
                    original_landmarks,
                    width,
                )
            )

        elif case in AP_KEEP_CASES:

            action = "KEEP"

            shutil.copy2(
                src_ap,
                dst_case / "AP.dcm",
            )

            corrected_landmarks = (
                copy.deepcopy(
                    original_landmarks
                )
            )

            n_flipped = 0

        else:

            raise RuntimeError(
                f"No AP rule for {case}"
            )

        # ----------------------------------------------------
        # Save corrected landmarks
        # ----------------------------------------------------

        with open(
            dst_case / "landmarks.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                corrected_landmarks,
                f,
                indent=2,
                ensure_ascii=False,
            )

        # ----------------------------------------------------
        # Validate output AP geometry
        # ----------------------------------------------------

        corrected_ds = pydicom.dcmread(
            dst_case / "AP.dcm"
        )

        corrected_arr = (
            corrected_ds.pixel_array
        )

        original_arr = (
            pydicom.dcmread(
                src_ap
            ).pixel_array
        )

        if corrected_arr.shape != original_arr.shape:

            raise RuntimeError(
                f"{case}: corrected AP shape changed"
            )

        if action == "FLIP":

            exact = np.array_equal(
                corrected_arr,
                original_arr[..., ::-1],
            )

            if not exact:
                raise RuntimeError(
                    f"{case}: AP flip validation failed"
                )

        else:

            exact = np.array_equal(
                corrected_arr,
                original_arr,
            )

            if not exact:
                raise RuntimeError(
                    f"{case}: AP KEEP validation failed"
                )

        corr_after = corr_u_x(
            corrected_landmarks
        )

        bad = check_landmarks_in_bounds(
            corrected_landmarks,
            width,
            height,
        )

        if bad:
            raise RuntimeError(
                f"{case}: corrected AP landmarks "
                f"out of bounds: {bad[:5]}"
            )

        # ----------------------------------------------------
        # Canonical target:
        #
        # after our image+landmark transformation,
        # all AP landmarks should have corr_u_x > 0.
        # ----------------------------------------------------

        if corr_after <= 0:

            raise RuntimeError(
                f"{case}: corrected corr_u_x "
                f"is not positive: {corr_after}"
            )

        # ----------------------------------------------------
        # Important xvr hidden-flip guard
        # ----------------------------------------------------

        patient_orientation = getattr(
            corrected_ds,
            "PatientOrientation",
            None,
        )

        primary_angle = getattr(
            corrected_ds,
            "PositionerPrimaryAngle",
            None,
        )

        would_xvr_pf_flip = False

        try:
            would_xvr_pf_flip = (
                list(patient_orientation)
                == ["P", "F"]
                and
                float(primary_angle) < 0
            )
        except Exception:
            pass

        if would_xvr_pf_flip:

            raise RuntimeError(
                f"{case}: corrected AP would still trigger "
                "xvr PF->AF hidden horizontal flip. "
                "Do not continue until read_xray behavior "
                "is made explicitly controllable."
            )

        # ----------------------------------------------------
        # case01 sanity check against supplied AP_flip.dcm
        # ----------------------------------------------------

        supplied_ap_flip_match = None

        supplied = (
            src_case
            / "AP_flip.dcm"
        )

        if supplied.exists():

            supplied_arr = (
                pydicom.dcmread(
                    supplied
                ).pixel_array
            )

            supplied_ap_flip_match = (
                np.array_equal(
                    corrected_arr,
                    supplied_arr,
                )
            )

            if (
                case == "case01"
                and
                not supplied_ap_flip_match
            ):

                raise RuntimeError(
                    "case01 generated AP does not match "
                    "the supplied AP_flip.dcm"
                )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        manifest["cases"][case] = {
            "ap_action":
                action,

            "ap_landmark_action":
                action,

            "width_px":
                width,

            "height_px":
                height,

            "corr_u_x_before":
                corr_before,

            "corr_u_x_after":
                corr_after,

            "landmarks_transformed":
                n_flipped,

            "supplied_AP_flip_match":
                supplied_ap_flip_match,

            "lat_action":
                "KEEP",

            "intensity_action":
                "NONE_IN_DATASET",
        }

        rows.append(
            (
                case,
                action,
                corr_before,
                corr_after,
                n_flipped,
            )
        )

        print(
            f"AP image      : {action}"
        )

        print(
            f"AP landmarks  : {action}"
        )

        print(
            f"corr_u_x      : "
            f"{corr_before:+.6f} "
            f"-> "
            f"{corr_after:+.6f}"
        )

        print(
            f"landmarks     : "
            f"{n_flipped} transformed"
        )

        print(
            f"LAT           : KEEP"
        )

        if supplied_ap_flip_match is not None:

            print(
                "supplied AP_flip:",
                supplied_ap_flip_match,
            )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest_path = (
        DST_ROOT
        / "correction_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL AP CANONICALIZATION SUMMARY")
    print("=" * 90)

    print(
        f"{'case':8s} "
        f"{'action':8s} "
        f"{'corr before':>14s} "
        f"{'corr after':>14s} "
        f"{'LM changed':>12s}"
    )

    for (
        case,
        action,
        before,
        after,
        n,
    ) in rows:

        print(
            f"{case:8s} "
            f"{action:8s} "
            f"{before:+14.6f} "
            f"{after:+14.6f} "
            f"{n:12d}"
        )

    print()
    print(
        "Corrected dataset:",
        DST_ROOT,
    )

    print(
        "Manifest:",
        manifest_path,
    )

    print()
    print(
        "PASS: all 20 corrected AP landmarks "
        "use the canonical corr_u_x > 0 convention."
    )

    print(
        "PASS: reverse_x_axis=False is the "
        "canonical AP renderer setting."
    )


if __name__ == "__main__":
    main()


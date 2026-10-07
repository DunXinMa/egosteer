#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# Allow direct execution:
#   python scripts/convert_openarm_o6_sim_to_egosteer.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.lerobot.openarm_o6_kinematics import (
    EgoSteerOpenArmO6Kinematics,
)


STATE_KEY = "observation.state"
ACTION_KEY = "action"
EXTRINSIC_KEY = "observation.camera.head_world2cam"
HEAD_KEY = "observation.images.head"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--calibration", required=True)

    p.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N episodes for smoke testing.",
    )

    p.add_argument(
        "--val-episodes",
        type=int,
        default=0,
        help="Reserve the final N episodes as validation.",
    )

    p.add_argument(
        "--video-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def fixed_list_float32(values: np.ndarray):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    width = values.shape[-1]

    flat = pa.array(
        values.reshape(-1),
        type=pa.float32(),
    )

    return pa.FixedSizeListArray.from_arrays(
        flat,
        width,
    )


def replace_or_add_column(
    table: pa.Table,
    name: str,
    array,
):
    idx = table.schema.get_field_index(name)

    if idx >= 0:
        return table.set_column(
            idx,
            name,
            array,
        )

    return table.append_column(
        name,
        array,
    )


def find_unique(root: Path, pattern: str):
    matches = sorted(root.rglob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one match for {pattern!r} "
            f"under {root}, got {len(matches)}:\n"
            + "\n".join(str(x) for x in matches[:20])
        )

    return matches[0]


def install_file(
    src: Path,
    dst: Path,
    mode: str,
):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)

    elif mode == "symlink":
        dst.symlink_to(
            src.resolve()
        )

    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    else:
        raise ValueError(mode)


def file_sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def parse_tasks(path: Path):
    rows = read_jsonl(path)
    result = {}

    for row in rows:
        if "task_index" not in row:
            raise RuntimeError(
                f"task row missing task_index: {row}"
            )

        name = (
            row.get("task")
            or row.get("name")
            or row.get("instruction")
        )

        if not isinstance(name, str) or not name:
            raise RuntimeError(
                f"Cannot resolve task text: {row}"
            )

        result[int(row["task_index"])] = name

    return result


def parse_episode_rows(path: Path):
    rows = read_jsonl(path)

    return {
        int(row["episode_index"]): row
        for row in rows
        if "episode_index" in row
    }


def parent_link_of_joint(
    urdf: Path,
    joint_name: str,
):
    root = ET.parse(urdf).getroot()

    joint = root.find(
        f"./joint[@name='{joint_name}']"
    )

    if joint is None:
        raise RuntimeError(
            f"URDF joint missing: {joint_name}"
        )

    return joint.find("parent").attrib["link"]


def validate_calibration(payload):
    K = np.asarray(
        payload["intrinsics"],
        dtype=np.float64,
    ).reshape(3, 3)

    world2cam = np.asarray(
        payload["head_world2cam"],
        dtype=np.float64,
    ).reshape(4, 4)

    cam2world = np.asarray(
        payload["head_cam2world"],
        dtype=np.float64,
    ).reshape(4, 4)

    if not np.isfinite(K).all():
        raise RuntimeError("K contains NaN/Inf")

    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise RuntimeError("Invalid focal length")

    if not np.allclose(
        world2cam @ cam2world,
        np.eye(4),
        atol=1e-5,
    ):
        raise RuntimeError(
            "world2cam/cam2world inverse mismatch"
        )

    if not np.allclose(
        world2cam[3],
        [0, 0, 0, 1],
        atol=1e-7,
    ):
        raise RuntimeError(
            "Invalid world2cam bottom row"
        )

    det = np.linalg.det(
        world2cam[:3, :3]
    )

    if not np.isclose(
        det,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            f"Invalid camera rotation det={det}"
        )

    return K, world2cam, cam2world


def main():
    args = parse_args()

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    calibration_path = (
        Path(args.calibration)
        .expanduser()
        .resolve()
    )

    if not src.is_dir():
        raise FileNotFoundError(src)

    if not urdf.is_file():
        raise FileNotFoundError(urdf)

    if not calibration_path.is_file():
        raise FileNotFoundError(
            calibration_path
        )

    if dst.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Output already exists: {dst}\n"
                "Use --overwrite only for a disposable smoke output."
            )

        shutil.rmtree(dst)

    src_info = json.loads(
        (src / "meta/info.json")
        .read_text(encoding="utf-8")
    )

    if src_info.get("codebase_version") != "v2.0":
        raise RuntimeError(
            "This converter expects LeRobot v2.0 input, "
            f"got {src_info.get('codebase_version')!r}"
        )

    fps = float(src_info["fps"])

    if not np.isclose(
        fps,
        30.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"EgoSteer pipeline expects 30 fps, got {fps}"
        )

    source_total = int(
        src_info["total_episodes"]
    )

    total_episodes = source_total

    if args.max_episodes is not None:
        total_episodes = min(
            total_episodes,
            int(args.max_episodes),
        )

    if total_episodes <= 0:
        raise RuntimeError(
            "No episodes selected"
        )

    val_episodes = int(
        args.val_episodes
    )

    if not (
        0 <= val_episodes < total_episodes
    ):
        raise RuntimeError(
            "val-episodes must satisfy "
            "0 <= val < total_episodes"
        )

    source_episode_rows = parse_episode_rows(
        src / "meta/episodes.jsonl"
    )

    # EgoSteer uses one high-level task per episode.
    # The source LeRobot-v2 tasks.jsonl contains phase-conditioned
    # prompts (e.g. place mug / grasp bottle / pour / return bottle),
    # which must NOT become EgoSteer task identities.
    task_text_to_index = {}
    episode_task_map = {}

    for eid in range(total_episodes):
        source_episode = source_episode_rows.get(eid)

        if source_episode is None:
            raise RuntimeError(
                f"episode {eid}: missing from meta/episodes.jsonl"
            )

        task_text = source_episode.get("task")

        if not isinstance(task_text, str) or not task_text.strip():
            raise RuntimeError(
                f"episode {eid}: missing non-empty high-level "
                "task in meta/episodes.jsonl"
            )

        task_text = task_text.strip()

        if task_text not in task_text_to_index:
            task_text_to_index[task_text] = len(task_text_to_index)

        canonical_task_index = task_text_to_index[task_text]

        episode_task_map[eid] = (
            canonical_task_index,
            task_text,
        )

    tasks = {
        index: text
        for text, index in task_text_to_index.items()
    }

    calibration = json.loads(
        calibration_path.read_text(
            encoding="utf-8"
        )
    )

    K, world2cam, cam2world = (
        validate_calibration(
            calibration
        )
    )

    kin = EgoSteerOpenArmO6Kinematics(
        urdf
    )

    q_zero = np.zeros(
        26,
        dtype=np.float32,
    )

    left_base_link = parent_link_of_joint(
        urdf,
        "openarm_left_joint1",
    )

    right_base_link = parent_link_of_joint(
        urdf,
        "openarm_right_joint1",
    )

    left_base2world = kin.link_matrix(
        q_zero,
        left_base_link,
    )

    right_base2world = kin.link_matrix(
        q_zero,
        right_base_link,
    )

    # T_cam_to_base maps optical-camera coordinates into
    # the corresponding arm-base coordinate system.
    head_cam_to_left_base = (
        np.linalg.inv(left_base2world)
        @ cam2world
    )

    head_cam_to_right_base = (
        np.linalg.inv(right_base2world)
        @ cam2world
    )

    source_head_feature = (
        src_info["features"][HEAD_KEY]
    )

    source_video_info = dict(
        source_head_feature.get("info")
        or {}
    )

    height = int(
        source_video_info.get(
            "video.height",
            480,
        )
    )

    width = int(
        source_video_info.get(
            "video.width",
            640,
        )
    )

    channels = int(
        source_video_info.get(
            "video.channels",
            3,
        )
    )

    output_features = {}

    # Preserve non-video auxiliary columns.
    for name, feature in (
        src_info["features"].items()
    ):
        if feature.get("dtype") == "video":
            continue

        if name in (
            STATE_KEY,
            ACTION_KEY,
        ):
            continue

        output_features[name] = feature

    output_features[STATE_KEY] = {
        "dtype": "float32",
        "shape": [74],
    }

    output_features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": [74],
    }

    output_features[EXTRINSIC_KEY] = {
        "dtype": "float32",
        "shape": [16],
    }

    output_features[HEAD_KEY] = {
        "dtype": "video",
        "shape": [
            height,
            width,
            channels,
        ],
        "names": [
            "height",
            "width",
            "channels",
        ],
        "info": source_video_info,
    }

    dst.mkdir(
        parents=True,
        exist_ok=False,
    )

    (dst / "meta").mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []
    total_frames = 0

    camera_flat = (
        world2cam.astype(np.float32)
        .reshape(16)
    )

    print(
        f"Converting {total_episodes} episodes..."
    )

    for eid in range(total_episodes):
        source_parquet = find_unique(
            src / "data",
            f"episode_{eid:06d}.parquet",
        )

        table = pq.read_table(
            source_parquet
        )

        n = len(table)

        if n <= 0:
            raise RuntimeError(
                f"episode {eid}: empty"
            )

        states26 = np.stack(
            table[STATE_KEY].to_numpy()
        ).astype(
            np.float32,
            copy=False,
        )

        if states26.shape != (n, 26):
            raise RuntimeError(
                f"episode {eid}: "
                f"state shape={states26.shape}"
            )

        if not np.isfinite(
            states26
        ).all():
            raise RuntimeError(
                f"episode {eid}: "
                "state has NaN/Inf"
            )

        frame_index = np.asarray(
            table["frame_index"]
        )

        if not np.array_equal(
            frame_index,
            np.arange(
                n,
                dtype=frame_index.dtype,
            ),
        ):
            raise RuntimeError(
                f"episode {eid}: "
                "non-contiguous frame_index"
            )

        state74 = (
            kin.native26_to_storage74(
                states26
            )
        )

        next_states26 = np.concatenate(
            [
                states26[1:],
                states26[-1:],
            ],
            axis=0,
        )

        action74 = (
            kin.native26_to_storage74(
                next_states26
            )
        )

        if (
            state74.shape != (n, 74)
            or action74.shape != (n, 74)
        ):
            raise RuntimeError(
                f"episode {eid}: "
                "74D conversion shape failure"
            )

        if not np.array_equal(
            state74[:, :26],
            states26,
        ):
            raise RuntimeError(
                f"episode {eid}: native26 "
                "not preserved in state74"
            )

        if not np.array_equal(
            action74[:, :26],
            next_states26,
        ):
            raise RuntimeError(
                f"episode {eid}: native26 "
                "not preserved in action74"
            )

        extrinsics = np.repeat(
            camera_flat[None],
            n,
            axis=0,
        )

        table = replace_or_add_column(
            table,
            STATE_KEY,
            fixed_list_float32(
                state74
            ),
        )

        table = replace_or_add_column(
            table,
            ACTION_KEY,
            fixed_list_float32(
                action74
            ),
        )

        table = replace_or_add_column(
            table,
            EXTRINSIC_KEY,
            fixed_list_float32(
                extrinsics
            ),
        )

        # Canonicalize indexing rather than trusting stale v2 globals.
        table = replace_or_add_column(
            table,
            "timestamp",
            pa.array(
                np.arange(
                    n,
                    dtype=np.float32,
                )
                / np.float32(fps),
                type=pa.float32(),
            ),
        )

        table = replace_or_add_column(
            table,
            "frame_index",
            pa.array(
                np.arange(
                    n,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
        )

        table = replace_or_add_column(
            table,
            "episode_index",
            pa.array(
                np.full(
                    n,
                    eid,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
        )

        table = replace_or_add_column(
            table,
            "index",
            pa.array(
                np.arange(
                    total_frames,
                    total_frames + n,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
        )

        canonical_task_index, task_text = (
            episode_task_map[eid]
        )

        # Collapse source phase-conditioned task ids into one
        # high-level EgoSteer task id for the whole episode.
        table = replace_or_add_column(
            table,
            "task_index",
            pa.array(
                np.full(
                    n,
                    canonical_task_index,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
        )

        chunk_index = (
            eid // args.chunk_size
        )

        file_index = eid

        output_data = (
            dst
            / "data"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:06d}.parquet"
        )

        output_data.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pq.write_table(
            table,
            output_data,
            compression="zstd",
        )

        # Source v2 contains head + two wrist videos with the same
        # episode filename.  Select the head stream explicitly.
        head_matches = [
            p
            for p in (
                src / "videos"
            ).rglob(
                f"episode_{eid:06d}.mp4"
            )
            if p.parent.name == HEAD_KEY
        ]

        if len(head_matches) != 1:
            raise RuntimeError(
                f"episode {eid}: expected exactly one "
                f"{HEAD_KEY} video, got {len(head_matches)}:\n"
                + "\n".join(
                    str(x)
                    for x in head_matches
                )
            )

        source_video = head_matches[0]

        output_video = (
            dst
            / "videos"
            / HEAD_KEY
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:06d}.mp4"
        )

        install_file(
            source_video,
            output_video,
            args.video_mode,
        )

        source_episode = source_episode_rows[eid]

        # EgoSteer episode metadata uses the same high-level task
        # that was written into every frame's canonical task_index.
        episode_tasks = [
            task_text
        ]

        source_instructions = (
            source_episode.get(
                "instructions"
            )
        )

        if (
            isinstance(
                source_instructions,
                list,
            )
            and source_instructions
            and all(
                isinstance(x, str)
                and x
                for x
                in source_instructions
            )
        ):
            instructions = (
                source_instructions
            )
        else:
            instructions = [
                episode_tasks[0]
            ]

        metadata_rows.append(
            {
                "episode_index": eid,
                "length": n,
                "tasks": episode_tasks,
                "instructions": instructions,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + n,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "calibration/head_intrinsics":
                    K.reshape(-1).tolist(),
                "calibration/head_cam_to_left_base":
                    head_cam_to_left_base.reshape(-1).tolist(),
                "calibration/head_cam_to_right_base":
                    head_cam_to_right_base.reshape(-1).tolist(),
                f"videos/{HEAD_KEY}/chunk_index":
                    chunk_index,
                f"videos/{HEAD_KEY}/file_index":
                    file_index,
                f"videos/{HEAD_KEY}/from_timestamp":
                    0.0,
                f"videos/{HEAD_KEY}/to_timestamp":
                    float(n / fps),
            }
        )

        total_frames += n

        if (
            eid < 3
            or (eid + 1) % 50 == 0
            or eid + 1 == total_episodes
        ):
            print(
                f"[{eid + 1}/{total_episodes}] "
                f"frames={n} "
                f"total_frames={total_frames}"
            )

    # ------------------------------------------------------------------
    # tasks.parquet
    # ------------------------------------------------------------------

    task_rows = [
        {
            "task_index": idx,
            "task": text,
        }
        for idx, text
        in sorted(tasks.items())
    ]

    pq.write_table(
        pa.Table.from_pylist(
            task_rows
        ),
        dst / "meta/tasks.parquet",
        compression="zstd",
    )

    # ------------------------------------------------------------------
    # episode metadata
    # ------------------------------------------------------------------

    episodes_dir = (
        dst
        / "meta"
        / "episodes"
        / "chunk-000"
    )

    episodes_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        pa.Table.from_pylist(
            metadata_rows
        ),
        episodes_dir
        / "file-000.parquet",
        compression="zstd",
    )

    train_stop = (
        total_episodes
        - val_episodes
    )

    splits = {
        "train": f"0:{train_stop}",
    }

    if val_episodes:
        splits["val"] = (
            f"{train_stop}:"
            f"{total_episodes}"
        )

    output_info = {
        "codebase_version": "v3.0",
        "robot_type":
            src_info.get(
                "robot_type",
                "openarm_linkerhand_o6",
            ),
        "fps": fps,
        "total_episodes":
            total_episodes,
        "total_frames":
            total_frames,
        "total_videos":
            total_episodes,
        "total_chunks":
            (
                total_episodes
                + args.chunk_size
                - 1
            )
            // args.chunk_size,
        "chunks_size":
            args.chunk_size,
        "data_path":
            "data/chunk-{chunk_index:03d}/"
            "file-{file_index:06d}.parquet",
        "video_path":
            "videos/{video_key}/"
            "chunk-{chunk_index:03d}/"
            "file-{file_index:06d}.mp4",
        "splits": splits,
        "features":
            output_features,
    }

    (
        dst / "meta/info.json"
    ).write_text(
        json.dumps(
            output_info,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    provenance = {
        "schema":
            "openarm_o6_to_egosteer_v1",
        "source_dataset":
            str(src),
        "source_codebase_version":
            src_info.get(
                "codebase_version"
            ),
        "urdf":
            str(urdf),
        "calibration":
            str(calibration_path),
        "calibration_sha256":
            file_sha256(
                calibration_path
            ),
        "action_semantics":
            "realized_next_state_q_t_plus_1",
        "task_semantics":
            "episode_high_level_task",
        "source_phase_task_ids_dropped":
            True,
        "last_action":
            "repeat_final_realized_state",
        "world_frame":
            calibration.get(
                "world_frame"
            ),
        "camera_convention":
            calibration.get(
                "coordinate_convention"
            ),
        "left_arm_base_link":
            left_base_link,
        "right_arm_base_link":
            right_base_link,
        "video_mode":
            args.video_mode,
    }

    (
        dst
        / "meta"
        / "openarm_o6_egosteer_conversion.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print("output        =", dst)
    print("episodes      =", total_episodes)
    print("frames        =", total_frames)
    print("splits        =", splits)
    print("left base     =", left_base_link)
    print("right base    =", right_base_link)
    print(
        "calib sha256  =",
        provenance[
            "calibration_sha256"
        ],
    )
    print(
        "OPENARM_O6_TO_EGOSTEER_PASS"
    )


if __name__ == "__main__":
    main()

"""OpenArm + LinkerHand O6 kinematics for the OpenWAM Alpha 80-D adapter.

This module intentionally has no Pinocchio / Isaac Sim dependency.
It parses the validated OpenArm URDF and computes FK from the native
7-DoF arm joint positions to the corresponding hand-base frame.

Native OpenArm O6:
    [L arm 7, R arm 7, L hand 6, R hand 6] = 26-D

Semantic OpenWAM raw representation:
    [L xyz 3, L rot6d 6, L hand 6,
     R xyz 3, R rot6d 6, R hand 6] = 30-D
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


NATIVE_DIM = 26
SEMANTIC_DIM = 30

LEFT_ARM_SLICE = slice(0, 7)
RIGHT_ARM_SLICE = slice(7, 14)
LEFT_HAND_SLICE = slice(14, 20)
RIGHT_HAND_SLICE = slice(20, 26)

LEFT_POSE_SLICE = slice(0, 9)
LEFT_SEM_HAND_SLICE = slice(9, 15)
RIGHT_POSE_SLICE = slice(15, 24)
RIGHT_SEM_HAND_SLICE = slice(24, 30)

ROT6D_DIMS_SEMANTIC30 = tuple(range(3, 9)) + tuple(range(18, 24))


def _vec(text: str | None, n: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(n, default, dtype=np.float64)
    arr = np.fromstring(text, sep=" ", dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"Expected {n} values, got {text!r}")
    return arr


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, cr, -sr],
         [0.0, sr, cr]],
        dtype=np.float64,
    )

    ry = np.array(
        [[cp, 0.0, sp],
         [0.0, 1.0, 0.0],
         [-sp, 0.0, cp]],
        dtype=np.float64,
    )

    rz = np.array(
        [[cy, -sy, 0.0],
         [sy, cy, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    return rz @ ry @ rx


def _origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _rpy_matrix(rpy)
    t[:3, 3] = xyz
    return t


def _axis_angle_batch(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))

    if norm < 1e-12:
        raise ValueError("Revolute joint has zero-length axis")

    axis = axis / norm
    x, y, z = axis

    k = np.array(
        [[0.0, -z, y],
         [z, 0.0, -x],
         [-y, x, 0.0]],
        dtype=np.float64,
    )

    outer = np.outer(axis, axis)
    eye = np.eye(3, dtype=np.float64)

    angle = np.asarray(angle, dtype=np.float64)
    c = np.cos(angle)[..., None, None]
    s = np.sin(angle)[..., None, None]

    return c * eye + (1.0 - c) * outer + s * k


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """OpenWAM rot6d convention: first two rotation-matrix columns."""
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected (...,3,3) rotation matrix, got {matrix.shape}"
        )

    return np.concatenate(
        [matrix[..., :, 0], matrix[..., :, 1]],
        axis=-1,
    ).astype(np.float32)


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray | None
    q_index: int | None


class OpenArmO6Kinematics:
    """URDF FK for both OpenArm 7-DoF arms."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path).expanduser().resolve()

        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)

        root = ET.parse(self.urdf_path).getroot()

        self._joint_by_child = {}

        for elem in root.findall("joint"):
            name = elem.attrib["name"]
            joint_type = elem.attrib["type"]

            parent = elem.find("parent").attrib["link"]
            child = elem.find("child").attrib["link"]

            origin_elem = elem.find("origin")

            xyz = _vec(
                origin_elem.attrib.get("xyz") if origin_elem is not None else None,
                3,
            )
            rpy = _vec(
                origin_elem.attrib.get("rpy") if origin_elem is not None else None,
                3,
            )

            axis_elem = elem.find("axis")

            if joint_type in ("revolute", "continuous"):
                axis = _vec(
                    axis_elem.attrib.get("xyz") if axis_elem is not None else "1 0 0",
                    3,
                )
            else:
                axis = None

            if child in self._joint_by_child:
                raise ValueError(
                    f"URDF child link {child!r} has more than one parent joint"
                )

            self._joint_by_child[child] = {
                "name": name,
                "joint_type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _origin_transform(xyz, rpy),
                "axis": axis,
            }

        self._chains = {
            "left": self._build_chain(
                target="L_hand_base_link",
                expected_arm_joints=[
                    f"openarm_left_joint{i}" for i in range(1, 8)
                ],
            ),
            "right": self._build_chain(
                target="R_hand_base_link",
                expected_arm_joints=[
                    f"openarm_right_joint{i}" for i in range(1, 8)
                ],
            ),
        }

    def _build_chain(
        self,
        *,
        target: str,
        expected_arm_joints: list[str],
    ) -> tuple[JointSpec, ...]:
        chain_raw = []
        cur = target

        while cur in self._joint_by_child:
            entry = self._joint_by_child[cur]
            chain_raw.append(entry)
            cur = entry["parent"]

        chain_raw.reverse()

        q_index = {
            name: i
            for i, name in enumerate(expected_arm_joints)
        }

        variable_names = [
            j["name"]
            for j in chain_raw
            if j["joint_type"] in ("revolute", "continuous")
        ]

        if variable_names != expected_arm_joints:
            raise ValueError(
                f"{target} arm chain mismatch.\n"
                f"expected={expected_arm_joints}\n"
                f"actual={variable_names}"
            )

        chain = []

        for j in chain_raw:
            jt = j["joint_type"]

            if jt not in ("fixed", "revolute", "continuous"):
                raise NotImplementedError(
                    f"Unsupported joint type {jt!r} in {target} chain"
                )

            idx = q_index.get(j["name"])

            if jt in ("revolute", "continuous") and idx is None:
                raise ValueError(
                    f"Unexpected actuated joint {j['name']!r} in {target} chain"
                )

            chain.append(
                JointSpec(
                    name=j["name"],
                    joint_type=jt,
                    parent=j["parent"],
                    child=j["child"],
                    origin=j["origin"],
                    axis=j["axis"],
                    q_index=idx,
                )
            )

        return tuple(chain)

    def fk_matrix(self, q7: np.ndarray, side: str) -> np.ndarray:
        q = np.asarray(q7, dtype=np.float64)

        if q.shape[-1] != 7:
            raise ValueError(
                f"{side} arm q must end in 7 dims, got {q.shape}"
            )

        if side not in self._chains:
            raise ValueError(f"Unknown side={side!r}")

        batch_shape = q.shape[:-1]

        t = np.broadcast_to(
            np.eye(4, dtype=np.float64),
            batch_shape + (4, 4),
        ).copy()

        for joint in self._chains[side]:
            t = np.matmul(t, joint.origin)

            if joint.q_index is not None:
                rot = _axis_angle_batch(
                    joint.axis,
                    q[..., joint.q_index],
                )

                motion = np.zeros(
                    batch_shape + (4, 4),
                    dtype=np.float64,
                )
                motion[..., :3, :3] = rot
                motion[..., 3, 3] = 1.0

                t = np.matmul(t, motion)

        return t

    def pose9(self, q7: np.ndarray, side: str) -> np.ndarray:
        t = self.fk_matrix(q7, side)

        pos = t[..., :3, 3].astype(np.float32)
        rot6d = matrix_to_rot6d(t[..., :3, :3])

        out = np.concatenate([pos, rot6d], axis=-1)

        if out.shape[-1] != 9 or not np.isfinite(out).all():
            raise RuntimeError(
                f"Invalid {side} FK pose: shape={out.shape}, "
                f"finite={np.isfinite(out).all()}"
            )

        return out.astype(np.float32)

    def native26_to_semantic30(self, native: np.ndarray) -> np.ndarray:
        arr = np.asarray(native, dtype=np.float32)

        if arr.shape[-1] != NATIVE_DIM:
            raise ValueError(
                f"native OpenArm action/state must end in 26 dims, got {arr.shape}"
            )

        left_pose = self.pose9(arr[..., LEFT_ARM_SLICE], "left")
        right_pose = self.pose9(arr[..., RIGHT_ARM_SLICE], "right")

        out = np.concatenate(
            [
                left_pose,
                arr[..., LEFT_HAND_SLICE],
                right_pose,
                arr[..., RIGHT_HAND_SLICE],
            ],
            axis=-1,
        ).astype(np.float32)

        if out.shape[-1] != SEMANTIC_DIM:
            raise RuntimeError(
                f"semantic conversion produced {out.shape}, expected (...,30)"
            )

        if not np.isfinite(out).all():
            raise RuntimeError("semantic30 contains NaN/Inf")

        return out


__all__ = [
    "NATIVE_DIM",
    "SEMANTIC_DIM",
    "ROT6D_DIMS_SEMANTIC30",
    "OpenArmO6Kinematics",
    "matrix_to_rot6d",
]


# ============================================================================
# EgoSteer OpenArm + LinkerHand O6 extension
#
# Raw EgoSteer LeRobot storage:
#
#   native26
#   + left wrist xyz+rot6d  9
#   + right wrist xyz+rot6d 9
#   + left fingertips       15
#   + right fingertips      15
#   = 74D
#
# Fingertip order:
#   thumb, index, middle, ring, pinky
#
# IMPORTANT:
# Runtime Isaac Sim geometry validation showed that the O6 mimic distal
# rotations use the opposite sign from the exported URDF <mimic multiplier>.
# The runtime-equivalent FK therefore uses:
#
#   four fingers: DIP = +0.89 * MCP
#   left thumb:    DIP = +2.29 * CMC pitch
#   right thumb:   DIP = +1.86 * CMC pitch
#
# Do not apply the exported mimic-joint position limits as clipping.
# ============================================================================

EGOSTEER_STORAGE48_DIM = 48
EGOSTEER_LEROBOT74_DIM = 74

FINGERTIP_ORDER = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
)

_LEFT_HAND_DRIVER_NAMES = (
    "L_index_mcp_pitch",
    "L_middle_mcp_pitch",
    "L_pinky_mcp_pitch",
    "L_ring_mcp_pitch",
    "L_thumb_cmc_pitch",
    "L_thumb_cmc_yaw",
)

_RIGHT_HAND_DRIVER_NAMES = (
    "R_index_mcp_pitch",
    "R_middle_mcp_pitch",
    "R_pinky_mcp_pitch",
    "R_ring_mcp_pitch",
    "R_thumb_cmc_pitch",
    "R_thumb_cmc_yaw",
)

_NATIVE_JOINT_NAMES = (
    *(f"openarm_left_joint{i}" for i in range(1, 8)),
    *(f"openarm_right_joint{i}" for i in range(1, 8)),
    *_LEFT_HAND_DRIVER_NAMES,
    *_RIGHT_HAND_DRIVER_NAMES,
)

_RUNTIME_MIMIC_JOINTS = {
    "L_index_dip",
    "L_middle_dip",
    "L_pinky_dip",
    "L_ring_dip",
    "L_thumb_dip",
    "R_index_dip",
    "R_middle_dip",
    "R_pinky_dip",
    "R_ring_dip",
    "R_thumb_dip",
}

_FINGERTIP_LINK = {
    ("left", "thumb"): "L_thumb_distal",
    ("left", "index"): "L_index_distal",
    ("left", "middle"): "L_middle_distal",
    ("left", "ring"): "L_ring_distal",
    ("left", "pinky"): "L_pinky_distal",
    ("right", "thumb"): "R_thumb_distal",
    ("right", "index"): "R_index_distal",
    ("right", "middle"): "R_middle_distal",
    ("right", "ring"): "R_ring_distal",
    ("right", "pinky"): "R_pinky_distal",
}

# Distal-mesh terminal-cap centroids from the validated Sim6 O6 meshes.
# Coordinates are expressed in each distal-link local frame.
_FINGERTIP_LOCAL = {
    ("left", "thumb"): np.array(
        [-0.00708365, -0.00008035, 0.04816084],
        dtype=np.float64,
    ),
    ("left", "index"): np.array(
        [0.01532759, 0.00002511, 0.03710665],
        dtype=np.float64,
    ),
    ("left", "middle"): np.array(
        [0.01532759, 0.00002511, 0.03710665],
        dtype=np.float64,
    ),
    ("left", "ring"): np.array(
        [0.01532759, 0.00002511, 0.03710665],
        dtype=np.float64,
    ),
    ("left", "pinky"): np.array(
        [0.01532759, 0.00002511, 0.03710665],
        dtype=np.float64,
    ),
    ("right", "thumb"): np.array(
        [-0.00705329, -0.00004820, 0.04816366],
        dtype=np.float64,
    ),
    ("right", "index"): np.array(
        [0.01532296, 0.00000789, 0.03710350],
        dtype=np.float64,
    ),
    ("right", "middle"): np.array(
        [0.01532296, 0.00000789, 0.03710350],
        dtype=np.float64,
    ),
    ("right", "ring"): np.array(
        [0.01532296, 0.00000789, 0.03710350],
        dtype=np.float64,
    ),
    ("right", "pinky"): np.array(
        [0.01532296, 0.00000789, 0.03710350],
        dtype=np.float64,
    ),
}


class EgoSteerOpenArmO6Kinematics(OpenArmO6Kinematics):
    """Runtime-validated OpenArm+O6 FK for EgoSteer LeRobot conversion."""

    def __init__(self, urdf_path: str | Path):
        super().__init__(urdf_path)

        root = ET.parse(self.urdf_path).getroot()

        self._full_joint_by_child = {}

        for elem in root.findall("joint"):
            name = elem.attrib["name"]
            joint_type = elem.attrib["type"]

            parent = elem.find("parent").attrib["link"]
            child = elem.find("child").attrib["link"]

            origin_elem = elem.find("origin")

            xyz = _vec(
                origin_elem.attrib.get("xyz")
                if origin_elem is not None
                else None,
                3,
            )

            rpy = _vec(
                origin_elem.attrib.get("rpy")
                if origin_elem is not None
                else None,
                3,
            )

            axis_elem = elem.find("axis")

            if joint_type in ("revolute", "continuous"):
                axis = _vec(
                    axis_elem.attrib.get("xyz")
                    if axis_elem is not None
                    else "1 0 0",
                    3,
                )
            else:
                axis = None

            mimic_elem = elem.find("mimic")

            mimic = None

            if mimic_elem is not None:
                mimic = {
                    "driver": mimic_elem.attrib["joint"],
                    "multiplier": float(
                        mimic_elem.attrib.get("multiplier", "1")
                    ),
                    "offset": float(
                        mimic_elem.attrib.get("offset", "0")
                    ),
                }

            if child in self._full_joint_by_child:
                raise ValueError(
                    f"URDF child link {child!r} has multiple parent joints"
                )

            self._full_joint_by_child[child] = {
                "name": name,
                "joint_type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _origin_transform(xyz, rpy),
                "axis": axis,
                "mimic": mimic,
            }

    @staticmethod
    def _native_named_q(native: np.ndarray) -> dict[str, np.ndarray]:
        arr = np.asarray(native, dtype=np.float64)

        if arr.shape[-1] != NATIVE_DIM:
            raise ValueError(
                f"native OpenArm state must end in 26 dims, got {arr.shape}"
            )

        return {
            name: arr[..., i]
            for i, name in enumerate(_NATIVE_JOINT_NAMES)
        }

    def _raw_chain(self, target: str) -> tuple[dict, ...]:
        chain = []
        cur = target

        while cur in self._full_joint_by_child:
            joint = self._full_joint_by_child[cur]
            chain.append(joint)
            cur = joint["parent"]

        chain.reverse()
        return tuple(chain)

    def link_matrix(
        self,
        native: np.ndarray,
        target: str,
    ) -> np.ndarray:
        """FK from URDF root to target link using runtime-equivalent O6 mimic signs."""

        arr = np.asarray(native, dtype=np.float64)

        if arr.shape[-1] != NATIVE_DIM:
            raise ValueError(
                f"native OpenArm state must end in 26 dims, got {arr.shape}"
            )

        q_named = self._native_named_q(arr)
        batch_shape = arr.shape[:-1]

        t = np.broadcast_to(
            np.eye(4, dtype=np.float64),
            batch_shape + (4, 4),
        ).copy()

        for joint in self._raw_chain(target):
            t = np.matmul(t, joint["origin"])

            jt = joint["joint_type"]

            if jt == "fixed":
                continue

            if jt not in ("revolute", "continuous"):
                raise NotImplementedError(
                    f"Unsupported joint type {jt!r} in chain to {target!r}"
                )

            mimic = joint["mimic"]

            if mimic is not None:
                name = joint["name"]

                if name not in _RUNTIME_MIMIC_JOINTS:
                    raise RuntimeError(
                        f"Unexpected mimic joint {name!r}"
                    )

                # Runtime geometry gate:
                # simulator/physical link orientation corresponds to
                # the sign-flipped exported URDF multiplier.
                urdf_multiplier = float(mimic["multiplier"])

                if urdf_multiplier >= 0.0:
                    raise RuntimeError(
                        f"{name}: expected negative exported mimic "
                        f"multiplier, got {urdf_multiplier}"
                    )

                if abs(float(mimic["offset"])) > 1e-12:
                    raise RuntimeError(
                        f"{name}: unexpected non-zero mimic offset "
                        f"{mimic['offset']}"
                    )

                q = (
                    -urdf_multiplier
                    * q_named[mimic["driver"]]
                )
            else:
                name = joint["name"]

                if name not in q_named:
                    raise RuntimeError(
                        f"No native OpenArm q value for joint {name!r}"
                    )

                q = q_named[name]

            rot = _axis_angle_batch(
                joint["axis"],
                q,
            )

            motion = np.zeros(
                batch_shape + (4, 4),
                dtype=np.float64,
            )

            motion[..., :3, :3] = rot
            motion[..., 3, 3] = 1.0

            t = np.matmul(t, motion)

        return t

    @staticmethod
    def _apply_world_root(
        t_root: np.ndarray,
        T_world_root: np.ndarray | None,
    ) -> np.ndarray:
        if T_world_root is None:
            return t_root

        transform = np.asarray(
            T_world_root,
            dtype=np.float64,
        )

        if transform.shape != (4, 4):
            raise ValueError(
                "T_world_root must have shape (4,4), "
                f"got {transform.shape}"
            )

        return np.matmul(
            transform,
            t_root,
        )

    @staticmethod
    def _matrix_to_pose9(t: np.ndarray) -> np.ndarray:
        pos = t[..., :3, 3].astype(np.float32)

        rot6d = matrix_to_rot6d(
            t[..., :3, :3]
        )

        return np.concatenate(
            [pos, rot6d],
            axis=-1,
        ).astype(np.float32)

    def fingertips30(
        self,
        native: np.ndarray,
        *,
        T_world_root: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return L then R fingertips, each thumb/index/middle/ring/pinky XYZ."""

        arr = np.asarray(
            native,
            dtype=np.float64,
        )

        points = []

        for side in ("left", "right"):
            for finger in FINGERTIP_ORDER:
                target = _FINGERTIP_LINK[
                    (side, finger)
                ]

                local = _FINGERTIP_LOCAL[
                    (side, finger)
                ]

                t = self.link_matrix(
                    arr,
                    target,
                )

                t = self._apply_world_root(
                    t,
                    T_world_root,
                )

                point = (
                    np.einsum(
                        "...ij,j->...i",
                        t[..., :3, :3],
                        local,
                    )
                    + t[..., :3, 3]
                )

                points.append(
                    point.astype(np.float32)
                )

        out = np.concatenate(
            points,
            axis=-1,
        ).astype(np.float32)

        if (
            out.shape[-1] != 30
            or not np.isfinite(out).all()
        ):
            raise RuntimeError(
                f"Invalid fingertip30 output: {out.shape}"
            )

        return out

    def native26_to_storage48(
        self,
        native: np.ndarray,
        *,
        T_world_root: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert native26 to EgoSteer raw 48D LeRobot storage representation."""

        arr = np.asarray(
            native,
            dtype=np.float32,
        )

        if arr.shape[-1] != NATIVE_DIM:
            raise ValueError(
                f"native OpenArm state must end in 26 dims, got {arr.shape}"
            )

        left_t = self.fk_matrix(
            arr[..., LEFT_ARM_SLICE],
            "left",
        )

        right_t = self.fk_matrix(
            arr[..., RIGHT_ARM_SLICE],
            "right",
        )

        left_t = self._apply_world_root(
            left_t,
            T_world_root,
        )

        right_t = self._apply_world_root(
            right_t,
            T_world_root,
        )

        left_pose9 = self._matrix_to_pose9(
            left_t
        )

        right_pose9 = self._matrix_to_pose9(
            right_t
        )

        fingertip30 = self.fingertips30(
            arr,
            T_world_root=T_world_root,
        )

        out = np.concatenate(
            [
                left_pose9,
                right_pose9,
                fingertip30,
            ],
            axis=-1,
        ).astype(np.float32)

        if (
            out.shape[-1] != EGOSTEER_STORAGE48_DIM
            or not np.isfinite(out).all()
        ):
            raise RuntimeError(
                f"Invalid EgoSteer storage48 output: {out.shape}"
            )

        return out

    def native26_to_storage74(
        self,
        native: np.ndarray,
        *,
        T_world_root: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return official EgoSteer LeRobot 74D storage vector."""

        arr = np.asarray(
            native,
            dtype=np.float32,
        )

        storage48 = self.native26_to_storage48(
            arr,
            T_world_root=T_world_root,
        )

        out = np.concatenate(
            [
                arr,
                storage48,
            ],
            axis=-1,
        ).astype(np.float32)

        if (
            out.shape[-1] != EGOSTEER_LEROBOT74_DIM
            or not np.isfinite(out).all()
        ):
            raise RuntimeError(
                f"Invalid EgoSteer storage74 output: {out.shape}"
            )

        return out

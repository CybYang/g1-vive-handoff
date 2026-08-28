#!/usr/bin/env python3
"""Inspire G2 mapping and rate-limited serial output for live retargeting.

The serial operations reuse the range-checked adaptation of the vendor
``demo_serial_064.py`` in ``tools/demo_serial_064_safe.py``.  Importing this
module never opens a serial port and never sends a command.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEFT_MAPPING_PATH = Path(__file__).with_name("inspire_g2_left_mapping.yaml")
RIGHT_MAPPING_PATH = Path(__file__).with_name("inspire_g2_right_mapping.yaml")
# Backwards compatibility for the existing left-hand camera entry point.
DEFAULT_MAPPING_PATH = LEFT_MAPPING_PATH


@dataclass(frozen=True)
class JointMapping:
    hardware_name: str
    urdf_joint: str
    urdf_anchors: np.ndarray
    hardware_anchors: np.ndarray


def load_joint_mapping(
    path: Path | str = DEFAULT_MAPPING_PATH,
) -> tuple[JointMapping, ...]:
    """Load and validate the ordered piecewise-linear anchor mapping."""
    mapping_path = Path(path)
    data = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    order = data["hardware_order"]
    entries = data["joints"]

    result = []
    for hardware_name in order:
        entry = entries[hardware_name]
        anchors = np.asarray(entry["anchors"], dtype=np.float64)
        if anchors.ndim != 2 or anchors.shape[1] != 2 or anchors.shape[0] < 2:
            raise ValueError(
                f"{hardware_name}: anchors must contain at least two [q, value] pairs"
            )
        if not np.all(np.diff(anchors[:, 0]) > 0):
            raise ValueError(
                f"{hardware_name}: URDF anchor coordinates must strictly increase"
            )
        result.append(
            JointMapping(
                hardware_name=hardware_name,
                urdf_joint=entry["urdf_joint"],
                urdf_anchors=anchors[:, 0],
                hardware_anchors=anchors[:, 1],
            )
        )
    return tuple(result)


def qpos_to_angle_set(
    qpos: Sequence[float],
    joint_names: Sequence[str],
    mapping: Sequence[JointMapping],
) -> list[int]:
    """Convert a full DexRetargeting qpos to the vendor's 13-value order."""
    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.shape != (len(joint_names),):
        raise ValueError(
            f"qpos has shape {qpos_array.shape}, expected {(len(joint_names),)}"
        )
    if not np.isfinite(qpos_array).all():
        raise ValueError("qpos contains NaN or infinity")

    q_by_name = dict(zip(joint_names, qpos_array, strict=True))
    values = []
    for item in mapping:
        if item.urdf_joint not in q_by_name:
            raise ValueError(f"missing URDF joint: {item.urdf_joint}")
        value = np.interp(
            q_by_name[item.urdf_joint],
            item.urdf_anchors,
            item.hardware_anchors,
        )
        values.append(round(float(value)))
    return values


def slew_limited_command(
    current: Sequence[int],
    desired: Sequence[int],
    limits: np.ndarray,
    max_step_units: int,
) -> list[int]:
    """Clamp a target to hardware limits and to one packet's maximum change."""
    current_array = np.asarray(current, dtype=np.int64)
    desired_array = np.asarray(desired, dtype=np.int64)
    limit_array = np.asarray(limits, dtype=np.int64)
    if current_array.shape != (13,) or desired_array.shape != (13,):
        raise ValueError("current and desired must each contain 13 values")
    if limit_array.shape != (13, 2):
        raise ValueError("limits must have shape (13, 2)")
    if max_step_units <= 0:
        raise ValueError("max_step_units must be positive")

    desired_array = np.clip(
        desired_array,
        limit_array[:, 0],
        limit_array[:, 1],
    )
    delta = np.clip(
        desired_array - current_array,
        -max_step_units,
        max_step_units,
    )
    command = current_array + delta
    command = np.clip(command, limit_array[:, 0], limit_array[:, 1])
    return command.astype(int).tolist()


class InspireG2LiveController:
    """Rate- and step-limited hand output using the verified vendor frame."""

    def __init__(
        self,
        port: str,
        hand_id: int = 1,
        update_rate_hz: float = 10.0,
        max_step_units: int = 20,
        health_check_period_s: float = 2.0,
    ):
        if update_rate_hz <= 0:
            raise ValueError("update_rate_hz must be positive")
        if max_step_units <= 0:
            raise ValueError("max_step_units must be positive")
        if not 1 <= hand_id <= 254:
            raise ValueError("hand_id must be in the range 1..254")
        if health_check_period_s <= 0:
            raise ValueError("health_check_period_s must be positive")

        # Keep the vendor-derived protocol in one place.  PROJECT_ROOT is
        # inserted only at construction time so importing this module is inert.
        import sys

        root_string = str(PROJECT_ROOT)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        from tools.demo_serial_064_safe import (
            ANGLE_LIMITS,
            DemoError,
            openSerial,
            read13,
            validate_angle_values,
            validate_pre_motion,
            write13,
        )

        self._angle_limits = np.asarray(ANGLE_LIMITS, dtype=np.int64)
        self._demo_error = DemoError
        self._read13 = read13
        self._validate_angle_values = validate_angle_values
        self._validate_pre_motion = validate_pre_motion
        self._write13 = write13
        self.port = port
        self.hand_id = hand_id
        self.update_period_s = 1.0 / update_rate_hz
        self.max_step_units = max_step_units
        self.health_check_period_s = health_check_period_s
        self.last_send_time = 0.0
        self.last_health_check_time = 0.0
        self.sent_count = 0

        self.serial = openSerial(port, 115200)
        try:
            actual = self._quiet_read("angleAct")
            self._run_health_check()
        except Exception:
            self.serial.close()
            raise

        # The observed thumb value can be slightly outside the nominal manual
        # range (1711 was observed once).  Clamp the internal starting point;
        # every transmitted target is still validated against table 35.
        self.last_command = np.clip(
            np.asarray(actual, dtype=np.int64),
            self._angle_limits[:, 0],
            self._angle_limits[:, 1],
        )

    def _quiet_read(self, register: str) -> list[int]:
        with contextlib.redirect_stdout(io.StringIO()):
            return self._read13(self.serial, self.hand_id, register)

    def _run_health_check(self) -> None:
        errors = self._quiet_read("errCode")
        statuses = self._quiet_read("statusCode")
        temperatures = self._quiet_read("temp")
        with contextlib.redirect_stdout(io.StringIO()):
            self._validate_pre_motion(errors, statuses, temperatures)
        self.last_health_check_time = time.monotonic()

    def read_actual(self) -> list[int]:
        """Read all measured joint positions without issuing motion."""
        return self._quiet_read("angleAct")

    def send_if_due(self, desired: Sequence[int]) -> list[int] | None:
        """Send one slew-limited command when the configured period has elapsed."""
        now = time.monotonic()
        if now - self.last_send_time < self.update_period_s:
            return None
        if now - self.last_health_check_time >= self.health_check_period_s:
            self._run_health_check()

        command = slew_limited_command(
            self.last_command,
            desired,
            self._angle_limits,
            self.max_step_units,
        )
        self._validate_angle_values(command)

        # write13 is the checked vendor-demo path.  Suppress its per-packet hex
        # dump here because a live stream would otherwise print dozens of lines
        # per second.
        with contextlib.redirect_stdout(io.StringIO()):
            self._write13(self.serial, self.hand_id, "angleSet", command)

        self.last_command = np.asarray(command, dtype=np.int64)
        self.last_send_time = now
        self.sent_count += 1
        return command

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()

    def __enter__(self) -> "InspireG2LiveController":  # noqa: PYI034, UP037
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

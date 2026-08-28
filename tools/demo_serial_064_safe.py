#!/usr/bin/env python3
"""Safety-limited RH5DG2 demo derived from the vendor demo_serial_064.py.

Vendor source:
  compare_works/因时G2灵巧手URDF和SDK/URDF和SDK/demo_serial_064.py

Kept from the vendor demo:
  * pyserial connection at 115200 baud
  * register dictionary
  * 0x11 read-register framing
  * 0x12 write-register framing
  * write13()/read13() style API
  * little-endian INT16 values and additive checksum

Changes made for the first physical motion:
  * no movement on ordinary startup
  * validate every angle against RH5DG2 manual table 35
  * read angle/error/status/temperature before movement
  * reject non-zero errors, documented stop/fault states, or high temperature
  * preserve the vendor's 13-joint demo pose while clamping its four
    out-of-range groups to the V1.0 manual limits
  * read feedback again after movement
  * require explicit operator confirmation

This file is a vendor-demo adaptation, not the production retargeting
controller. Hardware execution remains the responsibility of the on-site
operator.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass

import serial


REGISTERS = {
    "ID": 1000,
    "baudrate": 1001,
    "clearErr": 1003,
    "forceClb": 1007,
    "angleSet": 1080,
    "forceSet": 1094,
    "speedSet": 1108,
    "angleAct": 1136,
    "forceAct": 1150,
    "errCode": 1178,
    "statusCode": 1192,
    "temp": 1206,
    "mode": 1220,
    "actionSeq": 2160,
    "actionRun": 2162,
}

JOINT_NAMES = (
    "pinky_mcp",
    "ring_mcp",
    "middle_mcp",
    "middle_yaw",
    "index_mcp",
    "index_yaw",
    "pinky_pip",
    "ring_pip",
    "middle_pip",
    "index_pip",
    "thumb_yaw",
    "thumb_mcp",
    "thumb_dip",
)

# RH5DG2 user manual V1.0, section 2.5.11, table 35.
ANGLE_LIMITS = (
    (965, 1800),
    (965, 1800),
    (965, 1800),
    (-150, 150),
    (965, 1800),
    (-150, 150),
    (1050, 1850),
    (1050, 1850),
    (1050, 1850),
    (1050, 1850),
    (600, 1700),
    (1000, 1550),
    (1450, 2040),
)

FIRST_TEST_MAX_TEMPERATURE_C = 60
CONFIRMATION_TEXT = "RUN_SAFE_VENDOR_DEMO"

# Original vendor demo:
# [1800, 1800, 1800, 0, 1514, 0,
#  1900, 1900, 1900, 1514, 1750, 1600, 2080]
#
# Only the values outside table 35 are clamped:
#   finger PIP 1900 -> 1850
#   thumb_yaw 1750 -> 1700
#   thumb_mcp 1600 -> 1550
#   thumb_dip 2080 -> 2040
SAFE_VENDOR_DEMO_ANGLE_SET = (
    1800,
    1800,
    1800,
    0,
    1514,
    0,
    1850,
    1850,
    1850,
    1514,
    1700,
    1550,
    2040,
)


class DemoError(RuntimeError):
    """Serial protocol, validation, or safety-check failure."""


@dataclass(frozen=True)
class RegisterReply:
    address: int
    values: tuple[int, ...]
    raw: bytes


def openSerial(port: str, baudrate: int) -> serial.Serial:
    """Vendor-demo API: open the serial connection."""
    return serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.02,
        write_timeout=0.5,
    )


def _checksum(frame_without_checksum: bytes) -> int:
    return sum(frame_without_checksum[2:]) & 0xFF


def _read_complete_response(
    ser: serial.Serial,
    timeout: float,
) -> bytes:
    deadline = time.monotonic() + timeout
    buffer = bytearray()
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buffer.extend(chunk)

        header = buffer.find(b"\x90\xEB")
        if header < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            continue
        if header:
            del buffer[:header]
        if len(buffer) < 4:
            continue

        total_length = buffer[3] + 5
        if len(buffer) >= total_length:
            return bytes(buffer[:total_length])

    raw = " ".join(f"{value:02X}" for value in buffer) or "<empty>"
    raise DemoError(f"response timeout; received: {raw}")


def _validate_common_response(
    response: bytes,
    hand_id: int,
    command: int,
    address: int,
) -> None:
    if len(response) < 8:
        raise DemoError(f"response is too short: {len(response)} bytes")
    if response[:2] != b"\x90\xEB":
        raise DemoError("invalid response header")
    if response[2] != hand_id:
        raise DemoError(
            f"response hand ID {response[2]}, expected {hand_id}"
        )
    if response[4] != command:
        raise DemoError(
            f"response command 0x{response[4]:02X}, expected 0x{command:02X}"
        )
    response_address = response[5] | (response[6] << 8)
    if response_address != address:
        raise DemoError(
            f"response address {response_address}, expected {address}"
        )
    if _checksum(response[:-1]) != response[-1]:
        raise DemoError("response checksum mismatch")


def readRegister(
    ser: serial.Serial,
    hand_id: int,
    address: int,
    byte_count: int,
    timeout: float = 0.5,
) -> RegisterReply:
    """Vendor-demo 0x11 read operation with complete-frame validation."""
    request_without_checksum = bytes(
        (
            0xEB,
            0x90,
            hand_id,
            0x04,
            0x11,
            address & 0xFF,
            (address >> 8) & 0xFF,
            byte_count,
        )
    )
    request = request_without_checksum + bytes(
        (_checksum(request_without_checksum),)
    )

    ser.reset_input_buffer()
    ser.write(request)
    ser.flush()
    response = _read_complete_response(ser, timeout)
    _validate_common_response(response, hand_id, 0x11, address)

    payload = response[7:-1]
    if len(payload) != byte_count:
        raise DemoError(
            f"payload length {len(payload)}, expected {byte_count}"
        )
    if len(payload) % 2:
        raise DemoError("INT16 payload has an odd byte count")
    values = struct.unpack(f"<{len(payload) // 2}h", payload)
    return RegisterReply(address=address, values=values, raw=response)


def writeRegister(
    ser: serial.Serial,
    hand_id: int,
    address: int,
    values: list[int],
    timeout: float = 0.5,
) -> bytes:
    """Vendor-demo 0x12 write operation with acknowledgement validation."""
    payload = b"".join(
        struct.pack("<h", value)
        for value in values
    )
    request_without_checksum = bytes(
        (
            0xEB,
            0x90,
            hand_id,
            len(payload) + 3,
            0x12,
            address & 0xFF,
            (address >> 8) & 0xFF,
        )
    ) + payload
    request = request_without_checksum + bytes(
        (_checksum(request_without_checksum),)
    )

    ser.reset_input_buffer()
    print("TX:", " ".join(f"{value:02X}" for value in request))
    ser.write(request)
    ser.flush()
    response = _read_complete_response(ser, timeout)
    _validate_common_response(response, hand_id, 0x12, address)
    if len(response) != 9 or response[3] != 0x04 or response[7] != 0x01:
        raise DemoError(
            "write acknowledgement does not match the vendor format"
        )
    print("RX:", " ".join(f"{value:02X}" for value in response))
    return response


def validate_angle_values(values: list[int]) -> None:
    if len(values) != 13:
        raise DemoError(f"angleSet needs 13 values, received {len(values)}")
    for name, value, (lower, upper) in zip(
        JOINT_NAMES,
        values,
        ANGLE_LIMITS,
        strict=True,
    ):
        if value == -1:
            continue
        if not lower <= value <= upper:
            raise DemoError(
                f"{name}={value} is outside manual range "
                f"[-1 or {lower}..{upper}]"
            )


def write13(
    ser: serial.Serial,
    hand_id: int,
    param_name: str,
    values: list[int],
) -> bytes:
    """Vendor-demo API with RH5DG2 manual range validation."""
    if param_name != "angleSet":
        raise DemoError(
            "safe demo permits only angleSet; force/speed/mode writes are disabled"
        )
    validate_angle_values(values)
    print("Validated angleSet:")
    for name, value in zip(JOINT_NAMES, values, strict=True):
        suffix = "no action" if value == -1 else f"{value / 10:.1f} deg"
        print(f"  {name:>11}: {value:>5} ({suffix})")
    return writeRegister(ser, hand_id, REGISTERS[param_name], values)


def read13(
    ser: serial.Serial,
    hand_id: int,
    param_name: str,
) -> list[int]:
    """Vendor-demo API corrected to read 13 x INT16 = 26 bytes."""
    allowed = ("angleAct", "forceAct", "errCode", "statusCode", "temp")
    if param_name not in allowed:
        raise DemoError(f"unsupported read register: {param_name}")
    reply = readRegister(ser, hand_id, REGISTERS[param_name], 26)
    values = list(reply.values)
    print(f"{param_name}:")
    for name, value in zip(JOINT_NAMES, values, strict=True):
        unit = ""
        if param_name == "angleAct":
            unit = f" ({value / 10:.1f} deg)"
        elif param_name == "temp":
            unit = " deg C"
        print(f"  {name:>11}: {value}{unit}")
    return values


def validate_pre_motion(
    errors: list[int],
    statuses: list[int],
    temperatures: list[int],
) -> None:
    if any(errors):
        raise DemoError(f"non-zero error code; refusing movement: {errors}")
    if any(value in (5, 6, 7) for value in statuses):
        raise DemoError(
            f"protective/fault stop status; refusing movement: {statuses}"
        )
    if max(temperatures) > FIRST_TEST_MAX_TEMPERATURE_C:
        raise DemoError(
            f"temperature exceeds {FIRST_TEST_MAX_TEMPERATURE_C} C; "
            f"refusing movement: {temperatures}"
        )
    if all(value == 255 for value in statuses):
        print(
            "WARNING: status=255 is returned by all joints but is not defined "
            "in the supplied manual."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Range-checked full-pose adaptation of vendor demo_serial_064.py"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open the serial port and perform the corrected vendor demo pose",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f"must exactly equal {CONFIRMATION_TEXT!r}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Vendor-demo safe adaptation")
    print("Planned action: move all 13 joints to the range-corrected vendor pose.")
    print("No force, speed, mode, calibration, or saved parameter is changed.")
    validate_angle_values(list(SAFE_VENDOR_DEMO_ANGLE_SET))
    print("Validated full target pose:")
    for name, value in zip(
        JOINT_NAMES,
        SAFE_VENDOR_DEMO_ANGLE_SET,
        strict=True,
    ):
        print(f"  {name:>11}: {value:>5} ({value / 10:.1f} deg)")

    if not args.execute:
        print("Dry run only. The serial port was not opened.")
        return 0
    if args.confirm != CONFIRMATION_TEXT:
        print(
            f"ERROR: add --confirm {CONFIRMATION_TEXT}",
            file=sys.stderr,
        )
        return 2

    try:
        ser = openSerial(args.port, 115200)
    except serial.SerialException as exc:
        print(f"ERROR: cannot open {args.port}: {exc}", file=sys.stderr)
        return 3

    try:
        print(f"Opened {args.port}: 115200 8N1, hand ID {args.hand_id}")
        before = read13(ser, args.hand_id, "angleAct")
        errors = read13(ser, args.hand_id, "errCode")
        statuses = read13(ser, args.hand_id, "statusCode")
        temperatures = read13(ser, args.hand_id, "temp")
        validate_pre_motion(errors, statuses, temperatures)

        target_values = list(SAFE_VENDOR_DEMO_ANGLE_SET)
        print("Planned changes from the live pose:")
        for name, current, target in zip(
            JOINT_NAMES,
            before,
            target_values,
            strict=True,
        ):
            print(
                f"  {name:>11}: {current / 10:>6.1f} -> "
                f"{target / 10:>6.1f} deg "
                f"({(target - current) / 10:+.1f})"
            )
        write13(ser, args.hand_id, "angleSet", target_values)

        time.sleep(0.5)
        after = read13(ser, args.hand_id, "angleAct")
        after_errors = read13(ser, args.hand_id, "errCode")
        after_temperatures = read13(ser, args.hand_id, "temp")
        print("Observed changes:")
        for name, current, observed, target in zip(
            JOINT_NAMES,
            before,
            after,
            target_values,
            strict=True,
        ):
            print(
                f"  {name:>11}: moved "
                f"{(observed - current) / 10:+.1f} deg; "
                f"target error {(observed - target) / 10:+.1f} deg"
            )
        if any(after_errors):
            raise DemoError(f"post-motion error code: {after_errors}")
        if max(after_temperatures) > FIRST_TEST_MAX_TEMPERATURE_C:
            raise DemoError(
                f"post-motion temperature exceeds "
                f"{FIRST_TEST_MAX_TEMPERATURE_C} C: {after_temperatures}"
            )
    except (DemoError, serial.SerialException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

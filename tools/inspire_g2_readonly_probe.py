#!/usr/bin/env python3
"""Read-only RS485 probe for one Inspire RH5DG2 hand.

Safety boundary
---------------
This program can only construct the vendor's read-register command (0x11).
It has no implementation of the write-register command (0x12), so it cannot
set angles, force, speed, modes, actions, calibration, or device parameters.
By default it only prints the frames. The serial port is opened only when the
operator explicitly passes ``--execute-read``.

Protocol sources
----------------
Vendor SDK:
  src/document/RH5DG2_485协议格式说明.md, sections 3 and 5
  src/document/因时机器人-RH5DG2系列灵巧手用户手册V1.0.pdf,
  register table and sections 2.5.15, 2.5.18, 2.5.19, 2.5.20

The query frame is:
  EB 90 ID 04 11 ADDR_L ADDR_H BYTE_COUNT CHECKSUM
where CHECKSUM is the low byte of the sum from ID through BYTE_COUNT.
"""

from __future__ import annotations

import argparse
import os
import select
import struct
import sys
import termios
import time
from dataclasses import dataclass


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


@dataclass(frozen=True)
class ReadableRegister:
    address: int
    byte_count: int
    unit: str = ""


# This allow-list contains only registers marked R (read-only) by the RH5DG2
# user manual. No caller-supplied address is accepted.
READABLE_REGISTERS = {
    "angle_actual": ReadableRegister(1136, 26, "0.1 deg"),
    "error_code": ReadableRegister(1178, 26),
    "status": ReadableRegister(1192, 26),
    "temperature": ReadableRegister(1206, 26, "deg C"),
}


class ProbeError(RuntimeError):
    """A communication or validation failure."""


def build_read_frame(hand_id: int, register: ReadableRegister) -> bytes:
    if not 1 <= hand_id <= 254:
        raise ValueError("hand ID must be in [1, 254]")
    if not 1 <= register.byte_count <= 255:
        raise ValueError("byte count must be in [1, 255]")

    frame_without_checksum = bytes(
        (
            0xEB,
            0x90,
            hand_id,
            0x04,
            0x11,
            register.address & 0xFF,
            (register.address >> 8) & 0xFF,
            register.byte_count,
        )
    )
    checksum = sum(frame_without_checksum[2:]) & 0xFF
    return frame_without_checksum + bytes((checksum,))


def configure_serial(fd: int) -> None:
    """Configure 115200 baud, 8 data bits, no parity, 1 stop bit."""
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
    attrs[3] = 0
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)


def extract_complete_frame(buffer: bytearray) -> bytes | None:
    header = buffer.find(b"\x90\xEB")
    if header < 0:
        if len(buffer) > 1:
            del buffer[:-1]
        return None
    if header:
        del buffer[:header]
    if len(buffer) < 8:
        return None

    data_length = buffer[3]
    if data_length < 3:
        raise ProbeError(f"invalid response data length: {data_length}")
    total_length = data_length + 5
    if len(buffer) < total_length:
        return None
    return bytes(buffer[:total_length])


def validate_and_decode(
    response: bytes,
    hand_id: int,
    register: ReadableRegister,
) -> list[int]:
    expected_length = register.byte_count + 8
    if len(response) != expected_length:
        raise ProbeError(
            f"response length {len(response)}, expected {expected_length}"
        )
    if response[:2] != b"\x90\xEB":
        raise ProbeError("invalid response header")
    if response[2] != hand_id:
        raise ProbeError(
            f"response hand ID {response[2]}, expected {hand_id}"
        )
    if response[4] != 0x11:
        raise ProbeError(
            f"response command 0x{response[4]:02X}, expected read command 0x11"
        )

    response_address = response[5] | (response[6] << 8)
    if response_address != register.address:
        raise ProbeError(
            f"response address {response_address}, expected {register.address}"
        )
    if (sum(response[2:-1]) & 0xFF) != response[-1]:
        raise ProbeError("response checksum mismatch")

    payload = response[7:-1]
    if len(payload) != register.byte_count or len(payload) % 2:
        raise ProbeError("invalid register payload length")
    return list(struct.unpack(f"<{len(payload) // 2}h", payload))


def query_register(
    fd: int,
    hand_id: int,
    register: ReadableRegister,
    timeout: float,
) -> tuple[list[int], bytes, bytes]:
    request = build_read_frame(hand_id, register)
    termios.tcflush(fd, termios.TCIFLUSH)
    os.write(fd, request)

    deadline = time.monotonic() + timeout
    buffer = bytearray()
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            break
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            continue
        if chunk:
            buffer.extend(chunk)
            response = extract_complete_frame(buffer)
            if response is not None:
                values = validate_and_decode(response, hand_id, register)
                return values, request, response

    raw = " ".join(f"{byte:02X}" for byte in buffer) or "<empty>"
    raise ProbeError(f"timeout waiting for a complete response; received: {raw}")


def format_frame(frame: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Inspire RH5DG2 RS485 communication probe"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument(
        "--register",
        action="append",
        choices=tuple(READABLE_REGISTERS),
        help="read only this register; repeat to select several (default: all)",
    )
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument(
        "--execute-read",
        action="store_true",
        help="open the serial port and send 0x11 read requests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_names = args.register or list(READABLE_REGISTERS)

    print("Safety: this tool implements only command 0x11 (read register).")
    for name in selected_names:
        request = build_read_frame(args.hand_id, READABLE_REGISTERS[name])
        print(f"planned {name:>14}: {format_frame(request)}")

    if not args.execute_read:
        print("Dry run only. No serial port was opened and no frame was sent.")
        print("Add --execute-read only after the operator confirms the hardware.")
        return 0

    try:
        fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as exc:
        print(f"ERROR: cannot open {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        configure_serial(fd)
        print(f"Opened {args.port}: 115200 8N1, hand ID {args.hand_id}")
        for index, name in enumerate(selected_names):
            register = READABLE_REGISTERS[name]
            values, request, response = query_register(
                fd, args.hand_id, register, args.timeout
            )
            print(f"\n{name}")
            print(f"  TX: {format_frame(request)}")
            print(f"  RX: {format_frame(response)}")
            if len(values) == len(JOINT_NAMES):
                for joint_name, value in zip(JOINT_NAMES, values, strict=True):
                    suffix = f" {register.unit}" if register.unit else ""
                    print(f"  {joint_name:>11}: {value}{suffix}")
            else:
                print(f"  values: {values}")
            if index + 1 < len(selected_names):
                time.sleep(0.05)
    except (OSError, ProbeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        os.close(fd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

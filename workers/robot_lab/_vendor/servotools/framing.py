"""Pure framing subset of pinned Enigma servotools bus.py; MIT, see LICENSE.

Only checksum/packet/find_frame are extracted. No serial import or I/O occurs.
"""
from __future__ import annotations

def checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def packet(sid: int, inst: int, params: bytes = b"") -> bytes:
    body = bytes([sid, len(params) + 2, inst]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])





def find_frame(resp: bytes, sid: int, length: int, check_error: bool = False) -> bytes | None:
    """Return the params of the first checksum-valid status packet from `sid`, else None."""
    i = 0
    while i < len(resp) - 3:
        if resp[i] == 0xFF and resp[i + 1] == 0xFF and resp[i + 2] == sid:
            ln = resp[i + 3]
            end = i + 4 + ln
            if ln == length + 2 and end <= len(resp):
                pkt = resp[i:end]
                if checksum(pkt[2:-1]) == pkt[-1] and not (check_error and pkt[4] != 0):
                    return pkt[5:-1]
        i += 1
    return None

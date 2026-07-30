"""ASTM F3411 / OpenDroneID Remote ID decoder reference (passive receive only).

ASTM F3411 is the baseline Remote ID standard behind the FAA Broadcast Remote ID
rule: a drone broadcasts its ID, location, speed and heading over Bluetooth LE
and/or WiFi beacons, receivable by an ordinary smartphone. OpenDroneID is the
open reference implementation. This module documents the broadcast message
fields and provides a small stdlib **decoder** for the 25-byte OpenDroneID
message layout (Basic ID and Location messages fully decoded; other types
surfaced as typed reference records).

CRITICAL CAVEAT (attached to every decoded message): Remote ID only reveals
*compliant, un-spoofed* platforms. It is an **identification aid that is
spoofable** and MUST NOT be treated as trusted authentication.

Scope: passive receive/decode reference only. This module performs no radio
transmission. The ``pack_*`` helpers merely assemble a byte string for decoder
validation / offline simulation - they do not transmit anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SPOOFABLE_CAVEAT = (
    "ASTM F3411 Remote ID is a spoofable IDENTIFICATION AID, not trusted "
    "authentication; treat all fields as unverified."
)

MESSAGE_SIZE = 25  # bytes, per the OpenDroneID single-message layout

# OpenDroneID message types (header high nibble).
MESSAGE_TYPES: dict[int, str] = {
    0x0: "basic_id",
    0x1: "location",
    0x2: "authentication",
    0x3: "self_id",
    0x4: "system",
    0x5: "operator_id",
    0xF: "message_pack",
}

# Basic ID: ID Type (high nibble of byte 1).
ID_TYPES: dict[int, str] = {
    0: "none",
    1: "serial_number",       # ANSI/CTA-2063-A serial number
    2: "caa_registration",    # CAA-assigned registration ID
    3: "utm_uuid",            # UTM (USS) assigned UUID
    4: "specific_session",    # Specific Session ID
}

# Basic ID: UA (unmanned aircraft) Type (low nibble of byte 1).
UA_TYPES: dict[int, str] = {
    0: "none",
    1: "aeroplane",
    2: "multirotor",
    3: "gyroplane",
    4: "hybrid_lift",
    5: "ornithopter",
    6: "glider",
    7: "kite",
    8: "free_balloon",
    9: "captive_balloon",
    10: "airship",
    11: "free_fall_parachute",
    12: "rocket",
    13: "tethered_powered",
    14: "ground_obstacle",
    15: "other",
}

# Location: operational status (high nibble of byte 1).
OPERATIONAL_STATUS: dict[int, str] = {
    0: "undeclared",
    1: "ground",
    2: "airborne",
    3: "emergency",
    4: "remote_id_system_failure",
}

# Field reference for the documentation page / analysts. Each entry: the field,
# which message carries it, and a one-line description. Every field is an aid.
FIELD_REFERENCE: tuple[dict[str, str], ...] = (
    {"field": "uas_id", "message": "basic_id", "desc": "Broadcast UAS ID (serial / registration / UUID)."},
    {"field": "id_type", "message": "basic_id", "desc": "Which ID scheme the uas_id uses."},
    {"field": "ua_type", "message": "basic_id", "desc": "Airframe class (multirotor, aeroplane, ...)."},
    {"field": "latitude", "message": "location", "desc": "Current latitude, degrees (int32 * 1e-7)."},
    {"field": "longitude", "message": "location", "desc": "Current longitude, degrees (int32 * 1e-7)."},
    {"field": "geodetic_altitude_m", "message": "location", "desc": "Geodetic altitude, metres."},
    {"field": "height_m", "message": "location", "desc": "Height above takeoff/ground, metres."},
    {"field": "speed_mps", "message": "location", "desc": "Ground speed, m/s."},
    {"field": "direction_deg", "message": "location", "desc": "Track direction, degrees (0-359)."},
    {"field": "operational_status", "message": "location", "desc": "Ground / airborne / emergency."},
)


@dataclass(frozen=True)
class RemoteIDMessage:
    """A decoded ASTM F3411 / OpenDroneID broadcast message.

    ``trusted`` is always False and ``spoofable`` always True: a decoded Remote
    ID message is an identification aid, never authenticated truth.
    """

    message_type: str
    protocol_version: int
    fields: dict = field(default_factory=dict)
    trusted: bool = False
    spoofable: bool = True
    caveat: str = SPOOFABLE_CAVEAT

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict, caveat included."""
        return {
            "message_type": self.message_type,
            "protocol_version": self.protocol_version,
            "fields": dict(self.fields),
            "trusted": self.trusted,
            "spoofable": self.spoofable,
            "caveat": self.caveat,
        }


def _decode_basic_id(payload: bytes) -> dict:
    b1 = payload[1]
    id_type = b1 >> 4
    ua_type = b1 & 0x0F
    uas_id = payload[2:22].rstrip(b"\x00").decode("ascii", "replace")
    return {
        "id_type": ID_TYPES.get(id_type, f"reserved_{id_type}"),
        "ua_type": UA_TYPES.get(ua_type, f"reserved_{ua_type}"),
        "uas_id": uas_id,
    }


def _decode_altitude(raw: int) -> float:
    """OpenDroneID altitude encoding: metres = raw * 0.5 - 1000."""
    return raw * 0.5 - 1000.0


def _decode_location(payload: bytes) -> dict:
    status_byte = payload[1]
    status = status_byte >> 4
    height_type = (status_byte >> 2) & 0x1
    ew_segment = (status_byte >> 1) & 0x1
    speed_mult = status_byte & 0x1

    track_raw = payload[2]
    speed_raw = payload[3]
    vspeed_raw = int.from_bytes(payload[4:5], "little", signed=True)

    lat = int.from_bytes(payload[5:9], "little", signed=True) / 1e7
    lon = int.from_bytes(payload[9:13], "little", signed=True) / 1e7

    pressure_alt = _decode_altitude(int.from_bytes(payload[13:15], "little"))
    geodetic_alt = _decode_altitude(int.from_bytes(payload[15:17], "little"))
    height = _decode_altitude(int.from_bytes(payload[17:19], "little"))

    # Track direction: 0-179 plus a 180-degree East/West segment flag.
    direction = track_raw + (180 if ew_segment else 0)
    # Speed multiplier segment encoding.
    if speed_mult == 0:
        speed = speed_raw * 0.25
    else:
        speed = speed_raw * 0.75 + (255 * 0.25)

    return {
        "operational_status": OPERATIONAL_STATUS.get(status, f"reserved_{status}"),
        "height_type": "geodetic" if height_type else "above_takeoff",
        "direction_deg": direction,
        "speed_mps": round(speed, 3),
        "vertical_speed_mps": round(vspeed_raw * 0.5, 3),
        "latitude": round(lat, 7),
        "longitude": round(lon, 7),
        "pressure_altitude_m": pressure_alt,
        "geodetic_altitude_m": geodetic_alt,
        "height_m": height,
    }


def decode_message(payload: bytes) -> RemoteIDMessage:
    """Decode one 25-byte OpenDroneID message from ``payload``.

    Basic ID (type 0) and Location (type 1) are fully decoded; other message
    types return a typed record with the raw hex for reference. Raises
    ``ValueError`` when the payload is not exactly :data:`MESSAGE_SIZE` bytes.
    """
    if len(payload) != MESSAGE_SIZE:
        raise ValueError(
            f"Remote ID message must be {MESSAGE_SIZE} bytes, got {len(payload)}"
        )
    header = payload[0]
    mtype = header >> 4
    version = header & 0x0F
    name = MESSAGE_TYPES.get(mtype, f"reserved_{mtype}")

    if mtype == 0x0:
        fields = _decode_basic_id(payload)
    elif mtype == 0x1:
        fields = _decode_location(payload)
    else:
        fields = {"raw_hex": payload.hex()}

    return RemoteIDMessage(message_type=name, protocol_version=version, fields=fields)


# --- Reference byte-layout builders (for decoder validation / simulation) -----
# These assemble a byte string only. They perform NO radio transmission; they
# exist so tests and analysts can round-trip the decoder offline.

def _header(mtype: int, version: int) -> int:
    return ((mtype & 0x0F) << 4) | (version & 0x0F)


def pack_basic_id(
    uas_id: str,
    *,
    id_type: int = 1,
    ua_type: int = 2,
    version: int = 2,
) -> bytes:
    """Assemble a 25-byte Basic ID message (byte-layout reference, no transmit)."""
    ascii_id = uas_id.encode("ascii", "replace")[:20].ljust(20, b"\x00")
    body = bytes([_header(0x0, version), ((id_type & 0x0F) << 4) | (ua_type & 0x0F)])
    body += ascii_id
    body += b"\x00" * (MESSAGE_SIZE - len(body))
    return body


def _encode_altitude(alt_m: float) -> int:
    return max(0, min(0xFFFF, int(round((alt_m + 1000.0) / 0.5))))


def pack_location(
    latitude: float,
    longitude: float,
    *,
    geodetic_altitude_m: float = 0.0,
    height_m: float = 0.0,
    speed_mps: float = 0.0,
    direction_deg: int = 0,
    status: int = 2,
    version: int = 2,
) -> bytes:
    """Assemble a 25-byte Location message (byte-layout reference, no transmit)."""
    ew_segment = 1 if direction_deg >= 180 else 0
    track_raw = (direction_deg - 180) if ew_segment else direction_deg
    track_raw = max(0, min(255, int(track_raw)))

    if speed_mps <= 255 * 0.25:
        speed_mult = 0
        speed_raw = max(0, min(255, int(round(speed_mps / 0.25))))
    else:
        speed_mult = 1
        speed_raw = max(0, min(255, int(round((speed_mps - 255 * 0.25) / 0.75))))

    status_byte = ((status & 0x0F) << 4) | (ew_segment << 1) | speed_mult

    out = bytearray(MESSAGE_SIZE)
    out[0] = _header(0x1, version)
    out[1] = status_byte
    out[2] = track_raw
    out[3] = speed_raw
    out[4] = 0  # vertical speed
    out[5:9] = int(round(latitude * 1e7)).to_bytes(4, "little", signed=True)
    out[9:13] = int(round(longitude * 1e7)).to_bytes(4, "little", signed=True)
    out[13:15] = _encode_altitude(0.0).to_bytes(2, "little")   # pressure alt
    out[15:17] = _encode_altitude(geodetic_altitude_m).to_bytes(2, "little")
    out[17:19] = _encode_altitude(height_m).to_bytes(2, "little")
    return bytes(out)

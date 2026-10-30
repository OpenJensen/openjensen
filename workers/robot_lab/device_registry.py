"""Persistent device identities. Discovery never opens a serial bus."""

import hashlib
import json
from pathlib import Path

from dataset import atomic_json


class DeviceRegistry:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "devices.json"

    def document(self):
        if self.path.is_symlink():
            raise ValueError("Device registry must be a regular file")
        if not self.path.exists():
            return {"version": 1, "devices": [], "removed_devices": []}
        if self.path.stat().st_size > 1024 * 1024:
            raise ValueError("Device registry is too large")
        value = json.loads(self.path.read_text())
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or not isinstance(value.get("devices"), list)
            or not isinstance(value.get("removed_devices", []), list)
        ):
            raise ValueError("Invalid device registry")
        value.setdefault("removed_devices", [])
        for device in value["devices"] + value["removed_devices"]:
            if (
                not isinstance(device, dict)
                or not all(
                    isinstance(device.get(k), str) for k in ("id", "port", "name", "robot_id")
                )
                or device.get("role") not in ("leader", "follower")
                or device.get("profile_role") not in ("leader", "follower")
            ):
                raise ValueError("Invalid saved device configuration")
        return value

    def read(self):
        return self.document()["devices"]

    def removed(self):
        return self.document()["removed_devices"]

    def remove(self, payload):
        if set(payload) != {"id"} or not isinstance(payload["id"], str):
            raise ValueError("Choose a saved device to remove")
        value = self.document()
        device = next((d for d in value["devices"] if d["id"] == payload["id"]), None)
        if device is None:
            raise ValueError("Refresh devices and choose a saved device")
        value["devices"].remove(device)
        value["removed_devices"] = [d for d in value["removed_devices"] if d["id"] != device["id"]]
        value["removed_devices"].append(device)
        atomic_json(self.path, value)

    def restore(self, payload):
        if set(payload) != {"id"} or not isinstance(payload["id"], str):
            raise ValueError("Choose a removed device to restore")
        value = self.document()
        device = next((d for d in value["removed_devices"] if d["id"] == payload["id"]), None)
        if device is None:
            raise ValueError("Refresh devices and choose a removed device")
        if any(
            d["id"] == device["id"]
            or d["robot_id"] == device["robot_id"]
            or (
                device["port"]
                and d["port"] == device["port"]
                and (
                    not device.get("serial_number")
                    or not d.get("serial_number")
                    or d["serial_number"] == device["serial_number"]
                )
            )
            for d in value["devices"]
        ):
            raise ValueError("Remove the current configuration for this device before restoring it")
        value["removed_devices"].remove(device)
        value["devices"].append(device)
        atomic_json(self.path, value)

    @staticmethod
    def key(port, ports):
        serial = port.get("serial_number")
        unique = serial and sum(p.get("serial_number") == serial for p in ports) == 1
        identity = ("serial:" + serial) if unique else ("port:" + port["device"])
        return hashlib.sha256(identity.encode()).hexdigest()[:24]

    def discover(self, ports, profiles):
        value = self.document()
        saved = value["devices"]
        changed = False
        # Recover existing calibration identities without renaming or moving their files.
        for profile in profiles:
            if any(
                d["robot_id"] == profile["robot_id"]
                or (
                    d.get("calibration_id") == profile["robot_id"]
                    and d["profile_role"] == profile["role"]
                )
                for d in saved + value["removed_devices"]
            ):
                continue
            port = profile.get("setup_port") or ""
            if port and any(d["port"] == port for d in saved):
                continue
            ambiguous = (
                len({p["role"] for p in profiles if p["robot_id"] == profile["robot_id"]}) > 1
            )
            logical_id = (
                (
                    "device-"
                    + hashlib.sha256(
                        (profile["role"] + "/" + profile["robot_id"]).encode()
                    ).hexdigest()[:16]
                )
                if ambiguous
                else profile["robot_id"]
            )
            saved.append(
                dict(
                    id="profile-" + profile["role"] + "-" + profile["robot_id"],
                    port=port,
                    serial_number=None,
                    name=profile["robot_id"],
                    robot_id=logical_id,
                    calibration_id=profile["robot_id"],
                    role=profile["role"],
                    profile_role=profile["role"],
                )
            )
            changed = True
        entries = []
        used = set()
        for port in ports:
            serial = port.get("serial_number")
            known_serial = [d for d in saved if serial and d.get("serial_number") == serial]
            unique_serial = (
                serial
                and len(known_serial) <= 1
                and sum(p.get("serial_number") == serial for p in ports) == 1
            )
            key = (
                self.key(port, ports)
                if unique_serial
                else hashlib.sha256(("port:" + port["device"]).encode()).hexdigest()[:24]
            )
            matches = [d for d in saved if d["id"] == key]
            if not matches and unique_serial:
                matches = known_serial
            if not matches:
                matches = [
                    d
                    for d in saved
                    if d["port"] == port["device"]
                    and (not d.get("serial_number") or d.get("serial_number") == serial)
                ]
            device = (
                dict(matches[0])
                if matches
                else dict(
                    id=key,
                    port=port["device"],
                    serial_number=port.get("serial_number"),
                    name="",
                    robot_id="device-" + key[:16],
                    role=None,
                    profile_role=None,
                )
            )
            if matches:
                used.add(device["id"])
                if device["port"] != port["device"] or device.get("serial_number") != port.get(
                    "serial_number"
                ):
                    matches[0].update(port=port["device"], serial_number=port.get("serial_number"))
                    device.update(matches[0])
                    changed = True
            entries.append(
                dict(device, connected=True, description=port.get("description", "USB device"))
            )
        entries.extend(
            dict(d, connected=False, description="Saved device")
            for d in saved
            if d["id"] not in used
        )
        if changed:
            atomic_json(self.path, value)
        return entries

    def save(self, payload, entries):
        if set(payload) - {"id", "port", "name", "role"}:
            raise ValueError("Unexpected device configuration fields")
        device = next((d for d in entries if d["id"] == payload.get("id")), None)
        if device is None:
            raise ValueError("Refresh devices and choose a known device")
        name, role = payload.get("name"), payload.get("role")
        if not isinstance(name, str) or len(name) > 64 or any(not c.isprintable() for c in name):
            raise ValueError("Device name must contain at most 64 printable characters")
        if role not in ("leader", "follower"):
            raise ValueError("Choose a default device role")
        port = payload.get("port", device["port"])
        if not isinstance(port, str) or (
            port != device["port"]
            and not any(d["connected"] and d["port"] == port for d in entries)
        ):
            raise ValueError("Choose a connected USB port")
        value = self.document()
        saved = value["devices"]
        if any(
            d["id"] != device["id"]
            and port
            and d["port"] == port
            and any(e["id"] == d["id"] and e["connected"] for e in entries)
            for d in saved
        ):
            raise ValueError("That USB port already belongs to another saved device")
        record = {
            k: device[k]
            for k in ("id", "port", "serial_number", "name", "robot_id", "role", "profile_role")
        }
        if device.get("calibration_id"):
            record["calibration_id"] = device["calibration_id"]
        record.update(
            name=name.strip(), role=role, port=port, profile_role=device["profile_role"] or role
        )
        connected = next((d for d in entries if d["connected"] and d["port"] == port), None)
        if connected:
            record["serial_number"] = connected.get("serial_number")
            # Adopt the stable serial identity when binding a formerly unbound profile.
            if not device["port"]:
                record["id"] = connected["id"]
        saved = [d for d in saved if d["id"] not in (device["id"], record["id"])]
        saved.append(record)
        value["devices"] = saved
        atomic_json(self.path, value)
        return record

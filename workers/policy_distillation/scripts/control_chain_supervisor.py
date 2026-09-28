"""Opt-in macOS supervisor for one generated CPU compression chain.

No model imports, installation, retry or deletion. Limits are sampled abort
thresholds, not an OS memory/disk reservation: short between-sample peaks remain
possible. Exact PID births, retained groups and a private kernel cwd identity
cover this fixed inherited-cwd graph. This is not a sandbox for a program that
deliberately changes cwd to escape discovery. This tooling is macOS-only.
"""

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MIB = 1024**2
LIMITS = {
    "work_seconds": 600,
    "cleanup_seconds": 60,
    "owned_rss_bytes": 1536 * MIB,
    "allocated_output_bytes": 320 * MIB,
    "free_disk_bytes": 10 * 1024**3,
    "swap_growth_bytes": 128 * MIB,
    "log_tail_bytes": 64 * 1024,
}
POLL = 0.1
MAX_PROCESSES = 4096
MAX_OWNED = 128
SOURCE_LIMIT = 2 * MIB
SOURCE_TOTAL_LIMIT = 32 * MIB
SOURCE_DIRS = (
    "workers/policy_distillation/src",
    "workers/act_optimizer/src",
    "workers/smolvla_qlora/src",
    "workers/firebird_quant/src",
    "workers/isaac_sim/sim_worker",
)
PYTHON_ROOTS = (*SOURCE_DIRS[:-1], "workers/isaac_sim")
EXTRA_SOURCES = (
    "workers/policy_distillation/scripts/control_chain_fixture.py",
    "workers/policy_distillation/scripts/control_chain_supervisor.py",
    "workers/policy_distillation/tests/native_fixture.py",
    "workers/policy_distillation/tests/semantics_fixture.py",
    "workers/act_optimizer/scripts/temporal_http_fixture.py",
)


class Refused(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def finite_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Refused("Duplicate final JSON key")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise Refused("Nonfinite final JSON value")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=number)


def file_hash(path, limit=SOURCE_LIMIT):
    """Bounded regular-file read; no final symlink or FIFO acceptance."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise Refused(f"Invalid bounded input: {path.name}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise Refused(f"Input changed while reading: {path.name}")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def source_identity(repo):
    paths = {repo / name for name in EXTRA_SOURCES}
    for name in SOURCE_DIRS:
        root = repo / name
        if not root.is_dir() or root.is_symlink():
            raise Refused(f"Missing source root: {name}")
        for base, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(d for d in directories if d != "__pycache__")
            if any((Path(base) / d).is_symlink() for d in directories):
                raise Refused("Source directory link is unsupported")
            paths.update(Path(base) / f for f in files if f.endswith(".py"))
    if len(paths) > 2048:
        raise Refused("Source inventory exceeds its bound")
    result, total = {}, 0
    for path in sorted(paths):
        record = file_hash(path)
        total += record["bytes"]
        if total > SOURCE_TOTAL_LIMIT:
            raise Refused("Source inventory byte budget exceeded")
        result[path.relative_to(repo).as_posix()] = record
    return result


def allocated_bytes(root):
    """Count allocated blocks, including temporary files and directory blocks."""
    total, count = 0, 0
    pending = [root]
    while pending:
        path = pending.pop()
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue  # Worker-owned temporary files may be concurrently removed.
        count += 1
        if count > 32768:
            raise Refused("Owned output entry budget exceeded")
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
        ):
            raise Refused("Owned output contains a link or special file")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise Refused("Owned output must not link to another file inventory")
        total += info.st_blocks * 512
        if stat.S_ISDIR(info.st_mode):
            try:
                pending.extend(path.iterdir())
            except FileNotFoundError:
                continue
    return total


def swap_bytes(raw):
    match = re.search(r"\bused\s*=\s*(\d+(?:\.\d+)?)([KMGT])\b", raw)
    if match is None:
        raise Refused("Cannot parse system swap use")
    return int(float(match[1]) * 1024 ** ("KMGT".index(match[2]) + 1))


def system_output(command):
    """Fixed small metadata commands only, without arguments/environment of other tasks."""
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=2,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C"},
        )
        if result.returncode or len(result.stdout) > MIB or len(result.stderr) > 4096:
            raise Refused("System resource observation failed or exceeded its output bound")
        return result.stdout.decode("ascii", errors="strict")
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise Refused("System resource observation is unavailable") from exc


def process_rows(raw):
    result = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 6 or not all(x.isdigit() for x in fields[:5]):
            raise Refused("Malformed process observation")
        pid, parent, group, rss, uid = map(int, fields[:5])
        if pid in result or len(result) >= MAX_PROCESSES:
            raise Refused("Process observation exceeds its bound or repeats a PID")
        result[pid] = (parent, group, rss * 1024, uid, fields[5])
    return result


class BsdInfo(ctypes.Structure):
    # proc_bsdinfo / PROC_PIDTBSDINFO=3 from Apple's sys/proc_info.h.
    # Exact kernel start seconds + microseconds, not rounded ps lstart strings.
    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in (
                "flags",
                "status",
                "xstatus",
                "pid",
                "ppid",
                "uid",
                "gid",
                "ruid",
                "rgid",
                "svuid",
                "svgid",
                "reserved",
            )
        ]
        + [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
        + [(name, ctypes.c_uint32) for name in ("nfiles", "pgid", "jobc", "dev", "tpgid")]
        + [("nice", ctypes.c_int32), ("seconds", ctypes.c_uint64), ("micros", ctypes.c_uint64)]
    )


class VnodeStat(ctypes.Structure):
    # Fixed-size vinfo_stat from Apple's sys/proc_info.h, not a guessed stat ABI.
    _fields_ = [
        ("dev", ctypes.c_uint32),
        ("mode", ctypes.c_uint16),
        ("nlink", ctypes.c_uint16),
        ("inode", ctypes.c_uint64),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        *[
            (name, ctypes.c_int64)
            for name in (
                "atime",
                "atime_nsec",
                "mtime",
                "mtime_nsec",
                "ctime",
                "ctime_nsec",
                "birth",
                "birth_nsec",
                "size",
                "blocks",
            )
        ],
        ("block_size", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        ("rdev", ctypes.c_uint32),
        ("reserved", ctypes.c_int64 * 2),
    ]


class VnodeInfo(ctypes.Structure):
    _fields_ = [
        ("stat", VnodeStat),
        ("type", ctypes.c_int32),
        ("pad", ctypes.c_int32),
        ("fsid", ctypes.c_int32 * 2),
    ]


class VnodePath(ctypes.Structure):
    _fields_ = [("info", VnodeInfo), ("path", ctypes.c_char * 1024)]


class VnodePaths(ctypes.Structure):
    _fields_ = [("cwd", VnodePath), ("root", VnodePath)]


@dataclass(frozen=True)
class Process:
    pid: int
    birth: tuple[int, int]
    parent: int
    group: int
    rss: int
    uid: int


class Darwin:
    def __init__(self):
        if sys.platform != "darwin":
            raise Refused("This experiment's resource observer supports macOS only")
        self.library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.info = self.library.proc_pidinfo
        self.info.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        self.info.restype = ctypes.c_int

    def process(self, pid, rss=0):
        value = BsdInfo()
        ctypes.set_errno(0)
        size = self.info(pid, 3, 0, ctypes.byref(value), ctypes.sizeof(value))
        if size == 0 and ctypes.get_errno() == errno.ESRCH:
            return None
        if size != ctypes.sizeof(value) or value.pid != pid:
            raise Refused(f"Cannot establish kernel process identity: {pid}")
        return Process(pid, (value.seconds, value.micros), value.ppid, value.pgid, rss, value.uid)

    def cwd_identity(self, pid):
        value = VnodePaths()
        ctypes.set_errno(0)
        size = self.info(pid, 9, 0, ctypes.byref(value), ctypes.sizeof(value))
        if size == 0 and ctypes.get_errno() == errno.ESRCH:
            return None
        if size != ctypes.sizeof(value):
            raise Refused(f"Cannot observe kernel cwd ownership: {pid}")
        # Do not decode or record cwd path strings, process arguments or environments.
        return value.cwd.info.stat.dev, value.cwd.info.stat.inode

    def rows(self):
        return process_rows(system_output(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,uid=,stat="]))

    def swap(self):
        return swap_bytes(system_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"]))


class Tracker:
    """Retain discovered PID births/groups even when owners exit or children reparent."""

    def __init__(self, observer, tag_path=None):
        self.observer = observer
        self.known = {}
        self.groups = set()
        self.history = []
        self.tag_path = tag_path
        self.tag = self.tag_identity() if tag_path is not None else None

    def tag_identity(self):
        info = self.tag_path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise Refused("Private cwd must remain an owned nonsymlink directory")
        return info.st_dev, info.st_ino

    def register(self, process):
        prior = self.known.get(process.pid)
        if prior is not None and prior.birth != process.birth:
            raise Refused("Owned PID reuse makes cleanup uncertain")
        if (
            process.uid != os.getuid()
            or process.pid == os.getpid()
            or process.group == os.getpgrp()
        ):
            raise Refused("Refusing ownership of the supervisor's process group")
        if prior is None:
            if len(self.history) >= MAX_OWNED:
                raise Refused("Owned process identity budget exceeded")
            self.history.append(asdict(process))
        self.known[process.pid] = process
        self.groups.add(process.group)

    def refresh(self):
        if self.tag is not None and self.tag_identity() != self.tag:
            raise Refused("Private cwd inode changed during execution")
        rows = self.observer.rows()
        active = {}
        for pid, known in self.known.items():
            actual = self.observer.process(pid, rows.get(pid, (0, 0, 0, 0, ""))[2])
            if actual is not None:
                if actual.birth != known.birth:
                    raise Refused("Owned PID was reused; no signals will target its successor")
                if pid not in rows:
                    raise Refused("Owned process missing from RSS observation")
                active[pid] = actual
        if self.tag is not None:
            # Reconcile after leader/group disappearance too: a new-session child
            # can outlive its parent before the first ancestry sample.
            for pid, (_parent, _group, rss, uid, _status) in rows.items():
                if uid != os.getuid():
                    continue
                before = self.observer.process(pid, rss)
                if before is None:
                    continue
                tag = self.observer.cwd_identity(pid)
                after = self.observer.process(pid, rss)
                if after is None:
                    continue
                if before.birth != after.birth:
                    raise Refused("Process birth changed during cwd ownership observation")
                if pid in active and tag != self.tag:
                    raise Refused("An owned process left the fixed inherited-cwd graph")
                if tag == self.tag:
                    self.register(after)
                    active[pid] = after
        changed = True
        while changed:
            changed = False
            anchored = {p.group for p in active.values()}
            for pid, (parent, group, rss, _uid, _status) in rows.items():
                if pid in active or not (parent in active or group in anchored):
                    continue
                actual = self.observer.process(pid, rss)
                if actual is None:
                    continue
                # Recheck ancestry after the ps/kernel observation interval.
                if actual.parent not in active and actual.group not in anchored:
                    continue
                self.register(actual)
                active[pid] = actual
                changed = True
        for process in active.values():
            self.register(process)
        return active

    def signal_known(self, sig):
        """Keep helping other verified owners when global discovery is unavailable."""
        errors, signalled = [], set()
        for known in list(self.known.values()):
            try:
                current = self.observer.process(known.pid)
                if current is None:
                    continue
                if current.birth != known.birth or current.uid != os.getuid():
                    raise Refused(f"Cannot signal reused/unowned PID {known.pid}")
                if current.group == os.getpgrp():
                    raise Refused("Cannot signal supervisor process group")
                self.groups.add(current.group)
                if current.group not in signalled:
                    confirm = self.observer.process(known.pid)
                    if confirm is None:
                        continue
                    if confirm.birth != known.birth or confirm.group != current.group:
                        raise Refused("Group identity changed before fallback signal")
                    try:
                        os.killpg(current.group, sig)
                    except ProcessLookupError:
                        pass
                    signalled.add(current.group)
            except Exception as exc:
                errors.append(str(exc)[:300])
        return errors

    def gone(self):
        if self.refresh():
            return False
        for group in self.groups:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                raise Refused("Process group disappearance is not kernel-confirmed") from exc
            raise Refused("An owned group exists without a verified birth identity")
        return True

    def signal_groups(self, sig):
        active = self.refresh()
        for group in sorted({p.group for p in active.values()}):
            # A second birth check just before signalling refuses PID/group reuse.
            anchors = [p for p in active.values() if p.group == group]
            for process in anchors:
                current = self.observer.process(process.pid)
                if current is None:
                    continue
                if current.birth != process.birth or current.group != group:
                    raise Refused("Owned group identity changed before signal")
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
                break


class Tail:
    def __init__(self, stream):
        self.stream, self.data, self.error = stream, bytearray(), None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            while block := self.stream.read(4096):
                with self.lock:
                    self.data.extend(block)
                    del self.data[: -LIMITS["log_tail_bytes"]]
        except OSError as exc:
            self.error = str(exc)

    def save(self, path):
        with self.lock:
            raw = bytes(self.data)
        with path.open("xb") as stream:
            stream.write(raw)


def clean_environment(repo, output):
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LC_ALL": "C",
        "LANG": "C",
        "HOME": str(output / "home"),
        "TMPDIR": str(output / "tmp"),
        "TMP": str(output / "tmp"),
        "TEMP": str(output / "tmp"),
        "HF_HOME": str(output / "home/hf"),
        "XDG_CACHE_HOME": str(output / "home/cache"),
        "PYTHONPATH": os.pathsep.join(str(repo / name) for name in PYTHON_ROOTS),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TOKENIZERS_PARALLELISM": "false",
        **{
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def stages(repo, python, output):
    root = output / "chain"
    helper = repo / "workers/policy_distillation/scripts/control_chain_fixture.py"
    http = repo / "workers/act_optimizer/scripts/temporal_http_fixture.py"
    result = [
        ("generate", [python, str(helper), "generate", str(root)]),
        (
            "distill",
            [
                python,
                "-m",
                "firebird_distill.application",
                str(root / "distill-request.json"),
                str(root / "distill-result.json"),
            ],
        ),
        ("verify-distill", [python, str(helper), "verify-distill", str(root)]),
    ]
    for bits in (8, 4):
        package = root / f"quant-{bits}/native-quantized"
        result.append(
            (
                f"quant-{bits}",
                [
                    python,
                    "-m",
                    "firebird_quant.native_application",
                    str(root / f"quant-{bits}-request.json"),
                    str(root / f"quant-{bits}-result.json"),
                ],
            )
        )
        result.append(
            (
                f"http-{bits}",
                [
                    python,
                    str(http),
                    str(package / "policy"),
                    str(package / "verification.json"),
                    str(root / f"http-{bits}.json"),
                    "--packed",
                    "--forbid",
                    str(root / "teacher"),
                    "--forbid",
                    str(root / "corpus"),
                    "--forbid",
                    str(root / "operation/distilled-policy/policy"),
                ],
            )
        )
    result.append(("verify-final", [python, str(helper), "verify-final", str(root)]))
    return result


def resource_check(sample, baseline_swap, elapsed):
    for key in ("owned_rss_bytes", "allocated_output_bytes"):
        if sample[key] > LIMITS[key]:
            raise Refused(f"Resource ceiling exceeded: {key}")
    if sample["free_disk_bytes"] < LIMITS["free_disk_bytes"]:
        raise Refused("Free disk fell below 10 GiB")
    if sample["swap_bytes"] - baseline_swap > LIMITS["swap_growth_bytes"]:
        raise Refused("System swap growth exceeded 128 MiB; cause is not attributed")
    if elapsed >= LIMITS["work_seconds"]:
        raise Refused("Total work deadline exceeded")


class Supervisor:
    def __init__(self, observer, output, started):
        self.observer, self.output, self.started = observer, output, started
        self.deadline = started + LIMITS["work_seconds"]
        self.tracker = Tracker(observer, output / "workdir")
        self.cancelled = None
        self.baseline_swap = observer.swap()
        self.peak = {"owned_rss_bytes": 0, "allocated_output_bytes": 0, "swap_growth_bytes": 0}
        self.samples, self.last = 0, None
        self.records = []

    def signal(self, number, _frame):
        self.cancelled = number  # Never raise inside Popen's registration window.

    def check(self):
        if self.cancelled is not None:
            raise Refused(f"Cancelled by signal {self.cancelled}")
        active = self.tracker.refresh()
        rows = self.observer.rows()
        if os.getpid() not in rows:
            raise Refused("Supervisor RSS is unavailable")
        sample = {
            "owned_rss_bytes": sum(p.rss for p in active.values()) + rows[os.getpid()][2],
            "allocated_output_bytes": allocated_bytes(self.output),
            "free_disk_bytes": shutil.disk_usage(self.output).free,
            "swap_bytes": self.observer.swap(),
        }
        self.samples += 1
        self.last = sample
        for key in ("owned_rss_bytes", "allocated_output_bytes"):
            self.peak[key] = max(self.peak[key], sample[key])
        self.peak["swap_growth_bytes"] = max(
            self.peak["swap_growth_bytes"], sample["swap_bytes"] - self.baseline_swap
        )
        resource_check(sample, self.baseline_swap, time.monotonic() - self.started)

    def cleanup(self, process):
        deadline = min(time.monotonic() + 15, self.deadline + LIMITS["cleanup_seconds"])
        errors = set()
        if process.pid not in self.tracker.known:
            # Registration failed while the stdlib bootstrap was still blocked.
            # The parent has closed its gate: the bootstrap must exit without exec.
            process.wait(timeout=2)
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            raise Refused("Unregistered bootstrap group disappearance is unverified")
        # Give the direct Python owner time to drain its separately-sessioned children.
        if process.poll() is None:
            try:
                known = self.tracker.known[process.pid]
                current = self.observer.process(process.pid)
                if current is not None:
                    if current.birth != known.birth:
                        raise Refused("Cannot verify leader identity for graceful cancellation")
                    os.kill(process.pid, signal.SIGTERM)
            except Exception as exc:
                errors.add(str(exc)[:300])
        for duration, sig in ((5, None), (2, signal.SIGTERM), (5, signal.SIGKILL)):
            if sig is not None:
                try:
                    self.tracker.refresh()
                except Exception as exc:
                    errors.add(str(exc)[:300])
                errors.update(self.tracker.signal_known(sig))
            until = min(time.monotonic() + duration, deadline)
            while time.monotonic() < until:
                process.poll()  # Reap direct child even after successful leader exit.
                try:
                    gone = self.tracker.gone()
                except Exception as exc:
                    errors.add(str(exc)[:300])
                    gone = False
                if gone:
                    process.wait(timeout=max(0.01, until - time.monotonic()))
                    if errors:
                        raise Refused(
                            "Cleanup observation was uncertain: " + "; ".join(sorted(errors)[:8])
                        )
                    return
                time.sleep(POLL)
        detail = "; ".join(sorted(errors)[:8])
        raise Refused("Cleanup is unverified; an owned process/group may remain. " + detail)

    def run(self, name, command, environment, cwd):
        self.check()
        record = {"stage": name, "command": command, "status": "starting"}
        self.records.append(record)
        read_fd, write_fd = os.pipe()
        process, tail, error = None, None, None
        # A stdlib-only bootstrap blocks until the parent records kernel PID birth.
        bootstrap = (
            "import os,sys;f=int(sys.argv[1]);v=os.read(f,1);os.close(f);"
            "sys.exit(125) if v!=b'1' else os.execve(sys.argv[2],sys.argv[2:],os.environ)"
        )
        try:
            process = subprocess.Popen(
                [command[0], "-I", "-S", "-c", bootstrap, str(read_fd), *command],
                env=environment,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(read_fd,),
            )
            os.close(read_fd)
            read_fd = None
            record["spawned_pid"] = process.pid
            identity = self.observer.process(process.pid)
            if identity is None or identity.group != process.pid:
                raise Refused("Cannot register the blocked stage process")
            self.tracker.register(identity)
            record["leader"] = asdict(identity)
            tail = Tail(process.stdout)
            self.check()
            os.write(write_fd, b"1")
            os.close(write_fd)
            write_fd = None
            record["status"] = "running"
            print(json.dumps({"stage": name, "pid": process.pid}), flush=True)
            while process.poll() is None:
                self.check()
                time.sleep(POLL)
            record["exit_code"] = process.returncode
            if process.returncode != 0:
                raise Refused(f"Stage {name} exited {process.returncode}; no retry")
            self.check()
            record["status"] = "passed"
        except BaseException as exc:
            error = exc
            record.update(status="failed", error=str(exc)[:2000])
        finally:
            for descriptor in (read_fd, write_fd):
                if descriptor is not None:
                    os.close(descriptor)
            if process is not None:
                try:
                    self.cleanup(process)
                    record["cleanup_confirmed"] = True
                except BaseException as exc:
                    record["cleanup_confirmed"] = False
                    record["cleanup_error"] = str(exc)[:2000]
                    error = Refused("Cleanup is unverified: " + str(exc))
                if tail is not None:
                    tail.thread.join(timeout=2)
                    if tail.thread.is_alive() or tail.error:
                        record["pipe_cleanup_confirmed"] = False
                        error = Refused("Owned output pipe did not drain cleanly")
                    else:
                        record["pipe_cleanup_confirmed"] = True
                        process.stdout.close()
                    tail.save(self.output / "logs" / f"{name}.log")
        if error is not None:
            raise error


def new_owner(path):
    if not path.is_absolute() or ".." in path.parts:
        raise Refused("Output must be an absolute new directory without traversal")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise Refused("Output path must not contain symbolic links")
    path.mkdir(mode=0o700)  # No parents creation, adoption or overwrite.
    for name in ("logs", "tmp", "home", "workdir"):
        (path / name).mkdir(mode=0o700)


def execute(repo, python, output):
    started = time.monotonic()
    new_owner(output)
    receipt = {
        "schema_version": 1,
        "status": "failed",
        "limits": LIMITS,
        "output": str(output),
        "fixture_root": str(output / "chain"),
        "scope": "Generated ACT CPU software chain; no calibration, simulator or quality claim",
        "resources": "Sampled abort ceilings; not kernel memory/disk reservations",
        "automatic_retries": 0,
        "files_deleted": 0,
    }
    owner, before, failure = None, None, None
    handlers = {}
    try:
        if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
            raise Refused("An existing absolute Python executable is required")
        receipt["python"] = {
            "path": str(python),
            "resolved": str(python.resolve()),
            **file_hash(python.resolve(), 32 * MIB),
        }
        if (
            shutil.disk_usage(output).free
            < LIMITS["free_disk_bytes"] + LIMITS["allocated_output_bytes"]
        ):
            raise Refused("Preflight needs 10 GiB free plus the full 320 MiB output budget")
        before = source_identity(repo)
        receipt["source_before"] = before
        owner = Supervisor(Darwin(), output, started)
        receipt["cwd_ownership"] = {
            "path": str(output / "workdir"),
            "device": owner.tracker.tag[0],
            "inode": owner.tracker.tag[1],
            "uid": os.getuid(),
            "scope": "Fixed no-chdir inherited-cwd graph only; no hostile daemon containment",
        }
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, owner.signal)
        for name, command in stages(repo, str(python), output):
            owner.run(name, command, clean_environment(repo, output), output / "workdir")
        owner.check()
        final = output / "chain/chain-verification.json"
        receipt["chain_verification"] = file_hash(final, MIB)
        raw = final.read_bytes()
        value = finite_json(raw)
        if not isinstance(value, dict) or value.get("status") != "passed":
            raise Refused("Final chain verification did not report passed")
        receipt["status"] = "passed"
    except BaseException as exc:
        failure = exc
        receipt["error"] = str(exc)[:2000]
    finally:
        if before is not None:
            try:
                receipt["source_unchanged"] = source_identity(repo) == before
                if not receipt["source_unchanged"]:
                    raise Refused("Source changed during the proof")
            except BaseException as exc:
                failure = exc
                receipt.update(status="failed", source_error=str(exc)[:2000])
        if owner is not None:
            try:
                owner.check()
            except BaseException as exc:
                failure = exc
                receipt.update(status="failed", final_resource_error=str(exc)[:2000])
            receipt.update(
                stages=owner.records,
                processes=owner.tracker.history,
                resource_peak=owner.peak,
                resource_samples=owner.samples,
                last_resource_sample=owner.last,
                baseline_swap_bytes=owner.baseline_swap,
            )
            try:
                receipt["owned_processes_gone"] = owner.tracker.gone()
                if not receipt["owned_processes_gone"]:
                    raise Refused("Owned process absence was not confirmed")
            except BaseException as exc:
                failure = exc
                receipt.update(status="failed", cleanup_error=str(exc)[:2000])
        receipt["elapsed_seconds"] = time.monotonic() - started
        if failure is not None:
            receipt["status"] = "failed"
        raw = canonical(receipt)
        if len(raw) > MIB:
            raise Refused("Receipt exceeds its bound; original output is preserved")
        with (output / "supervisor-receipt.json").open("xb") as stream:
            stream.write(raw)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        print(
            json.dumps(
                {"status": receipt["status"], "receipt": str(output / "supervisor-receipt.json")}
            ),
            flush=True,
        )
    return 0 if receipt["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if not args.execute:
        print(
            canonical(
                {
                    "execute": False,
                    "limits": LIMITS,
                    "stages": stages(repo, str(args.python), args.output),
                }
            ).decode()
        )
        return 0
    return execute(repo, args.python, args.output)


if __name__ == "__main__":
    raise SystemExit(main())

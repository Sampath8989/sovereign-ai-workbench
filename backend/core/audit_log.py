"""
Audit Logger: SHA-256 hash-chained JSONL logger for tamper-evident event recording.

Bug 2 Fix: Sequence numbers + periodic checkpoints detect truncation.
Bug 5 Fix: Singleton writer thread shared across ALL AuditLogger instances.
"""

import hashlib
import json
import os
import queue
import threading
import time
from typing import List, Optional

AUDIT_LOG_DIR = "data"
AUDIT_LOG_FILE = os.path.join(AUDIT_LOG_DIR, "audit_log.jsonl")
CHECKPOINT_FILE = os.path.join(AUDIT_LOG_DIR, "audit_checkpoints.jsonl")
CHECKPOINT_INTERVAL = 1  # Checkpoint every entry to prevent tail-truncation gaps


class _WriterThread:
    """
    Singleton writer thread shared by ALL AuditLogger instances.
    Processes queued writes sequentially, eliminating races between
    multiple AuditLogger instances writing to the same file.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._next_sequence = 1
        self._init_lock = threading.Lock()

    def _loop(self):
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break

            file_path, checkpoint_file, event_type, details, result_holder = item
            try:
                entry = self._write_entry(file_path, checkpoint_file, event_type, details)
                result_holder["result"] = entry
            except Exception as e:
                result_holder["error"] = e
            finally:
                result_holder["done"].set()

    def _get_last_hash(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return "GENESIS"
        last_hash = "GENESIS"
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        last_hash = entry.get("current_hash", last_hash)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            return "GENESIS"
        return last_hash

    def _compute_hash(self, prev_hash: str, entry_data: dict) -> str:
        payload = f"{prev_hash}{json.dumps(entry_data, sort_keys=True)}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _write_entry(self, file_path, checkpoint_file, event_type, details):
        prev_hash = self._get_last_hash(file_path)
        seq = self._next_sequence
        self._next_sequence += 1

        entry_data = {
            "timestamp": time.time(),
            "sequence": seq,
            "event_type": event_type,
            "details": details,
            "prev_hash": prev_hash,
        }
        current_hash = self._compute_hash(prev_hash, entry_data)
        entry_data["current_hash"] = current_hash

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "a") as f:
            f.write(json.dumps(entry_data, sort_keys=True) + "\n")

        if seq % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                "timestamp": time.time(),
                "sequence": seq,
                "chain_hash": current_hash,
                "checkpoint_type": "periodic",
            }
            with open(checkpoint_file, "a") as f:
                f.write(json.dumps(checkpoint, sort_keys=True) + "\n")

        return entry_data

    def enqueue(self, file_path, checkpoint_file, event_type, details):
        result_holder = {"done": threading.Event(), "result": None, "error": None}
        self._queue.put((file_path, checkpoint_file, event_type, details, result_holder))
        result_holder["done"].wait(timeout=10.0)
        if result_holder["error"]:
            raise result_holder["error"]
        return result_holder["result"]

    def sync_sequence_from_file(self, file_path):
        """Initialize sequence counter from existing log file."""
        with self._init_lock:
            if not os.path.exists(file_path):
                return
            last_seq = 0
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            seq = entry.get("sequence", 0)
                            if seq > last_seq:
                                last_seq = seq
                        except json.JSONDecodeError:
                            continue
            except IOError:
                pass
            if last_seq >= self._next_sequence:
                self._next_sequence = last_seq + 1


# Module-level singleton writer
_singleton_writer: Optional[_WriterThread] = None
_singleton_lock = threading.Lock()


def _get_writer() -> _WriterThread:
    global _singleton_writer
    if _singleton_writer is None:
        with _singleton_lock:
            if _singleton_writer is None:
                _singleton_writer = _WriterThread()
    return _singleton_writer


def _reset_singleton_for_testing() -> None:
    """
    Reset the singleton writer thread to a fresh state.
    Only for use in test fixtures — never call in production.
    """
    global _singleton_writer
    with _singleton_lock:
        _singleton_writer = _WriterThread()


class AuditLogger:
    """
    Append-only, hash-chained JSONL audit log with truncation detection.
    All instances share a single writer thread (singleton pattern).
    """

    def __init__(self, file_path: str = AUDIT_LOG_FILE):
        self.file_path = file_path
        self.checkpoint_file = os.path.join(
            os.path.dirname(file_path), "audit_checkpoints.jsonl"
        )
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        self._writer = _get_writer()
        # Sync sequence from existing file on first use
        self._writer.sync_sequence_from_file(self.file_path)

    def log_event(self, event_type: str, details: dict) -> dict:
        return self._writer.enqueue(
            self.file_path, self.checkpoint_file, event_type, details
        )

    def get_last_entry(self) -> Optional[dict]:
        if not os.path.exists(self.file_path):
            return None
        last_entry = None
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last_entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            return None
        return last_entry

    def read_all_entries(self) -> List[dict]:
        entries = []
        if not os.path.exists(self.file_path):
            return entries
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        return entries

    def clear(self):
        """Clear the audit log and checkpoints (for testing)."""
        global _singleton_writer
        with _singleton_lock:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            if _singleton_writer:
                _singleton_writer._next_sequence = 1

    @staticmethod
    def _reset_for_testing() -> None:
        """
        Reset the module-level singleton writer to a completely fresh state.
        Drops the old writer thread and creates a new one with _next_sequence=1.
        Only for use in test fixtures — never call in production.
        """
        _reset_singleton_for_testing()

    def shutdown(self):
        pass  # Singleton thread is daemon, dies with process


def verify_chain(file_path: str = AUDIT_LOG_FILE) -> dict:
    """Verify chain integrity + truncation detection. Returns dict."""
    checkpoint_file = os.path.join(os.path.dirname(file_path), "audit_checkpoints.jsonl")
    result = {
        "valid": True, "truncated": False, "sequence_gap": False,
        "checkpoint_valid": True, "entry_count": 0, "last_sequence": 0, "details": "",
    }

    if not os.path.exists(file_path):
        result["details"] = "Empty log — valid"
        return result

    entries = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    result["valid"] = False
                    result["details"] = f"Corrupted JSONL at entry {len(entries)}"
                    return result
    except IOError:
        result["details"] = "File read error"
        return result

    if not entries:
        result["details"] = "No entries — valid"
        return result

    result["entry_count"] = len(entries)

    expected_prev = "GENESIS"
    for i, entry in enumerate(entries):
        if entry.get("prev_hash") != expected_prev:
            result["valid"] = False
            result["details"] = f"Hash chain broken at entry {i}: prev_hash mismatch"
            return result

        entry_data = {
            "timestamp": entry["timestamp"],
            "sequence": entry.get("sequence", 0),
            "event_type": entry["event_type"],
            "details": entry["details"],
            "prev_hash": entry["prev_hash"],
        }
        computed_hash = hashlib.sha256(
            f"{expected_prev}{json.dumps(entry_data, sort_keys=True)}".encode()
        ).hexdigest()

        if entry.get("current_hash") != computed_hash:
            result["valid"] = False
            result["details"] = f"Hash mismatch at entry {i}"
            return result

        expected_prev = computed_hash

    sequences = [e.get("sequence", 0) for e in entries]
    result["last_sequence"] = max(sequences) if sequences else 0

    expected_seq = 1
    for seq in sequences:
        if seq != expected_seq:
            result["sequence_gap"] = True
            result["valid"] = False
            result["details"] = f"Sequence gap: expected {expected_seq}, got {seq}"
            break
        expected_seq = seq + 1

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cp = json.loads(line)
                        cp_seq = cp.get("sequence", 0)
                        cp_hash = cp.get("chain_hash", "")
                        if cp_seq > result["last_sequence"]:
                            result["truncated"] = True
                            result["valid"] = False
                            result["checkpoint_valid"] = False
                            result["details"] = (
                                f"TRUNCATION: checkpoint seq {cp_seq} > log seq {result['last_sequence']}"
                            )
                            return result
                        matching = [e for e in entries if e.get("sequence") == cp_seq]
                        if matching and matching[0].get("current_hash") != cp_hash:
                            result["valid"] = False
                            result["checkpoint_valid"] = False
                            result["details"] = f"Checkpoint hash mismatch at seq {cp_seq}"
                            return result
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

    if not result["details"]:
        if result["valid"]:
            result["details"] = (
                f"Chain valid. {result['entry_count']} entries, "
                f"last seq={result['last_sequence']}, no truncation."
            )
        else:
            result["details"] = (
                f"Chain INVALID. {result['entry_count']} entries, "
                f"last seq={result['last_sequence']}, "
                f"truncated={result['truncated']}, sequence_gap={result['sequence_gap']}."
            )
    return result

"""
Sandbox Manager: Docker-based code execution with gVisor (runsc) isolation.
Falls back to runc with resource limits if runsc unavailable.
Startup health-check verifies container actually started.
"""

import os
import platform
import shutil
import subprocess
import tempfile
import time
import logging
import uuid
from typing import Optional

from backend.core.audit_log import AuditLogger

logger = logging.getLogger(__name__)

try:
    import docker
    from docker.errors import APIError as DockerAPIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

CONTAINER_IMAGE = "python:3.10-slim"
EXECUTION_TIMEOUT = 30
SANDBOX_MEMORY_LIMIT = "256m"
SANDBOX_PIDS_LIMIT = 64
SANDBOX_SHM_SIZE = 32 * 1024 * 1024


class SandboxManager:
    def __init__(self):
        self.audit = AuditLogger()
        self.os_type = platform.system()
        self._docker_client = None
        self._runtime = "runc"
        self._runtime_checked = False

        if DOCKER_AVAILABLE:
            try:
                self._docker_client = docker.from_env()
                self._docker_client.ping()
                self._detect_runtime()
            except Exception as e:
                logger.warning(f"Docker daemon not available: {e}")
                self._docker_client = None

    def _detect_runtime(self):
        """Detect available runtimes. Prefer runsc (gVisor) over runc."""
        try:
            info = self._docker_client.info()
            runtimes = info.get("Runtimes", {})
            if "runsc" in runtimes:
                self._runtime = "runsc"
                logger.info("gVisor (runsc) runtime detected and available.")
            else:
                self._runtime = "runc"
                logger.info("gVisor not available. Using runc.")
            self._runtime_checked = True
        except Exception:
            self._runtime = "runc"
            self._runtime_checked = True

    def _verify_container_started(self, container) -> bool:
        """Verify container actually ran (not just created)."""
        try:
            container.reload()
            return container.status != "created"
        except Exception:
            return False

    def _get_container_kwargs(self) -> dict:
        kwargs = {
            "image": CONTAINER_IMAGE,
            "detach": True,
            "auto_remove": False,
            "network_disabled": True,  # Fail-closed: no network by default
            "mem_limit": SANDBOX_MEMORY_LIMIT,
            "pids_limit": SANDBOX_PIDS_LIMIT,
            "shm_size": SANDBOX_SHM_SIZE,
            "read_only": True,
        }

        if self._runtime == "runsc":
            kwargs["runtime"] = "runsc"
        # else: no runtime kwarg = Docker default (runc)

        return kwargs

    def execute_code(self, python_code: str) -> dict:
        start_time = time.time()

        if self._docker_client is None:
            return self._execute_stub(python_code)

        container = None
        code_dir = None
        writable_dir = None
        try:
            # Each execution gets TWO unique ephemeral directories:
            # 1. code_dir: read-only bind mount at /app (contains user script)
            # 2. writable_dir: read-write bind mount at /tmp (isolated writable space)
            # This guarantees no two executions can ever share writable storage.
            code_dir = tempfile.mkdtemp()
            writable_dir = tempfile.mkdtemp()
            code_file = os.path.join(code_dir, "script.py")
            with open(code_file, "w") as f:
                f.write(python_code)

            container_kwargs = self._get_container_kwargs()
            container = self._docker_client.containers.run(
                command="python /app/script.py",
                volumes={
                    code_dir: {"bind": "/app", "mode": "ro"},
                    writable_dir: {"bind": "/tmp", "mode": "rw"},
                },
                working_dir="/app",
                **container_kwargs,
            )

            if not self._verify_container_started(container):
                try: container.remove(force=True)
                except: pass
                return {
                    "stdout": "", "stderr": "Container failed to start (health-check)",
                    "exit_code": -1, "execution_time": round(time.time() - start_time, 4),
                    "method": f"{self._runtime}_startup_failure", "started": False,
                }

            result = container.wait(timeout=EXECUTION_TIMEOUT)
            exit_code = result.get("StatusCode", -1)
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            output = {
                "stdout": stdout, "stderr": stderr, "exit_code": exit_code,
                "execution_time": round(time.time() - start_time, 4),
                "method": self._runtime, "started": True,
                "runtime": self._runtime,
            }

            self.audit.log_event("SANDBOX_EXEC", {
                "code_preview": python_code[:200], "exit_code": exit_code,
                "runtime": self._runtime, "memory_limit": SANDBOX_MEMORY_LIMIT,
                "pids_limit": SANDBOX_PIDS_LIMIT,
            })
            return output

        except DockerAPIError as e:
            return {
                "stdout": "", "stderr": f"Docker API error: {e}",
                "exit_code": -1, "execution_time": round(time.time() - start_time, 4),
                "method": "docker_api_error", "started": False,
            }
        except Exception as e:
            return {
                "stdout": "", "stderr": str(e), "exit_code": -1,
                "execution_time": round(time.time() - start_time, 4),
                "method": "error", "started": False,
            }
        finally:
            if container:
                try: container.remove(force=True)
                except: pass
            # Always clean up ephemeral directories to prevent host-side residue
            for d in (code_dir, writable_dir):
                if d and os.path.exists(d):
                    try: shutil.rmtree(d)
                    except: pass

    def _execute_stub(self, python_code: str) -> dict:
        import sys
        try:
            result = subprocess.run(
                [sys.executable, "-c", python_code],
                capture_output=True, text=True, timeout=EXECUTION_TIMEOUT,
            )
            return {
                "stdout": result.stdout, "stderr": result.stderr,
                "exit_code": result.returncode, "execution_time": 0.0,
                "method": "subprocess_stub", "started": True,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "", "stderr": "Execution timed out",
                "exit_code": -1, "execution_time": EXECUTION_TIMEOUT,
                "method": "subprocess_stub_timeout", "started": True,
            }
        except Exception as e:
            return {
                "stdout": "", "stderr": str(e), "exit_code": -1,
                "execution_time": 0.0, "method": "subprocess_stub_error", "started": False,
            }

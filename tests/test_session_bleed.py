#!/usr/bin/env python3
"""
Session bleed test (FIXED version): uses per-execution unique bind mount for /tmp
instead of shared Docker tmpfs. Each container gets its own ephemeral writable dir.
"""
import docker
import uuid
import shutil
import tempfile
import os
import sys


def run_bleed_test(num_tests=10, runtime="runsc"):
    """Run session bleed test num_tests times and return pass/fail results."""
    c = docker.from_env()
    results = []

    for i in range(num_tests):
        marker = str(uuid.uuid4())

        # Container 1: Write a unique marker to /tmp
        code1 = f"""import os
marker = "{marker}"
path = '/tmp/bleed_test_' + marker
with open(path, 'w') as f:
    f.write(marker)
print(f'Wrote marker: ' + marker)
print(f'Files in /tmp: ' + str(os.listdir('/tmp')))
"""
        # Container 2: Try to read the marker from a DIFFERENT /tmp
        code2 = f"""import os
marker = "{marker}"
target = '/tmp/bleed_test_' + marker
print(f'Looking for: ' + target)
if os.path.exists(target):
    with open(target) as f:
        content = f.read()
    print(f'LEAKED! Content: ' + content)
    print('leaked=True')
else:
    print(f'File not found - isolation holds')
    print('leaked=False')
print(f'Files in /tmp: ' + str(os.listdir('/tmp')))
"""

        code_dir1 = tempfile.mkdtemp()
        writable_dir1 = tempfile.mkdtemp()
        with open(os.path.join(code_dir1, "script.py"), "w") as f:
            f.write(code1)

        # Container 1: Write marker using FIXED config (unique bind mount for /tmp)
        cont1 = c.containers.run(
            command="python /app/script.py",
            image="python:3.10-slim",
            detach=True, auto_remove=False, network_disabled=True,
            mem_limit="256m", pids_limit=64, shm_size=32 * 1024 * 1024,
            read_only=True,
            runtime=runtime,
            volumes={
                code_dir1: {"bind": "/app", "mode": "ro"},
                writable_dir1: {"bind": "/tmp", "mode": "rw"},
            },
            working_dir="/app",
        )
        r1 = cont1.wait(timeout=30)
        out1 = cont1.logs(stdout=True, stderr=True).decode()
        cont1.remove(force=True)
        shutil.rmtree(code_dir1, ignore_errors=True)
        shutil.rmtree(writable_dir1, ignore_errors=True)

        # Container 2: Try to read the marker using a BRAND NEW /tmp
        code_dir2 = tempfile.mkdtemp()
        writable_dir2 = tempfile.mkdtemp()
        with open(os.path.join(code_dir2, "script.py"), "w") as f:
            f.write(code2)

        cont2 = c.containers.run(
            command="python /app/script.py",
            image="python:3.10-slim",
            detach=True, auto_remove=False, network_disabled=True,
            mem_limit="256m", pids_limit=64, shm_size=32 * 1024 * 1024,
            read_only=True,
            runtime=runtime,
            volumes={
                code_dir2: {"bind": "/app", "mode": "ro"},
                writable_dir2: {"bind": "/tmp", "mode": "rw"},
            },
            working_dir="/app",
        )
        r2 = cont2.wait(timeout=30)
        out2 = cont2.logs(stdout=True, stderr=True).decode()
        cont2.remove(force=True)
        shutil.rmtree(code_dir2, ignore_errors=True)
        shutil.rmtree(writable_dir2, ignore_errors=True)

        leaked = "leaked=True" in out2
        results.append(leaked)

        status = "FAIL (LEAKED)" if leaked else "PASS (isolated)"
        print(f"  Iteration {i+1}: {status}")
        print(f"    Container 1 output: {out1.strip()}")
        print(f"    Container 2 output: {out2.strip()}")
        if leaked:
            print(f"    *** SESSION BLEED DETECTED ***")
        print()

    return results


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    runtime = sys.argv[2] if len(sys.argv) > 2 else "runsc"
    print(f"=== Session Bleed Test (FIXED): {num} iterations, runtime={runtime} ===")
    print(f"Fix: Each container gets unique ephemeral /tmp via bind mount (no Docker tmpfs)\n")

    results = run_bleed_test(num, runtime)

    passes = sum(1 for r in results if not r)
    fails = sum(1 for r in results if r)
    print(f"=== FINAL: {passes}/{num} passed, {fails}/{num} leaked ===")
    sys.exit(0 if fails == 0 else 1)

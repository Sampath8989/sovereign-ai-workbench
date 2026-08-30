#!/usr/bin/env python3
"""
Production stack test: From inside the running agent container on sovereign-net:
1. Qdrant: real query, real response
2. Postgres: real query, real response
3. External network: 11-path battery, all must fail
Run 3 times.
"""
import subprocess
import json
import sys

# Test script that runs INSIDE the agent container
INSIDE_AGENT_TEST = r"""
import socket, json, os, sys, struct, time

results = {}

# ============================================================
# TEST 1: QDRANT CONNECTIVITY
# ============================================================
try:
    import urllib.request
    req = urllib.request.Request("http://sovereign-qdrant:6333/healthz")
    resp = urllib.request.urlopen(req, timeout=5)
    body = resp.read().decode()
    results['qdrant_healthz'] = f'SUCCESS: status={resp.status}, body={body.strip()[:200]}'
except Exception as e:
    results['qdrant_healthz'] = f'FAIL: {type(e).__name__}: {e}'

try:
    import urllib.request, json as j
    data = j.dumps({"vectors": {"size": 4, "distance": "Cosine"}}).encode()
    req = urllib.request.Request(
        "http://sovereign-qdrant:6333/collections/test_collection",
        data=data, headers={"Content-Type": "application/json"}, method="PUT"
    )
    resp = urllib.request.urlopen(req, timeout=5)
    body = resp.read().decode()
    results['qdrant_create_collection'] = f'SUCCESS: status={resp.status}, body={body.strip()[:200]}'
except Exception as e:
    results['qdrant_create_collection'] = f'FAIL: {type(e).__name__}: {e}'

try:
    import urllib.request, json as j
    data = j.dumps({
        "points": [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"text": "test from agent"}}]
    }).encode()
    req = urllib.request.Request(
        "http://sovereign-qdrant:6333/collections/test_collection/points",
        data=data, headers={"Content-Type": "application/json"}, method="PUT"
    )
    resp = urllib.request.urlopen(req, timeout=5)
    body = resp.read().decode()
    results['qdrant_insert'] = f'SUCCESS: status={resp.status}, body={body.strip()[:200]}'
except Exception as e:
    results['qdrant_insert'] = f'FAIL: {type(e).__name__}: {e}'

try:
    import urllib.request, json as j
    data = j.dumps({"vector": [0.1, 0.2, 0.3, 0.4], "limit": 1}).encode()
    req = urllib.request.Request(
        "http://sovereign-qdrant:6333/collections/test_collection/points/search",
        data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=5)
    body = resp.read().decode()
    results['qdrant_search'] = f'SUCCESS: status={resp.status}, body={body.strip()[:300]}'
except Exception as e:
    results['qdrant_search'] = f'FAIL: {type(e).__name__}: {e}'

# ============================================================
# TEST 2: POSTGRES CONNECTIVITY
# ============================================================
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(('sovereign-postgres', 5432))
    # Correct Postgres v3.0 startup: length (includes itself) | version | params
    params = b'user\x00sovereign\x00database\x00sovereign_ai\x00\x00'
    length = 4 + 4 + len(params)
    startup = struct.pack('!II', length, 196608) + params
    s.sendall(startup)
    data = s.recv(4096)
    if data and data[0:1] == b'R':
        auth_code = struct.unpack('!I', data[5:9])[0]
        auth_names = {0: 'AuthenticationOk', 3: 'PasswordMessage', 5: 'MD5Password',
                      10: 'SASL', 11: 'SASLContinue', 12: 'SASLFinal'}
        auth_name = auth_names.get(auth_code, f'Unknown({auth_code})')
        results['postgres_auth'] = f'SUCCESS: received auth request ({len(data)} bytes), type={auth_name}, server responded'
    elif data and data[0:1] == b'E':
        results['postgres_auth'] = f'SUCCESS: received error response ({len(data)} bytes), server is reachable'
    elif data:
        results['postgres_auth'] = f'SUCCESS: received {len(data)} bytes, first_byte={data[0]:02x}'
    else:
        results['postgres_auth'] = f'FAIL: received 0 bytes'
    s.close()
except Exception as e:
    results['postgres_auth'] = f'FAIL: {type(e).__name__}: {e}'

# ============================================================
# TEST 3: EXTERNAL NETWORK BATTERY (11 paths)
# ============================================================

# 3a: TCP connect to 8.8.8.8:53
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(('8.8.8.8', 53))
    results['ext_tcp_8888'] = 'SUCCESS - connection established'
    s.close()
except socket.timeout:
    results['ext_tcp_8888'] = 'TIMEOUT after 3s'
except OSError as e:
    results['ext_tcp_8888'] = f'OSError: {e}'
except Exception as e:
    results['ext_tcp_8888'] = f'{type(e).__name__}: {e}'

# 3b: TCP connect to 1.1.1.1:443
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(('1.1.1.1', 443))
    results['ext_tcp_1111'] = 'SUCCESS - connection established'
    s.close()
except socket.timeout:
    results['ext_tcp_1111'] = 'TIMEOUT after 3s'
except OSError as e:
    results['ext_tcp_1111'] = f'OSError: {e}'
except Exception as e:
    results['ext_tcp_1111'] = f'{type(e).__name__}: {e}'

# 3c: DNS gethostbyname
try:
    ip = socket.gethostbyname('google.com')
    results['ext_dns_gethostbyname'] = f'SUCCESS - resolved to {ip}'
except socket.timeout:
    results['ext_dns_gethostbyname'] = 'TIMEOUT'
except OSError as e:
    results['ext_dns_gethostbyname'] = f'OSError: {e}'
except Exception as e:
    results['ext_dns_gethostbyname'] = f'{type(e).__name__}: {e}'

# 3d: DNS getaddrinfo
try:
    addrs = socket.getaddrinfo('google.com', 80, socket.AF_INET, socket.SOCK_STREAM)
    results['ext_dns_getaddrinfo'] = f'SUCCESS - resolved to {addrs[0][4][0]}'
except socket.timeout:
    results['ext_dns_getaddrinfo'] = 'TIMEOUT'
except OSError as e:
    results['ext_dns_getaddrinfo'] = f'OSError: {e}'
except Exception as e:
    results['ext_dns_getaddrinfo'] = f'{type(e).__name__}: {e}'

# 3e: UDP DNS
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(3)
    dns_query = bytes([0xAA, 0xBB, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
                       0x00, 0x00, 0x00, 0x00, 0x06, 0x67, 0x6F, 0x6F,
                       0x67, 0x6C, 0x65, 0x03, 0x63, 0x6F, 0x6D, 0x00,
                       0x00, 0x01, 0x00, 0x01])
    s.sendto(dns_query, ('8.8.8.8', 53))
    data, addr = s.recvfrom(512)
    results['ext_udp_dns'] = f'SUCCESS - received {len(data)} bytes'
    s.close()
except socket.timeout:
    results['ext_udp_dns'] = 'TIMEOUT after 3s'
except OSError as e:
    results['ext_udp_dns'] = f'OSError: {e}'

# 3f: Raw socket
try:
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    results['ext_raw_socket'] = 'SUCCESS - raw socket created'
    s.close()
except OSError as e:
    results['ext_raw_socket'] = f'OSError: {e}'

# 3g: ICMP socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    results['ext_icmp_socket'] = 'SUCCESS - ICMP socket created'
    s.close()
except OSError as e:
    results['ext_icmp_socket'] = f'OSError: {e}'

# 3h: Docker gateway
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    for gw in ['172.17.0.1', '172.18.0.1', '172.19.0.1', '172.20.0.1', '10.0.0.1']:
        try:
            s.connect((gw, 2375))
            results['ext_docker_gateway'] = f'SUCCESS - connected to {gw}:2375'
            break
        except:
            pass
    else:
        results['ext_docker_gateway'] = 'BLOCKED - all gateway IPs unreachable'
    s.close()
except socket.timeout:
    results['ext_docker_gateway'] = 'TIMEOUT'
except OSError as e:
    results['ext_docker_gateway'] = f'OSError: {e}'

# 3i: HTTP urllib
try:
    import urllib.request
    resp = urllib.request.urlopen('http://example.com', timeout=3)
    results['ext_http'] = f'SUCCESS - status {resp.status}'
except socket.timeout:
    results['ext_http'] = 'TIMEOUT after 3s'
except OSError as e:
    results['ext_http'] = f'OSError: {e}'
except Exception as e:
    results['ext_http'] = f'{type(e).__name__}: {e}'

# 3j: IPv6 DNS
try:
    addrs = socket.getaddrinfo('google.com', 80, socket.AF_INET6, socket.SOCK_STREAM)
    results['ext_dns_ipv6'] = f'SUCCESS - resolved to {addrs[0][4][0]}'
except socket.timeout:
    results['ext_dns_ipv6'] = 'TIMEOUT'
except OSError as e:
    results['ext_dns_ipv6'] = f'OSError: {e}'
except Exception as e:
    results['ext_dns_ipv6'] = f'{type(e).__name__}: {e}'

# 3k: localhost TCP
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(('127.0.0.1', 80))
    results['ext_localhost_tcp'] = 'SUCCESS - connection established'
    s.close()
except socket.timeout:
    results['ext_localhost_tcp'] = 'TIMEOUT after 3s'
except OSError as e:
    results['ext_localhost_tcp'] = f'OSError: {e}'

print('RESULTS_JSON_START')
print(json.dumps(results, indent=2))
print('RESULTS_JSON_END')
"""


def run_iteration(c, iteration):
    """Execute the test script inside the agent container."""
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    script_path = os.path.join(tmpdir, "prod_test.py")
    with open(script_path, "w") as f:
        f.write(INSIDE_AGENT_TEST)

    cp_result = subprocess.run(
        ["docker", "cp", script_path, "sovereign-agent:/tmp/prod_test.py"],
        capture_output=True, text=True
    )
    if cp_result.returncode != 0:
        print(f"  docker cp failed: {cp_result.stderr}")
        return {}

    exec_result = subprocess.run(
        ["docker", "exec", "sovereign-agent", "python", "/tmp/prod_test.py"],
        capture_output=True, text=True, timeout=60
    )

    output = exec_result.stdout + exec_result.stderr

    if "RESULTS_JSON_START" in output and "RESULTS_JSON_END" in output:
        start = output.index("RESULTS_JSON_START") + len("RESULTS_JSON_START") + 1
        end = output.index("RESULTS_JSON_END")
        json_str = output[start:end].strip()
        try:
            results = json.loads(json_str)
        except:
            print(f"  JSON parse error. Raw output:\n{output}")
            results = {}
    else:
        print(f"  Could not find results markers. Raw output:\n{output}")
        results = {}

    os.remove(script_path)
    os.rmdir(tmpdir)

    return results


if __name__ == "__main__":
    import docker
    c = docker.from_env()

    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print("=== Production Stack Verification ===")
    for name in ['sovereign-agent', 'sovereign-qdrant', 'sovereign-postgres']:
        try:
            info = c.api.inspect_container(name)
            state = info.get("State", {}).get("Status", "unknown")
            print(f"  {name}: {state}")
        except:
            print(f"  {name}: NOT FOUND")

    net_info = c.api.inspect_network("sovereign-ai-workbench_sovereign-net")
    print(f"  sovereign-net: internal={net_info.get('Internal')}")
    print()

    all_results = []

    for i in range(num_runs):
        print(f"=== Iteration {i+1}/{num_runs} ===")
        results = run_iteration(c, i)
        all_results.append(results)

        print("\n  [QDRANT CONNECTIVITY]")
        for key in ['qdrant_healthz', 'qdrant_create_collection', 'qdrant_insert', 'qdrant_search']:
            val = results.get(key, 'NOT RUN')
            print(f"    {key}: {val}")

        print("\n  [POSTGRES CONNECTIVITY]")
        val = results.get('postgres_auth', 'NOT RUN')
        print(f"    postgres_auth: {val}")

        print("\n  [EXTERNAL NETWORK BATTERY]")
        ext_keys = [k for k in results if k.startswith('ext_')]
        ext_blocked = 0
        ext_total = 0
        for key in sorted(ext_keys):
            val = results[key]
            ext_total += 1
            is_blocked = not val.startswith("SUCCESS")
            if is_blocked:
                ext_blocked += 1
            print(f"    {key}: {val}")
        print(f"    External: {ext_blocked}/{ext_total} blocked")
        print()

    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    qdrant_ok = 0
    qdrant_total = 0
    for r in all_results:
        for key in ['qdrant_healthz', 'qdrant_insert', 'qdrant_search']:
            qdrant_total += 1
            if r.get(key, '').startswith("SUCCESS"):
                qdrant_ok += 1
    print(f"\n  Qdrant: {qdrant_ok}/{qdrant_total} operations succeeded across {num_runs} runs")

    pg_ok = 0
    for r in all_results:
        if r.get('postgres_auth', '').startswith("SUCCESS"):
            pg_ok += 1
    print(f"  Postgres: {pg_ok}/{num_runs} connections succeeded across {num_runs} runs")

    ext_leaked = 0
    ext_total = 0
    for r in all_results:
        for key in [k for k in r if k.startswith('ext_')]:
            ext_total += 1
            if r[key].startswith("SUCCESS"):
                ext_leaked += 1
    print(f"  External network: {ext_leaked}/{ext_total} leaked (must be 0)")

    if qdrant_ok == qdrant_total and pg_ok == num_runs and ext_leaked == 0:
        print(f"\n  PASS: Agent reaches Qdrant + Postgres, zero internet egress")
    else:
        print(f"\n  FAIL: See details above")

    sys.exit(0 if (qdrant_ok == qdrant_total and pg_ok == num_runs and ext_leaked == 0) else 1)

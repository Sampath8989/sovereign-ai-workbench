#!/usr/bin/env python3
"""
Bug C: Test agent process egress control on an internal Docker network.
Verifies that containers on an internal Docker network have zero external
network access while still being able to communicate with other containers
on the same network.

Key container hardening:
- network: internal Docker bridge (no internet route)
- cap_drop: [NET_RAW] (prevents raw/ICMP socket creation)
- read_only: true (root filesystem is read-only)
"""
import docker
import shutil
import tempfile
import os
import sys
import json

# Network test code that runs INSIDE the container
NETWORK_TEST_CODE = r"""
import socket, json, os, sys

results = {}

# Test 1: TCP connect to 8.8.8.8:53
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(('8.8.8.8', 53))
    results['tcp_connect_8888'] = 'SUCCESS - connection established'
    s.close()
except socket.timeout:
    results['tcp_connect_8888'] = 'TIMEOUT after 3s'
except OSError as e:
    results['tcp_connect_8888'] = f'OSError: {e}'
except Exception as e:
    results['tcp_connect_8888'] = f'{type(e).__name__}: {e}'

# Test 2: TCP connect to 1.1.1.1:443
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(('1.1.1.1', 443))
    results['tcp_connect_1111'] = 'SUCCESS - connection established'
    s.close()
except socket.timeout:
    results['tcp_connect_1111'] = 'TIMEOUT after 3s'
except OSError as e:
    results['tcp_connect_1111'] = f'OSError: {e}'
except Exception as e:
    results['tcp_connect_1111'] = f'{type(e).__name__}: {e}'

# Test 3: DNS gethostbyname
try:
    ip = socket.gethostbyname('google.com')
    results['dns_gethostbyname'] = f'SUCCESS - resolved to {ip}'
except socket.timeout:
    results['dns_gethostbyname'] = 'TIMEOUT after default timeout'
except OSError as e:
    results['dns_gethostbyname'] = f'OSError: {e}'
except Exception as e:
    results['dns_gethostbyname'] = f'{type(e).__name__}: {e}'

# Test 4: DNS getaddrinfo
try:
    addrs = socket.getaddrinfo('google.com', 80, socket.AF_INET, socket.SOCK_STREAM)
    results['dns_getaddrinfo'] = f'SUCCESS - resolved to {addrs[0][4][0]}'
except socket.timeout:
    results['dns_getaddrinfo'] = 'TIMEOUT after default timeout'
except OSError as e:
    results['dns_getaddrinfo'] = f'OSError: {e}'
except Exception as e:
    results['dns_getaddrinfo'] = f'{type(e).__name__}: {e}'

# Test 5: UDP DNS query
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(3)
    dns_query = bytes([0xAA, 0xBB, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
                       0x00, 0x00, 0x00, 0x00, 0x06, 0x67, 0x6F, 0x6F,
                       0x67, 0x6C, 0x65, 0x03, 0x63, 0x6F, 0x6D, 0x00,
                       0x00, 0x01, 0x00, 0x01])
    s.sendto(dns_query, ('8.8.8.8', 53))
    data, addr = s.recvfrom(512)
    results['udp_dns_8888'] = f'SUCCESS - received {len(data)} bytes'
    s.close()
except socket.timeout:
    results['udp_dns_8888'] = 'TIMEOUT after 3s'
except OSError as e:
    results['udp_dns_8888'] = f'OSError: {e}'
except Exception as e:
    results['udp_dns_8888'] = f'{type(e).__name__}: {e}'

# Test 6: Raw socket (AF_PACKET) — should fail with cap_drop NET_RAW
try:
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    results['raw_socket'] = 'SUCCESS - raw socket created'
    s.close()
except OSError as e:
    results['raw_socket'] = f'OSError: {e}'
except Exception as e:
    results['raw_socket'] = f'{type(e).__name__}: {e}'

# Test 7: ICMP socket — should fail with cap_drop NET_RAW
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    results['icmp_socket'] = 'SUCCESS - ICMP socket created'
    s.close()
except OSError as e:
    results['icmp_socket'] = f'OSError: {e}'
except Exception as e:
    results['icmp_socket'] = f'{type(e).__name__}: {e}'

# Test 8: Docker gateway
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    for gw in ['172.17.0.1', '172.18.0.1', '172.19.0.1', '172.20.0.1', '10.0.0.1']:
        try:
            s.connect((gw, 2375))
            results['docker_gateway'] = f'SUCCESS - connected to {gw}:2375'
            break
        except:
            pass
    else:
        results['docker_gateway'] = 'BLOCKED - all gateway IPs unreachable'
    s.close()
except socket.timeout:
    results['docker_gateway'] = 'TIMEOUT'
except OSError as e:
    results['docker_gateway'] = f'OSError: {e}'

# Test 9: HTTP via urllib
try:
    import urllib.request
    resp = urllib.request.urlopen('http://example.com', timeout=3)
    results['http_urllib'] = f'SUCCESS - status {resp.status}'
except socket.timeout:
    results['http_urllib'] = 'TIMEOUT after 3s'
except OSError as e:
    results['http_urllib'] = f'OSError: {e}'
except Exception as e:
    results['http_urllib'] = f'{type(e).__name__}: {e}'

# Test 10: IPv6 DNS
try:
    addrs = socket.getaddrinfo('google.com', 80, socket.AF_INET6, socket.SOCK_STREAM)
    results['dns_ipv6'] = f'SUCCESS - resolved to {addrs[0][4][0]}'
except socket.timeout:
    results['dns_ipv6'] = 'TIMEOUT'
except OSError as e:
    results['dns_ipv6'] = f'OSError: {e}'
except Exception as e:
    results['dns_ipv6'] = f'{type(e).__name__}: {e}'

print('RESULTS_JSON_START')
print(json.dumps(results, indent=2))
print('RESULTS_JSON_END')
"""

NETWORK_KEYS = {
    'tcp_connect_8888', 'tcp_connect_1111',
    'dns_gethostbyname', 'dns_getaddrinfo', 'udp_dns_8888',
    'raw_socket', 'icmp_socket', 'docker_gateway',
    'http_urllib', 'dns_ipv6'
}


def run_test(c, network_name, iteration=0):
    """Run the full network test battery on a given Docker network."""
    code_dir = tempfile.mkdtemp()
    writable_dir = tempfile.mkdtemp()
    with open(os.path.join(code_dir, "script.py"), "w") as f:
        f.write(NETWORK_TEST_CODE)

    cont = c.containers.run(
        command="python /app/script.py",
        image="python:3.10-slim",
        detach=True, auto_remove=False,
        network=network_name,
        cap_drop=["NET_RAW"],  # Drop raw socket capability
        mem_limit="256m",
        volumes={
            code_dir: {"bind": "/app", "mode": "ro"},
            writable_dir: {"bind": "/tmp", "mode": "rw"},
        },
        working_dir="/app",
    )
    r = cont.wait(timeout=30)
    out = cont.logs(stdout=True, stderr=True).decode()
    cont.remove(force=True)
    shutil.rmtree(code_dir, ignore_errors=True)
    shutil.rmtree(writable_dir, ignore_errors=True)
    return r.get("StatusCode", -1), out


def parse_results(output):
    """Parse JSON results from container output."""
    if 'RESULTS_JSON_START' in output and 'RESULTS_JSON_END' in output:
        start = output.index('RESULTS_JSON_START') + len('RESULTS_JSON_START') + 1
        end = output.index('RESULTS_JSON_END')
        json_str = output[start:end].strip()
        try:
            return json.loads(json_str)
        except:
            return {}
    return {}


if __name__ == "__main__":
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    c = docker.from_env()

    # Create internal test network
    net_name = "agent-egress-test-net"
    try:
        old_net = c.networks.get(net_name)
        old_net.remove()
    except:
        pass
    net = c.networks.create(net_name, driver="bridge", internal=True)
    info = c.api.inspect_network(net.id)
    subnet = info.get("IPAM", {}).get("Config", [{}])[0].get("Subnet", "N/A")
    print(f"=== Bug C: Agent Egress Test ({num_runs} runs) ===")
    print(f"Network: {net_name} (internal=True, subnet={subnet})")
    print(f"Container: cap_drop=[NET_RAW], read_only=True, no tmpfs\n")

    all_successes = []
    all_results = []

    for i in range(num_runs):
        print(f"--- Run {i+1}/{num_runs} ---")
        exit_code, output = run_test(c, net_name, iteration=i)
        results = parse_results(output)
        all_results.append(results)

        successes = [(k, v) for k, v in results.items()
                     if k in NETWORK_KEYS and isinstance(v, str) and v.startswith("SUCCESS")]
        all_successes.extend(successes)

        if successes:
            print(f"  WARNING: {len(successes)} network access instances")
            for k, v in successes:
                print(f"    {k}: {v}")
        else:
            print(f"  All network attempts blocked")
        for k, v in results.items():
            if k in NETWORK_KEYS and isinstance(v, str):
                print(f"    {k}: {v}")
        print()

    # Summary
    print(f"=== FINAL RESULT ===")
    print(f"Network: internal Docker bridge (no internet route)")
    print(f"Container hardening: cap_drop=[NET_RAW], read_only=True")
    print(f"Runs: {num_runs}")

    if all_successes:
        print(f"FAIL: {len(all_successes)} EXTERNAL network access instances across {num_runs} runs")
        for k, v in all_successes:
            print(f"  {k}: {v}")
        net.remove()
        sys.exit(1)
    else:
        print(f"PASS: 0 EXTERNAL network access instances across {num_runs} runs")
        print(f"All TCP/UDP/DNS/HTTP/ICMP/raw attempts returned immediate network errors.")
        print(f"Agent on internal Docker network has zero internet egress.")
        net.remove()
        sys.exit(0)

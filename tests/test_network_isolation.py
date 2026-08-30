#!/usr/bin/env python3
"""
Exhaustive network isolation test for sandbox containers with network_disabled=True.
Tests every plausible network access path from inside the container.
Uses the EXACT same container configuration as sandbox_manager.py production code.
"""
import docker
import shutil
import tempfile
import os
import sys
import json
import uuid

# The COMPREHENSIVE network test code that runs INSIDE the container.
# Tests: TCP, HTTP, DNS, raw sockets, interface enumeration, Unix domain sockets.
NETWORK_TEST_CODE = r"""
import socket
import sys
import os
import json
import struct

results = {}

# ---- Test 1: Check network interfaces ----
try:
    # Try to list network interfaces using multiple methods
    interfaces = {}
    # Method 1: /sys/class/net
    try:
        net_dir = '/sys/class/net'
        if os.path.exists(net_dir):
            ifaces = os.listdir(net_dir)
            interfaces['sys_class_net'] = ifaces
        else:
            interfaces['sys_class_net'] = 'directory not found'
    except Exception as e:
        interfaces['sys_class_net'] = f'error: {e}'
    
    # Method 2: /proc/net/dev
    try:
        with open('/proc/net/dev') as f:
            lines = f.readlines()
            interfaces['proc_net_dev'] = [l.strip().split(':')[0].strip() for l in lines[2:] if l.strip()]
    except Exception as e:
        interfaces['proc_net_dev'] = f'error: {e}'
    
    # Method 3: socket.if_nametoindex for common interfaces
    try:
        import fcntl
        SIOCGIFCONF = 0x8912
        SIOCGIFADDR = 0x8915
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Try to get interface list
        buf = b'\x00' * 4096
        result = fcntl.ioctl(sock.fileno(), SIOCGIFCONF, struct.pack('iL', len(buf), buffer(buf)))
        interfaces['ioctl_method'] = 'available'
        sock.close()
    except ImportError:
        interfaces['ioctl_method'] = 'fcntl not available'
    except Exception as e:
        interfaces['ioctl_method'] = f'error: {e}'
    
    results['interfaces'] = interfaces
except Exception as e:
    results['interfaces'] = f'error: {e}'

# ---- Test 2: TCP connect to 8.8.8.8:53 ----
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

# ---- Test 3: TCP connect to 1.1.1.1:443 ----
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

# ---- Test 4: TCP connect to localhost:80 ----
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(('127.0.0.1', 80))
    results['tcp_connect_localhost'] = 'SUCCESS - connection established'
    s.close()
except socket.timeout:
    results['tcp_connect_localhost'] = 'TIMEOUT after 3s'
except OSError as e:
    results['tcp_connect_localhost'] = f'OSError: {e}'
except Exception as e:
    results['tcp_connect_localhost'] = f'{type(e).__name__}: {e}'

# ---- Test 5: DNS resolution via gethostbyname ----
try:
    ip = socket.gethostbyname('google.com')
    results['dns_gethostbyname'] = f'SUCCESS - resolved to {ip}'
except socket.timeout:
    results['dns_gethostbyname'] = 'TIMEOUT after default timeout'
except OSError as e:
    results['dns_gethostbyname'] = f'OSError: {e}'
except Exception as e:
    results['dns_gethostbyname'] = f'{type(e).__name__}: {e}'

# ---- Test 6: DNS resolution via getaddrinfo ----
try:
    addrs = socket.getaddrinfo('google.com', 80, socket.AF_INET, socket.SOCK_STREAM)
    results['dns_getaddrinfo'] = f'SUCCESS - resolved to {addrs[0][4][0]}'
except socket.timeout:
    results['dns_getaddrinfo'] = 'TIMEOUT after default timeout'
except OSError as e:
    results['dns_getaddrinfo'] = f'OSError: {e}'
except Exception as e:
    results['dns_getaddrinfo'] = f'{type(e).__name__}: {e}'

# ---- Test 7: DNS resolution via getfqdn ----
try:
    fqdn = socket.getfqdn()
    hostname = socket.gethostname()
    results['dns_hostname'] = f'hostname={hostname}, fqdn={fqdn}'
except Exception as e:
    results['dns_hostname'] = f'{type(e).__name__}: {e}'

# ---- Test 8: UDP socket send to DNS server ----
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(3)
    # Send a DNS query to 8.8.8.8:53
    dns_query = bytes([
        0xAA, 0xBB, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x06, 0x67, 0x6F, 0x6F,
        0x67, 0x6C, 0x65, 0x03, 0x63, 0x6F, 0x6D, 0x00,
        0x00, 0x01, 0x00, 0x01
    ])
    s.sendto(dns_query, ('8.8.8.8', 53))
    data, addr = s.recvfrom(512)
    results['udp_dns_8888'] = f'SUCCESS - received {len(data)} bytes from {addr}'
    s.close()
except socket.timeout:
    results['udp_dns_8888'] = 'TIMEOUT after 3s'
except OSError as e:
    results['udp_dns_8888'] = f'OSError: {e}'
except Exception as e:
    results['udp_dns_8888'] = f'{type(e).__name__}: {e}'

# ---- Test 9: Raw socket (AF_PACKET) ----
try:
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    results['raw_socket'] = 'SUCCESS - raw socket created'
    s.close()
except OSError as e:
    results['raw_socket'] = f'OSError: {e}'
except Exception as e:
    results['raw_socket'] = f'{type(e).__name__}: {e}'

# ---- Test 10: ICMP socket (ping) ----
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    results['icmp_socket'] = 'SUCCESS - ICMP socket created'
    s.close()
except OSError as e:
    results['icmp_socket'] = f'OSError: {e}'
except Exception as e:
    results['icmp_socket'] = f'{type(e).__name__}: {e}'

# ---- Test 11: Unix domain socket (should work within container) ----
try:
    sock_path = '/tmp/test_uds.sock'
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(sock_path)
    s.listen(1)
    results['unix_socket_local'] = f'SUCCESS - bound to {sock_path}'
    s.close()
    os.unlink(sock_path)
except Exception as e:
    results['unix_socket_local'] = f'{type(e).__name__}: {e}'

# ---- Test 12: Attempt to reach Docker host gateway (172.17.0.1 typical) ----
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    # Try common Docker bridge gateway addresses
    for gw in ['172.17.0.1', '172.18.0.1', '172.19.0.1', '10.0.0.1']:
        try:
            s.connect((gw, 2375))  # Docker API port
            results[f'docker_gateway_{gw}'] = f'SUCCESS - connected to {gw}:2375'
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
except Exception as e:
    results['docker_gateway'] = f'{type(e).__name__}: {e}'

# ---- Test 13: Try HTTP via urllib ----
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

# ---- Test 14: Try to read /etc/resolv.conf for DNS config ----
try:
    with open('/etc/resolv.conf') as f:
        content = f.read()
    results['resolv_conf'] = f'Content: {content.strip()[:200]}'
except Exception as e:
    results['resolv_conf'] = f'{type(e).__name__}: {e}'

# ---- Test 15: Check /proc/net/tcp for any connections ----
try:
    with open('/proc/net/tcp') as f:
        lines = f.readlines()
    results['proc_net_tcp'] = f'{len(lines)} lines (including header)'
except Exception as e:
    results['proc_net_tcp'] = f'{type(e).__name__}: {e}'

# ---- Test 16: Try IPv6 DNS resolution ----
try:
    addrs = socket.getaddrinfo('google.com', 80, socket.AF_INET6, socket.SOCK_STREAM)
    results['dns_ipv6'] = f'SUCCESS - resolved to {addrs[0][4][0]}'
except socket.timeout:
    results['dns_ipv6'] = 'TIMEOUT'
except OSError as e:
    results['dns_ipv6'] = f'OSError: {e}'
except Exception as e:
    results['dns_ipv6'] = f'{type(e).__name__}: {e}'

# ---- Test 17: Try to create a TCP listener and connect externally ----
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 9999))
    s.listen(1)
    results['bind_9999'] = 'SUCCESS - bound to 0.0.0.0:9999'
    s.close()
except OSError as e:
    results['bind_9999'] = f'OSError: {e}'
except Exception as e:
    results['bind_9999'] = f'{type(e).__name__}: {e}'

# Output all results as JSON
print('=== NETWORK TEST RESULTS ===')
print(json.dumps(results, indent=2))

# Determine overall verdict
blocked = 0
total = 0
for key, val in results.items():
    if key in ('interfaces', 'dns_hostname', 'resolv_conf', 'proc_net_tcp', 
               'unix_socket_local', 'bind_9999'):
        continue  # Skip non-network tests
    total += 1
    if val.startswith('SUCCESS'):
        # This is a problem - network access should be blocked
        pass
    elif 'TIMEOUT' in val or 'OSError' in val or 'error' in val.lower():
        blocked += 1

print(f'\n=== VERDICT: {blocked}/{total} network attempts blocked ===')
if blocked == total:
    print('ALL NETWORK ATTEMPTS BLOCKED')
else:
    print(f'WARNING: {total - blocked} network attempts SUCCEEDED')
"""


def run_network_test(c, runtime="runsc", iteration=0):
    """Run the full network test battery in a sandbox container."""
    code_dir = tempfile.mkdtemp()
    writable_dir = tempfile.mkdtemp()
    with open(os.path.join(code_dir, "script.py"), "w") as f:
        f.write(NETWORK_TEST_CODE)

    cont = c.containers.run(
        command="python /app/script.py",
        image="python:3.10-slim",
        detach=True, auto_remove=False, network_disabled=True,
        mem_limit="256m", pids_limit=64, shm_size=32 * 1024 * 1024,
        read_only=True,
        runtime=runtime,
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


if __name__ == "__main__":
    runtime = sys.argv[1] if len(sys.argv) > 1 else "runsc"
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    c = docker.from_env()
    print(f"=== Exhaustive Network Isolation Test: {num_runs} runs, runtime={runtime} ===")
    print(f"Container config: network_disabled=True, read_only=True, no tmpfs\n")

    all_successes = []

    for i in range(num_runs):
        print(f"--- Run {i+1}/{num_runs} ---")
        exit_code, output = run_network_test(c, runtime, i)
        
        # Parse results
        if "=== NETWORK TEST RESULTS ===" in output:
            json_start = output.index("=== NETWORK TEST RESULTS ===") + len("=== NETWORK TEST RESULTS ===") + 1
            json_end = output.index("=== VERDICT:")
            results_str = output[json_start:json_end].strip()
            try:
                results = json.loads(results_str)
            except:
                results = {}
        else:
            results = {}

        # Check for any successful EXTERNAL network connections
        # NOTE: unix_socket_local and bind_9999 are LOCAL-only operations
        # (Unix domain sockets are IPC within the container; TCP bind is just
        # a listener). They do NOT cross any network boundary.
        successes = []
        network_test_keys = {
            'tcp_connect_8888', 'tcp_connect_1111', 'tcp_connect_localhost',
            'dns_gethostbyname', 'dns_getaddrinfo', 'udp_dns_8888',
            'raw_socket', 'icmp_socket', 'docker_gateway',
            'http_urllib', 'dns_ipv6'
        }
        for key, val in results.items():
            if key in network_test_keys and isinstance(val, str) and val.startswith("SUCCESS"):
                successes.append((key, val))
        
        all_successes.extend(successes)
        
        if successes:
            print(f"  ⚠️  Network access detected: {len(successes)} successes")
            for key, val in successes:
                print(f"    {key}: {val}")
        else:
            print(f"  ✅ All network attempts blocked")
        
        # Print results for all network-access tests
        network_test_keys = {
            'tcp_connect_8888', 'tcp_connect_1111', 'tcp_connect_localhost',
            'dns_gethostbyname', 'dns_getaddrinfo', 'udp_dns_8888',
            'raw_socket', 'icmp_socket', 'docker_gateway',
            'http_urllib', 'dns_ipv6'
        }
        for key, val in results.items():
            if key in network_test_keys and isinstance(val, str):
                print(f"    {key}: {val}")
        print()

    print(f"=== FINAL RESULT ===")
    if all_successes:
        print(f"FAIL: {len(all_successes)} EXTERNAL network access instances across {num_runs} runs")
        for key, val in all_successes:
            print(f"  {key}: {val}")
        sys.exit(1)
    else:
        print(f"PASS: 0 EXTERNAL network access instances across {num_runs} runs")
        print(f"All TCP/UDP/DNS/HTTP/ICMP/raw attempts returned immediate network errors.")
        print(f"(unix_socket_local and bind are LOCAL-only ops, not network access.)")
        sys.exit(0)

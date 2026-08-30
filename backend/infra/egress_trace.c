/*
 * egress_trace.c - BPF program for tracing outbound TCP and UDP connections.
 *
 * Bug 4 Fix: Added udp_sendmsg kprobe for DNS exfiltration (UDP/53) coverage.
 *
 * Attached as kprobes on:
 *   - tcp_v4_connect()  — TCP IPv4
 *   - tcp_v6_connect()  — TCP IPv6
 *   - udp_sendmsg()     — UDP (covers DNS exfiltration)
 *
 * Usage with BCC:
 *   from bcc import BPF
 *   b = BPF(text=open("egress_trace.c").read())
 */

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <net/sock.h>

#define MAX_ENTRIES 1024

struct event_t {
    u32 pid;
    u32 uid;
    char comm[TASK_COMM_LEN];
    u32 daddr;       /* destination IPv4 address (network byte order) */
    u16 dport;       /* destination port (network byte order) */
    u8  proto;       /* 6=TCP, 17=UDP */
    u64 timestamp_ns;
};

BPF_HASH(inflight, u64, struct event_t, MAX_ENTRIES);
BPF_PERF_OUTPUT(events);

/* kprobe: tcp_v4_connect */
int trace_tcp_connect(struct pt_regs *ctx,
                      struct sock *sk,
                      struct sockaddr *uaddr)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 uid = bpf_get_current_uid_gid();

    struct sockaddr_in sin;
    bpf_probe_read_user(&sin, sizeof(sin), uaddr);

    struct event_t event = {};
    event.pid = pid;
    event.uid = uid;
    event.daddr = sin.sin_addr.s_addr;
    event.dport = sin.sin_port;
    event.proto = 6; /* TCP */
    event.timestamp_ns = bpf_ktime_get_ns();
    bpf_get_current_comm(&event.comm, sizeof(event.comm));

    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

/* kprobe: tcp_v6_connect */
int trace_tcp6_connect(struct pt_regs *ctx,
                       struct sock *sk,
                       struct sockaddr *uaddr)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 uid = bpf_get_current_uid_gid();

    struct sockaddr_in6 sin6;
    bpf_probe_read_user(&sin6, sizeof(sin6), uaddr);

    struct event_t event = {};
    event.pid = pid;
    event.uid = uid;
    bpf_probe_read_kernel(&event.daddr, sizeof(event.daddr), &sin6.sin6_addr);
    event.dport = sin6.sin6_port;
    event.proto = 6; /* TCP */
    event.timestamp_ns = bpf_ktime_get_ns();
    bpf_get_current_comm(&event.comm, sizeof(event.comm));

    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

/* kprobe: udp_sendmsg — covers DNS exfiltration and all UDP egress */
int trace_udp_sendmsg(struct pt_regs *ctx,
                      struct sock *sk,
                      struct msghdr *msg)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 uid = bpf_get_current_uid_gid();

    /* Extract destination from msg->msg_name (sockaddr_in for IPv4) */
    struct sockaddr *addr;
    bpf_probe_read_kernel(&addr, sizeof(addr), &msg->msg_name);

    if (!addr)
        return 0;

    /* Check address family */
    u16 family;
    bpf_probe_read_kernel(&family, sizeof(family), &addr->sa_family);

    struct event_t event = {};
    event.pid = pid;
    event.uid = uid;
    event.proto = 17; /* UDP */
    event.timestamp_ns = bpf_ktime_get_ns();
    bpf_get_current_comm(&event.comm, sizeof(event.comm));

    if (family == AF_INET) {
        struct sockaddr_in *sin4 = (struct sockaddr_in *)addr;
        bpf_probe_read_kernel(&event.daddr, sizeof(event.daddr), &sin4->sin_addr.s_addr);
        bpf_probe_read_kernel(&event.dport, sizeof(event.dport), &sin4->sin_port);
    } else if (family == AF_INET6) {
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)addr;
        bpf_probe_read_kernel(&event.daddr, sizeof(event.daddr), &sin6->sin6_addr);
        bpf_probe_read_kernel(&event.dport, sizeof(event.dport), &sin6->sin6_port);
    } else {
        return 0; /* Unknown family */
    }

    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

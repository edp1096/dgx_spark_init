"""Restore a dedicated two-node RoCE rail without changing existing addresses.
Kept identical in both standalone recipes; no dependency on a sibling checkout.
"""
import ipaddress
import json
import re
import shlex
import subprocess


def ensure_rail(worker, head_ip, worker_ip, head_if, worker_if, subnet):
    network = ipaddress.IPv4Network(subnet, strict=False)
    addresses = [ipaddress.IPv4Address(x) for x in (head_ip, worker_ip)]
    if addresses[0] == addresses[1] or any(x not in network or x in (network.network_address, network.broadcast_address) for x in addresses):
        raise ValueError("Distinct usable head/worker addresses in the rail subnet are required")
    if not worker or worker.startswith("-") or any(c.isspace() for c in worker):
        raise ValueError("Set the worker SSH target to its reachable management/LAN address")
    for iface in (head_if, worker_if):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", iface):
            raise ValueError("Automatic rail setup requires one exact interface per node")

    def run(rank, args):
        command = args if rank == 0 else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker, shlex.join(args)]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        return result.stdout

    # Inspect both sides first: a peer conflict must not mutate the local node.
    missing = []
    for rank, (iface, address) in enumerate(zip((head_if, worker_if), addresses)):
        links = json.loads(run(rank, ["ip", "-j", "address", "show", "dev", iface]))
        if len(links) != 1 or "LOWER_UP" not in links[0].get("flags", []):
            raise RuntimeError(f"Rank {rank}: {iface} has no active cable link")
        run(rank, ["test", "-d", f"/sys/class/net/{iface}/device/infiniband"])
        assigned = [x for x in links[0].get("addr_info", []) if x.get("family") == "inet"]
        if any(x["local"] != str(address) for x in assigned):
            raise RuntimeError(f"Rank {rank}: {iface} already has another IPv4 address; leaving it unchanged")
        if assigned:
            peer = addresses[1-rank]
            if any(peer not in ipaddress.IPv4Network(f"{x['local']}/{x['prefixlen']}", strict=False) for x in assigned):
                raise RuntimeError(f"Rank {rank}: existing prefix cannot reach the peer; leaving it unchanged")
        else:
            missing.append((rank, iface, str(address)))
    for rank, iface, address in missing:
        run(rank, ["docker", "run", "--rm", "--network", "host", "--cap-drop", "ALL", "--cap-add", "NET_ADMIN", "alpine:3.22", "ip", "address", "add", f"{address}/{network.prefixlen}", "dev", iface])
    for rank, iface, peer in ((0, head_if, worker_ip), (1, worker_if, head_ip)):
        run(rank, ["ping", "-I", iface, "-c", "2", "-W", "2", peer])
    print(f"RoCE rail ready: {head_ip} <-> {worker_ip}; restored {len(missing)} address(es)", flush=True)


if __name__ == "__main__":
    import os
    head = os.environ.get("VLLM_HOST_IP") or os.environ.get("MASTER_ADDR", "")
    if os.environ.get("MASTER_ADDR", head) != head:
        raise ValueError("MASTER_ADDR must match VLLM_HOST_IP for this two-node rail")
    ensure_rail(os.environ.get("WORKER_HOST", ""), head,
                os.environ.get("WORKER_VLLM_HOST_IP", ""),
                os.environ.get("NCCL_SOCKET_IFNAME", "").removeprefix("="),
                (os.environ.get("WORKER_NCCL_SOCKET_IFNAME") or os.environ.get("NCCL_SOCKET_IFNAME", "")).removeprefix("="),
                os.environ.get("NCCL_SUBNET") or str(ipaddress.IPv4Network(f"{head}/{os.environ.get('DSPARK_RAIL_PREFIX', '24')}", strict=False)))

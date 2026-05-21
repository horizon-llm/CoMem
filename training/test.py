import os, socket, subprocess, ray
ray.init(address="auto")

@ray.remote
def whoami():
    env = {k: os.environ.get(k) for k in [
        "NCCL_SOCKET_IFNAME","GLOO_SOCKET_IFNAME","NCCL_IB_DISABLE",
        "FI_PROVIDER","NCCL_IGNORE_IFNAME","NCCL_DEBUG","TORCH_NCCL_ASYNC_ERROR_HANDLING"
    ]}
    # show interfaces visible in the worker
    ip_out = subprocess.run(
        ["bash","-lc","ip -o -4 addr | awk '{print $2,$4}'"],
        capture_output=True, text=True
    ).stdout.strip()
    return {
        "node": socket.gethostname(),
        "env": env,
        "ifaces": ip_out
    }

# fan out a few workers so we likely hit both nodes
results = ray.get([whoami.remote() for _ in range(8)])
# de-dup by node
seen = {}
for r in results:
    seen.setdefault(r["node"], r)
for node, r in seen.items():
    print("==== NODE:", node, "====")
    print("ENV:", r["env"])
    print("IFACES:\n", r["ifaces"])
    print()
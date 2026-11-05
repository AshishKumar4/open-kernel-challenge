"""H200 SXM offers on Vast for booking a GPU box, the machines that already hold our image first (receipts/vast/
image_machines.json, written by onboard.py: on a fresh machine the 56 GB image takes ~17 min to pull, Japan 11-05; on one
that holds it, seconds), then by price.
    python3 harness/h200/vast_offers.py [on-demand | bid] [GPUs per box, default 2]"""
import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
V = "vastai"   # the vast.ai CLI, logged in
CACHED = ROOT / "receipts/vast/image_machines.json"   # {machine_id: {"t", "image", "pull_s"}}


def cached():
    return json.loads(CACHED.read_text()) if CACHED.exists() else {}


def offers(kind="on-demand", n=2):
    r = subprocess.run([V, "search", "offers", f"gpu_name=H200 num_gpus={n}", "--type", kind, "--raw"],
                       capture_output=True, text=True, timeout=120)
    have = cached()
    rows = [o for o in json.loads(r.stdout) if o.get("gpu_name") == "H200"]   # (not "H200 NVL")
    price = (lambda o: o.get("min_bid") or 1e9) if kind == "bid" else (lambda o: o.get("dph_total") or 1e9)
    return sorted(rows, key=lambda o: (str(o.get("machine_id")) not in have, price(o)))


if __name__ == "__main__":
    kind, n = (sys.argv[1:2] or ["on-demand"])[0], int((sys.argv[2:3] or [2])[0])
    have = cached()
    for o in offers(kind, n)[:15]:
        print(f"{o['id']:>10} machine {o.get('machine_id'):>7} {'image held' if str(o.get('machine_id')) in have else '          '} "
              f"{o.get('geolocation', '')[:16]:16s} rel {o.get('reliability2', 0):.3f} "
              f"${(o.get('min_bid') if kind == 'bid' else o.get('dph_total')) or 0:.2f}/h")

#!/usr/bin/env -S uv run python
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import yaml
import zmq
from matplotlib import colors

NROW, NCOL = 6, 3          # DG-5F-S: 18 taxels, row 1 distal -> row 6 proximal
DEFAULT_PREFIXES = ["", "hand.", "hand1."]


def load_taxel_spec(yml_path, sensor_name):
    with Path(yml_path).open() as f:
        cfg = yaml.safe_load(f)

    specs = cfg.get("tactiles", [])
    spec = next((s for s in specs if s["name"] == sensor_name), None)
    if spec is None:
        raise SystemExit(f"sensor {sensor_name!r} not found in {yml_path}")

    bodies = {b["name"]: b for b in cfg["bodies"]}
    body = bodies[spec["body"]]
    shape = body["shapes"][0]
    if shape["type"] != "capsule":
        raise SystemExit(f"expected {spec['body']} shape to be capsule")

    samples = np.asarray(spec["samples"], dtype=float)
    radius, hh = map(float, shape["param"])
    center = np.asarray(shape.get("pos", [0.0, 0.0, 0.0]), dtype=float)
    # `pos` is the capsule midpoint on a capsule rooted at the joint origin, so
    # it also gives the link axis. The thumb runs along y, the fingers along x.
    axis = center / np.linalg.norm(center)
    return spec, samples, center, axis, radius, hh


def connect(names):
    ctx = zmq.Context.instance()
    poller = zmq.Poller()
    socks = {}
    for name in names:
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.connect(f"ipc:///dev/shm/{name}")
        poller.register(s, zmq.POLLIN)
        socks[s] = name
    return poller, socks


def capsule_wire(ax, center, axis, radius, hh):
    """Wireframe capsule of half-length hh about `axis`, centred on `center`."""
    # Orthonormal frame with e0 along the link axis.
    e0 = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(e0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e0, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    theta = np.linspace(0.0, 2.0 * np.pi, 97)

    def draw(pts, color):
        pts = np.asarray(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=0.8)

    for a in np.linspace(-hh, hh, 4):                       # rings along the barrel
        draw([center + e0 * a + radius * (np.cos(t) * e1 + np.sin(t) * e2)
              for t in theta], "0.75")
    for t in [0.0, np.pi / 2.0, np.pi, 1.5 * np.pi]:        # barrel generators
        r = radius * (np.cos(t) * e1 + np.sin(t) * e2)
        draw([center - e0 * hh + r, center + e0 * hh + r], "0.75")

    for side in (-1.0, 1.0):                                # end caps
        cap = center + e0 * (side * hh)
        for phi in [np.pi / 6.0, np.pi / 3.0]:
            rr = radius * np.cos(phi)
            draw([cap + e0 * (side * radius * np.sin(phi))
                  + rr * (np.cos(t) * e1 + np.sin(t) * e2) for t in theta], "0.72")
        for t in [0.0, np.pi / 2.0, np.pi, 1.5 * np.pi]:
            r = np.cos(t) * e1 + np.sin(t) * e2
            draw([cap + e0 * (side * radius * np.sin(p)) + radius * np.cos(p) * r
                  for p in np.linspace(0.0, np.pi / 2.0, 40)], "0.72")


def main():
    ap = argparse.ArgumentParser(description="Live matplotlib view for one KIDA DG5F-S fingertip tactile pad.")
    ap.add_argument("--names", nargs="+", default=None,
                    help="tactile IPC names to subscribe; default is --prefix x --finger")
    ap.add_argument("--prefix", nargs="+", default=DEFAULT_PREFIXES,
                    help='socket name prefixes, e.g. hand1. hand2.')
    ap.add_argument("--finger", default="index",
                    choices=["thumb", "index", "middle", "ring", "little"],
                    help="which fingertip pad to plot")
    ap.add_argument("--yml", default=str(Path(__file__).resolve().parent.parent / "yaml" / "dg5f-s-left.yaml"),
                    help="YAML file containing the unprefixed tactile geometry")
    ap.add_argument("--max", dest="vmax", type=float, default=0.0,
                    help="fixed color max; 0 means auto-scale")
    ap.add_argument("--hz", type=float, default=20.0, help="plot refresh rate")
    ap.add_argument("--timeout", type=float, default=2.0, help="seconds before showing stale title")
    args = ap.parse_args()
    sensor = f"{args.finger}_fingertip_taxel"
    if args.names is None:
        args.names = [p + sensor for p in args.prefix]

    spec, samples, center, axis, radius, hh = load_taxel_spec(args.yml, sensor)
    pos = samples[:, 0:3]
    normal = samples[:, 3:6]
    ntaxel = pos.shape[0]
    if ntaxel != NROW * NCOL:
        raise SystemExit(f"expected {NROW * NCOL} taxels, got {ntaxel}")

    poller, socks = connect(args.names)
    data = {name: np.zeros(ntaxel, dtype=np.float32) for name in args.names}
    stamp = {name: 0.0 for name in args.names}
    active_name = args.names[0]

    cmap = plt.get_cmap("inferno")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    fig.canvas.manager.set_window_title("KIDA DG5F-S tactile plot viewer")

    capsule_wire(ax, center, axis, radius, hh)
    scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=np.zeros(ntaxel),
                      cmap=cmap, norm=norm, s=60, edgecolor="black", linewidth=0.3)
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
              normal[:, 0], normal[:, 1], normal[:, 2],
              length=0.004, color="navy", normalize=True)
    ax.set_title(f"{spec['body']} local frame")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    span = np.abs(axis) * 4.0 + (1.0 - np.abs(axis)) * 2.0
    ax.set_box_aspect(tuple(span))
    ax.view_init(elev=22, azim=-55)

    # Payload order is the DG-5F-S datasheet numbering (row-major, 3 per row,
    # row 1 distal -> row 6 proximal), so it reshapes straight into the layout.
    img = ax2.imshow(np.zeros((NROW, NCOL)), cmap=cmap, norm=norm,
                     origin="upper", aspect="equal")
    ax2.set_title("taxel layout (4.8 x 9.6 mm)")
    ax2.set_xlabel("columns: taxel 1, 2, 3")
    ax2.set_ylabel("rows: distal -> proximal")
    ax2.set_xticks(range(NCOL))
    ax2.set_yticks(range(NROW))
    ax2.set_xticklabels(["1", "2", "3"])
    ax2.set_yticklabels([f"{1 + NCOL * i}" for i in range(NROW)])
    for i in range(NROW):
        for j in range(NCOL):
            ax2.text(j, i, "", ha="center", va="center", color="white", fontsize=8)
    texts = ax2.texts
    cb = fig.colorbar(img, ax=[ax, ax2], shrink=0.78, pad=0.04)
    cb.set_label("normal force")

    period = 1.0 / args.hz if args.hz > 0 else 0.05

    def update(_frame):
        nonlocal active_name
        events = dict(poller.poll(0))
        now = time.time()
        for s in events:
            name = socks[s]
            arr = np.frombuffer(s.recv(), dtype="<f4")
            if arr.size == ntaxel:
                data[name] = arr.copy()
                stamp[name] = now
                active_name = name

        value = data[active_name]
        frame_max = float(value.max())
        vmax = args.vmax if args.vmax > 0 else max(frame_max, 1e-6)
        norm.vmax = vmax

        scat.set_array(value)
        layout = value.reshape(NROW, NCOL)
        img.set_data(layout)
        for text, v in zip(texts, layout.reshape(-1)):
            text.set_text(f"{v:.1f}")
            text.set_color("black" if v > 0.7 * vmax else "white")

        age = now - stamp[active_name] if stamp[active_name] else None
        state = "waiting" if age is None else ("stale" if age > args.timeout else "live")
        fig.suptitle(
            f"{active_name}  {state}  sum={value.sum():.3f}  max={frame_max:.3f}  color max={vmax:.3f}",
            fontsize=11,
        )
        return [scat, img, *texts]

    timer = fig.canvas.new_timer(interval=max(1, int(period * 1000)))
    timer.add_callback(lambda: (update(None), fig.canvas.draw_idle(), True)[-1])
    timer.start()
    update(None)
    plt.show()


if __name__ == "__main__":
    main()

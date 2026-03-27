"""
compare_bev_fusion.py — Side-by-side BEV: geometry-only vs geometry+semantic fusion.
P1-M4.6: Visualise the effect of SegFormer semantic labels on traversability scores.

Usage:
    python scripts/compare_bev_fusion.py                          # uses example frame
    python scripts/compare_bev_fusion.py <image.jpg> <cloud.bin>  # custom frame

Pipeline:
    1. Run pipeline_runner (C++) on LiDAR scan  -> geometry grid (risk, confidence …)
    2. Run SegFormer (Python) on camera image    -> ADE20K label map
    3. Project LiDAR points onto image           -> per-point semantic label
    4. Aggregate labels per BEV cell (modal)     -> per-cell ADE20K modifier
    5. Fused risk = geometry_risk × modifier
    6. Side-by-side BEV: geometry-only | fused | modifier map | camera+segmentation
"""

import sys
import os
import subprocess
import tempfile

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_BIN = os.path.join(REPO_ROOT,
    "build/construction_perception/pipeline_runner")
CALIB_YAML   = os.path.join(REPO_ROOT, "config/camera_lidar_calib.yaml")
EXAMPLE_IMG  = os.path.join(REPO_ROOT,
    "third_party/RELLIS-3D/utils/example/frame000104-1581624663_149.jpg")
EXAMPLE_BIN  = os.path.join(REPO_ROOT,
    "third_party/RELLIS-3D/utils/example/000104.bin")
OUT_DIR      = os.path.join(REPO_ROOT, "docs/assets")

# ---------------------------------------------------------------------------
# Grid parameters — must match pipeline_runner.cpp exactly
# ---------------------------------------------------------------------------
X_MIN, X_MAX = -30.0,  5.0   # metres forward (camera faces -X in LiDAR frame)
Y_MIN, Y_MAX = -15.0, 15.0   # metres lateral
RESOLUTION   = 0.5            # metres per cell
ROWS = int((X_MAX - X_MIN) / RESOLUTION)   # 70
COLS = int((Y_MAX - Y_MIN) / RESOLUTION)   # 60
N_FIELDS = 6                               # risk, conf, slope, rough, step, mean_z

# ---------------------------------------------------------------------------
# ADE20K class → traversability modifier
# Unknown classes default to 1.0 (no penalty — conservative, avoid false positives)
# ---------------------------------------------------------------------------
ADE20K_MODIFIERS = {
    2:  1.0,   # sky        — no LiDAR hits, irrelevant
    4:  0.0,   # tree       — lethal (trunk collision)
    9:  1.0,   # grass      — nominal
    12: 0.0,   # person     — dynamic obstacle, stop
    13: 0.9,   # earth      — dirt path, slight penalty
    17: 0.8,   # plant/bush — light vegetation
    21: 0.1,   # water      — high hazard
    29: 1.0,   # field      — open terrain
    32: 0.0,   # fence      — hard obstacle
    60: 1.0,   # road       — paved, safe
}
DEFAULT_MODIFIER = 1.0

# ---------------------------------------------------------------------------
# Step 1 — Run C++ pipeline_runner → geometry grid binary
# ---------------------------------------------------------------------------
def run_pipeline(bin_path: str) -> np.ndarray:
    """Returns (ROWS, COLS, N_FIELDS) float32 array."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        grid_path = f.name

    try:
        result = subprocess.run(
            [PIPELINE_BIN, bin_path, grid_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pipeline_runner failed:\n{result.stderr}"
            )
        print(result.stdout.strip())

        raw = np.fromfile(grid_path, dtype=np.float32)
        expected = ROWS * COLS * N_FIELDS
        if len(raw) != expected:
            raise RuntimeError(
                f"Grid size mismatch: expected {expected} floats, got {len(raw)}"
            )
        return raw.reshape(ROWS, COLS, N_FIELDS)

    finally:
        if os.path.exists(grid_path):
            os.unlink(grid_path)

# ---------------------------------------------------------------------------
# Step 2 — SegFormer inference
# ---------------------------------------------------------------------------
def run_segformer(image_path: str) -> np.ndarray:
    """Returns (H, W) uint8 ADE20K label array."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
    try:
        from segmentation_node import run_segformer as _run
        return _run(image_path)
    except ImportError as e:
        print(f"[WARN] Could not import segmentation_node: {e}")
        print("[WARN] Returning empty label map — fusion will use default modifier")
        img = np.array(Image.open(image_path))
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

# ---------------------------------------------------------------------------
# Step 3 — Load calibration
# ---------------------------------------------------------------------------
def load_calib(yaml_path: str):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    intr = cfg["camera_intrinsics"]
    K = np.array([
        [intr["fx"],        0, intr["cx"]],
        [       0,  intr["fy"], intr["cy"]],
        [       0,          0,          1],
    ], dtype=np.float64)
    T = np.array(cfg["extrinsic_T_cam_lidar"], dtype=np.float64)
    return K, T, intr["width"], intr["height"]

# ---------------------------------------------------------------------------
# Step 4 — Project LiDAR points → per-point pixel + cell indices
# ---------------------------------------------------------------------------
def project_and_assign(pts_lidar: np.ndarray, K, T_cam_lidar,
                        img_w: int, img_h: int):
    """
    Returns:
        pixels  : (M, 2) int   — (u, v) for each visible point
        cell_ix : (M,)   int   — BEV row index
        cell_iy : (M,)   int   — BEV col index
        valid   : (M,)   bool  — mask into original pts_lidar
    """
    N = len(pts_lidar)
    ones = np.ones((N, 1))
    pts_h = np.hstack([pts_lidar.astype(np.float64), ones])   # (N, 4)
    P_cam = (T_cam_lidar @ pts_h.T).T                          # (N, 4)

    # depth check
    depth_ok = P_cam[:, 2] > 0

    # perspective projection
    p = (K @ P_cam[:, :3].T).T                                 # (N, 3)
    u = np.where(depth_ok, p[:, 0] / np.where(depth_ok, p[:, 2], 1), -1)
    v = np.where(depth_ok, p[:, 1] / np.where(depth_ok, p[:, 2], 1), -1)

    # bounds check
    in_bounds = (depth_ok &
                 (u >= 0) & (u < img_w) &
                 (v >= 0) & (v < img_h))

    # BEV cell assignment for ALL points (not just visible ones)
    x = pts_lidar[:, 0]
    y = pts_lidar[:, 1]
    cell_ix = np.floor((x - X_MIN) / RESOLUTION).astype(int)
    cell_iy = np.floor((y - Y_MIN) / RESOLUTION).astype(int)
    in_grid = ((cell_ix >= 0) & (cell_ix < ROWS) &
               (cell_iy >= 0) & (cell_iy < COLS))

    valid = in_bounds & in_grid

    pixels  = np.stack([u[valid].astype(int), v[valid].astype(int)], axis=1)
    return pixels, cell_ix[valid], cell_iy[valid], valid

# ---------------------------------------------------------------------------
# Step 5 — Build per-cell modifier map from semantic labels
# ---------------------------------------------------------------------------
def build_modifier_map(pixels, cell_ix, cell_iy, label_map) -> np.ndarray:
    """Returns (ROWS, COLS) float32 modifier array."""
    modifier_grid = np.full((ROWS, COLS), DEFAULT_MODIFIER, dtype=np.float32)

    # accumulate labels per cell
    cell_labels = {}
    for (u, v), ix, iy in zip(pixels, cell_ix, cell_iy):
        label = int(label_map[v, u])
        key = (ix, iy)
        if key not in cell_labels:
            cell_labels[key] = []
        cell_labels[key].append(label)

    # modal label → modifier
    for (ix, iy), labels in cell_labels.items():
        counts = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        modal_label = max(counts, key=counts.get)
        modifier_grid[ix, iy] = ADE20K_MODIFIERS.get(modal_label, DEFAULT_MODIFIER)

    return modifier_grid

# ---------------------------------------------------------------------------
# Step 6 — Visualise
# ---------------------------------------------------------------------------
def plot_comparison(geometry_grid, fused_grid, modifier_grid,
                    camera_image, label_map, out_path: str):

    risk_geo  = geometry_grid[:, :, 0]   # (ROWS, COLS)
    conf      = geometry_grid[:, :, 1]

    # mask unknown cells (confidence == 0)
    unknown = conf == 0.0

    # BEV: rotate so forward (x) is up in the plot
    def bev(arr):
        return np.flipud(arr.T)   # (COLS, ROWS) with forward=up

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle("P1-M4: Geometry-only vs Semantic Fusion BEV", fontsize=14)

    kw = dict(origin="upper", vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")

    # risk legend shared by panels 1 & 2
    risk_legend = [
        mpatches.Patch(color=plt.cm.RdYlGn_r(1.0), label="High risk"),
        mpatches.Patch(color=plt.cm.RdYlGn_r(0.5), label="Medium risk"),
        mpatches.Patch(color=plt.cm.RdYlGn_r(0.0), label="Low risk / safe"),
    ]

    # --- Panel 1: geometry-only risk ---
    geo_display = bev(np.where(unknown, np.nan, risk_geo))
    im0 = axes[0].imshow(geo_display, **kw)
    axes[0].set_title("Geometry-only risk\n(slope + roughness + step)")
    axes[0].set_xlabel("Lateral →"); axes[0].set_ylabel("← Forward")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].legend(handles=risk_legend, loc="lower right", fontsize=7)

    # --- Panel 2: fused risk ---
    fused_risk   = fused_grid[:, :, 0]
    fused_display = bev(np.where(unknown, np.nan, fused_risk))
    im1 = axes[1].imshow(fused_display, **kw)
    axes[1].set_title("Fused risk\n(geometry × semantic modifier)")
    axes[1].set_xlabel("Lateral →"); axes[1].set_ylabel("← Forward")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].legend(handles=risk_legend, loc="lower right", fontsize=7)

    # --- Panel 3: modifier map — discrete 4-band colormap for clear tier separation ---
    # Tiers: lethal(0.0) | water(0.1) | earth/plant(0.8-0.9) | safe(1.0)
    _mod_colors = ["#d73027", "#fc8d59", "#fee08b", "#1a9850"]   # red, orange, yellow, green
    _mod_cmap   = ListedColormap(_mod_colors)
    _mod_norm   = BoundaryNorm([0.0, 0.05, 0.5, 0.95, 1.01], _mod_cmap.N)
    mod_display = bev(np.where(unknown, np.nan, modifier_grid))
    im2 = axes[2].imshow(mod_display, origin="upper", aspect="auto",
                          cmap=_mod_cmap, norm=_mod_norm)
    axes[2].set_title("Semantic modifier\n(1.0=safe, 0.0=lethal)")
    axes[2].set_xlabel("Lateral →"); axes[2].set_ylabel("← Forward")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, ticks=[0.025, 0.275, 0.725, 0.98],
                 format=plt.FuncFormatter(lambda v, _: {0.025:"0.0×", 0.275:"0.1×",
                                                         0.725:"0.8-0.9×", 0.98:"1.0×"}.get(round(v,3), "")))

    mod_legend = [
        mpatches.Patch(color="#d73027", label="tree/fence/person (0.0×)"),
        mpatches.Patch(color="#fc8d59", label="water (0.1×)"),
        mpatches.Patch(color="#fee08b", label="earth/plant (0.8-0.9×)"),
        mpatches.Patch(color="#1a9850", label="grass/field (1.0×)"),
    ]
    axes[2].legend(handles=mod_legend, loc="lower right", fontsize=7)

    # --- Panel 4: camera + SegFormer overlay ---
    axes[3].imshow(np.array(camera_image))
    axes[3].imshow(label_map, cmap="tab20", alpha=0.45)
    axes[3].set_title("Camera + SegFormer ADE20K labels")
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(out_path[:out_path.rfind("/")], exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else EXAMPLE_IMG
    bin_path   = sys.argv[2] if len(sys.argv) > 2 else EXAMPLE_BIN

    print(f"Image  : {image_path}")
    print(f"LiDAR  : {bin_path}")

    # 1. Geometry grid from C++ pipeline
    print("\n[1/5] Running C++ pipeline_runner...")
    geometry_grid = run_pipeline(bin_path)
    print(f"       Grid shape: {geometry_grid.shape}  "
          f"(risk range [{geometry_grid[:,:,0].min():.2f}, "
          f"{geometry_grid[:,:,0].max():.2f}])")

    # 2. SegFormer labels
    print("\n[2/5] Running SegFormer...")
    label_map = run_segformer(image_path)
    unique_labels = sorted(set(label_map.flatten().tolist()))
    print(f"       Label map: {label_map.shape}  unique IDs: {unique_labels}")

    # 3. Load calibration + LiDAR points
    print("\n[3/5] Loading calibration and LiDAR points...")
    K, T_cam_lidar, img_w, img_h = load_calib(CALIB_YAML)
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    valid_mask = ~((pts[:, 0] == 0) & (pts[:, 1] == 0) & (pts[:, 2] == 0))
    pts = pts[valid_mask, :3]
    print(f"       LiDAR points: {len(pts):,}")

    # 4. Project → per-point cell + label
    print("\n[4/5] Projecting LiDAR → semantic labels per cell...")
    pixels, cell_ix, cell_iy, _ = project_and_assign(
        pts, K, T_cam_lidar, img_w, img_h)
    print(f"       Visible+in-grid points: {len(pixels):,}")

    modifier_grid = build_modifier_map(pixels, cell_ix, cell_iy, label_map)

    # 5. Fused grid: multiply risk by modifier
    fused_grid = geometry_grid.copy()
    fused_grid[:, :, 0] = geometry_grid[:, :, 0] * modifier_grid

    # report visible difference
    risk_geo   = geometry_grid[:, :, 0]
    risk_fused = fused_grid[:, :, 0]
    diff = np.abs(risk_fused - risk_geo)
    cells_changed = np.sum(diff > 0.01)
    print(f"\n       Cells with >1% risk change: {cells_changed} / {ROWS*COLS}")
    print(f"       Max risk reduction: {(risk_geo - risk_fused).max():.3f}")

    # 6. Visualise
    print("\n[5/5] Generating comparison plot...")
    camera_image = Image.open(image_path).convert("RGB")
    out_path = os.path.join(OUT_DIR, "bev_fusion_comparison.png")
    plot_comparison(geometry_grid, fused_grid, modifier_grid,
                    camera_image, label_map, out_path)


if __name__ == "__main__":
    main()

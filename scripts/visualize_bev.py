"""
visualize_bev.py — BEV traversability grid visualization.
P1-M3.6: The "wow" image for your README and demo.

Usage:
    python scripts/visualize_bev.py <traversability_grid.npy>

YOUR TASK:
  1. Load traversability grid (scores + confidence as numpy arrays)
  2. Color map:
     - Green: score > 0.7 and confidence > 0.5 (safe, confident)
     - Yellow: 0.3 < score < 0.7 and confidence > 0.5 (caution)
     - Red: score < 0.3 and confidence > 0.5 (hazard, confident)
     - Gray: confidence < 0.5 (uncertain / unknown)
  3. Overlay robot position marker at origin
  4. Save as PNG
"""

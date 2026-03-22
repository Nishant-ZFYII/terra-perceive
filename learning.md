---
# Terra Perceive — Learning Log

This file tracks what I learned, what broke, and what I now understand.
Written AFTER implementing each milestone, not before.
Anki cards are created from the "What broke" sections — not from reading.

---

## M1 — Environment + Data (Days 1-2)

### What I understood before starting
[Fill in after M1 complete]

### What I implemented
[Fill in after M1 complete]

### What broke and why
[Fill in — the most important section]

### What I now understand
[Fill in after M1 complete]

### Interview answer (whiteboard-ready)
[Fill in — one paragraph]

### Anki cards
Q:
A:

---

## M2 — Sector RANSAC (Days 3-4)

### What I understood before starting
[Fill in after M2 complete]

### What I implemented
[Fill in]

### What broke and why
[The global RANSAC failure case goes here — with the image reference]

### What I now understand
[Fill in]

### Interview answer
[e.g. "Global RANSAC fits a single plane to all points. On graded terrain, a slope that's driveable gets classified as an obstacle because it's 'above' the global fit plane. Sector RANSAC solves this by fitting an independent local plane per sector, so each sector evaluates traversability relative to its own local ground."]

### Anki cards
Q: What is the RANSAC iteration count formula and what does each variable mean?
A: N = log(1-p) / log(1-(1-e)^s). p = desired success probability (0.99), e = outlier ratio (estimated), s = sample size (3 for plane). For 99% confidence, 20% outliers: N≈17.

Q: Why does global RANSAC fail on graded terrain?
A: [Fill in from your debugging session]

Q: What does SVD refinement add to RANSAC?
A: [Fill in]

---

## M3 — Traversability Grid (Days 5-6)

### Anki cards
Q: What does the smallest PCA eigenvalue represent geometrically?
A: Variance in the direction of least spread. For a flat surface, points spread in x-y but barely in z, so the smallest eigenvalue eigenvector = surface normal.

Q: Why is the eigenvalue ratio a good roughness metric?
A: smallest_eigenvalue / sum_eigenvalues measures how "planar" the cell is. A perfectly flat cell has λ_min ≈ 0, ratio ≈ 0. A rough surface has significant variance in all directions, ratio approaches 1/3.

Q: Why is unknown traversability confidence=0 and NOT risk=0.5?
A: risk=0.5 implies "probably safe" — dangerous assumption with no data. confidence=0 signals "no measurement" — downstream planner can distinguish from measured-safe cells.

---

## M4 — Cam-LiDAR Projection (Days 7-8)

### Anki cards
Q: Write the LiDAR-to-pixel projection pipeline.
A: (1) P_cam = T_cam_lidar * [x,y,z,1]^T (2) depth check: P_cam.z > 0 (3) p = K * P_cam[:3] (4) u=p[0]/p[2], v=p[1]/p[2] (5) bounds check: 0<=u<W, 0<=v<H

---

## M5 — Safety Supervisor (Days 9-10)

### Anki cards
Q: What is a forward-arc lookahead for kinematic safety?
A: [Fill in from your implementation]

---

(Continue pattern for M6, M7)
---

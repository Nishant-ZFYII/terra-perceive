---
layout: post
title: "M2: Sector-Based RANSAC Ground Segmentation"
date: 2026-03-25
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Sector-Based RANSAC Ground Segmentation

*Part of the Terra Perceive series.*

Off-road terrain has slopes. A naive single-plane ground model fails — it classifies driveable slopes as obstacles. This milestone builds a sector-based RANSAC pipeline that handles graded terrain by fitting independent ground planes per spatial region.

---

## The Foundational Theory: Fischler & Bolles (1981)

The Random Sample Consensus (RANSAC) algorithm, introduced by Martin A. Fischler and Robert C. Bolles in 1981, remains the gold standard for robust model fitting in robotics. Before RANSAC, most systems relied on Least Squares, which is mathematically elegant but fails catastrophically in the presence of even a single "gross error" (outlier).

### The "Hypothesize-and-Verify" Paradigm

RANSAC's genius lies in its simplicity. Instead of trying to smooth out errors using all data points, it adopts a minimalist approach:

1. **Hypothesize**: Randomly select the *smallest possible* set of points (3 for a plane) to instantiate a model.
2. **Verify**: Count how many points in the entire cloud "consent" to this model within a noise tolerance $$\epsilon$$.
3. **Refine**: Once the "Largest Consensus Set" is found, re-fit the model using only the verified inliers.

### Mathematical Derivation of Iterations

How many times must we sample to be sure we've found the "true" ground? Let:
- $$w$$ = probability a point is an inlier (e.g., 0.8 for flat ground)
- $$n$$ = points per sample (3 for a plane)
- $$P$$ = desired success probability (e.g., 0.99)

The probability of choosing $$n$$ inliers in a single trial is $$w^n$$. The probability of failure across $$N$$ trials is $$(1 - w^n)^N$$. Setting this to $$1 - P$$ and solving for $$N$$:

$$N = \frac{\log(1 - P)}{\log(1 - w^n)}$$

For a typical RELLIS-3D frame with 20% outliers ($$w = 0.8$$), we only need **17 iterations** to achieve 99% confidence. This efficiency is what allows our C++ pipeline to run at 20Hz+.

### Noise vs. Outlier Gap

Fischler & Bolles distinguish between two types of errors:
- **Noise**: Local measurement jitter. Handled by the inlier threshold $$\epsilon$$.
- **Outliers**: Gross errors (e.g., a tree or a truck). Handled by the consensus count.

In the construction domain, this distinction is critical. A threshold $$\epsilon$$ too small rejects valid bumpy ground; too large, and the bottom of a tractor counts as "ground." Tuning this threshold is the primary challenge.

---

## Why Global RANSAC Fails on Slopes

A single RANSAC plane fit assumes the entire scene is locally planar. On flat ground, this works well. On a construction site with a 15-degree grade, the best single plane splits the difference — tilted at ~7.5 degrees. Both the flat portion and the sloped portion appear partially above the plane, and large sections of driveable terrain get misclassified as obstacles.

The root cause: a single plane cannot simultaneously represent two regions with different orientations. The algorithm is not broken — the model is wrong.

---

## The Sector-Based Fix

The solution is to divide the x-y plane into a grid of sectors and fit an independent RANSAC plane to each one. Each sector makes its own local planarity assumption, which holds even on graded terrain.

**Grid construction**: given cloud bounds $$[x_{min}, x_{max}]$$ and sector size $$s_x$$:

$$n_x = \left\lceil \frac{x_{max} - x_{min}}{s_x} \right\rceil$$

Each point is assigned to sector $$(i_x, i_y)$$ by:

$$i_x = \left\lfloor \frac{p_x - x_{min}}{s_x} \right\rfloor$$

**Continuity check**: adjacent sectors should agree on normal direction. If two neighboring sectors differ by more than 30 degrees, the boundary sector is flagged unreliable — this catches noisy sectors at the edges of the point cloud.

$$\theta = \arccos\left(\hat{n}_A \cdot \hat{n}_B\right)$$

If $$\theta > \theta_{thresh}$$, mark the sector unreliable.

---

## SVD Refinement

The 3-point RANSAC plane is noisy — it only uses 3 points to define a plane that may have thousands of inliers. SVD refinement re-fits a better plane using the full inlier set.

Given the set of inlier points $$\{p_i\}$$, compute the centroid $$\bar{p}$$ and center the points:

$$M = \begin{bmatrix} p_1 - \bar{p} & p_2 - \bar{p} & \cdots & p_N - \bar{p} \end{bmatrix}$$

The covariance matrix $$C = \frac{1}{N} M M^T$$ is 3×3. Its eigenvector corresponding to the **smallest eigenvalue** is the direction of minimum variance — perpendicular to the plane surface — i.e., the refined normal.

This is computed via `Eigen::SelfAdjointEigenSolver`, which returns eigenvalues in ascending order. Column 0 of the eigenvector matrix is the refined normal.

One subtlety: eigenvectors have no guaranteed sign. After refinement, we check `refined_normal.dot(ransac_normal) < 0` and flip if needed to maintain consistency.

---

## Implementation Notes

Key decisions made during implementation:

- **Default `min_inliers = 100`**: sectors with fewer points are marked unreliable before running RANSAC — prevents degenerate fits on sparse edge sectors.
- **`best_inlier_count < min_inliers` guard**: if no valid plane is found (e.g., all-collinear input), return an empty result rather than classify everything as ground with a zero normal.
- **Re-classify after SVD**: the inlier set changes slightly after refinement. Ground/obstacle points are re-partitioned using the refined plane to keep classification consistent.
- **Sector size 5m × 5m**: balances point density vs. local planarity assumption. At 30m range, a 5m sector still captures enough points for reliable RANSAC.

---

## Test Results

All 8 unit tests pass against synthetic point clouds with known geometry:

| Test | What it verifies |
|------|-----------------|
| `SyntheticFlatPlane` | Normal ≈ (0,0,1), inlier_ratio > 0.7, obstacles above ground |
| `TiltedPlane30Deg` | Normal within 5° of expected tilted normal |
| `SVDRefinement` | Refined normal within 3° of PCA ground truth |
| `DegenerateInput` | Empty cloud and collinear points return empty result, no crash |
| `MultiSlopeTerrain` | Flat and 15° sectors both detected as reliable |
| `ContinuityCheckPass` | 10° slope difference → both sectors reliable |
| `ContinuityCheckFail` | 60° slope difference → at least one sector unreliable |
| `EmptySector` | Sparse sectors marked unreliable, no crash |

---

## What I'd Do Differently

- **Adaptive sector size**: near the sensor, point density is high; at 50m range, a 5m sector may have only a handful of points. A range-adaptive sector size would improve reliability at distance.
- **Normal interpolation for unreliable sectors**: instead of discarding unreliable sectors entirely, interpolate from reliable neighbors.
- **Early termination per sector**: the current implementation uses a fixed `max_iterations`. Dynamic early termination using the RANSAC iteration formula $$N = \log(1-P) / \log(1-(1-e)^s)$$ — updated each iteration with the current estimated inlier ratio — would cut runtime significantly on easy sectors.

**Next Step**: [Milestone 3: Physics-Grounded Traversability Grid](m3-traversability)

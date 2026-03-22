---
# M2: Sector-Based RANSAC Ground Segmentation

*Part of the Terra Perceive series.*

## The problem

Off-road terrain has slopes. A naive single-plane ground model fails — it classifies driveable slopes as obstacles.

## The math

### RANSAC (Fischler & Bolles, 1981)

Given N points with unknown outlier ratio e, fitting a plane requires s=3 points.
Iteration count for probability p of finding a good model:

$$N = \frac{\log(1-p)}{\log(1-(1-e)^s)}$$

For p=0.99, e=0.20: N=17 iterations.

Plane equation: $\mathbf{n} \cdot \mathbf{x} + d = 0$ where $\mathbf{n} = (p_2 - p_1) \times (p_3 - p_1)$

Inlier condition: $|\mathbf{n} \cdot \mathbf{p} + d| < \epsilon$

### SVD refinement

[Fill in after M2]

### Why sector-based

[Fill in after M2 — with failure image]

## Global RANSAC failure

[Fill in — image goes here, explanation of why it fails]

## Sector-based fix

[Fill in]

## Results

[Fill in — side-by-side comparison images]

## What I'd do differently

[Fill in]
---

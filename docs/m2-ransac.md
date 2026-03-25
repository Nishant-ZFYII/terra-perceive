---
# M2: Sector-Based RANSAC Ground Segmentation

*Part of the Terra Perceive series.*

## The problem

Off-road terrain has slopes. A naive single-plane ground model fails — it classifies driveable slopes as obstacles.

## The math

## The Foundational Theory: Fischler & Bolles (1981)

The Random Sample Consensus (RANSAC) algorithm, introduced by Martin A. Fischler and Robert C. Bolles in 1981, remains the gold standard for robust model fitting in robotics. Before RANSAC, most systems relied on Least Squares, which is mathematically elegant but fails catastrophically in the presence of even a single "gross error" (outlier).

### The "Hypothesize-and-Verify" Paradigm

RANSAC's genius lies in its simplicity. Instead of trying to smooth out errors using all data points, it adopts a minimalist approach:
1.  **Hypothesize**: Randomly select the *smallest possible* set of points (3 for a plane) to instantiate a model.
2.  **Verify**: Count how many points in the entire cloud "consent" to this model within a noise tolerance $\epsilon$.
3.  **Refine**: Once the "Largest Consensus Set" is found, re-fit the model using only the verified inliers.

### Mathematical Derivation of Iterations

How many times must we sample to be sure we've found the "true" ground? The paper provides a probabilistic bound. Let:
- $w$ = Probability a point is an inlier (e.g., 0.8 for flat ground)
- $n$ = Points per sample (3 for a plane)
- $P$ = Desired success probability (e.g., 0.99)

The probability of choosing $n$ inliers in a single trial is $w^n$. The probability of failure across $N$ trials is $(1 - w^n)^N$. Setting this to $1-P$ and solving for $N$:

$$N = \frac{\log(1 - P)}{\log(1 - w^n)}$$

For a typical RELLIS-3D frame with 20% outliers ($w=0.8$), we only need **17 iterations** to achieve 99% confidence. This efficiency is what allows our C++ pipeline to run at 20Hz+.

### Handling Unstructured Terrain: The Noise vs. Outlier Gap

Fischler & Bolles distinguish between two types of errors:
- **Noise**: Local measurement jitter. Handled by the **Inlier Threshold** ($\epsilon$).
- **Outliers**: Gross errors (e.g., a tree or a truck). Handled by the **Consensus Count**.

In the construction domain, this distinction is critical. A "small" threshold $\epsilon$ may reject valid bumpy ground, while a "large" threshold might include the bottom of a tractor as "ground." Tuning this threshold is the primary challenge of Phase 1.

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

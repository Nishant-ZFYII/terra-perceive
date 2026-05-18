/**
 * Single source of truth for milestone metadata.
 *
 * Adding a new milestone post is a TWO-STEP operation:
 *   1. Drop the new markdown into `docs/m<N>-<slug>.md` (shadow rebuild
 *      handles it on the next `npm run dev` / `npm run build`).
 *   2. Add an entry to MILESTONES below with `slug`, `id`, `title`, `blurb`,
 *      `phase`, and `status`. The header dropdown, prev/next cards, and
 *      landing-page phase tables all derive from this.
 *
 * Notes:
 *   - `m5-pipeline` is M11 (tracker-safety perception loop). Filename
 *     predates the M11 numbering — kept as-is to avoid breaking links.
 *   - `m2-slam` and `m3-nats-bootstrap` are drafts/supplementary and
 *     deliberately excluded. They render at their URLs but don't appear in
 *     the nav or prev/next chain.
 */

export type Phase =
  | 'p1-core'
  | 'p2-odom-slam'
  | 'p2-mapping-tracking'
  | 'p2-refinement'
  | 'p2-cross-domain';

export type Status = 'shipped' | 'in-progress' | 'planned';

export interface Milestone {
  /** URL slug = markdown filename without the .md extension. */
  slug: string;
  /** Display ID like "M7" or "M15+". */
  id: string;
  /** Short post title (used in header dropdown and prev/next cards). */
  title: string;
  /** One-sentence summary for the landing-page row. */
  blurb: string;
  /** Which phase grouping this milestone belongs to. */
  phase: Phase;
  /** Shipping status — drives the colored dot on the landing page. */
  status: Status;
}

/**
 * The full milestone catalogue. Order matters: this is the linear sequence
 * used for prev/next navigation AND the row order in each phase table.
 */
export const MILESTONES: Milestone[] = [
  // ───── Phase 1 — Core perception & safety ─────
  { slug: 'm1-data', id: 'M1', title: 'Data ingestion',
    blurb: 'O(N) binary loader for RELLIS-3D and Open3D visualization',
    phase: 'p1-core', status: 'shipped' },
  { slug: 'm2-ransac', id: 'M2', title: 'Sector RANSAC',
    blurb: 'Ground segmentation for sloped and graded terrain',
    phase: 'p1-core', status: 'shipped' },
  { slug: 'm3-traversability', id: 'M3', title: 'Traversability grid',
    blurb: 'Risk / confidence maps using PCA surface normals',
    phase: 'p1-core', status: 'shipped' },
  { slug: 'm4-fusion', id: 'M4', title: 'Camera-LiDAR fusion',
    blurb: 'Homogeneous transforms and SegFormer semantic segmentation',
    phase: 'p1-core', status: 'shipped' },
  { slug: 'm5-safety', id: 'M5', title: 'Kinematic safety',
    blurb: 'Stopping distance, TTC, terrain-aware friction, priority interventions',
    phase: 'p1-core', status: 'shipped' },
  { slug: 'm6-docker', id: 'M6', title: 'Integration',
    blurb: 'Docker image, smoke test, end-to-end pipeline',
    phase: 'p1-core', status: 'shipped' },

  // ───── Phase 2 — Odometry & SLAM ─────
  { slug: 'm7-odometry', id: 'M7', title: 'Triple odometry',
    blurb: 'GPS/IMU, KISS-ICP, Cartographer benchmark with ATE/RPE comparison',
    phase: 'p2-odom-slam', status: 'shipped' },
  { slug: 'm8-slam', id: 'M8', title: 'LiDAR-inertial SLAM',
    blurb: 'From-scratch pose graph, IMU preintegration, Scan Context, manifold vs Euclidean ablation',
    phase: 'p2-odom-slam', status: 'shipped' },

  // ───── Phase 2 — Mapping & tracking ─────
  { slug: 'm9-bev-map', id: 'M9', title: 'Accumulated BEV map',
    blurb: 'World map from multi-source odometry, NATS transport',
    phase: 'p2-mapping-tracking', status: 'shipped' },
  { slug: 'm10-sort-tracker', id: 'M10', title: 'SORT tracker',
    blurb: 'Kalman + Hungarian + DBSCAN; IMM + Deep SORT cascade + Mahalanobis gate',
    phase: 'p2-mapping-tracking', status: 'shipped' },
  { slug: 'm5-pipeline', id: 'M11', title: 'Tracker-safety loop',
    blurb: 'YOLO + cam-LiDAR + SORT + safety supervisor + NATS + JetStream audit',
    phase: 'p2-mapping-tracking', status: 'shipped' },

  // ───── Phase 2 — Perception & safety refinement ─────
  { slug: 'm12-probabilistic-traversability', id: 'M12', title: 'Probabilistic traversability',
    blurb: 'Range-dependent LiDAR noise σ(r) propagated through per-cell PCA. Matches analytic prediction within 2%.',
    phase: 'p2-refinement', status: 'shipped' },
  { slug: 'm13-cbf-safety', id: 'M13', title: 'CBF safety',
    blurb: '1D scalar Control Barrier Function clamp; 6-scenario ablation vs the kinematic TTC step rule.',
    phase: 'p2-refinement', status: 'shipped' },

  // ───── Phase 2 — Cross-domain evaluation ─────
  // Add M14 / M15+ here as they ship. No other files need to change.
  { slug: 'm14-nuscenes', id: 'M14', title: 'nuScenes',
    blurb: 'Unified calibration adapter, second domain validation',
    phase: 'p2-cross-domain', status: 'in-progress' },
  { slug: 'm15-final-ship', id: 'M15+', title: 'MOTA eval, ROS2 live, final ship',
    blurb: '3D viz, ROS2 live pipeline, demo',
    phase: 'p2-cross-domain', status: 'planned' },
];

/** Headings for each phase grouping, in display order on the landing page. */
export const PHASE_HEADINGS: Record<Phase, string> = {
  'p1-core': 'Phase 1 — Core perception & safety',
  'p2-odom-slam': 'Phase 2 — Odometry & SLAM',
  'p2-mapping-tracking': 'Phase 2 — Mapping & tracking',
  'p2-refinement': 'Phase 2 — Perception & safety refinement',
  'p2-cross-domain': 'Phase 2 — Cross-domain evaluation',
};

// ─────────────────────────────────────────────────────────────────────────
// Derived constants. Components consume these — do not hand-maintain them.
// ─────────────────────────────────────────────────────────────────────────

/** Slugs of milestones that have a published post (excludes planned/in-progress). */
const PUBLISHED_SLUGS = MILESTONES.filter((m) => m.status === 'shipped').map((m) => m.slug);

/** Canonical sequence used for prev/next navigation (shipped posts only). */
export const POST_ORDER = PUBLISHED_SLUGS as readonly string[];

/** Slug → {id, title} lookup for prev/next cards and the header dropdown. */
export const POST_LABELS: Record<string, { id: string; title: string }> =
  Object.fromEntries(MILESTONES.map((m) => [m.slug, { id: m.id, title: m.title }]));

/** Group milestones by phase, preserving entry order. */
export function milestonesByPhase(): Array<{ phase: Phase; heading: string; rows: Milestone[] }> {
  const groups = new Map<Phase, Milestone[]>();
  for (const m of MILESTONES) {
    if (!groups.has(m.phase)) groups.set(m.phase, []);
    groups.get(m.phase)!.push(m);
  }
  return Array.from(groups.entries()).map(([phase, rows]) => ({
    phase,
    heading: PHASE_HEADINGS[phase],
    rows,
  }));
}

export function getNeighbors(slug: string): { prev: string | null; next: string | null } {
  const idx = POST_ORDER.indexOf(slug);
  if (idx === -1) return { prev: null, next: null };
  return {
    prev: idx > 0 ? POST_ORDER[idx - 1] : null,
    next: idx < POST_ORDER.length - 1 ? POST_ORDER[idx + 1] : null,
  };
}

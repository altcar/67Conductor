import type { Product, Vertical } from "../data/catalog";

export type Verdict = "like" | "nope" | "super";

export interface Taste {
  /** Learned affinity per tag. Negative means "keep this out of my feed". */
  weights: Record<string, number>;
  likes: string[];
  nopes: string[];
  supers: string[];
  /** Tags seeded from the onboarding chips. */
  seeds: string[];
}

export interface Decision {
  id: string;
  action: Verdict;
  at: number;
}

export interface Pick {
  product: Product;
  score: number;
  why: string;
}

const ACTION_WEIGHT: Record<Verdict, number> = { like: 1, super: 2.4, nope: -1.1 };

export function createTaste(seeds: string[] = []): Taste {
  const weights: Record<string, number> = {};
  for (const tag of seeds) weights[tag] = (weights[tag] ?? 0) + 1.6;
  return { weights, likes: [], nopes: [], supers: [], seeds: [...seeds] };
}

export function learn(taste: Taste, product: Product, action: Verdict): Taste {
  const next: Taste = {
    weights: { ...taste.weights },
    likes: [...taste.likes],
    nopes: [...taste.nopes],
    supers: [...taste.supers],
    seeds: [...taste.seeds],
  };

  const amount = ACTION_WEIGHT[action];
  for (const tag of product.tags) {
    const current = next.weights[tag] ?? 0;
    // Saturate so one obsessive category cannot drown everything else.
    next.weights[tag] = clampWeight(current + amount);
  }
  if (action !== "nope") {
    // A gentle category-level nudge makes recommendations converge faster.
    const categoryTag = `cat:${product.category}`;
    next.weights[categoryTag] = clampWeight((next.weights[categoryTag] ?? 0) + amount * 0.5);
  }

  next.likes = next.likes.filter((id) => id !== product.id);
  next.nopes = next.nopes.filter((id) => id !== product.id);
  next.supers = next.supers.filter((id) => id !== product.id);

  if (action === "like") next.likes.push(product.id);
  if (action === "nope") next.nopes.push(product.id);
  if (action === "super") next.supers.push(product.id);

  return next;
}

function clampWeight(value: number): number {
  return Math.max(-6, Math.min(6, value));
}

export function affinity(product: Product, taste: Taste): number {
  if (product.tags.length === 0) return 0;
  let dot = 0;
  let magnitude = 0;
  for (const tag of product.tags) {
    const weight = taste.weights[tag] ?? 0;
    dot += weight;
    magnitude += Math.abs(weight);
  }
  const category = taste.weights[`cat:${product.category}`] ?? 0;
  dot += category;
  magnitude += Math.abs(category);
  return magnitude === 0 ? 0 : dot / Math.sqrt(magnitude);
}

export interface RecommendOptions {
  vertical?: Vertical;
  exclude?: Set<string>;
  limit?: number;
  /** Products already seen get pushed down but are not banned. */
  seen?: Set<string>;
}

export function recommend(
  products: Product[],
  taste: Taste,
  options: RecommendOptions = {},
): Pick[] {
  const { vertical, exclude = new Set<string>(), limit = 3, seen = new Set<string>() } = options;
  const ranked = products
    .filter((product) => !exclude.has(product.id))
    .filter((product) => (vertical ? product.vertical === vertical : true))
    .map((product) => {
      const base = affinity(product, taste);
      const novelty = seen.has(product.id) ? 0.35 : 0;
      return { product, score: base - novelty };
    })
    .sort((a, b) => b.score - a.score || a.product.id.localeCompare(b.product.id));

  // Diversity pass: avoid three near-identical tags in a row.
  const picks: Pick[] = [];
  for (const entry of ranked) {
    if (picks.length >= limit) break;
    const overlap = picks.some(
      (pick) =>
        pick.product.category === entry.product.category &&
        pick.product.tags.filter((tag) => entry.product.tags.includes(tag)).length >= 3,
    );
    if (overlap && ranked.length > limit * 2) continue;
    picks.push({
      product: entry.product,
      score: Number(entry.score.toFixed(3)),
      why: explain(entry.product, taste),
    });
  }

  while (picks.length < limit && ranked.length > picks.length) {
    const fallback = ranked.find((entry) => !picks.some((pick) => pick.product.id === entry.product.id));
    if (!fallback) break;
    picks.push({
      product: fallback.product,
      score: Number(fallback.score.toFixed(3)),
      why: explain(fallback.product, taste),
    });
  }

  return picks;
}

export function explain(product: Product, taste: Taste): string {
  const scored = product.tags
    .map((tag) => ({ tag, weight: taste.weights[tag] ?? 0 }))
    .filter((entry) => entry.weight > 0.2)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 2);

  if (scored.length === 0) {
    return "Fresh territory — a wildcard from the rail.";
  }
  if (scored.length === 1) {
    return `Because you keep leaning toward ${scored[0].tag}.`;
  }
  return `Because you keep leaning toward ${scored[0].tag} and ${scored[1].tag}.`;
}

export function tasteSummary(taste: Taste, count = 3): string[] {
  return Object.entries(taste.weights)
    .filter(([tag, weight]) => !tag.includes(":") && weight > 0.5)
    .sort((a, b) => b[1] - a[1])
    .slice(0, count)
    .map(([tag]) => tag);
}

/** Blend the onboarding chips into an existing profile (idempotent-ish). */
export function applySeeds(taste: Taste, seeds: string[]): Taste {
  const next: Taste = {
    ...taste,
    weights: { ...taste.weights },
    seeds: [...new Set([...taste.seeds, ...seeds])],
  };
  for (const seed of seeds) {
    next.weights[seed] = clampWeight((next.weights[seed] ?? 0) + 1.2);
  }
  return next;
}

/**
 * Deck order: mostly unseen, with an injected recommendation every few cards.
 * Deterministic given the same inputs, which keeps the UI stable across renders.
 */
export function buildDeck(
  products: Product[],
  taste: Taste,
  decided: Set<string>,
  seen: Set<string>,
  size = 20,
): Product[] {
  const pool = products.filter((product) => product.vertical === "apparel");
  let unseen = pool.filter((product) => !decided.has(product.id) && !seen.has(product.id));
  if (unseen.length < 6) {
    unseen = pool.filter((product) => !decided.has(product.id));
  }
  if (unseen.length === 0) unseen = pool;

  const sorted = [...unseen].sort((a, b) => affinity(b, taste) - affinity(a, taste));
  const picks = recommend(pool, taste, {
    vertical: "apparel",
    exclude: decided,
    seen,
    limit: 6,
  }).map((pick) => pick.product);

  const deck: Product[] = [];
  const queue = [...sorted];
  const pickQueue = picks.filter((product) => !decided.has(product.id));
  while (deck.length < size && queue.length > 0) {
    if (deck.length % 4 === 3 && pickQueue.length > 0) {
      const candidate = pickQueue.shift()!;
      if (!deck.some((product) => product.id === candidate.id) && !decided.has(candidate.id)) {
        deck.push(candidate);
        continue;
      }
    }
    const next = queue.shift()!;
    if (!deck.some((product) => product.id === next.id)) deck.push(next);
  }
  return deck;
}

import { describe, expect, it } from "vitest";
import type { Product } from "../src/data/catalog";
import {
  affinity,
  applySeeds,
  buildDeck,
  createTaste,
  explain,
  learn,
  recommend,
  tasteSummary,
} from "../src/lib/recommender";

function product(id: string, tags: string[], category = "tee"): Product {
  return {
    id,
    name: `Product ${id}`,
    price: 40,
    vertical: "apparel",
    category,
    tags,
    palette: ["#d7ff3e", "#2f3a12"],
    blurb: "test",
  };
}

const knits = product("knit", ["cotton", "oversized", "everyday"]);
const leather = product("leather", ["leather", "tailored", "black"], "shoes");
const sport = product("sport", ["nylon", "sporty", "everyday"], "shorts");
const pool = [knits, leather, sport];

describe("taste learning", () => {
  it("moves affinity toward liked tags", () => {
    const taste = createTaste();
    const before = affinity(knits, taste);
    const after = learn(taste, knits, "like");
    expect(affinity(knits, after)).toBeGreaterThan(before);
    expect(after.likes).toContain("knit");
  });

  it("weights a 67 more heavily than a like", () => {
    const liked = learn(createTaste(), knits, "like");
    const supered = learn(createTaste(), knits, "super");
    expect(affinity(knits, supered)).toBeGreaterThan(affinity(knits, liked));
    expect(supered.supers).toContain("knit");
  });

  it("pushes a noped product down without banning its tags everywhere", () => {
    let taste = learn(createTaste(), leather, "nope");
    expect(taste.weights.leather).toBeLessThan(0);
    expect(affinity(leather, taste)).toBeLessThan(0);
    // A different product sharing only the everyday tag is barely affected.
    expect(affinity(sport, taste)).toBeGreaterThanOrEqual(0);
  });

  it("saturates instead of running away", () => {
    let taste = createTaste();
    for (let i = 0; i < 20; i += 1) taste = learn(taste, knits, "super");
    for (const value of Object.values(taste.weights)) expect(Math.abs(value)).toBeLessThanOrEqual(6);
  });
});

describe("recommendations", () => {
  it("ranks the matching product first and explains why", () => {
    let taste = createTaste();
    taste = learn(taste, knits, "super");
    const picks = recommend(pool, taste, { limit: 3 });
    expect(picks[0].product.id).toBe("knit");
    expect(picks[0].why).toMatch(/cotton|oversized|everyday/);
  });

  it("excludes decided products", () => {
    const taste = learn(createTaste(), knits, "like");
    const picks = recommend(pool, taste, { limit: 3, exclude: new Set(["knit"]) });
    expect(picks.map((pick) => pick.product.id)).not.toContain("knit");
  });

  it("keeps categories diverse when there is plenty to choose from", () => {
    const many = [
      ...Array.from({ length: 6 }, (_, i) => product(`tee${i}`, ["cotton", "oversized", "everyday"], "tee")),
      product("shoe1", ["leather", "everyday"], "shoes"),
    ];
    const taste = createTaste(["cotton", "oversized"]);
    const picks = recommend(many, taste, { limit: 3 });
    expect(new Set(picks.map((pick) => pick.product.category)).size).toBeGreaterThan(1);
  });

  it("applies onboarding seeds", () => {
    const taste = applySeeds(createTaste(), ["cotton", "oversized"]);
    expect(tasteSummary(taste)).toContain("oversized");
    expect(affinity(knits, taste)).toBeGreaterThan(0);
  });

  it("explains a wildcard", () => {
    expect(explain(sport, createTaste())).toMatch(/wildcard/i);
  });

  it("builds a deck without duplicates", () => {
    const deck = buildDeck(pool, createTaste(), new Set(), new Set(), 3);
    expect(deck.length).toBeGreaterThan(0);
    expect(new Set(deck.map((item) => item.id)).size).toBe(deck.length);
  });
});

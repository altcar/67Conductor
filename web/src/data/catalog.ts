import photos from "./photos.json";

export type Vertical = "apparel" | "music" | "fitness";
export type ArtKind =
  | "guitar"
  | "keys"
  | "pedal"
  | "headphones"
  | "amp"
  | "mic"
  | "dumbbell"
  | "kettlebell"
  | "band"
  | "mat"
  | "rack"
  | "timer";

export interface Product {
  id: string;
  name: string;
  price: number;
  vertical: Vertical;
  category: string;
  tags: string[];
  image?: string;
  art?: ArtKind;
  palette: [string, string];
  blurb: string;
  condition?: string;
  size?: string;
  oneOfOne?: boolean;
}

type PhotoEntry = { id: string; file: string };
type PhotoGroup = { label: string; entries: PhotoEntry[] };

const PALETTES: [string, string][] = [
  ["#d7ff3e", "#2f3a12"],
  ["#ff6a3d", "#3a1710"],
  ["#8bb4ff", "#111a2e"],
  ["#f3e7c9", "#3a3225"],
  ["#b8a1ff", "#1d1630"],
  ["#6ff0c0", "#0d2a22"],
];

const COLORS = [
  { name: "Ecru", hex: "#e7dfcf", tag: "ecru" },
  { name: "Bone", hex: "#f3efe6", tag: "bone" },
  { name: "Ink", hex: "#14151a", tag: "black" },
  { name: "Olive", hex: "#6b6f3f", tag: "olive" },
  { name: "Rust", hex: "#a9502f", tag: "rust" },
  { name: "Cobalt", hex: "#3e5a9e", tag: "blue" },
  { name: "Slate", hex: "#5b6470", tag: "grey" },
  { name: "Moss", hex: "#46543b", tag: "green" },
  { name: "Burgundy", hex: "#6e2b34", tag: "red" },
  { name: "Sand", hex: "#c9b18e", tag: "sand" },
  { name: "Chambray", hex: "#7c93ae", tag: "blue" },
  { name: "Charcoal", hex: "#34363c", tag: "black" },
];

const STYLES = [
  { name: "Minimal", tag: "minimal" },
  { name: "Utility", tag: "utility" },
  { name: "Retro", tag: "retro" },
  { name: "Street", tag: "streetwear" },
  { name: "Preppy", tag: "preppy" },
  { name: "Sport", tag: "sporty" },
  { name: "Studio", tag: "studio" },
  { name: "After Dark", tag: "after dark" },
];

const CONDITIONS = ["Excellent", "Good", "Very good", "Like new", "Worn in"];
const SIZES = ["XS", "S", "M", "L", "XL"];

interface CategorySpec {
  dir: string;
  noun: string;
  cuts: string[];
  baseTags: string[];
}

const APPAREL: CategorySpec[] = [
  { dir: "t-shirt", noun: "Tee", cuts: ["Boxy", "Oversized", "Relaxed", "Cropped"], baseTags: ["cotton", "jersey", "everyday"] },
  { dir: "longsleeve", noun: "Long Sleeve", cuts: ["Ribbed", "Relaxed", "Longline", "Slim"], baseTags: ["cotton", "layer", "everyday"] },
  { dir: "shirt", noun: "Shirt", cuts: ["Camp Collar", "Oversized", "Tailored", "Short Sleeve"], baseTags: ["poplin", "button-up", "smart"] },
  { dir: "outwear", noun: "Jacket", cuts: ["Utility", "Quilted", "Chore", "Cropped"], baseTags: ["layer", "utility", "outerwear"] },
  { dir: "dress", noun: "Dress", cuts: ["Slip", "Tiered", "Wrap", "Column"], baseTags: ["occasion", "statement", "flowy"] },
  { dir: "skirt", noun: "Skirt", cuts: ["Pleated", "Midi", "Denim", "Bias Cut"], baseTags: ["midi", "layer", "occasion"] },
  { dir: "pants", noun: "Trousers", cuts: ["Wide Leg", "Straight Leg", "Pleated", "Cropped"], baseTags: ["trouser", "everyday", "denim"] },
  { dir: "shorts", noun: "Shorts", cuts: ["Baggy", "Tailored", "Sport", "Denim"], baseTags: ["summer", "casual", "everyday"] },
  { dir: "shoes", noun: "Shoes", cuts: ["Derby", "Chunky", "Low Top", "Loafer"], baseTags: ["leather", "grounding", "everyday"] },
  { dir: "hat", noun: "Hat", cuts: ["Bucket", "Beanie", "Cap", "Sun"], baseTags: ["accessory", "finish", "unisex"] },
];

function hash(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(list: T[], random: () => number): T {
  return list[Math.floor(random() * list.length) % list.length];
}

function money(value: number): number {
  return Math.max(9, Math.round(value / 5) * 5 - 1);
}

function buildApparel(): Product[] {
  const groups = photos as Record<string, PhotoGroup>;
  const items: Product[] = [];
  const usedNames = new Set<string>();

  APPAREL.forEach((spec, categoryIndex) => {
    const group = groups[spec.dir];
    if (!group) return;
    for (const entry of group.entries) {
      const random = mulberry32(hash(entry.id));
      const color = pick(COLORS, random);
      const style = pick(STYLES, random);
      const cut = pick(spec.cuts, random);
      const condition = pick(CONDITIONS, random);
      const size = pick(SIZES, random);
      const basePrice = 16 + random() * 74;

      let name = `${cut} ${color.name} ${spec.noun}`;
      if (usedNames.has(name)) name = `${name} No. ${categoryIndex + 2}`;
      usedNames.add(name);

      const tags = [
        ...spec.baseTags,
        color.tag,
        style.tag,
        cut.split(" ").pop()!.toLowerCase(),
      ];
      const unique = [...new Set(tags)];

      items.push({
        id: entry.id,
        name,
        price: money(basePrice),
        vertical: "apparel",
        category: spec.dir,
        tags: unique,
        image: entry.file,
        palette: PALETTES[hash(entry.id) % PALETTES.length],
        blurb: `${condition} condition, one previous owner. ${style.name} ${spec.noun.toLowerCase()} in ${color.name.toLowerCase()}.`,
        condition,
        size,
        oneOfOne: true,
      });
    }
  });

  return items;
}

const MUSIC: Omit<Product, "vertical">[] = [
  {
    id: "mu-offset",
    name: "67 Offset Guitar",
    price: 289,
    category: "guitar",
    tags: ["analog", "warm", "stage", "guitar", "mid"],
    art: "guitar",
    palette: ["#d7ff3e", "#26300d"],
    blurb: "Short-scale offset with a wide, forgiving neck. Sits in a mix without a fight.",
  },
  {
    id: "mu-pocket-synth",
    name: "Pocket Synth MK67",
    price: 179,
    category: "synth",
    tags: ["digital", "compact", "studio", "synth", "beginner"],
    art: "keys",
    palette: ["#8bb4ff", "#101a2e"],
    blurb: "Battery-powered six-voice synth. The whole idea is that you actually carry it.",
  },
  {
    id: "mu-tape-echo",
    name: "Tape Echo Pedal",
    price: 129,
    category: "pedal",
    tags: ["analog", "warm", "effects", "pedal", "mid"],
    art: "pedal",
    palette: ["#ff6a3d", "#33130c"],
    blurb: "Wobbly, imperfect repeats. Turn one knob and sound like a record.",
  },
  {
    id: "mu-headphones",
    name: "Studio Headphones 67",
    price: 149,
    category: "headphones",
    tags: ["clean", "studio", "monitoring", "compact"],
    art: "headphones",
    palette: ["#f3e7c9", "#332b1e"],
    blurb: "Closed-back, flat-ish, honest. For checking the take, not flattering it.",
  },
  {
    id: "mu-valve-amp",
    name: "Valve Combo Amp",
    price: 349,
    category: "amp",
    tags: ["loud", "warm", "stage", "amp", "heavy"],
    art: "amp",
    palette: ["#ff6a3d", "#2b0f08"],
    blurb: "Twelve watts, one twelve-inch speaker, no master volume. Commitment.",
  },
  {
    id: "mu-mic",
    name: "Condenser Mic",
    price: 199,
    category: "mic",
    tags: ["clean", "studio", "recording", "mid"],
    art: "mic",
    palette: ["#b8a1ff", "#1c1530"],
    blurb: "Large-diaphragm condenser with a cardioid pattern. Voices and rooms.",
  },
  {
    id: "mu-keyboard",
    name: "67-Key Controller",
    price: 159,
    category: "keys",
    tags: ["digital", "compact", "studio", "keys", "beginner"],
    art: "keys",
    palette: ["#6ff0c0", "#0c2820"],
    blurb: "Slim 67-key MIDI controller with weighted-feel action. Desk-friendly.",
  },
  {
    id: "mu-fuzz",
    name: "Fuzz Unit",
    price: 99,
    category: "pedal",
    tags: ["loud", "gritty", "effects", "pedal", "beginner"],
    art: "pedal",
    palette: ["#d7ff3e", "#2a330c"],
    blurb: "Two transistors, no apologies. Cleans up when you roll the volume back.",
  },
  {
    id: "mu-acoustic",
    name: "Busker Acoustic",
    price: 219,
    category: "guitar",
    tags: ["warm", "acoustic", "beginner", "guitar", "compact"],
    art: "guitar",
    palette: ["#f3e7c9", "#34291b"],
    blurb: "Small-body acoustic, set up low. Loud enough for a kitchen, polite in a flat.",
  },
  {
    id: "mu-loop",
    name: "Loop Station",
    price: 189,
    category: "pedal",
    tags: ["digital", "looping", "stage", "pedal", "mid"],
    art: "pedal",
    palette: ["#8bb4ff", "#111a2e"],
    blurb: "One foot in, one loop out. The fastest way to hear an idea back.",
  },
  {
    id: "mu-wedge",
    name: "Monitoring Wedge",
    price: 269,
    category: "amp",
    tags: ["loud", "stage", "monitoring", "amp", "heavy"],
    art: "amp",
    palette: ["#b8a1ff", "#1a1430"],
    blurb: "Coaxial wedge for small stages. Aim it at your knees and trust it.",
  },
  {
    id: "mu-cable",
    name: "Cable Kit (3×)",
    price: 29,
    category: "cable",
    tags: ["essential", "stage", "compact", "beginner"],
    art: "pedal",
    palette: ["#6ff0c0", "#0c2820"],
    blurb: "Three braided instrument cables that coil without a fight. Spares matter.",
  },
];

const FITNESS: Omit<Product, "vertical">[] = [
  {
    id: "fi-bumpers",
    name: "67 kg Bumper Set",
    price: 249,
    category: "plates",
    tags: ["strength", "heavy", "home", "mid"],
    art: "dumbbell",
    palette: ["#ff6a3d", "#2b0f08"],
    blurb: "Rubber bumpers, calibrated collars. Safe to drop, easy to read.",
  },
  {
    id: "fi-adjustable",
    name: "Adjustable Dumbbell",
    price: 189,
    category: "dumbbell",
    tags: ["strength", "compact", "home", "beginner"],
    art: "dumbbell",
    palette: ["#d7ff3e", "#26300d"],
    blurb: "2.5 to 24 kg in one dial. The one purchase that survives a small flat.",
  },
  {
    id: "fi-kettlebell",
    name: "Kettlebell 16",
    price: 79,
    category: "kettlebell",
    tags: ["strength", "heavy", "conditioning", "mid"],
    art: "kettlebell",
    palette: ["#8bb4ff", "#101a2e"],
    blurb: "Wide handle, flat base, powder coat. Swings, carries, squats.",
  },
  {
    id: "fi-bands",
    name: "Resistance Band 67",
    price: 34,
    category: "band",
    tags: ["recovery", "compact", "travel", "beginner"],
    art: "band",
    palette: ["#6ff0c0", "#0c2820"],
    blurb: "Layered latex loop in a mid-heavy tension. Warms up anything.",
  },
  {
    id: "fi-mat",
    name: "Studio Mat",
    price: 49,
    category: "mat",
    tags: ["recovery", "studio", "mobility", "home"],
    art: "mat",
    palette: ["#f3e7c9", "#332b1e"],
    blurb: "Six millimetres, closed cell, actually grippy when your hands are wet.",
  },
  {
    id: "fi-rack",
    name: "Squat Rack",
    price: 379,
    category: "rack",
    tags: ["strength", "home", "heavy", "mid"],
    art: "rack",
    palette: ["#34363c", "#0d0e10"],
    blurb: "Bolt-together uprights with spotter arms. Fits under a normal ceiling.",
  },
  {
    id: "fi-rope",
    name: "Jump Rope",
    price: 24,
    category: "rope",
    tags: ["cardio", "compact", "travel", "beginner"],
    art: "band",
    palette: ["#ff6a3d", "#2b0f08"],
    blurb: "Bearing handles, coated cable. Fast enough to be annoying.",
  },
  {
    id: "fi-belt",
    name: "Lifting Belt",
    price: 59,
    category: "belt",
    tags: ["strength", "heavy", "form", "mid"],
    art: "band",
    palette: ["#f3e7c9", "#332b1e"],
    blurb: "Ten-millimetre nylon, four-inch width. Cue to brace, not a crutch.",
  },
  {
    id: "fi-roller",
    name: "Foam Roller",
    price: 39,
    category: "roller",
    tags: ["recovery", "mobility", "home", "beginner"],
    art: "mat",
    palette: ["#b8a1ff", "#1a1430"],
    blurb: "Firm EVA core with a grid surface. Honest about your calves.",
  },
  {
    id: "fi-metronome",
    name: "67 BPM Metronome",
    price: 45,
    category: "timer",
    tags: ["tempo", "studio", "form", "compact"],
    art: "timer",
    palette: ["#d7ff3e", "#26300d"],
    blurb: "Clip-on beat keeper. Set it to 67 and let the reps fall on the click.",
  },
  {
    id: "fi-wraps",
    name: "Wrist Wraps",
    price: 29,
    category: "wraps",
    tags: ["strength", "form", "compact", "mid"],
    art: "band",
    palette: ["#34363c", "#0d0e10"],
    blurb: "Stiff, thumb-loop wraps. Support for the last two sets only.",
  },
  {
    id: "fi-kettlebell-24",
    name: "Kettlebell 24",
    price: 109,
    category: "kettlebell",
    tags: ["strength", "heavy", "conditioning", "mid"],
    art: "kettlebell",
    palette: ["#6ff0c0", "#0c2820"],
    blurb: "The next bell up. Same geometry, considerably more opinion.",
  },
];

export const CATALOG: Product[] = [
  ...buildApparel(),
  ...MUSIC.map((item) => ({ ...item, vertical: "music" as const })),
  ...FITNESS.map((item) => ({ ...item, vertical: "fitness" as const })),
];

export const PRODUCT_MAP = new Map(CATALOG.map((product) => [product.id, product]));

export function productsFor(vertical: Vertical): Product[] {
  return CATALOG.filter((product) => product.vertical === vertical);
}

export const CATEGORY_LABELS: Record<string, string> = Object.fromEntries(
  APPAREL.map((spec) => [spec.dir, spec.noun]),
);

/** Style words used by the onboarding chips. Each maps onto catalog tags. */
export const VIBES = [
  { label: "Oversized", tags: ["oversized", "baggy", "boxy", "relaxed"] },
  { label: "Tailored", tags: ["tailored", "smart", "pleated", "slim"] },
  { label: "Cotton", tags: ["cotton", "jersey", "poplin"] },
  { label: "Utility", tags: ["utility", "layer", "cargo"] },
  { label: "Retro", tags: ["retro", "statement", "warm"] },
  { label: "Minimal", tags: ["minimal", "clean"] },
  { label: "Black", tags: ["black", "charcoal"] },
  { label: "Colour", tags: ["rust", "cobalt", "green", "red", "blue", "sand"] },
  { label: "Everyday", tags: ["everyday", "casual", "basic"] },
  { label: "Going out", tags: ["after dark", "occasion", "going out"] },
];

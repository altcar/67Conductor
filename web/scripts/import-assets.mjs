/**
 * One-shot asset importer.
 *
 * - copies the MediaPipe wasm runtime out of node_modules
 * - downloads the hand + pose landmarker models
 * - samples the repo's clothing dataset into web/public/products
 * - writes src/data/photos.json
 *
 * Run with: npm run import-assets
 */
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, readdir, writeFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, "..");
const repo = resolve(web, "..");
const dataset = join(repo, "67Conductor", "Server_src", "static", "dataset", "train");
const publicDir = join(web, "public");
const productsDir = join(publicDir, "products");
const modelsDir = join(publicDir, "models");
const wasmDir = join(publicDir, "wasm");

const MODELS = {
  "hand_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  "pose_landmarker_lite.task":
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
};

const CATEGORIES = [
  { dir: "t-shirt", label: "tee", count: 7 },
  { dir: "longsleeve", label: "long sleeve", count: 6 },
  { dir: "shirt", label: "shirt", count: 6 },
  { dir: "outwear", label: "outerwear", count: 6 },
  { dir: "dress", label: "dress", count: 6 },
  { dir: "skirt", label: "skirt", count: 5 },
  { dir: "pants", label: "trousers", count: 7 },
  { dir: "shorts", label: "shorts", count: 5 },
  { dir: "shoes", label: "shoes", count: 6 },
  { dir: "hat", label: "hat", count: 4 },
];

async function ensureDirs() {
  for (const dir of [productsDir, modelsDir, wasmDir, join(web, "src", "data")]) {
    await mkdir(dir, { recursive: true });
  }
}

async function download(url, destination) {
  if (existsSync(destination)) {
    const info = await stat(destination);
    if (info.size > 100_000) {
      console.log(`  cached ${destination.split(/[\\/]/).pop()} (${(info.size / 1e6).toFixed(1)} MB)`);
      return;
    }
  }
  console.log(`  downloading ${url}`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`GET ${url} -> ${response.status}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  await writeFile(destination, buffer);
  console.log(`  saved ${destination.split(/[\\/]/).pop()} (${(buffer.length / 1e6).toFixed(1)} MB)`);
}

async function copyWasm() {
  const source = join(web, "node_modules", "@mediapipe", "tasks-vision", "wasm");
  const files = await readdir(source);
  for (const file of files) {
    if (!file.endsWith(".js") && !file.endsWith(".wasm")) continue;
    await copyFile(join(source, file), join(wasmDir, file));
  }
  console.log(`  copied ${files.length} wasm runtime files`);
}

async function importPhotos() {
  const photos = {};
  for (const category of CATEGORIES) {
    const source = join(dataset, category.dir);
    if (!existsSync(source)) {
      console.warn(`  skipping ${category.dir}: not found`);
      continue;
    }
    const files = (await readdir(source)).filter((f) => f.toLowerCase().endsWith(".jpg")).sort();
    if (files.length === 0) continue;

    const picked = [];
    const stride = files.length / category.count;
    for (let i = 0; i < category.count; i += 1) {
      const file = files[Math.min(files.length - 1, Math.floor(i * stride + stride * 0.37))];
      if (!picked.includes(file)) picked.push(file);
    }

    const entries = [];
    for (const file of picked) {
      const hash = createHash("sha1").update(category.dir + file).digest("hex").slice(0, 10);
      const id = `${category.dir.replace(/[^a-z]/g, "")}-${hash}`;
      const output = `${id}.jpg`;
      await copyFile(join(source, file), join(productsDir, output));
      entries.push({ id, file: `/products/${output}` });
    }
    photos[category.dir] = { label: category.label, entries };
    console.log(`  ${category.dir}: ${entries.length} photos`);
  }
  await writeFile(join(web, "src", "data", "photos.json"), `${JSON.stringify(photos, null, 2)}\n`);
}

console.log("import-assets");
await ensureDirs();
console.log("- wasm runtime");
await copyWasm();
console.log("- landmarker models");
for (const [name, url] of Object.entries(MODELS)) {
  await download(url, join(modelsDir, name));
}
console.log("- dataset photos");
await importPhotos();
console.log("done");

// keep the linter honest about readFile being intentionally unused
void readFile;

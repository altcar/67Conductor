import type { ArtKind, Product } from "../data/catalog";

/**
 * Line-art product illustrations for the music + fitness verticals. They share
 * one geometry so the catalog still feels like a single shop.
 */
export function productArt(kind: ArtKind, product: Product): string {
  const [accent, base] = product.palette;
  return `<svg class="art" viewBox="0 0 200 250" role="img" aria-label="${escapeXml(product.name)}">
    <rect width="200" height="250" fill="${base}"/>
    <circle cx="42" cy="52" r="70" fill="${accent}" opacity="0.10"/>
    <circle cx="175" cy="215" r="85" fill="${accent}" opacity="0.08"/>
    <text x="10" y="232" font-family="JetBrains Mono, monospace" font-size="66" font-weight="500" fill="${accent}" opacity="0.10">67</text>
    <g stroke="${accent}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
      ${shape(kind)}
    </g>
  </svg>`;
}

function shape(kind: ArtKind): string {
  switch (kind) {
    case "guitar":
      return `
        <circle cx="96" cy="172" r="34"/>
        <circle cx="96" cy="132" r="24"/>
        <path d="M90 66 h14 l5 -16 h-24 z"/>
        <rect x="92" y="64" width="10" height="62" rx="3"/>
        <line x1="95" y1="78" x2="95" y2="204" stroke-width="1"/>
        <line x1="99" y1="78" x2="99" y2="204" stroke-width="1"/>
        <circle cx="96" cy="143" r="6"/>
        <line x1="80" y1="196" x2="112" y2="196" stroke-width="5"/>`;
    case "keys":
      return `
        <rect x="38" y="104" width="124" height="74" rx="7"/>
        <line x1="38" y1="132" x2="162" y2="132"/>
        ${[0, 1, 2, 3, 4, 5, 6].map((i) => `<line x1="${38 + (124 / 7) * (i + 1)}" y1="132" x2="${38 + (124 / 7) * (i + 1)}" y2="178"/>`).join("")}
        ${[0, 1, 2, 4, 5].map((i) => `<rect x="${38 + (124 / 7) * (i + 1) - 7}" y="132" width="14" height="26" fill="${"#0c0c0d"}" stroke-width="1.4"/>`).join("")}
        <circle cx="52" cy="92" r="4"/><circle cx="68" cy="92" r="4"/><circle cx="84" cy="92" r="4"/>`;
    case "pedal":
      return `
        <rect x="58" y="104" width="84" height="104" rx="10"/>
        <circle cx="80" cy="130" r="7"/><circle cx="100" cy="130" r="7"/><circle cx="120" cy="130" r="7"/>
        <line x1="72" y1="148" x2="128" y2="148"/>
        <circle cx="100" cy="186" r="10"/>
        <line x1="100" y1="176" x2="100" y2="167"/>
        <line x1="70" y1="208" x2="70" y2="218"/><line x1="130" y1="208" x2="130" y2="218"/>`;
    case "headphones":
      return `
        <path d="M52 148 v-26 a48 48 0 0 1 96 0 v26"/>
        <rect x="42" y="142" width="22" height="46" rx="10"/>
        <rect x="136" y="142" width="22" height="46" rx="10"/>
        <path d="M56 118 q44 -22 88 0" opacity="0.5"/>
        <line x1="64" y1="165" x2="136" y2="165" opacity="0.25"/>`;
    case "amp":
      return `
        <rect x="44" y="86" width="112" height="112" rx="8"/>
        <rect x="56" y="98" width="88" height="58" rx="4"/>
        ${[0, 1, 2, 3, 4, 5].map((i) => `<line x1="${58 + i * 15}" y1="102" x2="${58 + i * 15}" y2="152" opacity="0.45"/>`).join("")}
        <circle cx="68" cy="176" r="6"/><circle cx="88" cy="176" r="6"/><circle cx="108" cy="176" r="6"/>
        <line x1="124" y1="170" x2="140" y2="182"/>
        <path d="M84 86 v-10 h32 v10" />`;
    case "mic":
      return `
        <rect x="87" y="66" width="26" height="58" rx="13"/>
        ${[0, 1, 2, 3].map((i) => `<line x1="90" y1="${78 + i * 12}" x2="110" y2="${78 + i * 12}" opacity="0.55"/>`).join("")}
        <path d="M76 118 a24 24 0 0 0 48 0"/>
        <line x1="100" y1="142" x2="100" y2="188"/>
        <line x1="78" y1="192" x2="122" y2="192"/>
        <circle cx="100" cy="188" r="3"/>`;
    case "dumbbell":
      return `
        <line x1="58" y1="146" x2="142" y2="146" stroke-width="6"/>
        <rect x="40" y="118" width="16" height="56" rx="5"/>
        <rect x="144" y="118" width="16" height="56" rx="5"/>
        <rect x="30" y="130" width="8" height="32" rx="3"/>
        <rect x="162" y="130" width="8" height="32" rx="3"/>
        <line x1="76" y1="132" x2="76" y2="160" opacity="0.35"/>
        <line x1="124" y1="132" x2="124" y2="160" opacity="0.35"/>`;
    case "kettlebell":
      return `
        <path d="M82 122 v-10 a18 18 0 0 1 36 0 v10"/>
        <path d="M72 124 q28 -14 56 0 l6 54 q-34 16 -68 0 z"/>
        <line x1="82" y1="150" x2="118" y2="150" opacity="0.35"/>
        <circle cx="100" cy="160" r="8" opacity="0.4"/>`;
    case "band":
      return `
        <ellipse cx="100" cy="146" rx="42" ry="58" transform="rotate(-18 100 146)"/>
        <ellipse cx="100" cy="146" rx="28" ry="42" transform="rotate(-18 100 146)" opacity="0.5"/>
        <line x1="72" y1="196" x2="128" y2="196" stroke-width="4"/>
        <line x1="100" y1="196" x2="100" y2="212"/>`;
    case "mat":
      return `
        <rect x="62" y="152" width="92" height="30" rx="12"/>
        <ellipse cx="62" cy="167" rx="16" ry="15"/>
        <path d="M62 152 a15 15 0 0 1 0 30" opacity="0.5"/>
        <path d="M62 158 a9 9 0 0 1 0 18" opacity="0.35"/>
        <line x1="78" y1="182" x2="150" y2="182" opacity="0.3"/>
        <circle cx="138" cy="167" r="4" opacity="0.5"/>`;
    case "rack":
      return `
        <line x1="64" y1="66" x2="64" y2="206"/>
        <line x1="136" y1="66" x2="136" y2="206"/>
        <line x1="64" y1="92" x2="136" y2="92"/>
        <line x1="52" y1="206" x2="76" y2="206"/>
        <line x1="124" y1="206" x2="148" y2="206"/>
        <circle cx="84" cy="150" r="20"/><circle cx="84" cy="150" r="6"/>
        <circle cx="116" cy="150" r="20"/><circle cx="116" cy="150" r="6"/>
        <line x1="84" y1="150" x2="116" y2="150" stroke-width="4"/>`;
    case "timer":
      return `
        <circle cx="100" cy="142" r="46"/>
        <circle cx="100" cy="142" r="38" opacity="0.4"/>
        <line x1="100" y1="142" x2="100" y2="112" stroke-width="3"/>
        <line x1="100" y1="142" x2="120" y2="150" stroke-width="2"/>
        <circle cx="100" cy="142" r="3" fill="${"currentColor"}"/>
        <line x1="100" y1="88" x2="100" y2="78"/>
        <rect x="94" y="70" width="12" height="8" rx="2"/>
        <line x1="66" y1="142" x2="72" y2="142" opacity="0.5"/>
        <line x1="128" y1="142" x2="134" y2="142" opacity="0.5"/>`;
    default:
      return `<circle cx="100" cy="140" r="40"/>`;
  }
}

function escapeXml(value: string): string {
  return value.replace(/[<>&"]/g, (character) => {
    switch (character) {
      case "<": return "&lt;";
      case ">": return "&gt;";
      case "&": return "&amp;";
      default: return "&quot;";
    }
  });
}

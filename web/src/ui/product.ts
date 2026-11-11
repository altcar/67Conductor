import type { Product } from "../data/catalog";
import { productArt } from "./art";
import { esc, money, qs } from "./dom";
import type { RailOptions } from "./context";

export function productVisual(product: Product): string {
  if (product.image) {
    return `<img class="photo" src="${esc(product.image)}" alt="${esc(product.name)}" loading="lazy" decoding="async">`;
  }
  return productArt(product.art ?? "guitar", product);
}

export function renderRail(options: RailOptions, onAdd: (product: Product) => void): HTMLElement {
  const section = document.createElement("section");
  section.className = "rail";
  section.innerHTML = `
    <div class="rail-head">
      <div>
        <span class="eyebrow">${esc(options.eyebrow)}</span>
        <h2>${esc(options.title)}</h2>
      </div>
      <span class="rail-count">${options.products.length} pick${options.products.length === 1 ? "" : "s"}</span>
    </div>
    <div class="rail-grid"></div>
  `;

  const grid = qs(section, ".rail-grid");
  if (options.products.length === 0) {
    grid.innerHTML = `<p class="rail-empty">${esc(options.emptyText ?? "Swipe a few things first — the rail fills in as your taste lands.")}</p>`;
    return section;
  }

  for (const { product, why } of options.products) {
    const card = document.createElement("article");
    card.className = "pick";
    card.innerHTML = `
      <div class="pick-media ${product.image ? "is-photo" : "is-art"}">
        ${productVisual(product)}
        <span class="pick-price">${money(product.price)}</span>
      </div>
      <div class="pick-body">
        <h3>${esc(product.name)}</h3>
        <p class="pick-why">${esc(why ?? product.blurb)}</p>
        <div class="pick-foot">
          <span class="pick-tag">${esc(product.tags[0] ?? product.category)}</span>
          <button class="btn btn-ghost btn-sm" type="button" data-add="${esc(product.id)}">Add to bag</button>
        </div>
      </div>
    `;
    card.querySelector("[data-add]")?.addEventListener("click", () => onAdd(product));
    grid.appendChild(card);
  }
  return section;
}

export function chipList(tags: string[], limit = 4): string {
  return tags
    .slice(0, limit)
    .map((tag) => `<span class="chip">${esc(tag)}</span>`)
    .join("");
}

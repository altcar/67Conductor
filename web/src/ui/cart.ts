import { PRODUCT_MAP } from "../data/catalog";
import type { Order } from "../lib/store";
import { store } from "../lib/store";
import type { ViewId } from "./context";
import { esc, icon, money, qs, template } from "./dom";
import { productVisual } from "./product";

export class Cart {
  readonly el: HTMLElement;
  private body: HTMLElement;
  private foot: HTMLElement;
  private isOpen = false;

  constructor(
    private readonly onCheckout: (order: Order) => void,
    private readonly showView: (id: ViewId) => void,
  ) {
    this.el = template(`
      <div class="cart-layer" data-open="false">
        <div class="cart-backdrop" data-close></div>
        <aside class="cart" aria-label="Shopping bag" aria-hidden="true">
          <header class="cart-head">
            <div>
              <span class="eyebrow">Your bag</span>
              <h2 data-count>Empty</h2>
            </div>
            <button class="iconbtn" data-close type="button" aria-label="Close bag">${icon("close")}</button>
          </header>
          <div class="cart-body" data-body></div>
          <footer class="cart-foot" data-foot></footer>
        </aside>
      </div>
    `);

    this.body = qs(this.el, "[data-body]");
    this.foot = qs(this.el, "[data-foot]");

    this.el.addEventListener("click", (event) => {
      const target = event.target as HTMLElement;
      if (target.closest("[data-close]")) {
        this.close();
        return;
      }
      const inc = target.closest<HTMLElement>("[data-inc]")?.dataset.inc;
      const dec = target.closest<HTMLElement>("[data-dec]")?.dataset.dec;
      const remove = target.closest<HTMLElement>("[data-remove]")?.dataset.remove;
      if (inc) store.setQty(inc, (store.snapshot.cart.find((line) => line.id === inc)?.qty ?? 0) + 1);
      if (dec) store.setQty(dec, (store.snapshot.cart.find((line) => line.id === dec)?.qty ?? 0) - 1);
      if (remove) store.setQty(remove, 0);
      if (target.closest("[data-checkout]")) this.checkout();
      if (target.closest("[data-browse]")) {
        this.close();
        this.showView("rewear");
      }
    });
  }

  mount(parent: HTMLElement): void {
    parent.appendChild(this.el);
    store.subscribe(() => {
      if (this.isOpen) this.render();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.isOpen) this.close();
    });
  }

  open(): void {
    this.isOpen = true;
    this.el.dataset.open = "true";
    const aside = this.el.querySelector("aside");
    aside?.setAttribute("aria-hidden", "false");
    document.body.classList.add("no-scroll");
    this.render();
  }

  close(): void {
    this.isOpen = false;
    this.el.dataset.open = "false";
    const aside = this.el.querySelector("aside");
    aside?.setAttribute("aria-hidden", "true");
    document.body.classList.remove("no-scroll");
  }

  private checkout(): void {
    if (store.snapshot.cart.length === 0) return;
    const order = store.checkout();
    this.close();
    this.onCheckout(order);
  }

  private render(): void {
    const cart = store.snapshot.cart;
    const count = store.cartCount;
    const countEl = this.el.querySelector("[data-count]");
    if (countEl) countEl.textContent = count === 0 ? "Empty" : `${count} item${count === 1 ? "" : "s"}`;

    if (cart.length === 0) {
      this.body.innerHTML = `
        <div class="cart-empty">
          <div class="cart-empty-mark">6<span>7</span></div>
          <p>Nothing in the bag yet. The rail is tuned as you swipe — start there.</p>
          <button class="btn btn-acid" data-browse type="button">Back to the deck</button>
        </div>
      `;
      this.foot.innerHTML = "";
      return;
    }

    this.body.innerHTML = cart
      .map((line) => {
        const product = PRODUCT_MAP.get(line.id);
        if (!product) return "";
        return `
          <div class="cart-line">
            <div class="cart-thumb ${product.image ? "is-photo" : "is-art"}">${productVisual(product)}</div>
            <div class="cart-line-body">
              <div class="cart-line-top">
                <h3>${esc(product.name)}</h3>
                <b>${money(product.price * line.qty)}</b>
              </div>
              <p class="tiny">${esc(product.category)}${product.size ? ` · ${esc(product.size)}` : ""}${product.oneOfOne ? " · 1 of 1" : ""}</p>
              <div class="cart-line-actions">
                <div class="qty">
                  <button type="button" data-dec="${esc(product.id)}" aria-label="One fewer">−</button>
                  <span>${line.qty}</span>
                  <button type="button" data-inc="${esc(product.id)}" aria-label="One more">+</button>
                </div>
                <button class="linkbtn" data-remove="${esc(product.id)}" type="button">Remove</button>
              </div>
            </div>
          </div>
        `;
      })
      .join("");

    const totals = store.totals();
    this.foot.innerHTML = `
      <div class="totals">
        <div><span>Subtotal</span><b>${money(totals.subtotal)}</b></div>
        <div><span>Shipping</span><b>${totals.shipping === 0 ? "Free" : money(totals.shipping)}</b></div>
        <div class="total-row"><span>Total</span><b>${money(totals.total)}</b></div>
      </div>
      <button class="btn btn-acid btn-block" data-checkout type="button">Checkout · ${money(totals.total)}</button>
      <p class="tiny">Browser-only demo: no payment, no server, no shipping. The bag is kept in this tab's local storage.</p>
    `;
  }
}

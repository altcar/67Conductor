import { productsFor } from "../data/catalog";
import type { Product } from "../data/catalog";
import { GestureRouter } from "../lib/gestures";
import { buildDeck, recommend, tasteSummary } from "../lib/recommender";
import { store } from "../lib/store";
import type { FramePacket } from "../lib/vision";
import type { View, ViewContext } from "./context";
import { drawHand, fitOverlay } from "./draw";
import { esc, money, qs, template } from "./dom";
import { renderRail } from "./product";

export class RewearView implements View {
  readonly id = "rewear" as const;
  readonly label = "Rewear";

  private root: HTMLElement | null = null;
  private deckEl: HTMLElement | null = null;
  private deck: Product[] = [];
  private cursor = 0;
  private busy = false;
  private router = new GestureRouter();
  private unsubscribe: (() => void) | null = null;
  private keyHandler: ((event: KeyboardEvent) => void) | null = null;
  private drag: {
    pointerId: number;
    x: number;
    y: number;
    dx: number;
    dy: number;
    card: HTMLElement;
  } | null = null;
  private resetArmed = false;

  constructor(private readonly ctx: ViewContext) {}

  show(slot: HTMLElement): void {
    this.root = template(`
      <section class="view rewear" aria-label="Rewear">
        <div class="view-head">
          <div>
            <span class="eyebrow">01 — Rewear</span>
            <h1>Secondhand, <em>first instinct.</em></h1>
          </div>
          <p class="lede">One-of-one pieces. Swipe with a palm, superlike with the 67. Every move tunes the rail below.</p>
        </div>

        <div class="rewear-grid">
          <aside class="panel taste-panel" data-taste></aside>

          <div class="deck-col">
            <div class="deck-topline">
              <span class="mono-label">THE DECK</span>
              <span class="counter" data-counter>00 / 00</span>
            </div>
            <div class="deck" data-deck></div>
            <div class="deck-actions">
              <button class="act act-nope" data-act="nope" type="button" aria-label="Nope">
                <span class="act-key">←</span><span>Nope</span>
              </button>
              <button class="act act-super" data-act="super" type="button" aria-label="67 superlike">
                <span class="act-key">67</span><span>Superlike</span>
              </button>
              <button class="act act-like" data-act="like" type="button" aria-label="Like">
                <span class="act-key">→</span><span>Like</span>
              </button>
            </div>
            <p class="keyhint">Drag the card · <kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> · right palm like · left palm nope · both palms 67</p>
          </div>

          <div class="stage-slot" data-stage></div>
        </div>

        <div class="picks-host" data-picks></div>
        <p class="sr-only" data-announce aria-live="polite"></p>
      </section>
    `);

    slot.replaceChildren(this.root);
    const foot = template(`
      <div class="gesture-legend">
        <div class="raise-row" data-side="nope"><span>LEFT</span><div class="raise-bar"><i data-bar="nope"></i></div></div>
        <div class="gauge" data-gauge>
          <div class="gauge-ring"><b>67</b></div>
          <span>SUPERLIKE</span>
        </div>
        <div class="raise-row" data-side="like"><span>RIGHT</span><div class="raise-bar"><i data-bar="like"></i></div></div>
      </div>
    `);

    this.ctx.stage.attach(qs(this.root, "[data-stage]"), {
      mode: "hands",
      title: "67 CAM / REWEAR",
      hint: "One open palm to swipe · both palms to superlike",
      foot,
    });

    this.deckEl = qs<HTMLElement>(this.root, "[data-deck]");

    this.root.addEventListener("click", (event) => this.onClick(event));
    this.bindDeck();

    this.unsubscribe = store.subscribe(() => this.refreshPanels());
    this.refreshPanels();
    this.refill(true);
    this.renderDeck();
    if (this.top()) store.markSeen(this.top()!.id);

    this.keyHandler = (event) => this.onKey(event);
    window.addEventListener("keydown", this.keyHandler);
    this.router.reset();
  }

  hide(): void {
    if (this.keyHandler) window.removeEventListener("keydown", this.keyHandler);
    this.keyHandler = null;
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.ctx.stage.detach();
    this.root = null;
    this.deckEl = null;
    this.drag = null;
  }

  onFrame(packet: FramePacket): void {
    if (!this.root || !this.ctx.stage.vision.isCameraOn) return;
    const { width, height } = fitOverlay(this.ctx.stage.canvasEl, this.ctx.stage.vision.video);
    const ctx2d = this.ctx.stage.context;
    clearCanvas(ctx2d, width, height);

    for (const hand of packet.hands) {
      const single = packet.hands.length === 1;
      const center = hand.points[0];
      const side = center.x > 0.57 ? "LIKE" : center.x < 0.43 ? "NOPE" : "";
      drawHand(ctx2d, width, height, hand, {
        color: single && side === "LIKE" ? "#d7ff3e" : single && side === "NOPE" ? "#ff5a3d" : "#f4f1ea",
        label: single ? side : "67",
      });
    }

    const update = this.router.update(packet.hands, packet.t);
    const gauge = this.root.querySelector("[data-gauge]") as HTMLElement | null;
    gauge?.style.setProperty("--score", update.superScore.toFixed(3));

    for (const side of ["like", "nope"] as const) {
      const bar = this.root.querySelector(`[data-bar="${side}"]`) as HTMLElement | null;
      if (!bar) continue;
      const active = update.raiseSide === side;
      bar.style.transform = `scaleX(${active ? update.raiseProgress : 0})`;
    }

    if (update.action) this.act(update.action);
  }

  private onClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    const action = target.closest<HTMLElement>("[data-act]")?.dataset.act;
    if (action === "like" || action === "nope" || action === "super") {
      this.act(action);
      return;
    }
    if (target.closest("[data-reset]")) {
      if (!this.resetArmed) {
        this.resetArmed = true;
        this.ctx.toast("Tap reset again to wipe your taste profile.", "warn");
        window.setTimeout(() => (this.resetArmed = false), 2600);
        return;
      }
      this.resetArmed = false;
      store.resetTaste();
      this.refill(true);
      this.renderDeck();
      this.ctx.toast("Taste reset. Blank slate.", "default");
      return;
    }
    if (target.closest("[data-vibes]")) {
      this.ctx.openVibes();
      return;
    }
  }

  private onKey(event: KeyboardEvent): void {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (document.querySelector('.modal-layer[data-open="true"], .cart-layer[data-open="true"]')) return;
    if (event.key === "ArrowLeft") this.act("nope");
    else if (event.key === "ArrowRight") this.act("like");
    else if (event.key === "ArrowUp" || event.key.toLowerCase() === "s") this.act("super");
    else if (event.key.toLowerCase() === "c") this.ctx.openCart();
  }

  rerank(): void {
    this.refill(true);
    this.renderDeck();
    this.refreshPanels();
    if (this.top()) store.markSeen(this.top()!.id);
  }

  private top(): Product | null {
    return this.deck[this.cursor] ?? null;
  }

  private refill(force = false): void {
    if (!force && this.deck.length - this.cursor > 2) return;
    this.deck = buildDeck(productsFor("apparel"), store.snapshot.taste, store.decided, store.seen, 20);
    this.cursor = 0;
  }

  private renderDeck(): void {
    if (!this.deckEl) return;
    const current = this.top();
    if (!current) {
      this.deckEl.innerHTML = `<div class="deck-empty"><span class="eyebrow">Deck complete</span><p>You have been through the rail. Reset your taste or keep scrolling the picks.</p><button class="btn btn-acid" data-reset type="button">Reset taste</button></div>`;
      return;
    }

    const visible = this.deck.slice(this.cursor, this.cursor + 3);
    this.deckEl.innerHTML = "";
    for (let i = visible.length - 1; i >= 0; i -= 1) {
      const product = visible[i];
      const card = template(`
        <article class="card ${product.image ? "is-photo" : "is-art"}" data-card ${i === 0 ? "data-top" : ""} style="--i:${i}">
          <div class="card-media">
            ${
              product.image
                ? `<img class="photo" src="${esc(product.image)}" alt="${esc(product.name)}" draggable="false" decoding="async">`
                : ""
            }
            <span class="card-index">${String(this.cursor + i + 1).padStart(2, "0")}</span>
            <span class="stamp stamp-super">67</span>
            <span class="stamp stamp-like">LIKE</span>
            <span class="stamp stamp-nope">NOPE</span>
          </div>
          <div class="card-body">
            <div class="card-title">
              <h2>${esc(product.name)}</h2>
              <span class="price">${money(product.price)}</span>
            </div>
            <p class="card-meta">${esc(product.condition ?? "")}${product.size ? ` · Size ${esc(product.size)}` : ""}${product.oneOfOne ? " · 1 of 1" : ""}</p>
            <div class="card-chips">
              ${product.tags.slice(0, 4).map((tag) => `<span class="chip">${esc(tag)}</span>`).join("")}
            </div>
          </div>
          <div class="swipe-hint hint-like">LIKE</div>
          <div class="swipe-hint hint-nope">NOPE</div>
          <div class="swipe-hint hint-super">67</div>
        </article>
      `);
      this.deckEl.appendChild(card);
    }

    const counter = this.root?.querySelector("[data-counter]");
    if (counter) {
      counter.textContent = `${String(Math.min(this.cursor + 1, this.deck.length)).padStart(2, "0")} / ${String(this.deck.length).padStart(2, "0")}`;
    }

    const announce = this.root?.querySelector("[data-announce]");
    if (announce && this.top()) {
      announce.textContent = `${this.top()!.name}, ${money(this.top()!.price)}. Tags: ${this.top()!.tags.join(", ")}.`;
    }
  }

  private bindDeck(): void {
    if (!this.deckEl) return;
    this.deckEl.addEventListener("pointerdown", (event) => {
      const card = (event.target as HTMLElement).closest<HTMLElement>("[data-top]");
      if (!card || this.busy) return;
      if (event.button !== 0 && event.pointerType === "mouse") return;
      card.setPointerCapture(event.pointerId);
      this.drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, dx: 0, dy: 0, card };
      card.classList.add("dragging");
    });

    this.deckEl.addEventListener("pointermove", (event) => {
      if (!this.drag || event.pointerId !== this.drag.pointerId) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.dx = dx;
      this.drag.dy = dy;
      const rotate = Math.max(-14, Math.min(14, dx / 18));
      this.drag.card.style.transform = `translate(${dx}px, ${dy}px) rotate(${rotate}deg)`;
      const likeOpacity = Math.max(0, Math.min(1, dx / 110));
      const nopeOpacity = Math.max(0, Math.min(1, -dx / 110));
      const superOpacity = Math.max(0, Math.min(1, -dy / 120)) * (Math.abs(dx) < 80 ? 1 : 0.2);
      this.applyHint("like", likeOpacity);
      this.applyHint("nope", nopeOpacity);
      this.applyHint("super", superOpacity);
    });

    const end = (event: PointerEvent) => {
      if (!this.drag || event.pointerId !== this.drag.pointerId) return;
      const { dx, dy, card } = this.drag;
      this.drag = null;
      card.classList.remove("dragging");
      if (dy < -110 && Math.abs(dx) < 90) {
        this.act("super");
      } else if (dx > 92) {
        this.act("like");
      } else if (dx < -92) {
        this.act("nope");
      } else {
        card.style.transform = "";
        for (const key of ["like", "nope", "super"]) this.applyHint(key, 0);
      }
    };

    this.deckEl.addEventListener("pointerup", end);
    this.deckEl.addEventListener("pointercancel", end);
  }

  private applyHint(kind: string, value: number): void {
    const hint = this.deckEl?.querySelector(`.hint-${kind}`) as HTMLElement | null;
    if (hint) hint.style.opacity = String(value);
  }

  private act(action: "like" | "nope" | "super"): void {
    if (this.busy) return;
    const product = this.top();
    if (!product) return;
    this.busy = true;
    const card = this.deckEl?.querySelector("[data-top]") as HTMLElement | null;

    store.record(product.id, action);
    this.ctx.stage.flash(action);
    if (action === "like") this.ctx.toast(`Liked — ${product.name}.`, "acid");
    else if (action === "nope") this.ctx.toast("Nope. Noted.", "default");
    else this.ctx.toast("67! Superlike. That one sharpens the whole profile.", "acid");

    card?.classList.add(`out-${action}`);

    window.setTimeout(() => {
      this.cursor += 1;
      this.busy = false;
      this.refill();
      this.renderDeck();
      this.refreshPanels();
      store.markSeen(this.top()?.id ?? "");
    }, 320);
  }

  private refreshPanels(): void {
    if (!this.root) return;

    const tastePanel = this.root.querySelector("[data-taste]") as HTMLElement | null;
    if (tastePanel) tastePanel.innerHTML = this.renderTaste();

    const picks = this.root.querySelector("[data-picks]") as HTMLElement | null;
    if (picks) {
      const options = this.pickOptions();
      picks.replaceChildren(
        renderRail(
          {
            eyebrow: "67 picks",
            title: "The rail, tuned to you",
            products: options.map((pick) => ({ product: pick.product, why: pick.why })),
            emptyText: "No signal yet. The rail learns from every like, nope and 67.",
          },
          (product) => {
            store.addToCart(product.id);
            this.ctx.toast(`Added ${product.name}.`, "acid");
          },
        ),
      );
    }

    const counter = this.root.querySelector("[data-counter]");
    if (counter) {
      counter.textContent = `${String(Math.min(this.cursor + 1, Math.max(1, this.deck.length))).padStart(2, "0")} / ${String(this.deck.length).padStart(2, "0")}`;
    }
    this.updateTasteCounter();
  }

  private updateTasteCounter(): void {
    const el = this.root?.querySelector("[data-signal]");
    if (el) el.textContent = String(store.snapshot.decisions.length);
  }

  private pickOptions(): { product: Product; why: string }[] {
    const taste = store.snapshot.taste;
    const hasSignal = Object.values(taste.weights).some((weight) => Math.abs(weight) > 0.2);
    if (!hasSignal) {
      const pool = productsFor("apparel");
      return [0, 1, 2].map((i) => ({
        product: pool[(i * 17 + 5) % pool.length],
        why: "Fresh territory — a wildcard from the rail.",
      }));
    }
    return recommend(productsFor("apparel"), taste, {
      limit: 3,
      exclude: store.decided,
      seen: new Set(store.snapshot.seen.slice(-6)),
    }).map((pick) => ({ product: pick.product, why: pick.why }));
  }

  private renderTaste(): string {
    const taste = store.snapshot.taste;
    const summary = tasteSummary(taste, 4);
    const decisions = store.snapshot.decisions.length;
    const likes = taste.likes.length;
    const supers = taste.supers.length;
    const nopes = taste.nopes.length;
    return `
      <span class="eyebrow">Taste profile</span>
      <h2>${summary.length ? summary.map((tag) => `<span>${esc(tag)}</span>`).join("") : "Blank<br>slate."}</h2>
      <p class="panel-copy">${
        decisions === 0
          ? "Pick a vibe, then start swiping. The profile builds in the browser and stays there."
          : summary.length
            ? `Built from ${decisions} decision${decisions === 1 ? "" : "s"}. The rail keeps re-ranking itself.`
            : "Nothing has stuck yet — keep going."
      }</p>
      <div class="taste-stats">
        <div><b data-signal>${decisions}</b><span>decisions</span></div>
        <div><b>${likes}</b><span>likes</span></div>
        <div><b>${supers}</b><span>67s</span></div>
        <div><b>${nopes}</b><span>nopes</span></div>
      </div>
      <div class="taste-actions">
        <button class="btn btn-ghost btn-sm" data-vibes type="button">Adjust vibe</button>
        <button class="btn btn-ghost btn-sm" data-reset type="button">Reset</button>
      </div>
      <p class="tiny">Stored in this browser only. No account, no server, no uploads.</p>
    `;
  }
}

function clearCanvas(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  ctx.clearRect(0, 0, width, height);
}

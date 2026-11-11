import "./fonts.css";
import "./styles.css";
import { PRODUCT_MAP, VIBES } from "./data/catalog";
import { AudioEngine } from "./lib/audio";
import { store } from "./lib/store";
import type { Order } from "./lib/store";
import type { View, ViewId } from "./ui/context";
import { Cart } from "./ui/cart";
import { esc, money, qs, qsa } from "./ui/dom";
import { FitnessView } from "./ui/fitness";
import { MusicView } from "./ui/music";
import { openModal } from "./ui/modal";
import { RewearView } from "./ui/rewear";
import { Stage } from "./ui/stage";

const VALID_VIEWS: ViewId[] = ["rewear", "music", "fitness"];

class App {
  readonly stage = new Stage();
  readonly audio = new AudioEngine();
  readonly views: Record<ViewId, View>;
  private readonly cart: Cart;
  private current: ViewId = "rewear";
  private started = false;

  constructor() {
    this.views = {
      rewear: new RewearView({
        stage: this.stage,
        audio: this.audio,
        toast: (message, tone) => this.toast(message, tone),
        openCart: () => this.cart.open(),
        showView: (id) => this.showView(id),
        openVibes: () => this.openVibes(),
      }),
      music: new MusicView(
        {
          stage: this.stage,
          audio: this.audio,
          toast: (message, tone) => this.toast(message, tone),
          openCart: () => this.cart.open(),
          showView: (id) => this.showView(id),
          openVibes: () => this.openVibes(),
        },
        this.audio,
      ),
      fitness: new FitnessView(
        {
          stage: this.stage,
          audio: this.audio,
          toast: (message, tone) => this.toast(message, tone),
          openCart: () => this.cart.open(),
          showView: (id) => this.showView(id),
          openVibes: () => this.openVibes(),
        },
        this.audio,
      ),
    };

    this.cart = new Cart(
      (order) => this.showOrder(order),
      (id) => this.showView(id),
    );
  }

  mount(): void {
    const root = qs(document, "#view-root");
    this.cart.mount(qs(document, "#cart-root"));

    this.stage.vision.onFrame = (packet) => {
      this.views[this.current].onFrame(packet);
    };

    qsa<HTMLElement>(document, "[data-view]").forEach((button) => {
      button.addEventListener("click", () => {
        const id = button.dataset.view as ViewId;
        if (VALID_VIEWS.includes(id)) this.showView(id);
      });
    });
    qsa<HTMLElement>(document, "[data-cart]").forEach((button) =>
      button.addEventListener("click", () => this.cart.open()),
    );
    qsa<HTMLElement>(document, "[data-help]").forEach((button) =>
      button.addEventListener("click", () => this.openHelp()),
    );
    qs(document, "[data-home]").addEventListener("click", (event) => {
      event.preventDefault();
      this.showView("rewear");
    });

    store.subscribe(() => this.updateBag());
    this.updateBag();

    const hash = location.hash.replace("#", "") as ViewId;
    const initial = VALID_VIEWS.includes(hash) ? hash : "rewear";
    this.current = initial;
    this.views[initial].show(root);
    this.started = true;
    this.updateNav();

    window.addEventListener("hashchange", () => {
      const next = location.hash.replace("#", "") as ViewId;
      if (VALID_VIEWS.includes(next)) this.showView(next);
    });

    window.setTimeout(() => {
      if (store.snapshot.taste.seeds.length === 0 && store.snapshot.decisions.length === 0) {
        this.openVibes();
      }
    }, 700);
  }

  private showView(id: ViewId): void {
    if (!this.started) return;
    if (id === this.current) {
      this.cart.close();
      return;
    }
    this.views[this.current].hide();
    this.current = id;
    this.views[id].show(qs(document, "#view-root"));
    this.updateNav();
    history.replaceState(null, "", `#${id}`);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  private updateNav(): void {
    qsa<HTMLElement>(document, "[data-view]").forEach((button) => {
      const active = button.dataset.view === this.current;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-current", active ? "page" : "false");
    });
  }

  private updateBag(): void {
    const count = store.cartCount;
    qsa<HTMLElement>(document, "[data-bag]").forEach((el) => {
      el.textContent = String(count);
      el.classList.toggle("has-items", count > 0);
    });
  }

  private toast(message: string, tone: "default" | "acid" | "warn" = "default"): void {
    const host = qs(document, "#toasts");
    while (host.children.length >= 3) host.firstElementChild?.remove();
    const node = document.createElement("div");
    node.className = "toast";
    node.dataset.tone = tone;
    node.textContent = message;
    host.appendChild(node);
    window.setTimeout(() => {
      node.classList.add("is-out");
      window.setTimeout(() => node.remove(), 260);
    }, 2500);
  }

  private openVibes(): void {
    const seeds = new Set(store.snapshot.taste.seeds);
    const modal = openModal({
      kicker: "Start here",
      title: "Pick a vibe.<br><em>Not a size.</em>",
      body: `
        <p class="modal-lede">The rail starts somewhere. Choose a few instincts and the deck re-ranks around them — you can change this whenever.</p>
        <div class="vibe-grid">
          ${VIBES.map((vibe) => {
            const on = vibe.tags.some((tag) => seeds.has(tag));
            return `<button class="vibe ${on ? "is-on" : ""}" data-tags="${esc(vibe.tags.join("|"))}" type="button" aria-pressed="${on}">${esc(vibe.label)}</button>`;
          }).join("")}
        </div>
        <p class="tiny">Stored in this browser. No account, no tracking, no server.</p>
      `,
      actions: [
        {
          label: "Tune the rail",
          variant: "acid",
          onClick: () => {
            const picked = qsa<HTMLElement>(modal, ".vibe.is-on")
              .flatMap((chip) => (chip.dataset.tags ?? "").split("|"))
              .filter(Boolean);
            if (picked.length === 0) {
              this.toast("Pick at least one vibe — or skip.", "warn");
              return false;
            }
            store.setVibes(picked);
            this.views[this.current].rerank?.();
            this.toast("Vibe seeded. The deck just moved.", "acid");
          },
        },
        { label: "Skip for now", variant: "ghost" },
      ],
    });

    modal.addEventListener("click", (event) => {
      const chip = (event.target as HTMLElement).closest<HTMLElement>(".vibe");
      if (!chip) return;
      const on = chip.classList.toggle("is-on");
      chip.setAttribute("aria-pressed", String(on));
    });
  }

  private openHelp(): void {
    openModal({
      kicker: "Gesture guide",
      title: "Three gestures.<br><em>One store.</em>",
      wide: true,
      body: `
        <div class="help-grid">
          <div class="help-card">
            <div class="help-glyph">✋</div>
            <h3>Palm left / right</h3>
            <p>One open palm on the left side of frame = nope. On the right = like. Hold for about half a second; the bar fills as you commit.</p>
          </div>
          <div class="help-card help-acid">
            <div class="help-glyph">🤲</div>
            <h3>Two palms · 67</h3>
            <p>Both hands open, rocking in opposite directions — the 6…7. Hold the pose or keep it moving until the ring closes. That is a superlike: it counts double in the taste profile.</p>
          </div>
          <div class="help-card">
            <div class="help-glyph">🖐</div>
            <h3>Fingers · chords</h3>
            <p>Music mode reads 1–4 fingers as C, Am, F and G. Height on the left hand is volume, side-to-side is the filter, a flick is a strum, a fist fades out.</p>
          </div>
          <div class="help-card">
            <div class="help-glyph">💪</div>
            <h3>One arm · 67 BPM</h3>
            <p>Fitness mode counts curls and calls out drift, depth and tempo. Turn on the 67 BPM click and keep each rep on the beat.</p>
          </div>
        </div>
        <div class="help-keys">
          <span><kbd>←</kbd><kbd>→</kbd><kbd>↑</kbd> swipe</span>
          <span><kbd>1</kbd>–<kbd>5</kbd> chords</span>
          <span><kbd>space</kbd> strum</span>
          <span><kbd>m</kbd> mute</span>
          <span><kbd>c</kbd> bag</span>
          <span><kbd>esc</kbd> close</span>
        </div>
        <p class="tiny">Camera frames never leave this tab. The first time you enable the cam, the browser asks permission; models are served from this site, not a CDN.</p>
      `,
      actions: [
        {
          label: "Enable the 67 cam",
          variant: "acid",
          onClick: () => {
            void this.stage.enable();
          },
        },
        { label: "Got it", variant: "ghost" },
      ],
    });
  }

  private showOrder(order: Order): void {
    const lines = order.lines
      .map((line) => {
        const product = PRODUCT_MAP.get(line.id);
        if (!product) return "";
        return `<div class="order-line"><span>${esc(product.name)} × ${line.qty}</span><b>${money(product.price * line.qty)}</b></div>`;
      })
      .join("");

    openModal({
      kicker: "Order placed",
      title: `${order.number}`,
      body: `
        <div class="order">
          ${lines}
          <div class="order-line order-total"><span>Total</span><b>${money(order.total)}</b></div>
        </div>
        <p class="modal-lede">That was a real interaction and a fake transaction. No payment was taken, nothing ships — the bag, the taste profile and this receipt are all local to your browser.</p>
      `,
      actions: [{ label: "Back to the deck", variant: "acid" }]
    });
  }
}

const app = new App();
app.mount();

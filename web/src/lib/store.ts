import { PRODUCT_MAP } from "../data/catalog";
import {
  applySeeds,
  createTaste,
  learn,
  type Decision,
  type Taste,
  type Verdict,
} from "./recommender";

const STORAGE_KEY = "store67:v1";

export interface CartLine {
  id: string;
  qty: number;
}

export interface PersistedState {
  taste: Taste;
  decisions: Decision[];
  cart: CartLine[];
  seen: string[];
  createdAt: number;
  orderCount: number;
}

export interface Order {
  number: string;
  total: number;
  lines: CartLine[];
}

type Listener = () => void;

const EMPTY: PersistedState = {
  taste: createTaste(),
  decisions: [],
  cart: [],
  seen: [],
  createdAt: 0,
  orderCount: 0,
};

function freshState(): PersistedState {
  return { ...structuredClone(EMPTY), taste: createTaste(), createdAt: Date.now() };
}

function load(): PersistedState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return freshState();
    const parsed = JSON.parse(raw) as Partial<PersistedState>;
    return {
      taste: parsed.taste && typeof parsed.taste === "object"
        ? { ...createTaste(), ...parsed.taste }
        : createTaste(),
      decisions: Array.isArray(parsed.decisions) ? parsed.decisions : [],
      cart: Array.isArray(parsed.cart)
        ? parsed.cart.filter((line) => PRODUCT_MAP.has(line.id))
        : [],
      seen: Array.isArray(parsed.seen) ? parsed.seen : [],
      createdAt: typeof parsed.createdAt === "number" ? parsed.createdAt : Date.now(),
      orderCount: typeof parsed.orderCount === "number" ? parsed.orderCount : 0,
    };
  } catch {
    return freshState();
  }
}

function orderNumber(count: number): string {
  const stamp = Date.now().toString(36).slice(-4).toUpperCase();
  return `67-${String(count + 1).padStart(3, "0")}-${stamp}`;
}

class Store {
  private state: PersistedState = load();
  private listeners = new Set<Listener>();

  get snapshot(): PersistedState {
    return this.state;
  }

  get decided(): Set<string> {
    return new Set(this.state.decisions.map((decision) => decision.id));
  }

  get seen(): Set<string> {
    return new Set(this.state.seen);
  }

  get cartCount(): number {
    return this.state.cart.reduce((sum, line) => sum + line.qty, 0);
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private commit(next: PersistedState): void {
    this.state = next;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Private mode / quota — the session still works in memory.
    }
    this.listeners.forEach((listener) => listener());
  }

  record(productId: string, action: Verdict): void {
    const product = PRODUCT_MAP.get(productId);
    if (!product) return;
    this.commit({
      ...this.state,
      taste: learn(this.state.taste, product, action),
      decisions: [
        ...this.state.decisions.filter((decision) => decision.id !== productId),
        { id: productId, action, at: Date.now() },
      ],
    });
  }

  markSeen(productId: string): void {
    if (this.state.seen.includes(productId)) return;
    this.commit({ ...this.state, seen: [...this.state.seen, productId] });
  }

  seedVibes(tags: string[]): void {
    if (this.state.taste.seeds.length > 0) return;
    const taste = createTaste(tags);
    this.commit({ ...this.state, taste });
  }

  setVibes(tags: string[]): void {
    if (tags.length === 0) return;
    if (this.state.taste.seeds.length === 0) {
      this.commit({ ...this.state, taste: createTaste(tags) });
      return;
    }
    this.commit({ ...this.state, taste: applySeeds(this.state.taste, tags) });
  }

  resetTaste(): void {
    this.commit({
      ...this.state,
      taste: createTaste(),
      decisions: [],
      seen: [],
    });
  }

  addToCart(productId: string, qty = 1): void {
    if (!PRODUCT_MAP.has(productId)) return;
    const existing = this.state.cart.find((line) => line.id === productId);
    const cart = existing
      ? this.state.cart.map((line) =>
          line.id === productId ? { ...line, qty: Math.min(9, line.qty + qty) } : line,
        )
      : [...this.state.cart, { id: productId, qty }];
    this.commit({ ...this.state, cart });
  }

  setQty(productId: string, qty: number): void {
    const cart =
      qty <= 0
        ? this.state.cart.filter((line) => line.id !== productId)
        : this.state.cart.map((line) =>
            line.id === productId ? { ...line, qty: Math.min(9, qty) } : line,
          );
    this.commit({ ...this.state, cart });
  }

  clearCart(): void {
    this.commit({ ...this.state, cart: [] });
  }

  totals(): { subtotal: number; shipping: number; total: number } {
    const subtotal = this.state.cart.reduce((sum, line) => {
      const product = PRODUCT_MAP.get(line.id);
      return sum + (product ? product.price * line.qty : 0);
    }, 0);
    const shipping = subtotal === 0 || subtotal >= 120 ? 0 : 6;
    return { subtotal, shipping, total: subtotal + shipping };
  }

  checkout(): Order {
    const { total } = this.totals();
    const order: Order = {
      number: orderNumber(this.state.orderCount),
      total,
      lines: [...this.state.cart],
    };
    this.commit({
      ...this.state,
      cart: [],
      orderCount: this.state.orderCount + 1,
    });
    return order;
  }
}

export const store = new Store();

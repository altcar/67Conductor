import type { Product } from "../data/catalog";
import type { AudioEngine } from "../lib/audio";
import type { FramePacket } from "../lib/vision";
import type { Stage } from "./stage";

export type ViewId = "rewear" | "music" | "fitness";

export interface ViewContext {
  stage: Stage;
  audio: AudioEngine;
  toast: (message: string, tone?: "default" | "acid" | "warn") => void;
  openCart: () => void;
  showView: (id: ViewId) => void;
  openVibes: () => void;
}

export interface View {
  readonly id: ViewId;
  readonly label: string;
  show(slot: HTMLElement): void;
  hide(): void;
  onFrame(packet: FramePacket): void;
  rerank?(): void;
}

export interface RailOptions {
  title: string;
  eyebrow: string;
  products: { product: Product; why?: string }[];
  emptyText?: string;
}

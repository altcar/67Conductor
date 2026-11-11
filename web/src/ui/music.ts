import { productsFor } from "../data/catalog";
import type { Product } from "../data/catalog";
import { CHORDS, type AudioEngine } from "../lib/audio";
import { isFist, longFingerCount } from "../lib/gestures";
import { recommend } from "../lib/recommender";
import { store } from "../lib/store";
import type { FramePacket } from "../lib/vision";
import type { View, ViewContext } from "./context";
import { drawHand, fitOverlay } from "./draw";
import { qs, template } from "./dom";
import { renderRail } from "./product";

export class MusicView implements View {
  readonly id = "music" as const;
  readonly label = "Music";

  private root: HTMLElement | null = null;
  private spectrum: HTMLCanvasElement | null = null;
  private raf = 0;
  private active = false;
  private unsubscribe: (() => void) | null = null;
  private keyHandler: ((event: KeyboardEvent) => void) | null = null;

  private lastFingerCount = -1;
  private lastChordIndex = 0;
  private lastStrum = 0;
  private lastWrist: { y: number; t: number } | null = null;
  private lastUiUpdate = 0;
  private ducked = false;

  constructor(
    private readonly ctx: ViewContext,
    private readonly audio: AudioEngine,
  ) {}

  show(slot: HTMLElement): void {
    this.root = template(`
      <section class="view music" aria-label="Music">
        <div class="view-head">
          <div>
            <span class="eyebrow">02 — Music</span>
            <h1>Play the rack <em>with your hands.</em></h1>
          </div>
          <p class="lede">Fingers choose the chord. Height is volume. A flick is a strum. Nothing needs installing — the synth is Web Audio.</p>
        </div>

        <div class="music-grid">
          <aside class="panel chord-panel">
            <span class="eyebrow">Chord bank</span>
            <h2>Fingers → harmony</h2>
            <div class="chord-pads">
              ${CHORDS.map(
                (chord, index) => `
                <button class="chord-pad" data-chord-index="${index}" type="button">
                  <span class="pad-name">${chord.name}</span>
                  <span class="pad-fingers">${index + 1} ${index === 0 ? "finger" : "fingers"}</span>
                </button>`,
              ).join("")}
            </div>
            <ul class="steps">
              <li><b>Right hand</b><span>1–4 fingers pick C, Am, F, G. Side-to-side sweeps the filter.</span></li>
              <li><b>Left hand</b><span>Height sets volume. Keep it in frame.</span></li>
              <li><b>Flick down</b><span>Strum. A fist fades everything out.</span></li>
            </ul>
            <button class="btn btn-acid" data-sound type="button">Sound off — tap to start</button>
            <p class="tiny">Audio starts on your first tap. Headphones recommended.</p>
          </aside>

          <div class="music-center">
            <div class="readout">
              <span class="eyebrow">Now playing</span>
              <div class="chord-name" data-chord>—</div>
              <p class="chord-sub" data-sub>Raise 1–4 fingers on your right hand</p>
              <div class="meters">
                <div class="meter"><span>VOLUME</span><div class="meter-bar"><i data-vol></i></div><b data-vol-num>65%</b></div>
                <div class="meter"><span>FILTER</span><div class="meter-bar"><i data-filter></i></div><b data-filter-num>3.2k</b></div>
                <div class="meter"><span>HANDS</span><div class="meter-bar"><i data-hands></i></div><b data-hands-num>0</b></div>
              </div>
            </div>
            <canvas class="spectrum" data-spectrum aria-hidden="true"></canvas>
            <div class="music-tips">
              <span class="eyebrow">Fallback</span>
              <p>Keys <kbd>1</kbd>–<kbd>5</kbd> play a chord, <kbd>space</kbd> strums, <kbd>m</kbd> mutes. The pads work with a mouse.</p>
            </div>
          </div>

          <div class="stage-slot" data-stage></div>
        </div>

        <div data-kit></div>
      </section>
    `);

    slot.replaceChildren(this.root);
    this.spectrum = qs<HTMLCanvasElement>(this.root, "[data-spectrum]");

    const foot = template(`
      <div class="music-foot">
        <span class="mono-label">RIGHT HAND</span><b data-foot-count>—</b>
        <span class="mono-label">CHORD</span><b data-foot-chord>—</b>
        <span class="mono-label">STRUM</span><b data-foot-strum>—</b>
      </div>
    `);

    this.ctx.stage.attach(qs(this.root, "[data-stage]"), {
      mode: "hands",
      title: "67 CAM / SYNTH",
      hint: "Fingers pick the chord · height is volume",
      foot,
    });

    this.root.addEventListener("click", (event) => this.onClick(event));
    this.unsubscribe = store.subscribe(() => this.renderKit());
    this.renderKit();

    this.keyHandler = (event) => this.onKey(event);
    window.addEventListener("keydown", this.keyHandler);
    this.active = true;
    this.visualise();
    this.updateSoundButton();
    this.lastUiUpdate = 0;
    this.updateMeters(0, null, 3200);
  }

  hide(): void {
    this.active = false;
    cancelAnimationFrame(this.raf);
    if (this.keyHandler) window.removeEventListener("keydown", this.keyHandler);
    this.keyHandler = null;
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.audio.suspend();
    this.ctx.stage.detach();
    this.root = null;
    this.spectrum = null;
  }

  onFrame(packet: FramePacket): void {
    if (!this.root || !this.active) return;
    const ctx = this.ctx.stage.context;
    const { width, height } = fitOverlay(this.ctx.stage.canvasEl, this.ctx.stage.vision.video);
    ctx.clearRect(0, 0, width, height);

    const hands = [...packet.hands].sort(
      (a, b) => (a.points[0]?.x ?? 0) - (b.points[0]?.x ?? 0),
    );
    const chordHand = hands[hands.length - 1] ?? null;
    const volumeHand = hands.length > 1 ? hands[0] : null;

    for (const hand of hands) {
      const isChord = hand === chordHand;
      drawHand(ctx, width, height, hand, {
        color: isChord ? "#d7ff3e" : "#8bb4ff",
        label: isChord ? `${longFingerCount(hand.points)}F / CHORD` : "VOLUME",
      });
    }

    if (!chordHand) {
      this.updateMeters(hands.length, null, null);
      return;
    }

    const fingers = longFingerCount(chordHand.points);
    const fist = isFist(chordHand.points);

    if (fist && !this.ducked) {
      this.ducked = true;
      this.audio.duck();
    } else if (!fist && this.ducked) {
      this.ducked = false;
      this.audio.unduck();
    }

    if (fingers > 0 && fingers !== this.lastFingerCount) {
      this.lastFingerCount = fingers;
      const index = Math.min(CHORDS.length - 1, fingers - 1);
      this.lastChordIndex = index;
      if (this.audio.ready) {
        this.audio.playChord(index, "down", 0.9);
        this.pulseStrum();
      }
      this.setChordReadout(index);
    } else if (fist) {
      this.lastFingerCount = 0;
      this.setChordReadout(-1);
    }

    // A quick downward flick (or upward) retriggers the chord.
    const wrist = chordHand.points[0];
    const now = packet.t;
    if (this.lastWrist) {
      const dy = wrist.y - this.lastWrist.y;
      const dt = now - this.lastWrist.t;
      if (dt > 0 && dt < 220 && now - this.lastStrum > 320) {
        if (dy > 0.045) {
          this.lastStrum = now;
          if (this.audio.ready && this.lastChordIndex >= 0) this.audio.playChord(this.lastChordIndex, "down", 1);
          this.pulseStrum();
        } else if (dy < -0.05) {
          this.lastStrum = now;
          if (this.audio.ready && this.lastChordIndex >= 0) this.audio.playChord(this.lastChordIndex, "up", 0.8);
          this.pulseStrum();
        }
      }
    }
    this.lastWrist = { y: wrist.y, t: now };

    // Volume from the left hand height, filter from the chord hand position.
    let volume: number | null = null;
    if (volumeHand) {
      const y = volumeHand.points[0].y;
      volume = Math.max(0.08, Math.min(1, (0.88 - y) / 0.66));
      if (this.audio.ready && !this.ducked) this.audio.setVolume(0.2 + volume * 0.75);
    }
    const cutoff = 380 * (5800 / 380) ** Math.max(0, Math.min(1, wrist.x));
    if (this.audio.ready) this.audio.setFilterHz(cutoff);

    this.updateMeters(hands.length, volume, cutoff);
    this.updateFoot(fingers, this.lastChordIndex, now - this.lastStrum);
  }

  private onClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    const pad = target.closest<HTMLElement>("[data-chord-index]");
    if (pad) {
      const index = Number(pad.dataset.chordIndex);
      void this.ensureAudio().then((ok) => {
        if (!ok) return;
        this.lastChordIndex = index;
        this.audio.playChord(index, "down", 1);
        this.setChordReadout(index);
        this.pulseStrum();
      });
      return;
    }
    if (target.closest("[data-sound]")) {
      void this.toggleSound();
      return;
    }
  }

  private onKey(event: KeyboardEvent): void {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (document.querySelector('.modal-layer[data-open="true"], .cart-layer[data-open="true"]')) return;
    const tag = (event.target as HTMLElement | null)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (/^[1-5]$/.test(event.key)) {
      const index = Number(event.key) - 1;
      void this.ensureAudio().then((ok) => {
        if (!ok) return;
        this.lastChordIndex = index;
        this.audio.playChord(index, "down", 1);
        this.setChordReadout(index);
        this.pulseStrum();
      });
    } else if (event.code === "Space") {
      event.preventDefault();
      void this.ensureAudio().then((ok) => {
        if (ok && this.lastChordIndex >= 0) {
          this.audio.playChord(this.lastChordIndex, "down", 1);
          this.pulseStrum();
        }
      });
    } else if (event.key.toLowerCase() === "m") {
      if (this.ducked) {
        this.ducked = false;
        this.audio.unduck();
      } else {
        this.ducked = true;
        this.audio.duck();
      }
    }
  }

  private async ensureAudio(): Promise<boolean> {
    const ok = await this.audio.start();
    this.updateSoundButton();
    return ok;
  }

  private async toggleSound(): Promise<void> {
    if (this.audio.ready) {
      this.audio.suspend();
      this.ctx.toast("Sound suspended.", "default");
    } else {
      const ok = await this.audio.start();
      if (ok) {
        this.audio.playChord(Math.max(0, this.lastChordIndex), "down", 0.9);
        this.pulseStrum();
      }
      this.ctx.toast(ok ? "Sound on. Play the room." : "This browser blocked audio.", ok ? "acid" : "warn");
    }
    this.updateSoundButton();
  }

  private updateSoundButton(): void {
    const button = this.root?.querySelector("[data-sound]");
    if (!button) return;
    button.textContent = this.audio.ready ? "Sound on" : "Sound off — tap to start";
    button.classList.toggle("is-on", this.audio.ready);
  }

  private setChordReadout(index: number): void {
    const name = this.root?.querySelector("[data-chord]");
    const sub = this.root?.querySelector("[data-sub]");
    if (!name || !sub) return;
    if (index < 0) {
      name.textContent = "·";
      sub.textContent = "Fist: faded out. Open the hand to come back.";
      return;
    }
    const chord = CHORDS[index];
    name.textContent = chord.name;
    sub.textContent = `${chord.notes.length} notes · ${index + 1} finger${index === 0 ? "" : "s"} held up`;
  }

  private pulseStrum(): void {
    const foot = this.root?.querySelector("[data-foot-strum]");
    if (!foot) return;
    foot.textContent = "hit";
    window.setTimeout(() => {
      if (foot.isConnected) foot.textContent = "—";
    }, 240);
  }

  private updateMeters(handCount: number, volume: number | null, cutoff: number | null): void {
    const now = performance.now();
    if (now - this.lastUiUpdate < 80) return;
    this.lastUiUpdate = now;

    const vol = volume ?? (this.audio.volume * 100) / 100;
    const volBar = this.root?.querySelector("[data-vol]") as HTMLElement | null;
    if (volBar) volBar.style.transform = `scaleX(${Math.max(0.04, vol)})`;
    const volNum = this.root?.querySelector("[data-vol-num]");
    if (volNum) volNum.textContent = `${Math.round(vol * 100)}%`;

    const filter = cutoff ?? 3200;
    const filterBar = this.root?.querySelector("[data-filter]") as HTMLElement | null;
    if (filterBar) {
      const normalized = Math.log(filter / 380) / Math.log(5800 / 380);
      filterBar.style.transform = `scaleX(${Math.max(0.03, Math.min(1, normalized))})`;
    }
    const filterNum = this.root?.querySelector("[data-filter-num]");
    if (filterNum) filterNum.textContent = filter >= 1000 ? `${(filter / 1000).toFixed(1)}k` : `${Math.round(filter)}`;

    const handsBar = this.root?.querySelector("[data-hands]") as HTMLElement | null;
    if (handsBar) handsBar.style.transform = `scaleX(${handCount / 2})`;
    const handsNum = this.root?.querySelector("[data-hands-num]");
    if (handsNum) handsNum.textContent = String(handCount);
  }

  private updateFoot(fingers: number, chordIndex: number, sinceStrum: number): void {
    const count = this.root?.querySelector("[data-foot-count]");
    if (count) count.textContent = `${fingers} finger${fingers === 1 ? "" : "s"}`;
    const chord = this.root?.querySelector("[data-foot-chord]");
    if (chord) chord.textContent = chordIndex >= 0 ? CHORDS[chordIndex].name : "—";
    const strum = this.root?.querySelector("[data-foot-strum]");
    if (strum && sinceStrum > 300) strum.textContent = "ready";
  }

  private renderKit(): void {
    if (!this.root) return;
    const host = this.root.querySelector("[data-kit]");
    if (!host) return;
    const taste = store.snapshot.taste;
    const picks = recommend(productsFor("music"), taste, { limit: 3 });
    host.replaceChildren(
      renderRail(
        {
          eyebrow: "The kit",
          title: "What this setup sounds like",
          products: picks.map((pick) => ({ product: pick.product, why: pick.why })),
        },
        (product: Product) => {
          store.addToCart(product.id);
          this.ctx.toast(`Added ${product.name}.`, "acid");
        },
      ),
    );
  }

  private visualise = (): void => {
    if (!this.active) return;
    this.raf = requestAnimationFrame(this.visualise);
    const canvas = this.spectrum;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(320, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(120, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    const data = this.audio.spectrumData();
    const bars = 42;
    const gap = 3 * dpr;
    const barWidth = (width - gap * (bars - 1)) / bars;
    for (let i = 0; i < bars; i += 1) {
      const value = data ? data[Math.floor((i / bars) ** 1.4 * data.length)] / 255 : 0;
      const phase = performance.now() / 1100;
      const idle = this.audio.ready
        ? 0.045 + 0.02 * Math.sin(i * 0.55 + phase)
        : 0.035 + 0.028 * Math.sin(i * 0.55 + phase) + 0.012 * Math.sin(i * 1.7 - phase * 0.6);
      const level = Math.max(idle, value);
      const barHeight = Math.max(3, level * height * 0.92);
      ctx.fillStyle = i % 7 === 0 ? "#ff6a3d" : "#d7ff3e";
      ctx.globalAlpha = this.audio.ready ? 0.92 : 0.4;
      ctx.fillRect(i * (barWidth + gap), height - barHeight, barWidth, barHeight);
    }
    ctx.globalAlpha = 1;
  };
}


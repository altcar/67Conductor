import { productsFor } from "../data/catalog";
import type { AudioEngine } from "../lib/audio";
import { CurlCounter, type ArmPoints } from "../lib/gestures";
import { recommend } from "../lib/recommender";
import { store } from "../lib/store";
import { poseArm, type FramePacket } from "../lib/vision";
import type { View, ViewContext } from "./context";
import { drawPose, fitOverlay } from "./draw";
import { qs, qsa, template } from "./dom";
import { renderRail } from "./product";

export class FitnessView implements View {
  readonly id = "fitness" as const;
  readonly label = "Fitness";

  private root: HTMLElement | null = null;
  private counter = new CurlCounter();
  private side: "left" | "right" = "right";
  private scores: number[] = [];
  private active = false;
  private unsubscribe: (() => void) | null = null;
  private demoFrame = 0;
  private lastUi = 0;
  private lastArmAt = 0;
  private lastNotes: string[] = [];
  private metronomeOn = false;

  constructor(
    private readonly ctx: ViewContext,
    private readonly audio: AudioEngine,
  ) {}

  show(slot: HTMLElement): void {
    this.root = template(`
      <section class="view fitness" aria-label="Fitness">
        <div class="view-head">
          <div>
            <span class="eyebrow">03 — Fitness</span>
            <h1>Curl at <em>67 BPM.</em></h1>
          </div>
          <p class="lede">The pose tracker counts reps and calls out your form in real time. Same camera, same tab. No wearable, no app.</p>
        </div>

        <div class="fitness-grid">
          <div class="stage-slot stage-big" data-stage></div>

          <aside class="panel coach">
            <div class="rep-block">
              <span class="eyebrow">Reps</span>
              <div class="rep-count" data-reps aria-live="polite">0</div>
              <p class="rep-sub" data-phase>Extend your arm to start</p>
            </div>

            <div class="form-block">
              <div class="form-head">
                <span class="mono-label">FORM</span>
                <b data-form>—</b>
              </div>
              <div class="form-track"><i data-form-bar></i></div>
              <ul class="notes" data-notes><li class="muted">Waiting for a full arm.</li></ul>
            </div>

            <div class="coach-controls">
              <div class="seg" role="group" aria-label="Working arm">
                <button type="button" data-arm="left">Left arm</button>
                <button type="button" data-arm="right" class="is-active">Right arm</button>
              </div>
              <button class="btn btn-ghost btn-sm" data-metronome type="button">67 BPM click</button>
              <button class="btn btn-ghost btn-sm" data-demo type="button">Demo a rep</button>
              <button class="btn btn-ghost btn-sm" data-reset type="button">Reset</button>
            </div>
            <p class="tiny">Angle, tempo and elbow drift are computed in this tab. Video is never recorded.</p>
          </aside>
        </div>

        <div class="rep-log" data-log></div>
        <div data-gear></div>
      </section>
    `);

    slot.replaceChildren(this.root);

    const foot = template(`
      <div class="fitness-foot">
        <span class="mono-label">ELBOW</span><b data-angle>—</b>
        <span class="mono-label">ARM</span><b data-arm-label>RIGHT</b>
        <span class="mono-label">TEMPO</span><b>67 BPM</b>
      </div>
    `);

    this.ctx.stage.attach(qs(this.root, "[data-stage]"), {
      mode: "pose",
      title: "67 CAM / FORM",
      hint: "Stand back so your shoulder, elbow and wrist are visible",
      foot,
    });
    this.lastArmAt = 0;

    this.root.addEventListener("click", (event) => this.onClick(event));
    this.unsubscribe = store.subscribe(() => this.renderGear());
    this.renderGear();
    this.renderLog();
    this.active = true;
    this.counter.reset();
    this.scores = [];
    this.updateArmButtons();
  }

  hide(): void {
    this.active = false;
    cancelAnimationFrame(this.demoFrame);
    this.stopMetronome();
    this.audio.suspend();
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.ctx.stage.detach();
    this.root = null;
  }

  onFrame(packet: FramePacket): void {
    if (!this.root || !this.active) return;
    const ctx = this.ctx.stage.context;
    const { width, height } = fitOverlay(this.ctx.stage.canvasEl, this.ctx.stage.vision.video);
    ctx.clearRect(0, 0, width, height);
    drawPose(ctx, width, height, packet.pose, this.side);

    const arm = poseArm(packet.pose, this.side);
    if (!arm) {
      if (this.lastArmAt > 0 && packet.t - this.lastArmAt > 1400) {
        this.lastArmAt = 0;
        const phase = this.root.querySelector("[data-phase]");
        if (phase) phase.textContent = "Step back — shoulder, elbow and wrist need to be in frame";
      }
      return;
    }
    this.lastArmAt = packet.t;
    const update = this.counter.update(arm, packet.t);
    this.applyUpdate(update.angle, update.phase, update.liveNotes, update.rep);
  }

  private onClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    const arm = target.closest<HTMLElement>("[data-arm]")?.dataset.arm;
    if (arm === "left" || arm === "right") {
      this.side = arm;
      this.counter.reset();
      this.scores = [];
      this.updateArmButtons();
      this.renderLog();
      this.setUI({ reps: 0, form: null, notes: [], phase: "Extend your arm to start" });
      this.ctx.toast(`Tracking the ${arm} arm.`, "default");
      return;
    }
    if (target.closest("[data-demo]")) {
      this.demoRep();
      return;
    }
    if (target.closest("[data-metronome]")) {
      void this.toggleMetronome();
      return;
    }
    if (target.closest("[data-reset]")) {
      this.counter.reset();
      this.scores = [];
      this.renderLog();
      this.setUI({ reps: 0, form: null, notes: [], phase: "Extend your arm to start" });
      const list = this.root?.querySelector("[data-notes]");
      if (list) list.innerHTML = `<li class="muted">Waiting for a full arm.</li>`;
      this.ctx.toast("Rep counter reset.", "default");
      return;
    }
  }

  private applyUpdate(
    angle: number,
    phase: "extended" | "curled" | "moving",
    notes: string[],
    rep: { rep: number; score: number; notes: string[]; durationMs: number } | null,
  ): void {
    const now = performance.now();
    if (now - this.lastUi > 90 && this.root) {
      this.lastUi = now;
      const angleEl = this.root.querySelector("[data-angle]");
      if (angleEl) angleEl.textContent = angle > 0 ? `${Math.round(angle)}°` : "—";
      const phaseEl = this.root.querySelector("[data-phase]");
      if (phaseEl) {
        phaseEl.textContent =
          angle === 0
            ? "Step back — shoulder, elbow and wrist need to be in frame"
            : phase === "curled"
              ? "Squeeze at the top"
              : phase === "moving"
                ? "Lower under control"
                : "Ready — curl up";
      }
    }

    if (notes.join("|") !== this.lastNotes.join("|")) {
      this.lastNotes = notes;
      this.renderNotes(notes);
    }

    if (rep) {
      this.scores.push(rep.score);
      const average = Math.round(this.scores.reduce((sum, value) => sum + value, 0) / this.scores.length);
      this.setUI({ reps: rep.rep, form: average, notes: rep.notes, phase: null });
      this.renderLog();
      if (this.audio.ready) this.audio.click(rep.score >= 85);
    }
  }

  private setUI(patch: {
    reps: number;
    form: number | null;
    notes: string[];
    phase: string | null;
  }): void {
    if (!this.root) return;
    const reps = this.root.querySelector("[data-reps]");
    if (reps) reps.textContent = String(patch.reps);
    if (patch.phase !== null) {
      const phase = this.root.querySelector("[data-phase]");
      if (phase) phase.textContent = patch.phase;
    }
    const form = this.root.querySelector("[data-form]");
    if (form) form.textContent = patch.form === null ? "—" : `${patch.form}`;
    const bar = this.root.querySelector("[data-form-bar]") as HTMLElement | null;
    if (bar) bar.style.transform = `scaleX(${patch.form === null ? 0 : patch.form / 100})`;
    this.renderNotes(patch.notes);
  }

  private renderNotes(notes: string[]): void {
    if (!this.root) return;
    const list = this.root.querySelector("[data-notes]");
    if (!list) return;
    if (notes.length === 0) {
      list.innerHTML = `<li class="ok">Form looks clean from here.</li>`;
      return;
    }
    list.innerHTML = notes.map((note) => `<li>${note}</li>`).join("");
  }

  private renderLog(): void {
    if (!this.root) return;
    const host = this.root.querySelector("[data-log]");
    if (!host) return;
    if (this.scores.length === 0) {
      host.innerHTML = `<span class="eyebrow">Rep log</span><p class="muted small">No reps yet. Do one, or hit “Demo a rep”.</p>`;
      return;
    }
    const average = Math.round(this.scores.reduce((sum, value) => sum + value, 0) / this.scores.length);
    const recent = this.scores.slice(-8);
    host.innerHTML = `
      <span class="eyebrow">Rep log</span>
      <div class="log-strip">
        ${recent
          .map(
            (score, index) => `
          <div class="log-rep">
            <span>${String(this.scores.length - recent.length + index + 1).padStart(2, "0")}</span>
            <b>${score}</b>
            <i style="--v:${score / 100}"></i>
          </div>`,
          )
          .join("")}
        <div class="log-avg"><span>AVG</span><b>${average}</b></div>
      </div>
    `;
  }

  private updateArmButtons(): void {
    if (!this.root) return;
    qsa<HTMLElement>(this.root, "[data-arm]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.arm === this.side);
    });
    const label = this.root.querySelector("[data-arm-label]");
    if (label) label.textContent = this.side.toUpperCase();
  }

  private toggleMetronome(): Promise<void> {
    return (async () => {
      if (this.metronomeOn) {
        this.stopMetronome();
        return;
      }
      const ok = await this.audio.start();
      if (!ok) {
        this.ctx.toast("This browser blocked audio.", "warn");
        return;
      }
      this.audio.setVolume(0.35);
      this.audio.startMetronome(67);
      this.metronomeOn = true;
      const button = this.root?.querySelector("[data-metronome]");
      button?.classList.add("is-on");
      if (button) button.textContent = "67 BPM click — on";
      this.ctx.toast("67 BPM. Let each rep land on the click.", "acid");
    })();
  }

  private stopMetronome(): void {
    if (!this.metronomeOn) return;
    this.metronomeOn = false;
    this.audio.stopMetronome();
    const button = this.root?.querySelector("[data-metronome]");
    button?.classList.remove("is-on");
    if (button) button.textContent = "67 BPM click";
  }

  private demoRep(): void {
    if (!this.root) return;
    const start = performance.now();
    const duration = 2100;
    const step = () => {
      if (!this.active || !this.root) return;
      const elapsed = performance.now() - start;
      if (elapsed > duration) {
        const arm = this.demoArm(180);
        const update = this.counter.update(arm, performance.now());
        this.applyUpdate(update.angle, update.phase, update.liveNotes, update.rep);
        return;
      }
      const t = elapsed / duration;
      const angle =
        t < 0.38
          ? smooth(180, 42, t / 0.38)
          : t < 0.58
            ? 42 + Math.sin((t - 0.38) * 40) * 3
            : smooth(42, 178, (t - 0.58) / 0.42);
      const arm = this.demoArm(angle);
      const update = this.counter.update(arm, performance.now());
      this.applyUpdate(update.angle, update.phase, update.liveNotes, update.rep);
      this.demoFrame = requestAnimationFrame(step);
    };
    this.demoFrame = requestAnimationFrame(step);
    this.ctx.toast("Demo rep — synthetic arm, real counter.", "default");
  }

  private demoArm(angle: number): ArmPoints {
    const shoulder = { x: 0.56, y: 0.38 };
    const elbow = { x: 0.58, y: 0.55 };
    const vector = { x: shoulder.x - elbow.x, y: shoulder.y - elbow.y };
    const radians = ((180 - angle) * Math.PI) / 180;
    const wrist = {
      x: elbow.x + vector.x * Math.cos(radians) - vector.y * Math.sin(radians),
      y: elbow.y + vector.x * Math.sin(radians) + vector.y * Math.cos(radians),
    };
    return { shoulder, elbow, wrist, hip: { x: 0.55, y: 0.62 } };
  }

  private renderGear(): void {
    if (!this.root) return;
    const host = this.root.querySelector("[data-gear]");
    if (!host) return;
    const picks = recommend(productsFor("fitness"), store.snapshot.taste, { limit: 3 });
    host.replaceChildren(
      renderRail(
        {
          eyebrow: "The rack",
          title: "Gear that matches the session",
          products: picks.map((pick) => ({ product: pick.product, why: pick.why })),
        },
        (product) => {
          store.addToCart(product.id);
          this.ctx.toast(`Added ${product.name}.`, "acid");
        },
      ),
    );
  }
}

function smooth(from: number, to: number, t: number): number {
  const clamped = Math.max(0, Math.min(1, t));
  const eased = clamped * clamped * (3 - 2 * clamped);
  return from + (to - from) * eased;
}

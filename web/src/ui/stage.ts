import { VisionEngine, type VisionMode, type VisionStatus } from "../lib/vision";
import { icon, template } from "./dom";

export interface StageConfig {
  mode: Exclude<VisionMode, "off">;
  title: string;
  hint: string;
  foot?: HTMLElement;
  cta?: string;
}

/**
 * One camera panel, shared by every mode. It is re-parented into whichever
 * view is active so the stream and the models stay alive while switching tabs.
 */
export class Stage {
  readonly el: HTMLElement;
  readonly vision: VisionEngine;
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private statusEl: HTMLElement;
  private hintEl: HTMLElement;
  private footEl: HTMLElement;
  private ctaEl: HTMLButtonElement;
  private titleEl: HTMLElement;
  private stopBtn: HTMLButtonElement;
  private mode: VisionMode = "off";
  private starting = false;

  constructor() {
    this.el = template(`
      <section class="stage" data-state="off" aria-label="67 camera">
        <div class="stage-head">
          <span class="stage-title">67 CAM</span>
          <span class="stage-status"><i></i><span data-status>off</span></span>
        </div>
        <div class="stage-frame">
          <video data-video playsinline muted autoplay></video>
          <canvas data-canvas aria-hidden="true"></canvas>
          <div class="stage-hint" data-hint>Point the camera at your hands</div>
          <div class="stage-cta" data-cta-wrap>
            <div class="stage-cta-inner">
              <span class="eyebrow">On-device vision</span>
              <p data-cta-copy>Your camera feed never leaves this tab. MediaPipe runs locally.</p>
              <button class="btn btn-acid" data-enable type="button">${icon("camera")} Enable the 67 cam</button>
              <span class="tiny">Or keep browsing with mouse + keys</span>
            </div>
          </div>
          <div class="stage-reticle" aria-hidden="true"><span>6</span><span>7</span></div>
          <div class="stage-flash" aria-hidden="true"></div>
        </div>
        <div class="stage-foot" data-foot>
          <span class="muted small">Camera stays local · no uploads</span>
        </div>
      </section>
    `);

    this.video = this.el.querySelector("[data-video]") as HTMLVideoElement;
    this.canvas = this.el.querySelector("[data-canvas]") as HTMLCanvasElement;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D is not available");
    this.ctx = ctx;
    this.statusEl = this.el.querySelector("[data-status]") as HTMLElement;
    this.hintEl = this.el.querySelector("[data-hint]") as HTMLElement;
    this.footEl = this.el.querySelector("[data-foot]") as HTMLElement;
    this.ctaEl = this.el.querySelector("[data-enable]") as HTMLButtonElement;
    this.titleEl = this.el.querySelector(".stage-title") as HTMLElement;
    this.stopBtn = template(`<button class="linkbtn" type="button" data-stop>Stop camera</button>`) as HTMLButtonElement;
    this.stopBtn.addEventListener("click", () => this.stop());

    this.vision = new VisionEngine(this.video);
    this.vision.onStatus = (status) => this.renderStatus(status);
    this.ctaEl.addEventListener("click", () => void this.enable());
    this.renderStatus({ camera: "idle", model: "idle" });
  }

  get context(): CanvasRenderingContext2D {
    return this.ctx;
  }

  get canvasEl(): HTMLCanvasElement {
    return this.canvas;
  }

  attach(slot: HTMLElement, config: StageConfig): void {
    this.mode = config.mode;
    this.titleEl.textContent = config.title;
    this.hintEl.textContent = config.hint;
    this.footEl.replaceChildren(config.foot ?? document.createElement("span"));
    if (config.foot) {
      const stop = this.stopBtn.cloneNode(true) as HTMLButtonElement;
      stop.addEventListener("click", () => this.stop());
      config.foot.appendChild(stop);
    }
    slot.appendChild(this.el);
    this.el.dataset.state = "active";

    if (this.vision.isCameraOn) {
      void this.vision.setMode(this.mode);
    } else {
      this.renderStatus({ camera: "idle", model: "idle" });
    }
  }

  detach(): void {
    this.mode = "off";
    void this.vision.setMode("off");
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  async enable(): Promise<void> {
    if (this.starting) return;
    this.starting = true;
    this.ctaEl.disabled = true;
    const ok = await this.vision.enableCamera();
    this.starting = false;
    this.ctaEl.disabled = false;
    if (ok && this.mode !== "off") await this.vision.setMode(this.mode);
  }

  setHint(text: string): void {
    this.hintEl.textContent = text;
  }

  setFoot(node: HTMLElement): void {
    this.footEl.replaceChildren(node);
    const stop = this.stopBtn.cloneNode(true) as HTMLButtonElement;
    stop.addEventListener("click", () => this.stop());
    this.footEl.appendChild(stop);
  }

  flash(tone: "like" | "super" | "nope"): void {
    const flash = this.el.querySelector(".stage-flash") as HTMLElement;
    flash.dataset.tone = tone;
    flash.classList.remove("flash");
    void flash.offsetWidth;
    flash.classList.add("flash");
  }

  stop(): void {
    this.vision.stopCamera();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private renderStatus(status: VisionStatus): void {
    const cameraOn = status.camera === "on";
    const showCta = !cameraOn;
    (this.el.querySelector("[data-cta-wrap]") as HTMLElement).toggleAttribute("hidden", !showCta);

    let label = "off";
    if (status.camera === "requesting") label = "asking…";
    else if (status.camera === "denied") label = "blocked";
    else if (status.camera === "error") label = "unavailable";
    else if (status.camera === "on") {
      if (status.model === "loading") label = "loading…";
      else if (status.model === "ready") label = this.mode === "pose" ? "form ready" : "67 ready";
      else if (status.model === "error") label = "error";
      else label = "on";
    }
    this.statusEl.textContent = label;
    this.el.dataset.state = status.camera === "on" ? (status.model === "error" ? "error" : "on") : status.camera;

    if (status.message) this.hintEl.textContent = status.message;
    const copy = this.el.querySelector("[data-cta-copy]") as HTMLElement;
    if (status.message && (status.camera === "denied" || status.camera === "error")) {
      copy.textContent = status.message;
      this.ctaEl.innerHTML = `${icon("camera")} Try the camera again`;
    }
    this.ctaEl.disabled = status.camera === "requesting";
  }
}

import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { LandmarkSmoother, type Hand, type Point } from "./gestures";

type WasmFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

export type VisionMode = "off" | "hands" | "pose";
export type CameraState = "idle" | "requesting" | "on" | "denied" | "error";
export type ModelState = "idle" | "loading" | "ready" | "error";

export interface VisionStatus {
  camera: CameraState;
  model: ModelState;
  message?: string;
}

export interface PosePoint extends Point {
  visibility?: number;
}

export interface FramePacket {
  hands: Hand[];
  pose: PosePoint[] | null;
  t: number;
}

const HAND_MODEL = "/models/hand_landmarker.task";
const POSE_MODEL = "/models/pose_landmarker_lite.task";

let filesetPromise: Promise<WasmFileset> | null = null;
function fileset(): Promise<WasmFileset> {
  if (!filesetPromise) {
    filesetPromise = FilesetResolver.forVisionTasks("/wasm").catch((error) => {
      filesetPromise = null;
      throw error;
    });
  }
  return filesetPromise;
}

export class VisionEngine {
  readonly video: HTMLVideoElement;
  mode: VisionMode = "off";
  onFrame: ((packet: FramePacket) => void) | null = null;
  onStatus: ((status: VisionStatus) => void) | null = null;

  private hands: HandLandmarker | null = null;
  private pose: PoseLandmarker | null = null;
  private handsPromise: Promise<HandLandmarker> | null = null;
  private posePromise: Promise<PoseLandmarker> | null = null;
  private stream: MediaStream | null = null;
  private raf = 0;
  private running = false;
  private lastDetect = 0;
  private lastVideoTime = -1;
  private lastTimestamp = 0;
  private handSmoother = new LandmarkSmoother(0.5);
  private poseSmoother = new LandmarkSmoother(0.4);
  private handSlots = 0;
  private status: VisionStatus = { camera: "idle", model: "idle" };

  constructor(video: HTMLVideoElement) {
    this.video = video;
    this.video.setAttribute("playsinline", "");
    this.video.muted = true;
  }

  private emit(patch: Partial<VisionStatus>): void {
    this.status = { ...this.status, ...patch };
    this.onStatus?.(this.status);
  }

  get isCameraOn(): boolean {
    return this.status.camera === "on";
  }

  async enableCamera(): Promise<boolean> {
    if (this.stream) return true;
    this.emit({ camera: "requesting", message: "Waiting for camera permission…" });
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not expose a camera API. Try Chrome, Edge or Safari.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play().catch(() => undefined);
      this.emit({ camera: "on", message: "Camera on" });
      return true;
    } catch (error) {
      const name = error instanceof DOMException ? error.name : "";
      const message =
        name === "NotAllowedError" || name === "SecurityError"
          ? "Camera permission was denied. You can still browse with mouse, keys and the 67 button."
          : name === "NotFoundError"
            ? "No camera found. Mouse and keyboard controls are still live."
            : error instanceof Error
              ? error.message
              : "Could not start the camera.";
      this.emit({ camera: name === "NotAllowedError" ? "denied" : "error", message });
      return false;
    }
  }

  stopCamera(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.video.srcObject = null;
    this.handSmoother.reset();
    this.poseSmoother.reset();
    this.emit({ camera: "idle", message: undefined });
  }

  private async loadHands(): Promise<HandLandmarker> {
    if (this.hands) return this.hands;
    if (this.handsPromise) return this.handsPromise;
    this.handsPromise = (async () => {
      const wasm = await fileset();
      const options = {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" as const },
        runningMode: "VIDEO" as const,
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      };
      try {
        const landmarker = await HandLandmarker.createFromOptions(wasm, options);
        this.hands = landmarker;
        return landmarker;
      } catch {
        const landmarker = await HandLandmarker.createFromOptions(wasm, {
          ...options,
          baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" },
        });
        this.hands = landmarker;
        return landmarker;
      }
    })();
    try {
      return await this.handsPromise;
    } finally {
      this.handsPromise = null;
    }
  }

  private async loadPose(): Promise<PoseLandmarker> {
    if (this.pose) return this.pose;
    if (this.posePromise) return this.posePromise;
    this.posePromise = (async () => {
      const wasm = await fileset();
      const options = {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" as const },
        runningMode: "VIDEO" as const,
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      };
      try {
        const landmarker = await PoseLandmarker.createFromOptions(wasm, options);
        this.pose = landmarker;
        return landmarker;
      } catch {
        const landmarker = await PoseLandmarker.createFromOptions(wasm, {
          ...options,
          baseOptions: { modelAssetPath: POSE_MODEL, delegate: "CPU" },
        });
        this.pose = landmarker;
        return landmarker;
      }
    })();
    try {
      return await this.posePromise;
    } finally {
      this.posePromise = null;
    }
  }

  async setMode(mode: VisionMode): Promise<void> {
    this.mode = mode;
    if (mode === "off") {
      this.running = false;
      cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.emit({ model: "idle" });
      return;
    }
    this.emit({ model: "loading", message: mode === "hands" ? "Loading hand tracker…" : "Loading pose tracker…" });
    try {
      if (mode === "hands") await this.loadHands();
      else await this.loadPose();
      this.emit({ model: "ready", message: mode === "hands" ? "67 cam ready" : "Form cam ready" });
      this.start();
    } catch (error) {
      this.emit({
        model: "error",
        message:
          error instanceof Error
            ? `Model failed to load: ${error.message}`
            : "Model failed to load.",
      });
    }
  }

  private start(): void {
    if (this.running) return;
    this.running = true;
    this.lastVideoTime = -1;
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.loop);
  }

  private loop = (): void => {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this.loop);

    const video = this.video;
    if (video.readyState < 2 || video.videoWidth === 0) return;

    const now = performance.now();
    if (now - this.lastDetect < 30) return;
    if (video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = video.currentTime;
    this.lastDetect = now;
    this.lastTimestamp = Math.max(now, this.lastTimestamp + 1);

    try {
      if (this.mode === "hands" && this.hands) {
        const result = this.hands.detectForVideo(video, this.lastTimestamp);
        const hands = this.toHands(result.landmarks);
        this.onFrame?.({ hands, pose: null, t: now });
      } else if (this.mode === "pose" && this.pose) {
        const result = this.pose.detectForVideo(video, this.lastTimestamp);
        const first = result.landmarks?.[0] ?? null;
        const pose = first ? this.toPose(first) : null;
        this.onFrame?.({ hands: [], pose, t: now });
      }
    } catch (error) {
      this.running = false;
      this.emit({
        model: "error",
        message: error instanceof Error ? error.message : "Inference failed.",
      });
    }
  };

  private toHands(landmarks: NormalizedLandmark[][]): Hand[] {
    if (landmarks.length !== this.handSlots) {
      this.handSmoother.reset();
      this.handSlots = landmarks.length;
    }
    // Sorting by mirrored x keeps the smoother attached to the same physical
    // hand instead of swapping slots when MediaPipe reorders its output.
    const mirrored = landmarks
      .map((points) => points.map((point) => ({ x: 1 - point.x, y: point.y, z: point.z })))
      .sort((a, b) => (a[0]?.x ?? 0) - (b[0]?.x ?? 0));
    return mirrored.map((points) => ({ points: this.handSmoother.smooth(points) }) as Hand);
  }

  private toPose(landmarks: NormalizedLandmark[]): PosePoint[] {
    const mirrored = landmarks.map((point) => ({
      x: 1 - point.x,
      y: point.y,
      z: point.z,
      visibility: point.visibility,
    }));
    return this.poseSmoother.smooth(mirrored) as PosePoint[];
  }
}

/** Joint angles only make sense if all three points are actually visible. */
export function poseArm(pose: PosePoint[] | null, side: "left" | "right"): import("./gestures").ArmPoints | null {
  if (!pose || pose.length < 25) return null;
  const shoulderIndex = side === "left" ? 11 : 12;
  const elbowIndex = side === "left" ? 13 : 14;
  const wristIndex = side === "left" ? 15 : 16;
  const hipIndex = side === "left" ? 23 : 24;
  const shoulder = pose[shoulderIndex];
  const elbow = pose[elbowIndex];
  const wrist = pose[wristIndex];
  const hip = pose[hipIndex];
  if (!shoulder || !elbow || !wrist) return null;
  const visible = [shoulder, elbow, wrist].every((point) => (point.visibility ?? 1) > 0.55);
  if (!visible) return null;
  const centered = (point: PosePoint): Point => ({ x: point.x, y: point.y });
  return {
    shoulder: centered(shoulder),
    elbow: centered(elbow),
    wrist: centered(wrist),
    hip: hip && (hip.visibility ?? 1) > 0.4 ? centered(hip) : undefined,
  };
}

export const POSE_CONNECTIONS: [number, number][] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
];

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

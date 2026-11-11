/**
 * Pure gesture math. No DOM, no MediaPipe types in here so the whole thing is
 * unit-testable with synthetic hand data.
 *
 * Coordinates are normalised 0..1 in *display* space (already mirrored), so
 * x grows to the viewer's right, y grows downwards.
 */

export interface Point {
  x: number;
  y: number;
  z?: number;
}

export interface Hand {
  points: Point[];
  score?: number;
}

export type SwipeAction = "like" | "nope" | "super";
export type RaiseSide = "like" | "nope";

export function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value));
}

export function distance(a: Point, b: Point): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

const TIPS = [4, 8, 12, 16, 20];
const PIPS = [3, 6, 10, 14, 18];
const MCPS = [2, 5, 9, 13, 17];

/**
 * Which of the five fingers look extended. Thumb is intentionally generous
 * because it sits at a different angle to the rest of the hand.
 */
export function fingerExtensions(points: Point[]): boolean[] {
  if (points.length < 21) return [false, false, false, false, false];
  const wrist = points[0];

  const fingers: boolean[] = [];
  for (let i = 0; i < 5; i += 1) {
    const tip = points[TIPS[i]];
    const pip = points[PIPS[i]];
    const mcp = points[MCPS[i]];
    if (!tip || !pip || !mcp) {
      fingers.push(false);
      continue;
    }
    if (i === 0) {
      // Thumb: compare how far the tip sits from the index knuckle and wrist.
      const tipFromIndex = distance(tip, points[5]);
      const ipFromIndex = distance(pip, points[5]);
      const tipFromWrist = distance(tip, wrist);
      const mcpFromWrist = distance(mcp, wrist);
      fingers.push(tipFromIndex > ipFromIndex * 1.18 || tipFromWrist > mcpFromWrist * 1.45);
    } else {
      const tipFromWrist = distance(tip, wrist);
      const pipFromWrist = distance(pip, wrist);
      const mcpFromWrist = distance(mcp, wrist);
      fingers.push(tipFromWrist > pipFromWrist * 1.08 && tipFromWrist > mcpFromWrist * 1.25);
    }
  }
  return fingers;
}

/** Number of extended fingers across the whole hand (0..5). */
export function fingerCount(points: Point[]): number {
  return fingerExtensions(points).filter(Boolean).length;
}

/** Count of extended fingers excluding the thumb (0..4). Used for chords. */
export function longFingerCount(points: Point[]): number {
  return fingerExtensions(points).slice(1).filter(Boolean).length;
}

export function isPalmOpen(points: Point[]): boolean {
  return fingerExtensions(points).slice(1).filter(Boolean).length >= 3;
}

export function isFist(points: Point[]): boolean {
  return fingerExtensions(points).filter(Boolean).length <= 1;
}

export function handCenter(points: Point[]): Point {
  if (points.length === 0) return { x: 0.5, y: 0.5 };
  const palm = [0, 5, 9, 13, 17];
  let x = 0;
  let y = 0;
  for (const index of palm) {
    x += points[index]?.x ?? 0;
    y += points[index]?.y ?? 0;
  }
  return { x: x / palm.length, y: y / palm.length };
}

export function handSpan(points: Point[]): number {
  const wrist = points[0];
  const middle = points[12] ?? wrist;
  return wrist && middle ? distance(wrist, middle) : 0;
}

/** Exponential smoothing for jittery landmark streams. */
export class LandmarkSmoother {
  private previous: Point[] | null = null;
  constructor(private readonly alpha = 0.45) {}

  smooth(points: Point[]): Point[] {
    if (!this.previous || this.previous.length !== points.length) {
      this.previous = points.map((point) => ({ ...point }));
      return points;
    }
    const next = points.map((point, index) => {
      const old = this.previous![index];
      return {
        x: old.x + (point.x - old.x) * this.alpha,
        y: old.y + (point.y - old.y) * this.alpha,
        z: point.z,
      };
    });
    this.previous = next;
    return next;
  }

  reset(): void {
    this.previous = null;
  }
}

export interface SixtySevenResult {
  /** 0..1 readiness of the "6...7" motion. */
  score: number;
  triggered: boolean;
  /** Peak-to-peak vertical anti-phase movement over the window. */
  range: number;
  /** Reversals counted in the window. */
  reversals: number;
  hands: number;
}

const ZERO_67: SixtySevenResult = { score: 0, triggered: false, range: 0, reversals: 0, hands: 0 };

/**
 * The meme: two open hands, palms facing each other, rocking in opposite
 * directions. We score the anti-phase vertical motion of the two wrists, and
 * accept a long static hold of the pose as a fallback.
 */
export class SixtySevenDetector {
  private samples: { t: number; d: number }[] = [];
  private lastTrigger = -Infinity;
  private holdSince: number | null = null;
  private armed = true;

  constructor(
    private readonly options: { windowMs?: number; cooldownMs?: number; trigger?: number } = {},
  ) {}

  reset(): void {
    this.samples = [];
    this.holdSince = null;
    this.armed = true;
  }

  update(hands: Hand[], t: number): SixtySevenResult {
    if (hands.length < 2) {
      this.samples = [];
      this.holdSince = null;
      this.armed = true;
      return { ...ZERO_67, hands: hands.length };
    }

    const sorted = [...hands].sort((a, b) => handCenter(a.points).x - handCenter(b.points).x);
    const left = sorted[0];
    const right = sorted[1];
    const leftCenter = handCenter(left.points);
    const rightCenter = handCenter(right.points);
    const separation = rightCenter.x - leftCenter.x;

    const open = isPalmOpen(left.points) && isPalmOpen(right.points);
    const framed =
      leftCenter.y > 0.02 &&
      leftCenter.y < 0.96 &&
      rightCenter.y > 0.02 &&
      rightCenter.y < 0.96 &&
      separation > 0.06 &&
      separation < 0.9;

    if (!open || !framed) {
      this.samples = [];
      this.holdSince = null;
      return { score: 0, triggered: false, range: 0, reversals: 0, hands: hands.length };
    }

    const delta = leftCenter.y - rightCenter.y;
    this.samples.push({ t, d: delta });
    const windowMs = this.options.windowMs ?? 1500;
    while (this.samples.length > 0 && t - this.samples[0].t > windowMs) this.samples.shift();

    if (this.samples.length < 5) {
      return { score: 0, triggered: false, range: 0, reversals: 0, hands: 2 };
    }

    const values = this.samples.map((sample) => sample.d);
    const range = Math.max(...values) - Math.min(...values);
    const reversals = countReversals(values);

    const motion =
      clamp((range - 0.02) / 0.09) * (reversals >= 2 ? 1 : reversals === 1 ? 0.58 : 0.15);

    if (range < 0.05) {
      if (this.holdSince === null) this.holdSince = t;
    } else {
      this.holdSince = null;
    }
    const hold =
      this.holdSince === null ? 0 : clamp((t - this.holdSince - 1000) / 800) * 0.82;

    const score = clamp(Math.max(motion, hold));
    const triggerAt = this.options.trigger ?? 0.72;
    const cooldownMs = this.options.cooldownMs ?? 1800;

    if (score < 0.25) this.armed = true;

    let triggered = false;
    if (this.armed && score >= triggerAt && t - this.lastTrigger > cooldownMs) {
      triggered = true;
      this.armed = false;
      this.lastTrigger = t;
      this.samples = [];
      this.holdSince = null;
    }

    return { score, triggered, range, reversals, hands: 2 };
  }
}

function countReversals(values: number[], epsilon = 0.004): number {
  let reversals = 0;
  let direction = 0;
  let last = values[0];
  for (let i = 1; i < values.length; i += 1) {
    const diff = values[i] - last;
    if (Math.abs(diff) < epsilon) continue;
    const next = Math.sign(diff);
    if (direction !== 0 && next !== direction) reversals += 1;
    direction = next;
    last = values[i];
  }
  return reversals;
}

export interface RaiseResult {
  side: RaiseSide | null;
  progress: number;
  triggered: boolean;
}

/** A single open palm parked on the left or right of frame for a moment. */
export class RaiseDetector {
  private since: number | null = null;
  private currentSide: RaiseSide | null = null;
  private armed = true;

  constructor(
    private readonly options: { dwellMs?: number; leftEdge?: number; rightEdge?: number } = {},
  ) {}

  reset(): void {
    this.since = null;
    this.currentSide = null;
    this.armed = true;
  }

  update(hands: Hand[], t: number): RaiseResult {
    if (hands.length === 0) {
      this.armed = true;
      this.since = null;
      this.currentSide = null;
      return { side: null, progress: 0, triggered: false };
    }
    if (hands.length > 1) {
      this.since = null;
      this.currentSide = null;
      return { side: null, progress: 0, triggered: false };
    }

    const hand = hands[0];
    if (!isPalmOpen(hand.points)) {
      this.armed = true;
      this.since = null;
      this.currentSide = null;
      return { side: null, progress: 0, triggered: false };
    }

    const center = handCenter(hand.points);
    if (center.y > 0.8) {
      this.since = null;
      this.currentSide = null;
      return { side: null, progress: 0, triggered: false };
    }
    const rightEdge = this.options.rightEdge ?? 0.57;
    const leftEdge = this.options.leftEdge ?? 0.43;
    const side: RaiseSide | null = center.x >= rightEdge ? "like" : center.x <= leftEdge ? "nope" : null;

    if (!side) {
      this.since = null;
      this.currentSide = null;
      return { side: null, progress: 0, triggered: false };
    }

    if (this.currentSide !== side) {
      this.currentSide = side;
      this.since = t;
    }
    if (this.since === null) this.since = t;

    const dwell = this.options.dwellMs ?? 620;
    const progress = clamp((t - this.since) / dwell);
    let triggered = false;
    if (this.armed && progress >= 1) {
      triggered = true;
      this.armed = false;
      this.since = t;
    }
    return { side, progress, triggered };
  }
}

export interface GestureUpdate {
  action: SwipeAction | null;
  superScore: number;
  raiseSide: RaiseSide | null;
  raiseProgress: number;
  hands: number;
}

/** Priority routing between the two hand gestures. */
export class GestureRouter {
  readonly sixtySeven = new SixtySevenDetector();
  readonly raise = new RaiseDetector();
  private suppressUntil = 0;

  reset(): void {
    this.sixtySeven.reset();
    this.raise.reset();
    this.suppressUntil = 0;
  }

  update(hands: Hand[], t: number): GestureUpdate {
    if (hands.length >= 2) {
      const result = this.sixtySeven.update(hands, t);
      this.raise.update([], t);
      let action: SwipeAction | null = null;
      if (result.triggered && t > this.suppressUntil) {
        action = "super";
        this.suppressUntil = t + 900;
      }
      return {
        action,
        superScore: result.score,
        raiseSide: null,
        raiseProgress: 0,
        hands: hands.length,
      };
    }

    this.sixtySeven.update(hands, t);
    const raised = this.raise.update(hands, t);
    let action: SwipeAction | null = null;
    if (raised.triggered && t > this.suppressUntil) {
      action = raised.side;
      this.suppressUntil = t + 900;
    }
    return {
      action,
      superScore: 0,
      raiseSide: raised.side,
      raiseProgress: raised.progress,
      hands: hands.length,
    };
  }
}

export interface ArmPoints {
  shoulder: Point;
  elbow: Point;
  wrist: Point;
  hip?: Point;
}

/** Interior angle at the elbow, in degrees (180 = straight arm, 0 = fully folded). */
export function elbowAngle(arm: ArmPoints): number {
  const a = Math.atan2(arm.shoulder.y - arm.elbow.y, arm.shoulder.x - arm.elbow.x);
  const b = Math.atan2(arm.wrist.y - arm.elbow.y, arm.wrist.x - arm.elbow.x);
  let between = Math.abs(((a - b) * 180) / Math.PI);
  if (between > 180) between = 360 - between;
  return 180 - between;
}

export interface RepResult {
  rep: number;
  score: number;
  notes: string[];
  durationMs: number;
  minAngle: number;
}

export interface CurlUpdate {
  reps: number;
  angle: number;
  phase: "extended" | "curled" | "moving";
  rep: RepResult | null;
  liveNotes: string[];
}

/**
 * Bicep curl rep counter with form feedback. The "67" idea shows up here as a
 * tempo target: 67 BPM, about one full rep every 1.8 seconds.
 */
export class CurlCounter {
  reps = 0;
  private phase: "extended" | "curled" = "extended";
  private minAngle = 180;
  private repStart = 0;
  private curlStart = 0;

  reset(): void {
    this.reps = 0;
    this.phase = "extended";
    this.minAngle = 180;
    this.repStart = 0;
    this.curlStart = 0;
  }

  update(arm: ArmPoints | null, t: number): CurlUpdate {
    if (!arm) {
      return { reps: this.reps, angle: 0, phase: "moving", rep: null, liveNotes: [] };
    }
    if (this.repStart === 0) this.repStart = t;
    const angle = elbowAngle(arm);

    const notes = this.formNotes(arm, angle);

    if (this.phase === "extended" && angle < 100) {
      this.phase = "curled";
      this.curlStart = t;
      this.minAngle = angle;
    } else if (this.phase === "curled") {
      this.minAngle = Math.min(this.minAngle, angle);
      if (angle > 150) {
        const duration = t - this.repStart;
        const repNotes = [...notes];
        if (this.minAngle > 78) repNotes.push("Curl higher — squeeze at the top");
        if (this.curlStart - this.repStart < 320) repNotes.push("Slow the lift down");
        if (duration < 900) repNotes.push("Full rep in under a second — own the negative");

        const uniqueNotes = [...new Set(repNotes)];
        const score = Math.max(
          20,
          100 - uniqueNotes.length * 12 + (this.minAngle < 55 ? 6 : 0),
        );
        this.reps += 1;
        const rep: RepResult = {
          rep: this.reps,
          score: Math.min(100, score),
          notes: uniqueNotes,
          durationMs: duration,
          minAngle: this.minAngle,
        };
        this.phase = "extended";
        this.repStart = t;
        this.minAngle = 180;
        return { reps: this.reps, angle, phase: "extended", rep, liveNotes: [] };
      }
    } else if (this.phase === "extended" && angle > 150) {
      if (this.repStart === 0) this.repStart = t;
    }

    return {
      reps: this.reps,
      angle,
      phase: this.phase === "curled" ? "curled" : angle < 140 && angle > 70 ? "moving" : "extended",
      rep: null,
      liveNotes: notes,
    };
  }

  private formNotes(arm: ArmPoints, angle: number): string[] {
    const notes: string[] = [];
    const drift = Math.abs(arm.elbow.x - arm.shoulder.x);
    if (drift > 0.12) notes.push("Pin your elbow closer to your side");
    if (arm.hip && Math.abs(arm.shoulder.x - arm.hip.x) > 0.13) notes.push("Keep your torso stacked");
    if (arm.elbow.y < arm.shoulder.y - 0.02) notes.push("Drop the elbow below the shoulder");
    if (angle < 25) notes.push("Ease off the top — don't crank the joint");
    return notes;
  }
}

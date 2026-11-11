import { describe, expect, it } from "vitest";
import {
  CurlCounter,
  RaiseDetector,
  SixtySevenDetector,
  elbowAngle,
  fingerCount,
  fingerExtensions,
  isFist,
  isPalmOpen,
  longFingerCount,
  type Hand,
  type Point,
} from "../src/lib/gestures";

function makeHand(extended: number[], centerY = 0.5, centerX = 0.5): Hand {
  const points: Point[] = new Array(21).fill(null).map(() => ({ x: centerX, y: centerY }));
  const wristY = 0.88;
  points[0] = { x: centerX, y: wristY };
  points[1] = { x: centerX - 0.04, y: 0.84 };
  points[2] = { x: centerX - 0.07, y: 0.81 };
  points[3] = { x: centerX - 0.1, y: 0.79 };
  points[4] = extended.includes(0)
    ? { x: centerX - 0.14, y: 0.765 }
    : { x: centerX + 0.01, y: 0.79 };

  const mcpX = [0.46, 0.5, 0.54, 0.58];
  for (let finger = 0; finger < 4; finger += 1) {
    const index = 5 + finger * 4;
    const x = centerX + (mcpX[finger] - 0.5);
    const isExtended = extended.includes(finger + 1);
    const mcpY = 0.73;
    points[index] = { x, y: mcpY };
    points[index + 1] = { x, y: 0.65 };
    points[index + 2] = { x, y: 0.61 };
    points[index + 3] = isExtended ? { x, y: 0.57 } : { x, y: 0.66 };
  }

  const palm = [0, 5, 9, 13, 17].reduce(
    (sum, index) => ({ x: sum.x + points[index].x, y: sum.y + points[index].y }),
    { x: 0, y: 0 },
  );
  const dx = centerX - palm.x / 5;
  const dy = centerY - palm.y / 5;
  return { points: points.map((point) => ({ x: point.x + dx, y: point.y + dy })) };
}

describe("hand math", () => {
  it("counts extended fingers", () => {
    expect(fingerCount(makeHand([0, 1, 2, 3, 4]).points)).toBe(5);
    expect(fingerCount(makeHand([1, 2]).points)).toBe(2);
    expect(longFingerCount(makeHand([0, 1, 2, 3, 4]).points)).toBe(4);
  });

  it("detects open palms and fists", () => {
    const open = makeHand([0, 1, 2, 3, 4]);
    const fist = makeHand([]);
    expect(isPalmOpen(open.points)).toBe(true);
    expect(isPalmOpen(fist.points)).toBe(false);
    expect(isFist(fist.points)).toBe(true);
    expect(fingerExtensions(open.points)).toHaveLength(5);
  });
});

describe("67 detector", () => {
  it("fires on two open hands rocking in opposite directions", () => {
    const detector = new SixtySevenDetector();
    let fired = false;
    for (let frame = 0; frame < 70; frame += 1) {
      const t = frame * 33;
      const phase = (frame / 30) * Math.PI * 2;
      const left = makeHand([0, 1, 2, 3, 4], 0.48 + Math.sin(phase) * 0.11, 0.3);
      const right = makeHand([0, 1, 2, 3, 4], 0.48 - Math.sin(phase) * 0.11, 0.7);
      const result = detector.update([left, right], t);
      if (result.triggered) fired = true;
    }
    expect(fired).toBe(true);
  });

  it("does not fire on one hand or two fists", () => {
    const detector = new SixtySevenDetector();
    expect(detector.update([makeHand([0, 1, 2, 3, 4])], 0).triggered).toBe(false);
    let fired = false;
    for (let frame = 0; frame < 50; frame += 1) {
      const t = frame * 33;
      const a = makeHand([], 0.5 + Math.sin(frame) * 0.1, 0.3);
      const b = makeHand([], 0.5 - Math.sin(frame) * 0.1, 0.7);
      if (detector.update([a, b], t).triggered) fired = true;
    }
    expect(fired).toBe(false);
  });
});

describe("palm raise", () => {
  it("commits a like on a right-side palm after the dwell", () => {
    const detector = new RaiseDetector();
    const hand = makeHand([0, 1, 2, 3, 4], 0.5, 0.78);
    let triggered = false;
    for (let t = 0; t <= 700; t += 50) {
      const result = detector.update([hand], t);
      if (result.triggered) triggered = true;
    }
    expect(triggered).toBe(true);
  });

  it("maps the left side to nope and ignores a fist", () => {
    const detector = new RaiseDetector();
    const hand = makeHand([0, 1, 2, 3, 4], 0.5, 0.2);
    let side: string | null = null;
    for (let t = 0; t <= 700; t += 50) {
      const result = detector.update([hand], t);
      side = result.side ?? side;
    }
    expect(side).toBe("nope");

    const fists = new RaiseDetector();
    let fired = false;
    for (let t = 0; t <= 800; t += 50) {
      if (fists.update([makeHand([], 0.5, 0.8)], t).triggered) fired = true;
    }
    expect(fired).toBe(false);
  });
});

describe("curl counter", () => {
  const arm = (angle: number) => {
    const shoulder = { x: 0.56, y: 0.38 };
    const elbow = { x: 0.58, y: 0.55 };
    const vector = { x: shoulder.x - elbow.x, y: shoulder.y - elbow.y };
    const radians = ((180 - angle) * Math.PI) / 180;
    return {
      shoulder,
      elbow,
      wrist: {
        x: elbow.x + vector.x * Math.cos(radians) - vector.y * Math.sin(radians),
        y: elbow.y + vector.x * Math.sin(radians) + vector.y * Math.cos(radians),
      },
      hip: { x: 0.55, y: 0.62 },
    };
  };

  it("measures elbow angles", () => {
    expect(elbowAngle(arm(180))).toBeCloseTo(180, 0);
    expect(elbowAngle(arm(90))).toBeCloseTo(90, 0);
  });

  it("counts a full rep and flags a shallow curl", () => {
    const counter = new CurlCounter();
    counter.update(arm(175), 0);
    counter.update(arm(150), 100);
    counter.update(arm(88), 300);
    const finish = counter.update(arm(172), 1200);
    expect(finish.reps).toBe(1);
    expect(finish.rep?.notes.join(" ")).toContain("Curl higher");
  });

  it("does not count a half rep that never extends again", () => {
    const counter = new CurlCounter();
    counter.update(arm(175), 0);
    counter.update(arm(45), 400);
    const held = counter.update(arm(50), 900);
    expect(held.reps).toBe(0);
  });
});

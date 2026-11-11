import type { Hand, Point } from "../lib/gestures";
import { HAND_CONNECTIONS, POSE_CONNECTIONS, type PosePoint } from "../lib/vision";

export interface HandDrawOptions {
  color?: string;
  label?: string;
  accentIndex?: number;
}

export function drawHand(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  hand: Hand,
  options: HandDrawOptions = {},
): void {
  const color = options.color ?? "#d7ff3e";
  const points = hand.points;
  if (points.length < 21) return;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.5, width / 480);

  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(points[a].x * width, points[a].y * height);
    ctx.lineTo(points[b].x * width, points[b].y * height);
    ctx.globalAlpha = 0.55;
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const tip = [4, 8, 12, 16, 20].includes(i);
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, tip ? 5 : 2.6, 0, Math.PI * 2);
    ctx.fillStyle = tip ? color : "#0c0c0d";
    ctx.fill();
    if (!tip) {
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }

  if (options.label) {
    const wrist = points[0];
    const labelX = Math.min(width - 84, Math.max(8, wrist.x * width + 14));
    const labelY = Math.min(height - 12, Math.max(18, wrist.y * height - 10));
    ctx.font = `600 ${Math.max(12, width / 48)}px "JetBrains Mono", monospace`;
    ctx.textBaseline = "middle";
    const textWidth = ctx.measureText(options.label).width;
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "#0c0c0d";
    ctx.fillRect(labelX - 6, labelY - 10, textWidth + 12, 20);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.fillText(options.label, labelX, labelY);
  }
  ctx.restore();
}

export function drawPose(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  pose: PosePoint[] | null,
  side: "left" | "right",
): void {
  if (!pose) return;
  const color = "#d7ff3e";
  const working = new Set(side === "left" ? [11, 13, 15] : [12, 14, 16]);

  ctx.save();
  ctx.lineCap = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;

  for (const [a, b] of POSE_CONNECTIONS) {
    const start = pose[a];
    const end = pose[b];
    if (!start || !end) continue;
    if ((start.visibility ?? 1) < 0.4 || (end.visibility ?? 1) < 0.4) continue;
    const active = working.has(a) && working.has(b);
    ctx.globalAlpha = active ? 1 : 0.32;
    ctx.strokeStyle = active ? color : "#f4f1ea";
    ctx.lineWidth = active ? Math.max(3, width / 220) : Math.max(1.2, width / 520);
    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  for (const [index, point] of pose.entries()) {
    if ((point.visibility ?? 1) < 0.45) continue;
    const isWorking = working.has(index);
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, isWorking ? 5 : 2.4, 0, Math.PI * 2);
    ctx.fillStyle = isWorking ? color : "#0c0c0d";
    ctx.fill();
    if (!isWorking) {
      ctx.strokeStyle = "rgba(244,241,234,0.55)";
      ctx.lineWidth = 1.1;
      ctx.stroke();
    }
  }

  const wrist = pose[side === "left" ? 15 : 16];
  if (wrist && (wrist.visibility ?? 1) > 0.5) {
    ctx.font = `600 ${Math.max(12, width / 48)}px "JetBrains Mono", monospace`;
    ctx.fillStyle = color;
    ctx.fillText(`${side === "left" ? "L" : "R"} ARM`, wrist.x * width + 12, wrist.y * height - 12);
  }
  ctx.restore();
}

export function clearCanvas(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  ctx.clearRect(0, 0, width, height);
}

/**
 * Match the overlay canvas to the video's *object-fit: cover* rect so landmarks
 * line up at any frame aspect ratio (4:3, 16:10, square on phones).
 */
export function fitOverlay(canvas: HTMLCanvasElement, video: HTMLVideoElement): { width: number; height: number } {
  const width = video.videoWidth || 960;
  const height = video.videoHeight || 720;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const frame = canvas.parentElement;
  if (frame) {
    const boxWidth = frame.clientWidth;
    const boxHeight = frame.clientHeight;
    if (boxWidth > 0 && boxHeight > 0) {
      const scale = Math.max(boxWidth / width, boxHeight / height);
      const displayWidth = width * scale;
      const displayHeight = height * scale;
      const left = (boxWidth - displayWidth) / 2;
      const top = (boxHeight - displayHeight) / 2;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
      canvas.style.left = `${left}px`;
      canvas.style.top = `${top}px`;
      canvas.style.right = "auto";
      canvas.style.bottom = "auto";
    }
  }

  return { width, height };
}

export function line(ctx: CanvasRenderingContext2D, a: Point, b: Point, width: number): void {
  ctx.beginPath();
  ctx.moveTo(a.x * width, a.y * width);
  ctx.lineTo(b.x * width, b.y * width);
  ctx.stroke();
}

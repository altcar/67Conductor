/**
 * Small Web Audio playground: a plucked-string-ish polysynth, a filter, a
 * feedback delay and a 67 BPM click track for the curl coach.
 */

export interface Chord {
  name: string;
  notes: number[];
  fingers: number;
}

export const CHORDS: Chord[] = [
  { name: "C", notes: [48, 55, 60, 64, 67], fingers: 1 },
  { name: "Am", notes: [45, 52, 57, 60, 64], fingers: 2 },
  { name: "F", notes: [41, 48, 53, 57, 60], fingers: 3 },
  { name: "G", notes: [43, 50, 55, 59, 62], fingers: 4 },
  { name: "Dm", notes: [50, 57, 62, 65, 69], fingers: 5 },
];

export function midiToFrequency(midi: number): number {
  return 440 * 2 ** ((midi - 69) / 12);
}

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private filter: BiquadFilterNode | null = null;
  private analyser: AnalyserNode | null = null;
  private noise: AudioBuffer | null = null;
  private spectrum: Uint8Array<ArrayBuffer> | null = null;
  private level = 0.7;
  private ducked = false;
  private clickTimer: number | null = null;
  private clickNext = 0;
  private audioLog: { chord: string; at: number }[] = [];

  get ready(): boolean {
    return this.ctx !== null && this.ctx.state === "running";
  }

  async start(): Promise<boolean> {
    if (!this.ctx) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return false;
      this.ctx = new Ctor();

      const master = this.ctx.createGain();
      master.gain.value = this.level;

      const compressor = this.ctx.createDynamicsCompressor();
      compressor.threshold.value = -16;
      compressor.knee.value = 18;
      compressor.ratio.value = 4;
      compressor.attack.value = 0.004;
      compressor.release.value = 0.22;

      const filter = this.ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.value = 3200;
      filter.Q.value = 0.7;

      const delay = this.ctx.createDelay(1.2);
      delay.delayTime.value = 0.23;
      const feedback = this.ctx.createGain();
      feedback.gain.value = 0.34;
      const wet = this.ctx.createGain();
      wet.gain.value = 0.18;

      const analyser = this.ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.82;
      this.spectrum = new Uint8Array(analyser.frequencyBinCount);

      filter.connect(master);
      filter.connect(wet);
      wet.connect(delay);
      delay.connect(feedback);
      feedback.connect(delay);
      delay.connect(master);
      master.connect(compressor);
      compressor.connect(analyser);
      analyser.connect(this.ctx.destination);

      this.master = master;
      this.filter = filter;
      this.analyser = analyser;

      const length = Math.floor(this.ctx.sampleRate * 0.09);
      const noise = this.ctx.createBuffer(1, length, this.ctx.sampleRate);
      const data = noise.getChannelData(0);
      for (let i = 0; i < length; i += 1) {
        data[i] = (Math.random() * 2 - 1) * (1 - i / length) ** 3;
      }
      this.noise = noise;
    }

    if (this.ctx.state === "suspended") await this.ctx.resume();
    return this.ctx.state === "running";
  }

  private now(): number {
    return this.ctx?.currentTime ?? 0;
  }

  setVolume(value: number): void {
    this.level = Math.max(0, Math.min(1, value));
    if (this.master && this.ctx && !this.ducked) {
      this.master.gain.setTargetAtTime(0.0001 + this.level * 0.85, this.now(), 0.05);
    }
  }

  get volume(): number {
    return this.level;
  }

  setFilterHz(hz: number): void {
    if (!this.filter || !this.ctx) return;
    this.filter.frequency.setTargetAtTime(Math.max(180, Math.min(9000, hz)), this.now(), 0.08);
  }

  duck(): void {
    if (!this.master || !this.ctx) return;
    this.ducked = true;
    this.master.gain.setTargetAtTime(0.0001, this.now(), 0.12);
  }

  unduck(): void {
    if (!this.master || !this.ctx) return;
    this.ducked = false;
    this.master.gain.setTargetAtTime(0.0001 + this.level * 0.85, this.now(), 0.18);
  }

  private pluck(midi: number, when: number, duration = 1.9, gain = 0.16): void {
    const ctx = this.ctx;
    const filter = this.filter;
    if (!ctx || !filter) return;

    const frequency = midiToFrequency(midi);
    const envelope = ctx.createGain();
    envelope.gain.setValueAtTime(0.0001, when);
    envelope.gain.exponentialRampToValueAtTime(gain, when + 0.012);
    envelope.gain.exponentialRampToValueAtTime(0.0001, when + duration);
    envelope.connect(filter);

    const voices: OscillatorNode[] = [];
    const shapes: OscillatorType[] = ["sawtooth", "triangle", "sawtooth"];
    const detunes = [-5, 0, 7];
    const levels = [0.45, 0.75, 0.22];
    for (let i = 0; i < shapes.length; i += 1) {
      const osc = ctx.createOscillator();
      osc.type = shapes[i];
      osc.frequency.value = frequency;
      osc.detune.value = detunes[i];
      const voiceGain = ctx.createGain();
      voiceGain.gain.value = levels[i];
      osc.connect(voiceGain);
      voiceGain.connect(envelope);
      osc.start(when);
      osc.stop(when + duration + 0.05);
      voices.push(osc);
    }

    if (this.noise) {
      const pick = ctx.createBufferSource();
      pick.buffer = this.noise;
      const band = ctx.createBiquadFilter();
      band.type = "bandpass";
      band.frequency.value = Math.min(8000, frequency * 3.5);
      band.Q.value = 0.8;
      const pickGain = ctx.createGain();
      pickGain.gain.value = 0.16;
      pick.connect(band);
      band.connect(pickGain);
      pickGain.connect(filter);
      pick.start(when);
    }

    // Keep a reference alive until the tail has finished.
    voices[0].onended = () => envelope.disconnect();
  }

  strum(notes: number[], direction: "up" | "down" = "down", velocity = 1): void {
    if (!this.ctx) return;
    const base = this.now() + 0.01;
    const ordered = direction === "down" ? notes : [...notes].reverse();
    ordered.forEach((note, index) => {
      this.pluck(note, base + index * 0.035, 1.7 + index * 0.05, 0.11 + 0.04 * velocity);
    });
  }

  playChord(index: number, direction: "up" | "down" = "down", velocity = 1): void {
    const chord = CHORDS[((index % CHORDS.length) + CHORDS.length) % CHORDS.length];
    this.strum(chord.notes, direction, velocity);
    this.audioLog.push({ chord: chord.name, at: performance.now() });
    if (this.audioLog.length > 24) this.audioLog.shift();
  }

  get recentChords(): { chord: string; at: number }[] {
    return this.audioLog;
  }

  click(accent = false): void {
    const ctx = this.ctx;
    if (!ctx || !this.master) return;
    const when = this.now() + 0.005;
    const osc = ctx.createOscillator();
    osc.type = "square";
    osc.frequency.value = accent ? 1760 : 1175;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, when);
    gain.gain.exponentialRampToValueAtTime(accent ? 0.05 : 0.028, when + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, when + 0.055);
    osc.connect(gain);
    gain.connect(this.master);
    osc.start(when);
    osc.stop(when + 0.08);
  }

  startMetronome(bpm = 67): void {
    this.stopMetronome();
    if (!this.ctx) return;
    this.clickNext = this.now();
    const interval = 60 / bpm;
    const tick = () => {
      if (!this.ctx) return;
      while (this.clickNext < this.now() + 0.15) {
        this.click(Math.round(this.clickNext / interval) % 4 === 0);
        this.clickNext += interval;
      }
      this.clickTimer = window.setTimeout(tick, 60);
    };
    tick();
  }

  stopMetronome(): void {
    if (this.clickTimer !== null) {
      window.clearTimeout(this.clickTimer);
      this.clickTimer = null;
    }
  }

  spectrumData(): Uint8Array<ArrayBuffer> | null {
    if (!this.analyser || !this.spectrum) return null;
    this.analyser.getByteFrequencyData(this.spectrum);
    return this.spectrum;
  }

  suspend(): void {
    this.stopMetronome();
    void this.ctx?.suspend();
  }
}

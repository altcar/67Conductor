# 67Conductor
Repository for HackNotts 67Conductor

**Browser implementation:** the [`web/`](./web) folder is a working, static rebuild of the 6'7'' store concept — gesture-driven resale, a hand-controlled synth and a form-coached curl counter, running entirely in the browser with MediaPipe + Web Audio.

**Live:** https://67conductor.hheylau.workers.dev — every push to `main` is built and deployed by Cloudflare Workers Builds.

Start locally:

```bash
cd web && npm install && npm run dev
```

The original hackathon prototype (Python/MediaPipe experiments, an Astro sketch and a local LLM server) is kept below for reference.



curl -LsSf https://astral.sh/uv/install.sh | sh
uv init app
uv run main.py
uv add mediapipe
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
wget -q -O image.jpg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg

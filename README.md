# 67Conductor
Repository for HackNotts 67Conductor



curl -LsSf https://astral.sh/uv/install.sh | sh
uv init app
uv run main.py
uv add mediapipe
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
wget -q -O image.jpg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg

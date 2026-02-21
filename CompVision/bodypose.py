from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import re
import time
import os
import urllib.request
import urllib.error
from collections import deque

# thresholds for classification
UP_MARGIN = 0.03
DOWN_MARGIN = 0.03

def classify_AB_from_landmarks(landmarks):
  """Return 'A' or 'B' or None from a single pose landmarks list.

  A = left hand up, right hand down
  B = left hand down, right hand up
  """
  try:
    l_sh = landmarks[11].y
    r_sh = landmarks[12].y
    l_wr = landmarks[15].y
    r_wr = landmarks[16].y
  except Exception:
    return None

  left_up = l_wr < (l_sh - UP_MARGIN)
  left_down = l_wr > (l_sh + DOWN_MARGIN)
  right_up = r_wr < (r_sh - UP_MARGIN)
  right_down = r_wr > (r_sh + DOWN_MARGIN)

  if left_up and right_down:
    return 'A'
  if left_down and right_up:
    return 'B'
  return None


def main():
  if len(sys.argv) < 2:
    print("Usage: python poseture.py <camera_index>")
    return 
  run_video_poselandmarker()

def run_video_poselandmarker():
  logger = logging.getLogger(__name__)
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

  # STEP 2: Create a PoseLandmarker object.
  base_options = python.BaseOptions(model_asset_path='lib/pose_landmarker_heavy.task')
  options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=1
  )
  detector = vision.PoseLandmarker.create_from_options(options)

  # front-end API to notify when pattern detected
  FRONTEND_API_URL = os.environ.get('BODYPOSE_FRONTEND_API')

  # Open the camera stream.
  cap = cv2.VideoCapture(int(sys.argv[1]))
  if not cap.isOpened():
    logger.error("Could not open camera (index %s)", sys.argv[1])
    return

  # detection state
  state = {
    'seq': deque(maxlen=128),
    'last_code': None,
    'signaled': False,
    'stable_frames': 0,
    'last_lw_y': None,
    'last_rw_y': None,
  }

  PATTERN = re.compile(r'^(?:AB){2,}(?:A)?$|^(?:BA){2,}(?:B)?$')
  RESET_STABLE_FRAMES = 25
  # extend state for horizontal tracking and swipe signaling
  state['last_lw_x'] = None
  state['last_rw_x'] = None
  state['swipe_signaled_left'] = False
  state['swipe_signaled_right'] = False

  # wrap detector.detect to compute swipe logic and draw a top percentage bar directly onto the image,
  # so later drawing/copying preserves the overlay
  _orig_detect = detector.detect

  def _detect_with_swipe(image):
    detection_result = _orig_detect(image)

    try:
      if not detection_result.pose_landmarks:
        # nothing to do
        return detection_result

      lm = detection_result.pose_landmarks[0]

      # midpoint between shoulders
      mid_x = (lm[11].x + lm[12].x) / 2.0
      shoulder_dx = abs(lm[11].x - lm[12].x)
      shoulder_dx = max(shoulder_dx, 0.05)  # avoid tiny spans
      span = shoulder_dx  # how far left/right we consider for full progress
      margin = span * 0.15  # crossing hysteresis margin

      # wrist x positions
      lw_x = lm[15].x
      rw_x = lm[16].x

      # helper to map x to 0..1 across [mid-span, mid+span]
      min_x = mid_x - span
      max_x = mid_x + span
      def norm_pct(x):
        return max(0.0, min(1.0, (x - min_x) / (max_x - min_x)))

      l_pct = norm_pct(lw_x)
      r_pct = norm_pct(rw_x)

      # choose the hand that is currently further from the midpoint for the top bar
      active_hand = 'L' if abs(lw_x - mid_x) > abs(rw_x - mid_x) else 'R'
      active_pct = l_pct if active_hand == 'L' else r_pct
      active_x = lw_x if active_hand == 'L' else rw_x

      # detect crossing events for both hands (left hand index 15, right hand index 16)
      def check_and_signal(hand_name, last_x_key, cur_x, signaled_key, hand_label):
        last_x = state.get(last_x_key)
        # crossing left->right
        if last_x is not None and not state.get(signaled_key, False):
          if last_x < (mid_x - margin) and cur_x > (mid_x + margin):
            # moved from left of mid to right -> swipe right
            try:
              notify_frontend({'event': 'swipe', 'direction': 'right', 'hand': hand_label})
            except Exception:
              pass
            state[signaled_key] = True
          elif last_x > (mid_x + margin) and cur_x < (mid_x - margin):
            # moved from right of mid to left -> swipe left
            try:
              notify_frontend({'event': 'swipe', 'direction': 'left', 'hand': hand_label})
            except Exception:
              pass
            state[signaled_key] = True
        # reset signal when hand returns close to midpoint (neutral)
        if abs(cur_x - mid_x) < (margin * 0.6):
          state[signaled_key] = False

      check_and_signal('left', 'last_lw_x', lw_x, 'swipe_signaled_left', 'left')
      check_and_signal('right', 'last_rw_x', rw_x, 'swipe_signaled_right', 'right')

      # draw a horizontal percentage bar at the very top of the image onto the mp.Image's numpy view
      try:
        img = image.numpy_view()  # RGB image that will later be copied into annotated_image
        h_img, w_img, _ = img.shape
        bar_h = 18
        pad = 8
        bx1 = pad
        by1 = pad
        bx2 = w_img - pad
        by2 = pad + bar_h

        # background bar
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (40, 40, 40), -1)

        # fill according to active_pct (left->right)
        fill_w = int((bx2 - bx1) * active_pct)
        fill_x = bx1 + fill_w
        # color: green for rightward (pct>0.5), blue for left-biased (pct<=0.5)
        color = (0, 200, 0) if active_hand == 'R' else (200, 100, 0)
        cv2.rectangle(img, (bx1, by1), (bx1 + fill_w, by2), color, -1)

        # center text like "Swipe: 34% (L)"
        txt = f"Swipe: {int(active_pct*100)}% ({active_hand})"
        txt_size = 0.5
        cv2.putText(img, txt, (bx1 + 6, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, txt_size, (255,255,255), 1, cv2.LINE_AA)
      except Exception:
        # swallow drawing exceptions to not break detection
        pass

      # update last positions
      state['last_lw_x'] = lw_x
      state['last_rw_x'] = rw_x

      # also store some derived values for potential later use
      detection_result._swipe = {
        'mid_x': mid_x,
        'left_pct': l_pct,
        'right_pct': r_pct,
        'active_hand': active_hand,
        'active_pct': active_pct,
      }
    except Exception:
      # keep original detect behavior on any failure
      state['last_lw_x'] = state.get('last_lw_x')
      state['last_rw_x'] = state.get('last_rw_x')
    return detection_result

  # install wrapper
  detector.detect = _detect_with_swipe
  def notify_frontend(payload: dict):
    if not FRONTEND_API_URL:
      logger.info("Frontend notify (disabled): %s", payload)
      return
    try:
      data = bytes(str(payload).replace("'", '"'), 'utf-8')
      req = urllib.request.Request(FRONTEND_API_URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')
      with urllib.request.urlopen(req, timeout=2) as resp:
        logger.info("Notify sent, status=%s", resp.status)
    except Exception:
      logger.exception("Failed to notify frontend")

  while True:
    ret, frame = cap.read()
    if not ret:
      logger.error("Could not read frame from camera")
      break

    rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(image)

    if detection_result.pose_landmarks:
      annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    else:
      annotated_image = image.numpy_view().copy()

    # classification per frame
    current_code = None
    if detection_result.pose_landmarks:
      current_code = None
      try:
        current_code = classify_AB_from_landmarks(detection_result.pose_landmarks[0])
      except Exception:
        logger.debug("Failed to classify frame")

    # Draw two vertical percentage bars (top-left): left hand and right hand.
    try:
      if detection_result.pose_landmarks:
        lm = detection_result.pose_landmarks[0]

        def hand_percent(elbow_i, wrist_i, shoulder_i):
          try:
            elbow_y = lm[elbow_i].y
            wrist_y = lm[wrist_i].y
            shoulder_y = lm[shoulder_i].y
          except Exception:
            return None
          d = abs(elbow_y - shoulder_y)
          if d < 1e-4:
            d = 0.1
          up_limit = elbow_y - d
          down_limit = elbow_y + d
          # percent: 1.0 when wrist at up_limit (high), 0.0 when at down_limit (low)
          pct = (down_limit - wrist_y) / (down_limit - up_limit)
          if pct != pct:  # NaN
            return None
          return max(0.0, min(1.0, pct))

        left_pct = hand_percent(13, 17, 11)
        right_pct = hand_percent(14, 18, 12)

        # draw bars on annotated_image (RGB). Reserve a small panel at top-left.
        h_img, w_img, _ = annotated_image.shape
        panel_x = 10
        panel_y = 10
        bar_w = 30
        bar_h = 120
        gap = 10

        # left bar rectangle background
        lx = panel_x
        ly = panel_y
        rx = lx + bar_w
        ry = ly + bar_h
        # background (dark gray)
        cv2.rectangle(annotated_image, (lx, ly), (rx, ry), (50, 50, 50), -1)
        # right bar
        lx2 = rx + gap
        rx2 = lx2 + bar_w
        cv2.rectangle(annotated_image, (lx2, ly), (rx2, ry), (50, 50, 50), -1)

        # fill based on percent (from bottom)
        if left_pct is not None:
          fill_h = int(bar_h * left_pct)
          top_fill = ry - fill_h
          cv2.rectangle(annotated_image, (lx, top_fill), (rx, ry), (0, 200, 0), -1)
          cv2.putText(annotated_image, f"L {int(left_pct*100)}%", (lx, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        else:
          cv2.putText(annotated_image, "L --%", (lx, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        if right_pct is not None:
          fill_h = int(bar_h * right_pct)
          top_fill = ry - fill_h
          cv2.rectangle(annotated_image, (lx2, top_fill), (rx2, ry), (0, 100, 255), -1)
          cv2.putText(annotated_image, f"R {int(right_pct*100)}%", (lx2, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        else:
          cv2.putText(annotated_image, "R --%", (lx2, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    except Exception:
      logger.exception("Error drawing hand percentage bars")

# Display landmarks 0 to 18 on the camera streaming window.
    if detection_result.pose_landmarks:
      for i in range(19):  # Loop through landmarks 0 to 18.
        landmark = detection_result.pose_landmarks[0][i]
        h, w, _ = rgb_frame.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)  # Draw a green circle for each landmark.
        cv2.putText(annotated_image, str(i), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated_image, f"x:{landmark.x:.2f}, y:{landmark.y:.2f}, z:{landmark.z:.2f}", 
              (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


    cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


# def tutorial_onlystillpicture():
#   # STEP 2: Create a PoseLandmarker object with multi-pose detection enabled.
#   base_options = python.BaseOptions(model_asset_path='lib/pose_landmarker.task')
#   options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True,
#     num_poses=2  # Allow detection of up to 2 people (adjust as needed).
#   )
#   detector = vision.PoseLandmarker.create_from_options(options)

#   # STEP 3: Load the input image.
#   image = mp.Image.create_from_file("lib/pic/girl.jpg")

#   # STEP 4: Detect pose landmarks from the input image.
#   detection_result = detector.detect(image)

#   # STEP 5: Process the detection result. In this case, visualize it.
#   annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#   cv2.imshow("hi", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#   cv2.waitKey(0)

#   segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
#   visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
#   cv2.imshow("hi", visualized_mask)
#   cv2.waitKey(0)


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


if __name__ == "__main__":
  main()
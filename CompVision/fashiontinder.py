import cv2
import mediapipe as mp
import numpy as np
import pygame
import logging

pygame.mixer.init()
like_sound = pygame.mixer.Sound('lib/sounds/like.wav')
dislike_sound = pygame.mixer.Sound('lib/sounds/like.wav')

def likeClothes():
    if not pygame.mixer.get_busy():
        like_sound.play()

def dislikeClothes():
    if not pygame.mixer.get_busy():
        dislike_sound.play()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            if handedness == 'Right':
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                y_pos = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                likeClothes()
            elif handedness == 'Left':
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                dislikeClothes()

    cv2.imshow("Fashion Tinder", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

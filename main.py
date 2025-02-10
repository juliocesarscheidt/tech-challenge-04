import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from deepface import DeepFace
from collections import deque

class PosesDetector:
  def __init__(self, buffer_size=10, waving_movement_threshold=0.02):
    self.left_hand_positions = deque(maxlen=buffer_size)
    self.right_hand_positions = deque(maxlen=buffer_size)
    self.waving_movement_threshold = waving_movement_threshold

    self.motion_history = deque(maxlen=10)

  def is_waving(self, mp_pose, landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    # Check if wrists are above shoulders with a small threshold
    left_arm_up = left_hand.y < (left_shoulder.y - 0.02)
    right_arm_up = right_hand.y < (right_shoulder.y - 0.02)
    # Store hand X positions to track movement over time
    self.left_hand_positions.append(left_hand.x)
    self.right_hand_positions.append(right_hand.x)
    # Detect waving if there's oscillation in X direction
    def detect_waving(hand_positions):
      if len(hand_positions) < 2:
        return False
      movement_range = max(hand_positions) - min(hand_positions)
      return movement_range > self.waving_movement_threshold

    left_waving = left_arm_up and detect_waving(self.left_hand_positions)
    right_waving = right_arm_up and detect_waving(self.right_hand_positions)

    return left_waving, right_waving  # Return whether each hand is waving

  def is_arm_up(self, mp_pose, landmarks, threshold=0.05):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    # Check if elbows are above shoulders with a threshold
    left_arm_up = left_elbow.y < (left_shoulder.y - threshold)
    right_arm_up = right_elbow.y < (right_shoulder.y - threshold)

    return left_arm_up, right_arm_up  # Return a tuple for individual detection

  def is_dancing(self, mp_pose, landmarks):
    # Extract only .y positions for movement tracking
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y

    # Store first frame without calculating movement
    if len(self.motion_history) == 0:
      self.motion_history.append({
        'left_hip': left_hip,
        'right_hip': right_hip,
        'left_hand': left_hand,
        'right_hand': right_hand,
        'left_foot': left_foot,
        'right_foot': right_foot
      })
      return False

    prev = self.motion_history[-1]
    # Compute movement distances (absolute difference)
    hip_movement = abs(left_hip - prev['left_hip']) + abs(right_hip - prev['right_hip'])
    hand_movement = abs(left_hand - prev['left_hand']) + abs(right_hand - prev['right_hand'])
    foot_movement = abs(left_foot - prev['left_foot']) + abs(right_foot - prev['right_foot'])
    # Store movement data in motion history
    self.motion_history.append({
      'left_hip': left_hip,
      'right_hip': right_hip,
      'left_hand': left_hand,
      'right_hand': right_hand,
      'left_foot': left_foot,
      'right_foot': right_foot,
      'hip_movement': hip_movement,
      'hand_movement': hand_movement,
      'foot_movement': foot_movement
    })

    # Ensure we have enough frames for movement analysis
    if len(self.motion_history) > 1:
      avg_hip_movement = np.mean([m['hip_movement'] for m in self.motion_history if 'hip_movement' in m])
      avg_hand_movement = np.mean([m['hand_movement'] for m in self.motion_history if 'hand_movement' in m])
      avg_foot_movement = np.mean([m['foot_movement'] for m in self.motion_history if 'foot_movement' in m])
      # Define better thresholds for dancing detection
      if avg_hand_movement > 0.07 and avg_foot_movement > 0.07 and avg_hip_movement > 0.04:
        return True

    return False

def detect_hands(mp_hands, mp_drawing, mp_drawing_styles, frame, rgb_frame):
  with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    result_hands = hands.process(rgb_frame)
    result_hands_landmarks = result_hands.multi_hand_landmarks
    if result_hands_landmarks:
      for hand_landmarks in result_hands_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())

def write_summary(total_frames, left_arm_movements_count, right_arm_movements_count,
                  waves_count, dancing_count, frames_without_emotions, detected_emotions):
  text = f"""Total de frames analisados {total_frames}
Total de frames sem emoções detectadas: {frames_without_emotions}
Movimentos com o braço esquerdo: {left_arm_movements_count}
Movimentos com o braço direito: {right_arm_movements_count}
Acenos: {waves_count}
Danças: {dancing_count}
Total de emoções detectadas:
"""
  for key, value in detected_emotions.items():
    text += f"\t{key}: {value}\n"
  with open(os.path.join(script_dir, "summary.txt"), "w", encoding='utf-8') as f:
    f.write(text)

def detect_faces_and_emotions(video_path, output_path):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print('Error: Could not open video.')
    return

  poses_detector = PosesDetector()

  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  print('fps', fps)
  print('total_frames', total_frames)

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  frames_without_emotions = 0

  left_arm_up = False
  right_arm_up = False
  left_arm_movements_count = 0
  right_arm_movements_count = 0
  waves_count = 0
  dancing_count = 0

  is_waving = False
  is_dancing = False

  detected_emotions = {
    'sad': 0,
    'happy': 0,
    'angry': 0,
    'fear': 0,
    'surprise': 0,
    'disgust': 0,
    'neutral': 0,
  }
  prev_emotion = None

  for _ in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
      break

    # print("current second", "%.2f" % (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect emotions
    result_emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    for emotions in result_emotions:
      emotions = result_emotions[0]
      x, y, w, h = emotions['region']['x'], emotions['region']['y'], emotions['region']['w'], emotions['region']['h']
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      # Define a threshold for "emotionless" (e.g., neutral > 90%)
      if emotions['emotion']['neutral'] > 90 or max(emotions['emotion'].values()) < 20:
        frames_without_emotions += 1
        prev_emotion = None
      else:
        dominant_emotion = emotions['dominant_emotion']
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # Count emotion only if it is a new distinct occurrence
        if dominant_emotion != prev_emotion:
          detected_emotions[dominant_emotion] += 1
        prev_emotion = dominant_emotion # Update previous emotion

    # detect hands
    detect_hands(mp_hands, mp_drawing, mp_drawing_styles, frame, rgb_frame)

    # detect poses
    result_poses = pose.process(rgb_frame)
    pose_landmarks = result_poses.pose_landmarks
    if pose_landmarks:
      mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
      # detect dancing
      detected_dancing = poses_detector.is_dancing(mp_pose, pose_landmarks.landmark)
      if detected_dancing:
        if not is_dancing:
          is_dancing = True
          dancing_count += 1
      else:
        is_dancing = False

      # detect waves - if not dancing
      if not is_dancing:
        detected_left_waving, detected_right_waving = poses_detector \
          .is_waving(mp_pose, pose_landmarks.landmark)
        if detected_left_waving:
          if not is_waving:
            is_waving = True
            waves_count += 1
        if detected_right_waving:
          if not is_waving:
            is_waving = True
            waves_count += 1

        if not detected_left_waving and not detected_right_waving:
          is_waving = False

        # detect arms up - if not dancing
        detected_left_arm_up, detected_right_arm_up = poses_detector \
          .is_arm_up(mp_pose, pose_landmarks.landmark)

        if detected_left_arm_up or detected_left_waving:
          if not left_arm_up:
            left_arm_up = True
            left_arm_movements_count += 1
        else:
          left_arm_up = False

        if detected_right_arm_up or detected_right_waving:
          if not right_arm_up:
            right_arm_up = True
            right_arm_movements_count += 1
        else:
          right_arm_up = False

    out.write(frame)

  write_summary(total_frames, left_arm_movements_count, right_arm_movements_count,
                waves_count, dancing_count, frames_without_emotions, detected_emotions)

  cap.release()

if __name__ == "__main__":
  script_dir = os.path.dirname(os.path.abspath(__file__))
  video_path = os.path.join(script_dir, "video.mp4")
  output_path = os.path.join(script_dir, "output_video.mp4")
  detect_faces_and_emotions(video_path, output_path)

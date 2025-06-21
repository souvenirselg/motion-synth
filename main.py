import cv2
import mediapipe as mp
from synth_state import SynthState, ADSR
from audio import AudioEngine
import numpy as np
import time

# Finger indices controlling notes (index, middle, ring)
NOTE_FINGER_INDICES = [8, 12, 16]

# Initialize
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sample_rate = 44100

# Setup synth state and audio engine
synth_state = SynthState(sample_rate=sample_rate)
audio_engine = AudioEngine(synth_state)
audio_engine.start()

def get_finger_coords(hand_landmarks, indices, width, height):
    coords = []
    for idx in indices:
        lm = hand_landmarks.landmark[idx]
        coords.append((int(lm.x * width), int(lm.y * height)))
    return coords

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

print("Press 'w' to switch waveform, 'q' to quit.")

waveforms = ["sine", "saw", "square"]
waveform_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            synth_state.notes = []
            # Loop through detected hands
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                # Get handedness info
                handedness_label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display handedness label near wrist
                wrist = hand_landmarks.landmark[0]
                wrist_pos = (int(wrist.x * w), int(wrist.y * h))
                cv2.putText(frame, handedness_label, wrist_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Get finger notes
                finger_points = get_finger_coords(hand_landmarks, NOTE_FINGER_INDICES, w, h)
                for (x, y) in finger_points:
                    freq = 200 + (1 - y / h) * 800
                    synth_state.notes.append(freq)
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)


            # Use hand distance for volume control if two hands detected
            if len(results.multi_hand_landmarks) == 2:
                h1 = results.multi_hand_landmarks[0]
                h2 = results.multi_hand_landmarks[1]
                p1 = (int(h1.landmark[0].x * w), int(h1.landmark[0].y * h))
                p2 = (int(h2.landmark[0].x * w), int(h2.landmark[0].y * h))
                hand_dist = distance(p1, p2)
                max_dist = w * 0.7
                vol = np.clip(hand_dist / max_dist, 0, 1)
                synth_state.volume = vol

            # Use index finger y of first hand for ADSR attack time control (mapped 0.01-0.5s)
            first_hand = results.multi_hand_landmarks[0]
            idx_tip = first_hand.landmark[8]
            attack = np.interp(idx_tip.x, [1, 0], [0.01, 0.5])
            synth_state.adsr.attack = attack

        else:
            synth_state.notes = []
            # Release notes envelope on no hands
            synth_state.adsr.note_off()

        cv2.putText(frame, f"Waveform: {synth_state.waveform}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Volume: {synth_state.volume:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Attack: {synth_state.adsr.attack:.2f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("GestureSynth", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('w'):
            waveform_idx = (waveform_idx + 1) % len(waveforms)
            synth_state.waveform = waveforms[waveform_idx]

finally:
    audio_engine.stop()
    cap.release()
    cv2.destroyAllWindows()

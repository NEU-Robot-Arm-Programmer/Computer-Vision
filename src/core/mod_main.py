import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque

class HandDetector:
	def __init__(self, mode=False, max_hands=2, model_complexity=1,
				 detection_confidence=0.5, tracking_confidence=0.5):
		self.mode = mode
		self.max_hands = max_hands
		self.model_complexity = model_complexity
		self.detection_confidence = detection_confidence
		self.tracking_confidence = tracking_confidence

		# MediaPipe setup
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			static_image_mode=self.mode,
			max_num_hands=self.max_hands,
			model_complexity=self.model_complexity,
			min_detection_confidence=self.detection_confidence,
			min_tracking_confidence=self.tracking_confidence
		)
		self.mp_draw = mp.solutions.drawing_utils
		self.mp_draw_styles = mp.solutions.drawing_styles

		# keep the latest results
		self.result = None
		self._result_lock = threading.Lock()

		# FPS smoothing
		self._fps_buffer = deque(maxlen=30)

	def find_hands(self, img, draw=True):
		"""
		Detect hands in the image/view
		Return the annotated image and set self.result
		"""
		rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		rgb_img.flags.writeable = False
		result = self.hands.process(rgb_img)
		rgb_img.flags.writeable = True

		with self._result_lock:
			self.result = result

		if result.multi_hand_landmarks:
			for i, hands_lms in enumerate(result.multi_hand_landmarks):
				label = result.multi_handedness[i].classification[0].label

				side = self.get_orientation(hands_lms, label)
				color = (0, 255, 0) if side == "Palm" else (0, 0, 255)

				if draw:
					# draw the landmarks with default style connections
					self.mp_draw.draw_landmarks(
						img,
						hands_lms,
						self.mp_hands.HAND_CONNECTIONS,
						self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
						self.mp_draw.DrawingSpec(color=color, thickness=2)
					)
		return img

	def get_orientation(self, hand_landmarks, label):
		"""
		Determines if the palm or back of the hand is visible to the camera
		Corrected for the mirrored positions between the left and right hands using cross
		product of vectors
		"""
		lm = hand_landmarks.landmark

		# 0: Wrist, 5: Index Finger Base, 17: Pinky Base
		# Gonna make 2 vectors for originating from the wrist
		#vector 1 is the wrist to index
		v1 = np.array([lm[5].x - lm[0].x, lm[5].y - lm[0].y, lm[5].z - lm[0].z])
		# Vector 2: Wrist to Pinky Base
		v2 = np.array([lm[17].x - lm[0].x, lm[17].y - lm[0].y, lm[17].z - lm[0].z])

		normal = np.cross(v1, v2)

		if label == "Right":
			return "Palm" if normal[2] > 0 else "Back"
		else:
			# the other hand, should be inverted
			return "Palm" if normal[2] < 0 else "Back"

	def find_positions(self, img, hands_num=0, draw=False):
		"""
		Returns a list of [id, x+px, y_px, z_norm] for every landmark
		on the requested hand.
		Returns [] if no hands is detected.
		Note: z is MediaPipe's normalized depth (relative to teh wrist, real depth will
		be added with RealSense)
		"""
		lm_list = []

		with self._result_lock:
			result = self.result

		if result is None or not result.multi_hand_landmarks:
			return lm_list

		if hands_num >= len(result.multi_hand_landmarks):
			return lm_list

		my_hand = result.multi_hand_landmarks[hands_num]
		h, w, _ = img.shape

		for lm_id, lm in enumerate(my_hand.landmark):
			cx = int(lm.x * w)
			cy = int(lm.y * h)
			lm_list.append([lm_id, cx, cy, lm.z])

			if draw:
				cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

		return lm_list

	def get_hand_label(self, hands_num=0):
		"""
		Return 'left' or 'right', otherwise None if no hand is detected
		"""
		with self._result_lock:
			result = self.result

		if result is None or not result.multi_hand_landmarks:
			return None

		if hands_num >= len(result.multi_hand_landmarks):
			return None

		# MediaPipe labels are from the subject perspective, rightnow base camera
		# is mirrored, gonna glip the table, and it should match with what you see
		# on teh screen

		label = result.multi_handedness[hands_num].classification[0].label
		return "Left" if label == "Right" else "Right"

	def calculate_finger_angles(self, hand_landmarks):
		"""
		Calculate finger angles between the fingertip to base vertex and the base
		to wrist vertex

		Returns a dict {'Thumb': float, 'Index': float, ... , 'Wrist_Flex': float }
		"""
		angles = {}

		# (tip_index, pip_index, base_index)  — pip gives a better lever arm
		finger_triplets = {
			"Thumb": (4, 3, 2),
			"Index": (8, 7, 5),
			"Middle": (12, 11, 9),
			"Ring": (16, 15, 13),
			"Pinky": (20, 19, 17),
		}

		lm = hand_landmarks.landmark

		def to_vec(idx):
			return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

		wrist_vec = to_vec(0)

		for name, (tip_idx, pip_idx, base_idx) in finger_triplets.items():
			tip = to_vec(tip_idx)
			pip = to_vec(pip_idx)
			base = to_vec(base_idx)

			# fingertip direction
			v1 = tip - pip
			# the palm direction
			v2 = base - wrist_vec

			norm1 = np.linalg.norm(v1)
			norm2 = np.linalg.norm(v2)

			if norm1 > 1e-6 and norm2 > 1e-6:
				cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
				angles[name] = float(np.degrees(np.arccos(cos_a)))
			else:
				angles[name] = float("nan")

		# wrist flexion / estimation
		wrist_tip = to_vec(20)
		wrist_base = to_vec(0)
		wrist_dir = wrist_tip - wrist_base
		ref = np.array([0.0, 0.0, 1.0])
		n = np.linalg.norm(wrist_dir)
		if n > 1e-6:
			cos_a = np.clip(np.dot(wrist_dir / n, ref), -1.0, 1.0)
			angles["Wrist_Flex"] = float(np.degrees(np.arccos(cos_a)))
		else:
			angles["Wrist_Flex"] = float("nan")
		return angles

	def calculate_hand_rotation(self, hand_landmarks, label):
		"""
		Going to be used to build a 3-axis coordinate frame with the help of the RealSense to get
		3 angles. Roll (Hand rotating around its own axis), Pitch (Fingers tilting up or down),
		and Yaw (the wrist abduction/adduction)
		"""
		lm = hand_landmarks.landmark

		def lm_vec(idx):
			return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

		# right hand frame
		wrist = lm_vec(0)
		mid_mcp = lm_vec(9)
		idx_mcp = lm_vec(5)
		pnk_mcp = lm_vec(17)

		palm_axis = mid_mcp - wrist  # "up the palm"
		thumb_axis = idx_mcp - pnk_mcp  # "across the palm"

		# normalize
		def safe_norm(vec):
			n = np.linalg.norm(vec)
			return vec / n if n > 1e-6 else vec

		palm_axis = safe_norm(palm_axis)
		thumb_axis = safe_norm(thumb_axis)

		# cross product of the two
		normal = np.cross(palm_axis, thumb_axis)
		normal = safe_norm(normal)

		# flip the normal to be consistent towards the palm face
		if label == "Right" and normal[2] > 0:
			normal = -normal
		elif label == "Left" and normal[2] < 0:
			normal = -normal

		# re-orthogonal the thumb-axis
		thumb_axis = np.cross(normal, palm_axis)
		thumb_axis = safe_norm(thumb_axis)

		# Project the normal onto the plane perpendicular to palm_axis
		n_proj = normal - np.dot(normal, palm_axis) * palm_axis
		n_proj = safe_norm(n_proj)

		# Reference direction = +Z (toward camera) projected out of palm_axis
		ref_z = np.array([0.0, 0.0, 1.0])
		ref_z = ref_z - np.dot(ref_z, palm_axis) * palm_axis
		ref_z = safe_norm(ref_z)

		cos_roll = np.clip(np.dot(n_proj, ref_z), -1.0, 1.0)
		roll = np.degrees(np.arccos(cos_roll))
		# Determine sign using the cross product against palm_axis
		cross_roll = np.cross(ref_z, n_proj)
		if np.dot(cross_roll, palm_axis) < 0:
			roll = -roll

		# Pitch  (fingertips up / down relative to horizontal)
		#   Angle of palm_axis below/above the horizontal (XZ) plane.
		#   Positive = fingers pointing upward in camera view.
		pitch = np.degrees(np.arcsin(np.clip(-palm_axis[1], -1.0, 1.0)))

		# Yaw  (hand panning left / right)
		#   Angle of palm_axis projected onto XZ plane, from +X axis.
		yaw = np.degrees(np.arctan2(palm_axis[0], palm_axis[2]))

		return {
			"pitch": round(float(pitch), 1),
			"yaw": round(float(yaw), 1),
			"roll": round(float(roll), 1),
			"normal": normal,
			"palm_axis": palm_axis,
			"thumb_axis": thumb_axis,
		}


	def get_fingers_up(self, hand_landmarks):
		""" Return the list of 5 integers (1 for up and 0 for down)"""
		fingers = []
		lm = hand_landmarks.landmark

		# Tip IDs for Index, Middle, Ring, Pinky
		tip_ids = [4, 8, 12, 16, 20]

		# Thumb (special case, check X-coordinates)
		if lm[tip_ids[0]].x < lm[tip_ids[0] - 1].x:
			fingers.append(1)
		else:
			fingers.append(0)

		# Other 4 fingers
		for i in range (1, 5):
			if lm[tip_ids[i]].y < lm[tip_ids[i] - 2].y:  # Tip is above the joint
				fingers.append(1)
			else:
				fingers.append(0)
		return fingers  # Returns list like [1, 0, 0, 0, 0] for just Thumb up


class GestureRecognizer:
	"""
	Gesture Recognizer, each gesture is defined as a dict with a
	'fingers' : list of 5 ints
	'side': "Palm", "Back" or None (can be changed)
	'roll-range': (min_deg, max_deg) or None
	'name': display string
	'priority': higher is checked first

	Gestures are checked in descending order, the first match wins.

	you can add your own by doing something like
		recognizer.register(fingers=[1,0,0,0,0], side=None, name="Thumbs Up")
	"""
	def __init__(self):
		self._gestures = []
		self._register_defaults()

	def register(self, fingers, name, side=None, roll_range=None, priority=0):
		"""
		Add a new gesture rule
		"""
		self._gestures.append({
			"fingers": fingers,
			"side": side,
			"roll_range": roll_range,
			"name": name,
			"priority": priority,
		})
		self._gestures.sort(key=lambda g: g["priority"], reverse=True)

	def recognize(self, fingers, side, roll=0.0):
		"""
		fingers : list of 5 ints  [Thumb, Index, Middle, Ring, Pinky]
        side    : "Palm" or "Back"
        roll    : float (degrees) from calculate_hand_rotation

        Returns the gesture name string, or "Unknown" if nothing matches.
        """
		for g in self._gestures:
			if self._match(g, fingers, side, roll):
				return g["name"]
		return "Unknown"

	def _match(self, g, fingers, side, roll):
		# check if the finger pattern is none
		for expected, actual in zip(g["fingers"], fingers):
			if expected is not None and expected != actual:
				return False

		# check the orientation
		if g["side"] is not None and g["side"] != side:
			return False

		if g["roll_range"] is not None:
			lo, hi = g["roll_range"]
			if not (lo <= roll <= hi):
				return False

		return True

	def _register_defaults(self):
		"""
		Built-in gesture library
		# TODO: Extend or override these

		fingers = [Thumb, Index, Middle, Ring, Pinky]
		"""
		gestures = [
			# higher priorities
			dict(fingers=[0, 1, 1, 1, 1], side=None, name="Open Hand", priority=10),
			dict(fingers=[0, 0, 0, 0, 0], side="Palm", name="Fist (Palm)", priority=10),
			dict(fingers=[0, 0, 0, 0, 0], side="Back", name="Fist (Back)", priority=10),

			# == ointing / counting
			dict(fingers=[0, 1, 0, 0, 0], side=None, name="Point", priority=8),
			dict(fingers=[0, 1, 1, 0, 0], side=None, name="Peace / Two", priority=8),
			dict(fingers=[0, 1, 1, 1, 0], side=None, name="Three", priority=7),
			dict(fingers=[1, 1, 1, 1, 1], side=None, name="Four", priority=7),

			dict(fingers=[1, 0, 0, 0, 1], side=None, name="Hang Loose", priority=8),

			# == Thumb gestures
			dict(fingers=[1, 0, 0, 0, 0], side="Palm", name="Thumbs Up", priority=9),
			dict(fingers=[1, 0, 0, 0, 0], side="Back", name="Thumbs Up (Back)", priority=9),

			# == Pinky
			dict(fingers=[0, 0, 0, 0, 1], side=None, name="Pinky Up", priority=7),

			# == Rock
			dict(fingers=[1, 1, 0, 0, 1], side=None, name="Rock On", priority=8),

			# == OK / pinch (thumb + index, others closed)
			dict(fingers=[0, 0, 1, 1, 1], side=None, name="OK / Pinch", priority=8),

			# == L-shape
			dict(fingers=[1, 1, 0, 0, 0], side=None, name="L-Shape", priority=7),

			# == Rotation-based gestures (roll-angle aware)
			# Open hand rotated so palm faces down → "Stop / Halt"
			dict(fingers=[1, 1, 1, 1, 1], side="Back",
			     roll_range=(-30, 30), name="Stop (Back)", priority=11),
		]

		for g in gestures:
			self.register(**g)

class ThreadedCamera:
	"""
	Captures frames in a background thread so the main loop is never blocked waiting
	for the next frame. Eliminating largest latency component in the original code.
	"""
	def __init__(self, src=0, width=640, height=480):
		self.cap = cv2.VideoCapture(src)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		self.ret = False
		self.frame = None
		self._lock = threading.Lock()
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._reader, daemon=True)
		self._thread.start()

	def _reader(self):
		while not self._stop_event.is_set():
			ret, frame = self.cap.read()
			with self._lock:
				self.ret = ret
				self.frame = frame
	def read(self):
		with self._lock:
			return self.ret, (self.frame.copy() if self.frame is not None else None)
	def release(self):
		self._stop_event.set()
		self._thread.join()
		self.cap.release()

def draw_fps(img, fps):
	cv2.putText(img, f"FPS: {int(fps)}",
				(10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (0, 255, 0), 2)

def draw_finger_angles(img, angles, x_offset=10):
	""" Draw per-finger angles in teh bottom left corner"""
	y_start = img.shape[0] - 20 * len(angles) - 10
	for i, (name, val) in enumerate(angles.items()):
		text = f"{name}: {val:.1f} deg" if not np.isnan(val) else f"{name}: --"
		cv2.putText(img, text, (x_offset, y_start + i * 20),
					cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 255), 1)

def draw_rotation(img, rotation, hand_num=0, side="Palm"):
	"""
	Draw the pitch, yaw, and roll in the top right corner
	Stacks vertically per hand so two hands dont overlap
	"""
	pitch = rotation["pitch"]
	yaw = rotation["yaw"]
	roll = rotation["roll"]

	# choose the color to match
	color = (0, 240, 0) if side == "Palm" else (0, 0, 240)
	x = img.shape[1] - 200 # right side of frame
	y_base = 60 + hand_num * 80

	lines = [
		f"Pitch: {pitch:+.1f} deg",
		f"Yaw: {yaw:+.1f} deg",
		f"Roll: {roll:+.1f} deg",
	]

	for i, line in enumerate(lines):
		cv2.putText(img, line, (x, y_base + i * 22),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_rotation_arc(img, roll, cx, cy, radius=40, color=(255, 255, 0)):
	"""
	Draws a small arc + needle on the wrist to give a roll indication
	roll = 0   → needle points straight up
    roll = +90 → needle points right (supination)
    roll = -90 → needle points left  (pronation)
    """
	# Background circle
	cv2.circle(img, (cx, cy), radius, (60, 60, 60), 1)

	# Needle direction: roll=0 → up, positive roll → clockwise
	angle_rad = np.radians(-roll + 90)   # offset so 0 points up
	nx = int(cx + radius * np.cos(angle_rad))
	ny = int(cy - radius * np.sin(angle_rad))
	cv2.line(img, (cx, cy), (nx, ny), color, 2)
	cv2.circle(img, (nx, ny), 4, color, cv2.FILLED)

def draw_gesture(img, gesture, wrist_x, wrist_y, side="Palm"):
	"""
	Draws the guesture name in the filled rounded rectangle above the wrist
	"""
	bg_color = (0, 180, 0) if side == "Palm" else (180, 0, 0)
	text_color = (255, 255, 255)
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.65
	thickness = 2

	(tw, th), _ = cv2.getTextSize(gesture, font, scale, thickness)
	pad = 6
	x1 = wrist_x - tw // 2 - pad
	y1 = wrist_y - 90 - th - pad
	x2 = wrist_x + tw // 2 + pad
	y2 = wrist_y - 90 + pad

	# Clamp to image bounds
	h, w = img.shape[:2]
	x1, y1 = max(x1, 0), max(y1, 0)
	x2, y2 = min(x2, w - 1), min(y2, h - 1)

	cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
	cv2.putText(img, gesture,
	            (x1 + pad, y2 - pad),
	            font, scale, text_color, thickness)


def main():
	cam = ThreadedCamera(src=0, width=640, height=480)
	# check if the camera is actually opened
	if not cam.cap.isOpened():
		print(f"Failed to open camera at index {0}, try another index")
		cam.release()
		cam = ThreadedCamera(src=1, width=640, height=480)

	detector = HandDetector(
		max_hands=2,
		model_complexity=1,
		detection_confidence=0.7,
		tracking_confidence=0.6,
	)

	recognizer = GestureRecognizer()

	pTime = time.time()
	print("Hand Tracking v1 - Press 'q' to quit")
	print("Registed guestures", [g["name"] for g in recognizer._gestures])

	while True:
		ret, img = cam.read()
		if not ret or img is None:
			continue

		img = detector.find_hands(img, draw=True)

		with detector._result_lock:
			result = detector.result

		if result and result.multi_hand_landmarks:
			for hand_num, hand_lms in enumerate(result.multi_hand_landmarks):
				mp_label = result.multi_handedness[hand_num].classification[0].label

				# orientation
				side = detector.get_orientation(hand_lms, mp_label)

				# screen label
				screen_label = detector.get_hand_label(hand_num)

				# landmark positions
				lm_list = detector.find_positions(img, hands_num=hand_num, draw=False)

				angles = detector.calculate_finger_angles(hand_lms)
				draw_finger_angles(img, angles, x_offset=10 + hand_num * 200)

				# hand rotation
				rotation = detector.calculate_hand_rotation(hand_lms, mp_label)
				draw_rotation(img, rotation, hand_num=hand_num, side=side)

				# gestures
				fingers = detector.get_fingers_up(hand_lms)
				gesture = recognizer.recognize(fingers, side, roll=rotation["roll"])

				# wrist label + roll
				if lm_list:
					wrist_x, wrist_y = lm_list[0][1], lm_list[0][2]

					info_text = f"{screen_label or '?'} | {side}"
					cv2.putText(img, info_text, (wrist_x - 40, wrist_y - 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.55,
								(0, 255, 0) if side == "Palm" else (0, 0, 255), 2)

					draw_gesture(img, gesture, wrist_x, wrist_y, side)

					# roll the arc drawn at the wrist position
					draw_rotation_arc(img, rotation["roll"],
									  cx=wrist_x, cy=wrist_y - 60,
									  color=(0, 255, 0) if side == "Palm" else (0, 0, 255))

		#FPS
		cTime = time.time()
		fps = 1.0 / max(cTime - pTime, 1e-6)
		pTime = cTime
		draw_fps(img, fps)

		cv2.imshow("Hand Tracking v3", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

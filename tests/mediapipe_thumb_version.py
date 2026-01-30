"""This script uses mediapipe only to control the actual geometry of Hand V2 in pybullet

-install pybullet, download the required mediapipe model (see below), and download the URDF of the hand prior to use
-thumb tracking is spotty but somewhat working
-to use:
    -start with your hand in a fully open position with the thumb abducted as far as possible 
    (try to match the hand model's initial pose
    -press 'c' to calibrate 
    -you will see a target position based on the thumb tip marker, as well as an actual position on the thumb
    -the IK is solved with numerical methods and sometimes struggles near the joint limits
"""

import os
import time
import math
import numpy as np
import cv2

import pybullet as p
import pybullet_data

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ----------------------------
# USER CONFIG (change here)
# ----------------------------
URDF_PATH = r"C:/Users/epfis/Documents/NU robotics hand/hand teleop/hand_urdf/hand_2.urdf"
URDF_DIR = os.path.dirname(URDF_PATH)

MODEL = r"C:/Users/epfis/Documents/hand tracking/hand_landmarker.task" #download this model from google's mediapipe page
WINDOW = "Hand + PyBullet"

NUM_LANDMARKS = 21
NUM_HANDS = 1

# One Euro filter parameters - tuned for smoothness
ONEEURO_MIN_CUTOFF = 1.5  # Lower = smoother but more lag
ONEEURO_BETA = 0.4        # Lower = smoother but more lag
ONEEURO_D_CUTOFF = 1.0

INDEX_PIP_JOINT_NAME = "index_3"
INDEX_PIP_SIGN = 1.0

# Thumb joint indices
THUMB_J1 = 2
THUMB_J2 = 3
THUMB_ABD = 4
THUMB_J3 = 5
THUMB_J4 = 6

# Wrist joint indices
WRIST_1 = 0
WRIST_2 = 1

# Thumb links
THUMB_LINK_TIP = 6

# Target link for IK
IK_TARGET_LINK = 6

# MediaPipe landmark indices
MP_WRIST = 0
MP_THUMB_TIP = 4
MP_INDEX_MCP = 5
MP_PINKY_MCP = 17

MP_TARGET_LANDMARK = 4

# ----------------------------
# Calibrated transformation parameters
# ----------------------------
DEFAULT_SCALE = 1.28
DEFAULT_AXIS_MAPPING = {
    'robot_x': {'mp_axis': 1, 'sign': 1.0},
    'robot_y': {'mp_axis': 2, 'sign': 1.0},
    'robot_z': {'mp_axis': 0, 'sign': 1.0},
}
DEFAULT_EXTRA_ROTATION = np.array([0.0, 0.0, 0.0])
DEFAULT_OFFSET = np.array([0.0, 0.0, 0.0])
DEFAULT_LOCAL_LINK_OFFSET = np.array([0.026, 0.0, 0.0])


# ----------------------------
# Rotation utilities
# ----------------------------
def rotation_from_euler_deg(rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ----------------------------
# One Euro Filter
# ----------------------------
class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, np.zeros_like(x)
            return x
        dt = max(1e-6, t - self.t_prev)
        a_d = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * self.d_cutoff * dt))
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


# ----------------------------
# Utilities
# ----------------------------
def angle_ABC(A, B, C):
    BA, BC = A - B, C - B
    cos = np.clip(np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9), -1, 1)
    return np.arccos(cos)

def clamp_to_limits(robot_id, joint_idx, value):
    info = p.getJointInfo(robot_id, joint_idx)
    lo, hi = float(info[8]), float(info[9])
    return np.clip(value, lo, hi) if hi > lo else value


# ----------------------------
# MediaPipe to Robot Mapper
# ----------------------------
class MediaPipeToRobotMapper:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.calibrated = False
        
        # Load default calibrated parameters
        self.scale = DEFAULT_SCALE
        self.axis_mapping = DEFAULT_AXIS_MAPPING
        self.extra_rotation = DEFAULT_EXTRA_ROTATION.copy()
        self.offset = DEFAULT_OFFSET.copy()
        self.local_link_offset = DEFAULT_LOCAL_LINK_OFFSET.copy()
        
    def get_robot_palm_pose(self):
        state = p.getLinkState(self.robot_id, 1, computeForwardKinematics=True)
        pos = np.array(state[4])
        R = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        return pos, R
    
    def get_robot_link_pose(self, link_id):
        state = p.getLinkState(self.robot_id, link_id, computeForwardKinematics=True)
        pos = np.array(state[4])
        R = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        return pos, R
    
    def get_offset_target_position(self, link_id):
        pos, R = self.get_robot_link_pose(link_id)
        return pos + R @ self.local_link_offset
    
    def calibrate(self, mp_landmarks):
        fw = np.array([[lm.x, lm.y, lm.z] for lm in mp_landmarks])
        
        self.mp_neutral = fw[MP_TARGET_LANDMARK].copy()
        self.mp_wrist_neutral = fw[MP_WRIST].copy()
        
        mp_index = fw[MP_INDEX_MCP]
        mp_pinky = fw[MP_PINKY_MCP]
        mp_wrist = fw[MP_WRIST]
        
        self.mp_x_axis = mp_index - mp_pinky
        self.mp_x_axis /= np.linalg.norm(self.mp_x_axis)
        
        mp_up = mp_index - mp_wrist
        self.mp_z_axis = np.cross(self.mp_x_axis, mp_up)
        self.mp_z_axis /= np.linalg.norm(self.mp_z_axis)
        
        self.mp_y_axis = np.cross(self.mp_z_axis, self.mp_x_axis)
        self.R_mp_hand = np.column_stack([self.mp_x_axis, self.mp_y_axis, self.mp_z_axis])
        
        self.robot_palm_pos, self.robot_palm_R = self.get_robot_palm_pose()
        self.robot_neutral = self.get_offset_target_position(IK_TARGET_LINK)
        
        self.calibrated = True
        print("Calibrated!")
        
    def build_transform_matrix(self):
        P = np.zeros((3, 3))
        for i, axis in enumerate(['robot_x', 'robot_y', 'robot_z']):
            mp_idx = self.axis_mapping[axis]['mp_axis']
            sign = self.axis_mapping[axis]['sign']
            P[i, mp_idx] = sign
        R_extra = rotation_from_euler_deg(*self.extra_rotation)
        return R_extra @ P
    
    def transform_mp_to_robot(self, mp_landmarks):
        if not self.calibrated:
            return None
        
        fw = np.array([[lm.x, lm.y, lm.z] for lm in mp_landmarks])
        mp_target = fw[MP_TARGET_LANDMARK]
        mp_wrist = fw[MP_WRIST]
        
        mp_index = fw[MP_INDEX_MCP]
        mp_pinky = fw[MP_PINKY_MCP]
        
        mp_x = mp_index - mp_pinky
        mp_x /= np.linalg.norm(mp_x) + 1e-9
        
        mp_up = mp_index - mp_wrist
        mp_z = np.cross(mp_x, mp_up)
        mp_z /= np.linalg.norm(mp_z) + 1e-9
        
        mp_y = np.cross(mp_z, mp_x)
        R_mp_current = np.column_stack([mp_x, mp_y, mp_z])
        
        mp_target_rel = mp_target - mp_wrist
        mp_target_local = R_mp_current.T @ mp_target_rel
        
        mp_neutral_rel = self.mp_neutral - self.mp_wrist_neutral
        mp_neutral_local = self.R_mp_hand.T @ mp_neutral_rel
        
        delta_mp_local = mp_target_local - mp_neutral_local
        delta_scaled = delta_mp_local * self.scale
        
        T = self.build_transform_matrix()
        delta_robot_local = T @ delta_scaled + self.offset
        
        target_world = self.robot_neutral + self.robot_palm_R @ delta_robot_local
        return target_world


# ----------------------------
# Position IK Solver
# ----------------------------
class ThumbPositionIK:
    def __init__(self, robot_id, target_link=THUMB_LINK_TIP):
        self.robot_id = robot_id
        self.target_link = target_link
        self.limits_lo = np.array([0.0, -.35, 0.0])
        self.limits_hi = np.array([0.785, 1.67, 1.57])
        self.q = np.array([0.0, 0.0, 0.0])
        self.joint_weights = np.array([1.0, 2.0, 2.0])
        self.damping = 0.05  # Increased for stability
        self.max_iterations = 10
        self.tolerance = 0.002
        self.step_size = 0.4  # Reduced for smoother motion
        self.local_offset = DEFAULT_LOCAL_LINK_OFFSET.copy()
        
        # Joint limit avoidance parameters
        self.limit_margin = 0.05
        self.limit_gain = 5.0
        
        # Deadband - ignore small position errors (meters)
        self.deadband = 0.003  # 3mm
        
        # Joint smoothing - exponential moving average
        self.q_smoothed = np.array([0.0, 0.0, 0.0])
        self.smoothing_factor = 0.3  # 0 = no smoothing, 1 = no movement
        
    def set_joints(self, q, smooth=True):
        self.q = np.clip(q, self.limits_lo, self.limits_hi)
        
        if smooth:
            # Apply exponential smoothing to joint outputs
            self.q_smoothed = (self.smoothing_factor * self.q_smoothed + 
                              (1 - self.smoothing_factor) * self.q)
            q_output = self.q_smoothed
        else:
            q_output = self.q
            self.q_smoothed = self.q.copy()
        
        p.resetJointState(self.robot_id, THUMB_J1, q_output[0])
        p.resetJointState(self.robot_id, THUMB_J2, -q_output[0])
        p.resetJointState(self.robot_id, THUMB_ABD, q_output[1])
        p.resetJointState(self.robot_id, THUMB_J3, q_output[2])
        p.resetJointState(self.robot_id, THUMB_J4, 0.0)
    
    def get_link_pose(self, link_id=None):
        if link_id is None:
            link_id = self.target_link
        state = p.getLinkState(self.robot_id, link_id, computeForwardKinematics=True)
        pos = np.array(state[4])
        R = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        return pos, R
    
    def get_link_position(self, link_id=None):
        pos, R = self.get_link_pose(link_id)
        return pos + R @ self.local_offset
    
    def compute_limit_avoidance(self):
        dq_avoid = np.zeros(3)
        for j in range(3):
            range_j = self.limits_hi[j] - self.limits_lo[j]
            margin = min(self.limit_margin, range_j * 0.1)
            
            dist_to_lo = self.q[j] - self.limits_lo[j]
            dist_to_hi = self.limits_hi[j] - self.q[j]
            
            if dist_to_lo < margin:
                dq_avoid[j] = self.limit_gain * (margin - dist_to_lo) / margin
            elif dist_to_hi < margin:
                dq_avoid[j] = -self.limit_gain * (margin - dist_to_hi) / margin
                
        return dq_avoid
    
    def compute_clamped_weights(self):
        W = np.diag(self.joint_weights.copy())
        for j in range(3):
            range_j = self.limits_hi[j] - self.limits_lo[j]
            margin = min(self.limit_margin * 2, range_j * 0.15)
            
            dist_to_lo = self.q[j] - self.limits_lo[j]
            dist_to_hi = self.limits_hi[j] - self.q[j]
            
            if dist_to_lo < margin:
                scale = dist_to_lo / margin
                W[j, j] *= max(0.1, scale)
            elif dist_to_hi < margin:
                scale = dist_to_hi / margin
                W[j, j] *= max(0.1, scale)
                
        return W
    
    def compute_jacobian(self, eps=0.003):
        J = np.zeros((3, 3))
        q_save = self.q.copy()
        for j in range(3):
            q_plus = q_save.copy()
            q_plus[j] = min(q_plus[j] + eps, self.limits_hi[j])
            q_minus = q_save.copy()
            q_minus[j] = max(q_minus[j] - eps, self.limits_lo[j])
            self.set_joints(q_plus, smooth=False)
            pos_plus = self.get_link_position()
            self.set_joints(q_minus, smooth=False)
            pos_minus = self.get_link_position()
            actual_eps = q_plus[j] - q_minus[j]
            if actual_eps > 1e-9:
                J[:, j] = (pos_plus - pos_minus) / actual_eps
            else:
                J[:, j] = 0
        self.set_joints(q_save, smooth=False)
        return J
    
    def solve(self, target_pos, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iterations
            
        for _ in range(max_iter):
            current_pos = self.get_link_position()
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            # Deadband - ignore small errors to reduce jitter
            if error_norm < self.deadband:
                break
                
            if error_norm < self.tolerance:
                break
            
            # Adaptive damping
            adaptive_damping = self.damping * (1.0 + error_norm * 10)
            
            J = self.compute_jacobian()
            W = self.compute_clamped_weights()
            
            JW = J @ W
            damped = JW @ J.T + (adaptive_damping ** 2) * np.eye(3)
            
            try:
                dq = W @ J.T @ np.linalg.solve(damped, error)
            except np.linalg.LinAlgError:
                continue
            
            # Add limit avoidance
            dq_avoid = self.compute_limit_avoidance()
            dq += 0.1 * dq_avoid
            
            # Limit maximum joint velocity per step
            max_dq = 0.1  # Reduced for smoother motion
            dq_norm = np.linalg.norm(dq)
            if dq_norm > max_dq:
                dq = dq * max_dq / dq_norm
            
            q_new = self.q + self.step_size * dq
            self.set_joints(q_new, smooth=False)
        
        # Apply smoothing on final output
        self.set_joints(self.q, smooth=True)
        return self.q_smoothed.copy()
    
    def reset(self):
        self.q = np.array([0.0, 0.0, 0.0])
        self.q_smoothed = np.array([0.0, 0.0, 0.0])
        self.set_joints(self.q, smooth=False)


# ----------------------------
# Debug Visualization
# ----------------------------
class DebugViz:
    def __init__(self):
        self.items = {}
    
    def draw_point(self, pos, name, color=[1, 0, 1], size=0.008):
        for i, axis in enumerate([[1,0,0], [0,1,0], [0,0,1]]):
            a = np.array(axis) * size
            key = f"{name}_ax{i}"
            if key in self.items:
                self.items[key] = p.addUserDebugLine(pos - a, pos + a, color, 2, 
                                                      replaceItemUniqueId=self.items[key])
            else:
                self.items[key] = p.addUserDebugLine(pos - a, pos + a, color, 2)

viz = DebugViz()


# ----------------------------
# PyBullet / MediaPipe init
# ----------------------------
def init_pybullet():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    os.chdir(URDF_DIR)
    robot_id = p.loadURDF(URDF_PATH, useFixedBase=True)
    
    name_to_jidx = {}
    for j in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, j)[1].decode("utf-8")
        name_to_jidx[name] = j
    
    return robot_id, name_to_jidx

def init_mediapipe():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )
    return vision.HandLandmarker.create_from_options(options)


# ----------------------------
# MAIN
# ----------------------------
def main():
    robot_id, name_to_jidx = init_pybullet()
    landmarker = init_mediapipe()
    cap = cv2.VideoCapture(0) #use 0 for built in webcam, 1 for external webcam, and 2 for RGB only Realsense camera
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    
    p.resetJointState(robot_id, WRIST_1, 0)
    p.resetJointState(robot_id, WRIST_2, 0)
    
    ik_solver = ThumbPositionIK(robot_id, target_link=IK_TARGET_LINK)
    ik_solver.reset()
    
    mapper = MediaPipeToRobotMapper(robot_id)
    
    # Stronger filtering on target position for smoothness
    target_filter = OneEuro(min_cutoff=1.5, beta=0.3, d_cutoff=1.0)
    world_filters = [[OneEuro(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF) 
                      for _ in range(NUM_LANDMARKS)] for _ in range(NUM_HANDS)]
    
    t0 = time.perf_counter()
    
    print("\nHand Teleoperation")
    print("  'c' = Calibrate (hold hand flat, thumb extended)")
    print("  'r' = Reset")
    print("  ESC = Exit\n")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        t = time.perf_counter()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int((t - t0) * 1000))
        
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks[:NUM_HANDS]:
                proto = landmark_pb2.NormalizedLandmarkList(
                    landmark=[landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                             for lm in hand_lms])
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, proto, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
        
        target_pos = None
        if result.hand_world_landmarks:
            hw = result.hand_world_landmarks[0]
            fw = np.array([[lm.x, lm.y, lm.z] for lm in hw])
            
            for i in range(NUM_LANDMARKS):
                fw[i] = world_filters[0][i](fw[i], t)
            
            def setj(name, val):
                if name in name_to_jidx:
                    idx = name_to_jidx[name]
                    p.resetJointState(robot_id, idx, clamp_to_limits(robot_id, idx, val))
            
            # Index
            b = np.clip(np.pi - angle_ABC(fw[0], fw[5], fw[6]), -0.2, 1.6)
            setj("index_1", -0.6*b); setj("index_2", -0.6*b)
            setj(INDEX_PIP_JOINT_NAME, INDEX_PIP_SIGN * np.clip(
                1.8*(np.pi - angle_ABC(fw[5], fw[6], fw[7])), 0, 2.1))
            
            # Middle
            b = np.clip(np.pi - angle_ABC(fw[0], fw[9], fw[10]), -0.2, 1.6)
            setj("middle_1", -0.6*b); setj("middle_2", 0.6*b)
            setj("middle_3", np.clip(1.5*(np.pi - angle_ABC(fw[9], fw[10], fw[11])), 0, 2.1))
            
            # Ring
            b = np.clip(np.pi - angle_ABC(fw[0], fw[13], fw[14]), -0.2, 1.6)
            setj("ring_1", -0.6*b); setj("ring_2", 0.6*b)
            setj("ring_3", np.clip(1.5*(np.pi - angle_ABC(fw[13], fw[14], fw[15])), 0, 2.1))
            
            # Pinky
            b = np.clip(np.pi - angle_ABC(fw[0], fw[17], fw[18]), -0.2, 1.6)
            setj("pinky_1", -0.6*b); setj("pinky_2", 0.6*b)
            setj("pinky_3", -np.clip(1.5*(np.pi - angle_ABC(fw[17], fw[18], fw[19])), 0, 2.1))
            
            if mapper.calibrated:
                target_pos = mapper.transform_mp_to_robot(hw)
        
        if target_pos is not None:
            target_pos = target_filter(target_pos, t)
            q = ik_solver.solve(target_pos, max_iter=10)
            
            viz.draw_point(target_pos, "target", [1, 0, 1])  # Magenta = target
            viz.draw_point(ik_solver.get_link_position(), "actual", [0, 1, 1])  # Cyan = actual
            
            err = np.linalg.norm(target_pos - ik_solver.get_link_position()) * 1000
            cv2.putText(frame, f"Error: {err:.1f}mm", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            if not mapper.calibrated:
                cv2.putText(frame, "Press 'c' to calibrate", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        p.stepSimulation()
        p.resetJointState(robot_id, WRIST_1, 0)
        p.resetJointState(robot_id, WRIST_2, 0)
        
        cv2.imshow(WINDOW, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and result.hand_world_landmarks:
            ik_solver.reset()
            p.stepSimulation()
            mapper.calibrate(result.hand_world_landmarks[0])
            target_filter = OneEuro(min_cutoff=1.5, beta=0.3, d_cutoff=1.0)
            
        elif key == ord('r'):
            ik_solver.reset()
            mapper.calibrated = False
            target_filter = OneEuro(min_cutoff=1.5, beta=0.3, d_cutoff=1.0)
            print("Reset")
            
        elif key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()


if __name__ == "__main__":

    main()

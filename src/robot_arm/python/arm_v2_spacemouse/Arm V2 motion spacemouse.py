#!/usr/bin/env python3
"""
6-Axis Robot Teleoperation — SpaceMouse Edition (console only)
--------------------------------------------------------------
SpaceMouse axes → robot mapping (all tunable via constants below):
  Translation  : X/Y/Z stick  → robot X/Y/Z endpoint position
  Rotation     : roll/pitch/yaw stick → robot roll/pitch/yaw (RPY)
  ENTER        : clear IK fault and return to home
  r + ENTER    : reset RPY to zero
  q + ENTER    : quit
"""

import asyncio
import threading
import time
import math
import traceback

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import moteus
import pyspacemouse

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS


# ============================================================
# USER CONFIG — Motion
# ============================================================

CONTROL_DT  = 0.01          # seconds between motor commands
SM_DT       = 0.01          # seconds between SpaceMouse polls (~100 Hz)
IK_INTERVAL = 1.0 / 30.0   # max IK solve rate while SpaceMouse is moving

HOME_POS = np.array([0.4, 0.0, 0.3], dtype=float)
HOME_RPY = np.array([0.0,  0.0, 0.0], dtype=float)

POS_MIN = np.array([ 0.10, -0.40, 0.05])
POS_MAX = np.array([ 0.55,  0.40, 0.60])

SAFE_INTERP_TIME = 3.0   # seconds to blend to safe pose after IK fault


# ============================================================
# USER CONFIG — SpaceMouse sensitivity & axis mapping
# ============================================================

SM_TRANS_SCALE = 0.003   # metres moved per raw tick — reduce if too fast
SM_ROT_SCALE   = 0.008    # radians per raw tick
SM_DEADZONE    = 0.05     # raw values below this are ignored

# Which SpaceMouse axis drives each robot DOF.
# Flip 'sign' to invert an axis.
SM_AXIS_MAP = {
    'pos_x'    : ('y',     +1.0),   # push forward   -> +X
    'pos_y'    : ('x',     -1.0),   # push right      -> -Y
    'pos_z'    : ('z',     +1.0),   # push up         -> +Z
    'rpy_roll' : ('roll',  +1.0),
    'rpy_pitch': ('pitch', +1.0),
    'rpy_yaw'  : ('yaw',   +1.0),
}

SM_ROTATION_ENABLED = True   # set False for position-only control


# ============================================================
# USER CONFIG — Moteus
# ============================================================

CONTROLLER_IDS = {
    1: 5,   # base
    2: 4,   # shoulder motor A
    3: 3,   # shoulder motor B
    4: 2,   # elbow
    5: 1,   # forearm roll
}

ACTIVE_JOINTS = [1, 2, 3, 4, 5]

JOINT_CONFIG = {
    1: {"ratio": 10.0, "sign": -1.0},  # base
    2: {"ratio": 10.0, "sign": 1.0},  # shoulder A
    3: {"ratio": 10.0, "sign": 1.0},  # shoulder B
    4: {"ratio": 10.0, "sign": -1.0},  # elbow
    5: {"ratio":  6.0, "sign": -1.0},  # forearm roll
}

MAX_TORQUE = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.8}

# Shoulder differential tuning
SHOULDER_ROLL_SIGN  = -1.0
SHOULDER_PITCH_SIGN =  1.0
SHOULDER_MIX_SCALE  =  1.0


# ============================================================
# USER CONFIG — Dynamixel wrist (joint 6)
# ============================================================

DXL_DEVICENAME = "COM8"
DXL_BAUDRATE   = 57600
DXL_ID         = 1
DXL_SIGN       = -1.0
DXL_OFFSET_RAD = 0.0
DXL_GEAR_RATIO = 0.5

DXL_CENTER_POS     = 1861
DXL_COUNTS_PER_REV = 4096

DXL_PROTOCOL_VERSION    = 2.0
ADDR_TORQUE_ENABLE      = 64
ADDR_OPERATING_MODE     = 11
ADDR_PROFILE_ACCEL      = 108
ADDR_PROFILE_VELOCITY   = 112
ADDR_MAX_POSITION_LIMIT = 48
ADDR_MIN_POSITION_LIMIT = 52
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132
ADDR_PRESENT_CURRENT    = 126
ADDR_HARDWARE_ERROR     = 70

POSITION_CONTROL_MODE = 3
A_PER_LSB             = 0.00269


# ============================================================
# HELPERS — Moteus
# ============================================================

def joint_to_motor_rev(q_rad, gear_ratio=1.0, sign=1.0):
    return (sign * q_rad / (2.0 * math.pi)) * gear_ratio


def shoulder_mix(q_pitch, q_roll):
    qr = SHOULDER_ROLL_SIGN * q_roll
    qp = SHOULDER_PITCH_SIGN * q_pitch
    return SHOULDER_MIX_SCALE * (qr + qp), SHOULDER_MIX_SCALE * (qr - qp)


def cosine_interp(a, b, frac):
    alpha = 0.5 * (1.0 - math.cos(math.pi * frac))
    return {k: a[k] * (1.0 - alpha) + b[k] * alpha for k in a}


# ============================================================
# HELPERS — Dynamixel
# ============================================================

def s16(v):
    return v - 65536 if v > 32767 else v


def rad_to_dxl_pos(q_rad, sign=1.0, offset=0.0, gear_ratio=1.0,
                   center=DXL_CENTER_POS, counts_per_rev=DXL_COUNTS_PER_REV):
    ticks = int(round((sign * (q_rad + offset) * gear_ratio / (2.0 * math.pi)) * counts_per_rev))
    return center + ticks


def dxl_die(pk, msg, res=None, err=None):
    if res is not None:
        msg += f" | comm={pk.getTxRxResult(res)}"
    if err:
        msg += f" | err={pk.getRxPacketError(err)}"
    raise RuntimeError(msg)


# ============================================================
# DYNAMIXEL DRIVER
# ============================================================

class DynamixelWrist:
    def __init__(self, device, baudrate, dxl_id):
        self.dxl_id  = dxl_id
        self.port    = PortHandler(device)
        self.pk      = PacketHandler(DXL_PROTOCOL_VERSION)
        self._open   = False
        self.min_pos = 0
        self.max_pos = DXL_COUNTS_PER_REV - 1

    def connect(self):
        print(f"Dynamixel: opening {DXL_DEVICENAME} @ {DXL_BAUDRATE} baud...")
        if not self.port.openPort():
            raise RuntimeError(f"Dynamixel: failed to open {DXL_DEVICENAME}")
        if not self.port.setBaudRate(DXL_BAUDRATE):
            raise RuntimeError("Dynamixel: failed to set baudrate")
        print("Dynamixel: port open")

        hw_err = self._read1(ADDR_HARDWARE_ERROR)
        if hw_err != 0:
            print(f"Dynamixel: hardware error 0x{hw_err:02X} — rebooting to clear...")
            self.pk.reboot(self.port, self.dxl_id)
            time.sleep(0.5)
            hw_err = self._read1(ADDR_HARDWARE_ERROR)
            if hw_err != 0:
                raise RuntimeError(f"Dynamixel: error 0x{hw_err:02X} persists after reboot")
            print("Dynamixel: hardware error cleared")
        else:
            print("Dynamixel: no hardware errors")

        self._write1(ADDR_TORQUE_ENABLE, 0)
        self._write1(ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
        mode = self._read1(ADDR_OPERATING_MODE)
        if mode != POSITION_CONTROL_MODE:
            raise RuntimeError(f"Dynamixel: mode={mode}, expected {POSITION_CONTROL_MODE}")
        print(f"Dynamixel: operating mode = {mode} (position control) OK")

        self.min_pos = self._read4(ADDR_MIN_POSITION_LIMIT)
        self.max_pos = self._read4(ADDR_MAX_POSITION_LIMIT)
        print(f"Dynamixel: position limits [{self.min_pos}, {self.max_pos}] ticks "
              f"({self.min_pos / DXL_COUNTS_PER_REV * 360:.1f} deg - "
              f"{self.max_pos / DXL_COUNTS_PER_REV * 360:.1f} deg)")

        for addr, name, val in ((ADDR_PROFILE_ACCEL,   "profile_accel", 50),
                                (ADDR_PROFILE_VELOCITY, "profile_vel",  200)):
            try:
                self._write4(addr, val)
                print(f"Dynamixel: {name} set to {val}")
            except RuntimeError as e:
                print(f"Dynamixel: WARNING — could not write {name}: {e}")

        self._write1(ADDR_TORQUE_ENABLE, 1)
        if self._read1(ADDR_TORQUE_ENABLE) != 1:
            raise RuntimeError("Dynamixel: torque enable readback failed")
        self._open = True
        print(f"Dynamixel ID {self.dxl_id} ready — torque ON")

    def set_position(self, goal_tick):
        goal_tick = max(self.min_pos, min(self.max_pos, goal_tick))
        self._write4(ADDR_GOAL_POSITION, goal_tick)

    def query(self):
        return (self._read4(ADDR_PRESENT_POSITION),
                s16(self._read2(ADDR_PRESENT_CURRENT)) * A_PER_LSB)

    def stop(self):
        if self._open:
            try:
                self._write1(ADDR_TORQUE_ENABLE, 0)
            except Exception:
                pass
            self.port.closePort()
            self._open = False
            print("Dynamixel torque disabled, port closed")

    def _write1(self, addr, val):
        res, err = self.pk.write1ByteTxRx(self.port, self.dxl_id, addr, val)
        if res != COMM_SUCCESS or err != 0:
            dxl_die(self.pk, f"write1 addr={addr} val={val}", res, err)

    def _write4(self, addr, val):
        res, err = self.pk.write4ByteTxRx(self.port, self.dxl_id, addr, val & 0xFFFFFFFF)
        if res != COMM_SUCCESS or err != 0:
            dxl_die(self.pk, f"write4 addr={addr} val={val}", res, err)

    def _read1(self, addr):
        val, res, err = self.pk.read1ByteTxRx(self.port, self.dxl_id, addr)
        if res != COMM_SUCCESS or err != 0:
            dxl_die(self.pk, f"read1 addr={addr}", res, err)
        return val

    def _read2(self, addr):
        val, res, err = self.pk.read2ByteTxRx(self.port, self.dxl_id, addr)
        if res != COMM_SUCCESS or err != 0:
            dxl_die(self.pk, f"read2 addr={addr}", res, err)
        return val

    def _read4(self, addr):
        val, res, err = self.pk.read4ByteTxRx(self.port, self.dxl_id, addr)
        if res != COMM_SUCCESS or err != 0:
            dxl_die(self.pk, f"read4 addr={addr}", res, err)
        return val


# ============================================================
# SHARED STATE
# ============================================================

class TeleopState:
    def __init__(self):
        self.lock            = threading.Lock()
        self.pos             = HOME_POS.copy()
        self.rpy             = HOME_RPY.copy()
        self.q_current       = None
        self.q_cmd_joint     = {jid: 0.0 for jid in ACTIVE_JOINTS}
        self.q_cmd_wrist     = 0.0
        self.last_ik_ok      = False
        self.running         = True
        self.print_requested = True
        self.sm_active       = False

        # Safe pose — populated from initial IK in main()
        self.safe_q      = {jid: 0.0 for jid in ACTIVE_JOINTS}
        self.safe_wrist  = 0.0

        # Fault state
        self.faulted           = False
        self.fault_start_time  = 0.0
        self.fault_start_q     = {jid: 0.0 for jid in ACTIVE_JOINTS}
        self.fault_start_wrist = 0.0

    def trigger_fault(self):
        if not self.faulted:
            self.faulted           = True
            self.fault_start_time  = time.monotonic()
            self.fault_start_q     = dict(self.q_cmd_joint)
            self.fault_start_wrist = self.q_cmd_wrist
            self.sm_active         = False
            print("\n[!] IK FAULT — retreating to safe pose.")
            print("    Press ENTER to clear fault, or 'r' + ENTER to reset.\n")

    def clear_fault(self):
        self.faulted         = False
        self.last_ik_ok      = False
        self.sm_active       = False
        self.pos[:]          = HOME_POS
        self.rpy[:]          = HOME_RPY
        self.print_requested = True
        print("Fault cleared — re-solving IK from home.")


state = TeleopState()


# ============================================================
# ROBOT MODEL + IK
# ============================================================

from NURoboticsArm import NURoboticsArm
robot = NURoboticsArm()
robot.tool = SE3.Rx(np.pi / 2)


def solve_ik_and_update_command():
    with state.lock:
        if state.faulted:
            return False
        Tep = SE3.Trans(*state.pos) * SE3.RPY(state.rpy)
        q0  = robot.qr if state.q_current is None else state.q_current.copy()

    sol = robot.ik_LM(Tep, q0=q0)
    if not sol[1]:
        with state.lock:
            state.last_ik_ok = False
            state.trigger_fault()
        return False

    q_sol = np.array(sol[0], dtype=float)
    if len(q_sol) < 6:
        print(f"IK returned {len(q_sol)} joints, expected >=6")
        with state.lock:
            state.last_ik_ok = False
            state.trigger_fault()
        return False

    q1, q_roll, q_pitch, q4, q5, q6 = q_sol[:6]
    a2, a3 = shoulder_mix(q_roll, q_pitch)

    with state.lock:
        state.q_current   = q_sol
        state.q_cmd_joint = {1: q1, 2: a2, 3: a3, 4: q4, 5: q5}
        state.q_cmd_wrist = q6
        state.last_ik_ok  = True
    return True


# ============================================================
# SPACEMOUSE INPUT THREAD
# ============================================================

def _deadzone(value, dz):
    return 0.0 if abs(value) < dz else value


def spacemouse_thread():
    print("Opening SpaceMouse...")
    try:
        device = pyspacemouse.open()
        if not device:
            raise RuntimeError("pyspacemouse.open() returned None/False — device not found")
    except Exception:
        print("=" * 60)
        print("ERROR: SpaceMouse failed to open — position input disabled.")
        traceback.print_exc()
        print("=" * 60)
        return

    print("SpaceMouse ready.")
    print("  Push/pull/lift/slide : XYZ position")
    print("  Twist               : RPY orientation")

    last_ik_t    = 0.0

    try:
        while True:
            with state.lock:
                if not state.running:
                    break
                faulted = state.faulted

            sm = device.read()
            t  = time.perf_counter()

            if faulted:
                time.sleep(SM_DT)
                continue

            # ---- Axis reading -------------------------------------------
            raw = {
                'x':     _deadzone(sm.x,     SM_DEADZONE),
                'y':     _deadzone(sm.y,     SM_DEADZONE),
                'z':     _deadzone(sm.z,     SM_DEADZONE),
                'roll':  _deadzone(sm.roll,  SM_DEADZONE),
                'pitch': _deadzone(sm.pitch, SM_DEADZONE),
                'yaw':   _deadzone(sm.yaw,   SM_DEADZONE),
            }

            moving = any(v != 0.0 for v in raw.values())

            if moving:
                with state.lock:
                    for dof, (attr, sign) in (
                        ('pos_x', SM_AXIS_MAP['pos_x']),
                        ('pos_y', SM_AXIS_MAP['pos_y']),
                        ('pos_z', SM_AXIS_MAP['pos_z']),
                    ):
                        idx = {'pos_x': 0, 'pos_y': 1, 'pos_z': 2}[dof]
                        state.pos[idx] += sign * raw[attr] * SM_TRANS_SCALE

                    if SM_ROTATION_ENABLED:
                        for dof, (attr, sign) in (
                            ('rpy_roll',  SM_AXIS_MAP['rpy_roll']),
                            ('rpy_pitch', SM_AXIS_MAP['rpy_pitch']),
                            ('rpy_yaw',   SM_AXIS_MAP['rpy_yaw']),
                        ):
                            idx = {'rpy_roll': 0, 'rpy_pitch': 1, 'rpy_yaw': 2}[dof]
                            state.rpy[idx] += sign * raw[attr] * SM_ROT_SCALE

                    state.pos[:] = np.clip(state.pos, POS_MIN, POS_MAX)
                    state.sm_active       = True
                    state.print_requested = True

                if t - last_ik_t >= IK_INTERVAL:
                    solve_ik_and_update_command()
                    last_ik_t = t
            else:
                with state.lock:
                    state.sm_active = False

            time.sleep(SM_DT)

    except Exception:
        print("=" * 60)
        print("ERROR: SpaceMouse thread crashed")
        traceback.print_exc()
        print("=" * 60)
        with state.lock:
            state.sm_active = False
    finally:
        try:
            pyspacemouse.close()
        except Exception:
            pass
        print("SpaceMouse closed")


# ============================================================
# MOTEUS + DYNAMIXEL CONTROL LOOP
# ============================================================

async def moteus_control_loop():
    qr = moteus.QueryResolution()
    qr._extra = {moteus.Register.Q_CURRENT: moteus.F32}

    controllers = {
        jid: moteus.Controller(id=CONTROLLER_IDS[jid], query_resolution=qr)
        for jid in ACTIVE_JOINTS
    }

    moteus_ok = False
    zero_pos  = {jid: 0.0 for jid in ACTIVE_JOINTS}
    try:
        for jid, ctl in controllers.items():
            st = await ctl.query()
            zero_pos[jid] = st.values[moteus.Register.POSITION]
            print(f"Joint {jid} (CAN {CONTROLLER_IDS[jid]}): "
                  f"zero_pos = {zero_pos[jid]:.4f} rev")
        await asyncio.gather(*(c.set_recapture_position_velocity() for c in controllers.values()))
        await asyncio.gather(*(c.set_stop()                        for c in controllers.values()))
        moteus_ok = True
    except Exception:
        print("=" * 60)
        print("WARNING: Moteus init failed — running without CAN hardware")
        traceback.print_exc()
        print("=" * 60)

    print("Moteus init complete — starting Dynamixel setup...")

    loop   = asyncio.get_running_loop()
    dxl    = DynamixelWrist(DXL_DEVICENAME, DXL_BAUDRATE, DXL_ID)
    dxl_ok = False
    try:
        await loop.run_in_executor(None, dxl.connect)
        dxl_ok = True
    except Exception:
        print("=" * 60)
        print("ERROR: Dynamixel wrist init failed — wrist will be skipped")
        traceback.print_exc()
        print("=" * 60)

    print("Control loop started")
    last_print = time.time()

    try:
        while True:
            with state.lock:
                running            = state.running
                faulted            = state.faulted
                fault_start_time   = state.fault_start_time
                fault_start_q      = dict(state.fault_start_q)
                fault_start_wrist  = state.fault_start_wrist
                safe_q             = dict(state.safe_q)
                safe_wrist         = state.safe_wrist
                q_cmd_joint        = dict(state.q_cmd_joint)
                q_cmd_wrist        = state.q_cmd_wrist
                last_ik_ok         = state.last_ik_ok
                print_requested    = state.print_requested
                pos                = state.pos.copy()
                rpy                = state.rpy.copy()

            if not running:
                break

            if faulted:
                elapsed = time.monotonic() - fault_start_time
                frac    = min(1.0, elapsed / SAFE_INTERP_TIME)
                alpha   = 0.5 * (1.0 - math.cos(math.pi * frac))
                q_send  = cosine_interp(fault_start_q, safe_q, frac)
                w_send  = fault_start_wrist * (1.0 - alpha) + safe_wrist * alpha
            elif not last_ik_ok:
                await asyncio.sleep(CONTROL_DT)
                continue
            else:
                q_send = q_cmd_joint
                w_send = q_cmd_wrist

            results = []
            if moteus_ok:
                cmds = []
                for jid, ctl in controllers.items():
                    cfg = JOINT_CONFIG[jid]
                    cmds.append(ctl.set_position(
                        position=zero_pos[jid] + joint_to_motor_rev(
                            q_send[jid], gear_ratio=cfg["ratio"], sign=cfg["sign"]),
                        velocity=0.0,
                        maximum_torque=MAX_TORQUE[jid],
                        accel_limit=40.0,
                        velocity_limit=40.0,
                        query=True,
                    ))
                results = await asyncio.gather(*cmds)

            goal_tick = rad_to_dxl_pos(w_send, sign=DXL_SIGN,
                                        offset=DXL_OFFSET_RAD, gear_ratio=DXL_GEAR_RATIO)
            if dxl_ok:
                await loop.run_in_executor(None, dxl.set_position, goal_tick)

            now = time.time()
            if print_requested or (now - last_print > 0.5):
                with state.lock:
                    state.print_requested = False

                fault_tag     = " [FAULTED]" if faulted else ""
                dxl_status    = "dxl OFFLINE"
                moteus_status = "moteus OFFLINE"
                if dxl_ok:
                    dxl_pos, dxl_cur = await loop.run_in_executor(None, dxl.query)
                    dxl_status = f"dxl pos={dxl_pos} cur={dxl_cur:.3f}A"
                if moteus_ok and results:
                    moteus_status = " | ".join(
                        f"j{jid}/CAN{CONTROLLER_IDS[jid]}: "
                        f"pos={st.values[moteus.Register.POSITION]:.3f}rev "
                        f"tor={st.values[moteus.Register.TORQUE]:.3f}Nm "
                        f"cur={st.values[moteus.Register.Q_CURRENT]:.2f}A"
                        for (jid, _), st in zip(controllers.items(), results)
                    )
                print(
                    f"target pos={np.round(pos, 3)} rpy={np.round(rpy, 3)}{fault_tag} "
                    f"| cmds={[round(q_send[j], 3) for j in ACTIVE_JOINTS]} "
                    f"| wrist={w_send:.3f}rad -> tick={goal_tick} "
                    f"| {dxl_status} | {moteus_status}"
                )
                last_print = now

            await asyncio.sleep(CONTROL_DT)

    finally:
        print("Stopping motors...")
        if moteus_ok:
            try:
                await asyncio.gather(*(c.set_stop() for c in controllers.values()))
            except Exception:
                pass
        await loop.run_in_executor(None, dxl.stop)


def moteus_thread_entry():
    try:
        asyncio.run(moteus_control_loop())
    except Exception:
        print("=" * 60)
        print("FATAL: control thread crashed")
        traceback.print_exc()
        print("=" * 60)
        with state.lock:
            state.running = False



# ============================================================
# KEYBOARD INPUT THREAD
# ============================================================

def keyboard_thread():
    """
    Blocking stdin reader — runs in its own thread so it never blocks the
    control loop.  Commands:
      ENTER (blank line) : clear IK fault and return to home
      r + ENTER          : reset RPY orientation to zero
      q + ENTER          : quit
    """
    print("Keyboard commands (press ENTER after each):")
    print("  ENTER   : clear IK fault, return to home")
    print("  r       : reset orientation to zero")
    print("  q       : quit")

    while True:
        try:
            cmd = input().strip().lower()
        except (EOFError, OSError):
            break

        with state.lock:
            running = state.running
        if not running:
            break

        if cmd == "":                    # bare ENTER — clear fault
            with state.lock:
                if state.faulted:
                    state.clear_fault()
            solve_ik_and_update_command()

        elif cmd == "r":                 # reset orientation
            with state.lock:
                if not state.faulted:
                    state.rpy[:] = HOME_RPY
                    state.print_requested = True
            solve_ik_and_update_command()

        elif cmd == "q":                 # quit
            print("Quit requested — shutting down...")
            with state.lock:
                state.running = False
            break

        else:
            print(f"Unknown command '{cmd}'. ENTER=clear fault  r=reset RPY  q=quit")


# ============================================================
# MAIN
# ============================================================

def main():
    if not solve_ik_and_update_command():
        raise RuntimeError("Initial IK failed — check HOME_POS is reachable")

    with state.lock:
        state.safe_q     = dict(state.q_cmd_joint)
        state.safe_wrist = state.q_cmd_wrist

    print("\nControls — SpaceMouse:")
    print("  Push/pull/lift/slide : XYZ position")
    print("  Twist               : RPY orientation")
    print("Controls — Keyboard:")
    print("  ENTER   : clear IK fault, return to home")
    print("  r       : reset orientation to zero")
    print("  q       : quit\n")

    threading.Thread(target=moteus_thread_entry, daemon=True).start()
    threading.Thread(target=spacemouse_thread,   daemon=True).start()
    threading.Thread(target=keyboard_thread,     daemon=True).start()

    try:
        while state.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — shutting down...")
    finally:
        with state.lock:
            state.running = False


if __name__ == "__main__":
    main()
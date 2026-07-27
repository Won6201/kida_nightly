#!/usr/bin/env -S uv run python
# -*- coding: utf-8 -*-
"""
KIDA 통합 단일 파일 (GUI + 제어 실행기 + 컨트롤러)

이 한 파일이 기존의 여러 파이썬 실행/모듈 파일을 모두 포함한다:
  - kida-gui.py   → run_gui()            (기본 실행: 인자 없음)
  - kida-run      → run_kida(argv)        ("kida-run"   서브커맨드)
  - single-run    → run_single(argv)      ("single-run" 서브커맨드)
  - kida.py       → class KidaController
  - single.py     → class SingleController
  - dg5f.py       → class DG5FController
  - h9.py         → class H9Controller

역할 분기 (argv[1]):
    ./kida-app.py                     # GUI 런처
    ./kida-app.py kida-run   -g 1 -x  # 듀얼암 제어 루프 (GUI가 내부에서 재실행)
    ./kida-app.py single-run -t 0 -g 1  # 싱글암 제어 루프

GUI는 './kida-run' 대신 "자기 자신"을 kida-run 모드로 재실행한다. 제어 루프는
여전히 별도 프로세스로 돌아가므로 코어 affinity·렌더 창·ZMQ 소켓 격리가 그대로
유지된다. (subprocess 유지 방식)

C++ 바이너리(rs2/msender, vive/vmaster)와 셸 스크립트(can-up/down, steamvr-run)는
통합 대상이 아니며 지금처럼 subprocess로 실행된다.

필요 패키지: pyside6, pyzmq, opencv-python, numpy, tact(pytact)
"""
import os, sys, re, pty, fcntl, socket, signal, subprocess, threading, struct, termios
import faulthandler; faulthandler.enable()   # SIGSEGV 시 C/파이썬 백트레이스 덤프

import numpy as np
import zmq

# --- GUI 라이브러리 ---
try:
    import pyte                      # vmaster ncurses 화면 재현용
except ImportError:
    pyte = None
import cv2
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QLibraryInfo

# cv2가 덮어쓴 Qt 플러그인 경로를 PySide6 것으로 복원 (xcb 로드 실패 방지)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = \
    QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout,
    QHBoxLayout, QVBoxLayout, QPlainTextEdit, QMessageBox, QGroupBox, QFrame,
    QRadioButton)

# tact(pytact)와 ctypes는 제어 루프에서만 쓰므로 무겁지만 모듈 최상단에서 import
# 해둔다 (컨트롤러 클래스 메서드가 전역 `tact`를 참조). GUI 프로세스에서도 로드되나
# 메모리 비용뿐 부작용은 없다(창은 tact.Env(render=True)에서만 열림).
import tact

# ----------------------------- 설정 -----------------------------
REPO = os.path.dirname(os.path.abspath(__file__))   # repo 루트 기준
HAND_IP_RIGHT = '192.168.0.72'
HAND_IP_LEFT  = '192.168.0.73'
HAND_PORT     = 502
CAM_NAMES_REAL = ['head', 'left', 'right']          # msender 토큰 = ipc 소켓명
CAM_NAMES_SIM  = ['headcam', 'leftcam', 'rightcam'] # 시뮬 kida-run이 publish


def self_cmd(*extra):
    """자기 자신을 서브커맨드 모드로 재실행할 argv를 만든다.
    PyInstaller 등으로 얼렸을 때(sys.frozen)는 실행파일이 자신을 재실행하고,
    스크립트 상태에서는 현재 인터프리터로 이 파일을 실행한다."""
    if getattr(sys, 'frozen', False):
        return [sys.executable, *extra]
    return [sys.executable, os.path.abspath(__file__), *extra]


KIDA_BASE   = self_cmd('kida-run', '-g', '1')       # DG5F (-x는 실물 모드에서 추가)
MSENDER_CMD = ['./rs2/msender'] + CAM_NAMES_REAL
VMASTER_CMD = ['./vive/vmaster', '-t2', '-g1']
STEAMVR_CMD = ['./vive/steamvr-run']

ANSI = re.compile(rb'\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][0-9A-B]|[\x00-\x08\x0b-\x1f]')


# ======================================================================
#  컨트롤러 (구 kida.py / single.py / dg5f.py / h9.py)
# ======================================================================

class KidaController:                            # 구 kida.py: Controller
    n_y = 42 #number of outputs
    n_u = 14 #number of control input

    def __init__(self, env, ymlname, prefix='', rate=None, verbose=False):
        # TODO when has_pd=True is used (kida-run -b): wrap the self.m.ik(...) calls
        # in update()'s task/home-task/task-loop branches with try/except RuntimeError
        # → fall back to self.q_ref_old. IK can fail at workspace boundary / near
        # singularities and currently crashes kida-run. Skipped while we stay on
        # has_pd=False (cmode=0, JTC path) where IK is only called at __init__.
        #self.has_pd = False
        self.has_pd = env.has_pd
        self.verbose = verbose

        self.m = tact.Model(ymlname)
        self.env = env
        self.prefix = prefix

        # rate = control loop ticks/sec; runner passes this (kida-run → 240).
        # When None (e.g. start with CEnv-real where dt isn't exposed), fall
        # back to the kida HW pacing (240 Hz, eio usleep(3000)).
        self.rate = rate if rate is not None else 240

        self.shift(0)
        self.T = 0

        #kp = np.array([200, 200, 40, 40, 40, 15, 15]*2, dtype=float)
        #kd = np.array([4.0, 4.0, 1.0, 1.0, 1.0, 0.4, 0.4]*2, dtype=float)
        kp = np.array([150, 150, 35, 35, 35, 15, 15]*2, dtype=float)
        kd = np.array([4.0, 4.0, 1.0, 1.0, 1.0, 0.4, 0.4]*2, dtype=float)
        self.pid = tact.PIDController(kp, kd, 0, 0.005)
        # Implicit joint-PD gains for the has_pd path (tact backend) — per-decision
        # control outputs, returned from update() as part of the 5-tuple command
        # (tau, q_ref, qd_ref, kp, kd); these attrs are just the gain table.
        # Same values the former YAML `k:` entries carried (gains are control
        # policy, not plant — they moved out of the YAML).
        self.kp = kp
        self.kd = kd
        self.trj1 = tact.MovingAverageWaypointSmoother(10) #joint space trajectory

        Kp = np.array([600, 600, 600, 10, 10, 10]*2, dtype=float)
        Kd = np.array([10, 10, 10, 0.10, 0.10, 0.10]*2, dtype=float)
        #Kp = np.array([1500, 1500, 1500, 60, 60, 60]*2, dtype=float) # claude guide
        #Kd = np.array([80, 80, 80, 2.0, 2.0, 2.0]*2, dtype=float)
        self.jtc = tact.JacobianTransposeController(self.m, {'tcp1':'6d', 'tcp2':'6d'}, Kp, Kd)
        self.task_Kp = Kp
        self.task_Kd = Kd
        self.trj2 = tact.MovingAverageWaypointSmoother(5) #task space trajectory

        #self.sk = np.array([0, 0, 2.5, 0, 0, 0, 0]*2) #spring stiffness
        #self.rq = np.array([0, 0, -0.1, 0, 0, 0, 0]*2) #reference - q
        self.sk = np.array([0, 2.5, 0.0, 0, 0, 0, 0]*2) #spring stiffness
        self.rq = np.array([0, 0.1, 0.0, 0, 0, 0, 0]*2) #reference - q

        self.joint_err_w = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.3, 0.3]*2) #joint error weight
        self.task_err_w = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2]*2) #task error weight
        self.ik_tolerance = 0.002

        #self.init1 = [-1.0, 0.2, 0, 1.8, 0.2, 0.3, 0]*2
        #self.init2 = [-0.6, 0.2, 0, 1.7, 0.2, 0.3, 0]*2
        self.init1 = [-1.0, 0.2, 0, 2.1, 0.2, 0.3, 0]*2
        self.init2 = [-0.4, 0.2, 0, 1.7, 0.2, 0.3, 0]*2

        self.home_task = [0.30, 0.23, -0.40, 0, 0, 0, 0.30, -0.23, -0.40, 0, 0, 0]
        self.home = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, self.init2, self.home_task, tolerance=self.ik_tolerance)

        self.x_d1 = np.array([0.3, 0.15, -0.30, 0, 0, 0,   0.3, -0.35, -0.30, 0, 0, 0])
        self.x_d2 = np.array([0.3, 0.35, -0.30, 0, 0, 0,   0.3, -0.15, -0.30, 0, 0, 0])
        self.x_d3 = np.array([0.3, 0.35, -0.50, 0, 0, 0,   0.3, -0.15, -0.50, 0, 0, 0])
        self.x_d4 = np.array([0.3, 0.15, -0.50, 0, 0, 0,   0.3, -0.35, -0.50, 0, 0, 0])

        self.q_d1 = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, self.home, self.x_d1, tolerance=self.ik_tolerance)
        self.q_d2 = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, self.home, self.x_d2, tolerance=self.ik_tolerance)
        self.q_d3 = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, self.home, self.x_d3, tolerance=self.ik_tolerance)
        self.q_d4 = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, self.home, self.x_d4, tolerance=self.ik_tolerance)

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def scan(self):
        #depth = self.env.raymap('%sheadcam' %self.prefix, 30, 20, 1.5)
        rgb = self.env.get_image('%sheadcam' %self.prefix)

    #Postural spring sk*(rq-q) projected into null(J) so it can't push the TCP.
    #N = I - J^+ J with damped pinv (J full row-rank in regular configs; damping
    #keeps it well-behaved near kinematic singularities). For 7-DOF redundant
    #arm, null(J) is 1-D per arm, so only the component of the elbow bias that
    #lies in that 1-D space survives — small but task-orthogonal.
    def _null_space_postural(self, q, J=None, damping=1e-4):
        if J is None: J = self.m.jacob({'tcp1':'6d', 'tcp2':'6d'}, q)
        tau = self.sk * (self.rq - q)
        JJt = J @ J.T + (damping*damping) * np.eye(J.shape[0])
        return tau - J.T @ np.linalg.solve(JJt, J @ tau)

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1
        self.T += 1

    def msgproc(self, w):
        if w[0] == 'joint':
            if len(w[1:]) == 14:
                self.v = np.array(w[1:], dtype=float)
                lo = [-1.5, -1.5, -1.5, -1.0, -1.5, -1.5, -1.5]*2
                hi = [ 1.5,  1.5,  1.5,  2.5,  1.5,  1.5,  1.5]*2
                self.v = np.clip(self.v, lo, hi)
                self.shift(w[0])

        elif w[0] == 'task':
            if len(w[1:]) == 12:
                self.v = np.array(w[1:], dtype=float)
                lo = [ 0.0, -0.5, -0.6, -1.57, -1.30, -1.57]*2
                hi = [ 0.6,  0.5,  0.0,  1.57,  1.30,  1.57]*2
                self.v = np.clip(self.v, lo, hi)
                self.shift(w[0])

        elif w[0] in ['init', 'rest', 'home', 'home-task', 'joint-loop', 'task-loop']: self.shift(w[0])
        elif w[0] in ['free', 'gcomp'] and not self.has_pd: self.shift(w[0])
        elif w[0] == 'scan': self.scan()

    def update(self, y):
        q, qd = y[0:14], y[14:28]
        x = self.m.fk({'tcp1':'6d', 'tcp2':'6d'}, q)

        # Each state branch sets exactly one of (tau, q_ref). For 'free' (no branch fires)
        # has_pd holds last commanded position; external path stays at None (zero tau).
        tau = None
        q_ref = self.q_ref_old if (self.T > 0 and self.has_pd) else None

        if self.s == 'joint':
            if self.t == 0:
                e_eff = np.linalg.norm(self.joint_err_w*(self.v - q)) #effective joint error
                duration = int(4.0*self.rate*e_eff) + 1
                self.trj1.target(self.v.reshape((1, 14)), [duration], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'task':
            if self.t == 0:
                #e_eff = np.linalg.norm(self.task_err_w*self._task_error(q, self.v)) #effective task error
                e_eff = np.linalg.norm(self.task_err_w*self.m.error({'tcp1':'6d', 'tcp2':'6d'}, q, self.v)) #effective task error
                duration = int(4.0*self.rate*e_eff) + 1
                self.trj2.target(self.v.reshape((1, 12)), [duration], x, self.T)
            if self.has_pd:
                tau = self.m.gravity(q) + self._null_space_postural(q)
                q_ref = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, q, self.trj2.generate(), tolerance=self.ik_tolerance)
            else:
                J = self.m.jacob({'tcp1':'6d', 'tcp2':'6d'}, q)
                #tau = self._task_tau(self.trj2.generate(), q, qd, J) + self.m.gravity(q) + self._null_space_postural(q, J=J)
                tau = self.jtc.update(self.trj2.generate(), q, qd, J=J) + self.m.gravity(q) + self._null_space_postural(q, J=J)

        elif self.s == 'init':
            if self.t == 0: self.trj1.target(np.array([self.init1, self.init2, self.home]), [2*self.rate, self.rate, self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'rest':
            if self.t == 0: self.trj1.target([self.init2, self.init1, [0]*14], [self.rate, self.rate, 2*self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'home':
            if self.t == 0: self.trj1.target([self.home], [2*self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'home-task':
            if self.t == 0: self.trj2.target(([self.home_task]), [self.rate], x, self.T)
            if self.has_pd:
                tau = self.m.gravity(q) + self._null_space_postural(q)
                q_ref = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, q, self.trj2.generate(), tolerance=self.ik_tolerance)
            else:
                J = self.m.jacob({'tcp1':'6d', 'tcp2':'6d'}, q)
                #tau = self._task_tau(self.trj2.generate(), q, qd, J) + self.m.gravity(q) + self._null_space_postural(q, J=J)
                tau = self.jtc.update(self.trj2.generate(), q, qd, J=J) + self.m.gravity(q) + self._null_space_postural(q, J=J)

        elif self.s == 'joint-loop':
            if self.t % (self.rate*8) == 0: self.trj1.target(np.array([self.q_d1, self.q_d1, self.q_d2, self.q_d2, self.q_d3, self.q_d3, self.q_d4, self.q_d4]), [self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'task-loop':
            if self.t % (self.rate*8) == 0: self.trj2.target(np.array([self.x_d1, self.x_d1, self.x_d2, self.x_d2, self.x_d3, self.x_d3, self.x_d4, self.x_d4]), [self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate], x, self.T)
            if self.has_pd:
                tau = self.m.gravity(q) + self._null_space_postural(q)
                q_ref = self.m.ik({'tcp1':'6d', 'tcp2':'6d'}, q, self.trj2.generate(), tolerance=self.ik_tolerance)
            else:
                J = self.m.jacob({'tcp1':'6d', 'tcp2':'6d'}, q)
                #tau = self._task_tau(self.trj2.generate(), q, qd, J) + self.m.gravity(q) + self._null_space_postural(q, J=J)
                tau = self.jtc.update(self.trj2.generate(), q, qd, J=J) + self.m.gravity(q) + self._null_space_postural(q, J=J)

        elif self.s == 'gcomp':
            # gcomp only reachable when has_pd=False (gated in msgproc)
            tau = self.m.gravity(q)

        if self.verbose:
            #print('[%d] %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f' %(self.T, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]))
            print('[%d] %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f | %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f' %(self.T, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13]))

        self.one_step_forward()
        self.q_ref_old = q_ref
        return tau, q_ref, None, self.kp, self.kd


class SingleController:                           # 구 single.py: Controller
    n_y = 21 #number of outputs
    n_u = 7 #number of control input

    def __init__(self, env, ymlname, prefix='', rate=None, verbose=False):
        # TODO when has_pd=True is used (single-run -b): wrap the self.m.ik(...) calls
        # in update()'s task/home-task/task-loop branches with try/except RuntimeError
        # → fall back to self.q_ref_old. IK can fail at workspace boundary / near
        # singularities and currently crashes single-run. Skipped while we stay on
        # has_pd=False (cmode=0, JTC path) where IK is only called at __init__.
        #self.has_pd = False
        self.has_pd = env.has_pd
        self.verbose = verbose

        self.m = tact.Model(ymlname)
        self.env = env
        self.prefix = prefix

        # rate = control loop ticks/sec; runner passes this (single-run → 240).
        # When None (e.g. start with CEnv-real where dt isn't exposed), fall
        # back to the kida HW pacing (240 Hz, eio usleep(3000)).
        self.rate = rate if rate is not None else 240

        self.shift(0)
        self.T = 0

        kp = np.array([150, 150, 35, 35, 35, 15, 15], dtype=float)
        kd = np.array([4.0, 4.0, 1.0, 1.0, 1.0, 0.4, 0.4], dtype=float)
        self.pid = tact.PIDController(kp, kd, 0, 0.005)
        # Implicit joint-PD gains for the has_pd path (tact backend) — per-decision
        # control outputs, returned from update() as part of the 5-tuple command
        # (tau, q_ref, qd_ref, kp, kd); these attrs are just the gain table.
        # Same values the former YAML `k:` entries carried (gains are control
        # policy, not plant — they moved out of the YAML).
        self.kp = kp
        self.kd = kd
        self.trj1 = tact.MovingAverageWaypointSmoother(10) #joint space traj generator

        Kp = np.array([600, 600, 600, 10, 10, 10], dtype=float)
        Kd = np.array([10, 10, 10, 0.10, 0.10, 0.10], dtype=float)
        self.jtc = tact.JacobianTransposeController(self.m, {'tcp':'6d'}, Kp, Kd)
        self.task_Kp = Kp
        self.task_Kd = Kd
        self.trj2 = tact.MovingAverageWaypointSmoother(5) #task space traj generator

        #self.sk = np.array([0, 0, 2.5, 0, 0, 0, 0]) #spring stiffness
        #self.rq = np.array([0, 0, -0.1, 0, 0, 0, 0]) #reference - q
        self.sk = np.array([0, 2.5, 2.5, 0, 0, 0, 0]) #spring stiffness
        self.rq = np.array([0, 0.1, 0.0, 0, 0, 0, 0]) #reference - q

        self.joint_err_w = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.3, 0.3]) #joint error weight
        self.task_err_w = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2]) #task error weight
        self.ik_tolerance = 0.002

        if   '-left'  in ymlname: self.y_sign =  1.0
        elif '-right' in ymlname: self.y_sign = -1.0
        else: print('wrong ymlname=%s' %ymlname); exit(0)

        #self.init1 = [-1.0, 0.2, 0, 1.8, 0.2, 0.3, 0]
        #self.init2 = [-0.6, 0.2, 0, 1.7, 0.2, 0.3, 0]
        self.init1 = [-1.0, 0.2, 0, 2.1, 0.2, 0.3, 0]
        self.init2 = [-0.4, 0.2, 0, 1.7, 0.2, 0.3, 0]

        self.home_task = [0.30, self.y_sign*0.23, -0.40, 0, 0, 0]
        self.home = self.m.ik({'tcp':'6d'}, self.init2, self.home_task, tolerance=self.ik_tolerance)

        #for task-space loop test
        self.x_d1 = np.array([0.3, self.y_sign*0.15, -0.30, 0, 0, 0])
        self.x_d2 = np.array([0.3, self.y_sign*0.35, -0.30, 0, 0, 0])
        self.x_d3 = np.array([0.3, self.y_sign*0.35, -0.50, 0, 0, 0])
        self.x_d4 = np.array([0.3, self.y_sign*0.15, -0.50, 0, 0, 0])

        #for joint space loop test
        self.q_d1 = self.m.ik({'tcp':'6d'}, self.home, self.x_d1, tolerance=self.ik_tolerance)
        self.q_d2 = self.m.ik({'tcp':'6d'}, self.home, self.x_d2, tolerance=self.ik_tolerance)
        self.q_d3 = self.m.ik({'tcp':'6d'}, self.home, self.x_d3, tolerance=self.ik_tolerance)
        self.q_d4 = self.m.ik({'tcp':'6d'}, self.home, self.x_d4, tolerance=self.ik_tolerance)

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    #Project postural spring into null(J) so it doesn't disturb task-space JTC.
    #N = I - J^+ J via damped pinv (see KidaController._null_space_postural for rationale).
    def _null_space_postural(self, q, J=None, damping=1e-4):
        if J is None: J = self.m.jacob({'tcp':'6d'}, q)
        tau = self.sk * (self.rq - q)
        JJt = J @ J.T + (damping*damping) * np.eye(J.shape[0])
        return tau - J.T @ np.linalg.solve(JJt, J @ tau)

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1
        self.T += 1

    def msgproc(self, w):
        if w[0] == 'joint':
            if len(w[1:]) == 7:
                self.v = np.array(w[1:], dtype=float)
                lo = [-1.5, -1.5, -1.5, -1.0, -1.5, -1.5, -1.5]*2
                hi = [ 1.5,  1.5,  1.5,  2.5,  1.5,  1.5,  1.5]*2
                self.v = np.clip(self.v, lo, hi)
                self.shift(w[0])

        elif w[0] == 'task':
            if len(w[1:]) == 6:
                self.v = np.array(w[1:], dtype=float)
                lo = [ 0.0, -0.5, -0.6, -1.57, -1.30, -1.57]
                hi = [ 0.6,  0.5,  0.0,  1.57,  1.30,  1.57]
                self.v = np.clip(self.v, lo, hi)
                self.shift(w[0])

        elif w[0] in ['init', 'rest', 'home', 'home-task', 'joint-loop', 'task-loop']: self.shift(w[0])
        elif w[0] in ['free', 'gcomp'] and not self.has_pd: self.shift(w[0])
        elif w[0] == 'scan': self.scan()

    def update(self, y):
        q, qd, act = y[0:7], y[7:14], y[14:21]
        x = self.m.fk({'tcp':'6d'}, q)

        # Each state branch sets exactly one of (tau, q_ref). For 'free' (no branch fires)
        # has_pd holds last commanded position; external path stays at None (zero tau).
        tau = None
        q_ref = self.q_ref_old if (self.T > 0 and self.has_pd) else None

        if self.s == 'joint':
            if self.t == 0:
                e_eff = np.linalg.norm(self.joint_err_w*(self.v - q)) #effective joint error
                duration = int(4.0*self.rate*e_eff) + 1
                self.trj1.target(self.v.reshape((1, 7)), [duration], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'task':
            if self.t == 0:
                #e_eff = np.linalg.norm(self.task_err_w*self._task_error(q, self.v)) #effective task error
                e_eff = np.linalg.norm(self.task_err_w*self.m.error({'tcp':'6d'}, q, self.v)) #effective task error
                duration = int(4.0*self.rate*e_eff) + 1
                self.trj2.target(self.v.reshape((1, 6)), [duration], x, self.T)
            if self.has_pd:
                tau = self.m.gravity(q) + self._null_space_postural(q)
                q_ref = self.m.ik({'tcp':'6d'}, q, self.trj2.generate(), tolerance=self.ik_tolerance)
            else:
                J = self.m.jacob({'tcp':'6d'}, q)
                #tau = self._task_tau(self.trj2.generate(), q, qd, J) + self.m.gravity(q) + self._null_space_postural(q, J=J)
                tau = self.jtc.update(self.trj2.generate(), q, qd, J=J) + self.m.gravity(q) + self._null_space_postural(q, J=J)

        elif self.s == 'init':
            if self.t == 0: self.trj1.target(np.array([self.init1, self.init2, self.home]), [2*self.rate, self.rate, self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'rest':
            if self.t == 0: self.trj1.target(np.array([self.init2, self.init1, [0]*7]), [self.rate, self.rate, 2*self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'home':
            if self.t == 0: self.trj1.target(np.array([self.home]), [2*self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'home-task':
            if self.t == 0: self.trj2.target(np.array([self.home_task]), [self.rate], x, self.T)
            if self.has_pd:
                tau = self.m.gravity(q) + self._null_space_postural(q)
                q_ref = self.m.ik({'tcp':'6d'}, q, self.trj2.generate(), tolerance=self.ik_tolerance)
            else:
                J = self.m.jacob({'tcp':'6d'}, q)
                #tau = self._task_tau(self.trj2.generate(), q, qd, J) + self.m.gravity(q) + self._null_space_postural(q, J=J)
                tau = self.jtc.update(self.trj2.generate(), q, qd, J=J) + self.m.gravity(q) + self._null_space_postural(q, J=J)

        elif self.s == 'joint-loop':
            if self.t % (self.rate*8) == 0: self.trj1.target(np.array([self.q_d1, self.q_d1, self.q_d2, self.q_d2, self.q_d3, self.q_d3, self.q_d4, self.q_d4]), [self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate], q, self.T)
            if self.has_pd:
                tau = self.m.gravity(q)
                q_ref = self.trj1.generate()
            else: tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)

        elif self.s == 'task-loop':
            if self.t % (self.rate*8) == 0: self.trj2.target(np.array([self.x_d1, self.x_d1, self.x_d2, self.x_d2, self.x_d3, self.x_d3, self.x_d4, self.x_d4]), [self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate, self.rate], x, self.T)
            if self.has_pd:
                tau = self.m.gravity(q) + self._null_space_postural(q)
                q_ref = self.m.ik({'tcp':'6d'}, q, self.trj2.generate(), tolerance=self.ik_tolerance)
            else:
                J = self.m.jacob({'tcp':'6d'}, q)
                #tau = self._task_tau(self.trj2.generate(), q, qd, J) + self.m.gravity(q) + self._null_space_postural(q, J=J)
                tau = self.jtc.update(self.trj2.generate(), q, qd, J=J) + self.m.gravity(q) + self._null_space_postural(q, J=J)

        elif self.s == 'gcomp':
            # gcomp only reachable when has_pd=False (gated in msgproc)
            tau = self.m.gravity(q)

        if self.verbose:
            #print('[%d] %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f' %(self.T, x[0], x[1], x[2], x[3], x[4], x[5]))
            print('[%d] %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f' %(self.T, q[0], q[1], q[2], q[3], q[4], q[5], q[6]))

        self.one_step_forward()
        self.q_ref_old = q_ref
        return tau, q_ref, None, self.kp, self.kd


class DG5FController:                             # 구 dg5f.py: Controller
    n_y = 60 #number of outputs #pos, vel, current (activation torque)
    n_u = 20 #number of control input

    def __init__(self, env, ymlname, prefix='', rate=None, verbose=False):
        #self.has_pd = False
        self.has_pd = env.has_pd
        # Real-HW override: dg5f (dg5f hand) is intrinsically position-controlled — its
        # firmware always runs onboard PD on the q_ref it receives. The hand always
        # emits q_ref regardless of the arm's mode.
        if env.backend == 'real': self.has_pd = True
        self.verbose = verbose

        self.m = tact.Model(ymlname)
        self.env = env
        self.prefix = prefix

        # rate = control loop ticks/sec; runner passes this. Fall back to 240
        # (HW pacing) when None — covers direct construction and CEnv-real.
        self.rate = rate if rate is not None else 240

        if   '-left'  in ymlname: self.home = np.array([-0.4,  0.4, -0.4, -0.4,   0.4, 0.4, 0.4, 0.4,   0.2, 0.4, 0.4, 0.4,  0, 0.4, 0.4, 0.4,  -0.2, -0.2, 0.4, 0.4])
        elif '-right' in ymlname: self.home = np.array([ 0.4, -0.4,  0.4,  0.4,  -0.4, 0.4, 0.4, 0.4,  -0.2, 0.4, 0.4, 0.4,  0, 0.4, 0.4, 0.4,   0.2,  0.2, 0.4, 0.4])

        self.shift(0)
        self.T = 0

        kp = np.array([1.0, 1.0, 1.0, 1.0]*5, dtype=float)
        kd = np.array([0.01, 0.01, 0.01, 0.01]*5, dtype=float)
        self.pid = tact.PIDController(kp, kd, 0, 0.005)
        # Implicit joint-PD gains for the has_pd path (tact backend) — per-decision
        # control outputs, returned from update() as part of the 5-tuple command
        # (tau, q_ref, qd_ref, kp, kd); these attrs are just the gain table.
        # Same values the former YAML `k:` entries carried (gains are control
        # policy, not plant — they moved out of the YAML).
        self.kp = kp
        self.kd = kd
        self.trj = tact.MovingAverageWaypointSmoother(20)
        self.joint_err_w = np.array([1.0, 1.0, 0.7, 0.5,  1.0, 1.0, 0.7, 0.5,  1.0, 1.0, 0.7, 0.5,  1.0, 1.0, 0.7, 0.5,  1.0, 1.0, 0.7, 0.5]) #joint error weight

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1
        self.T += 1

    def msgproc(self, w):
        if w[0] in ['zero', 'home']:  self.shift(w[0])
        elif w[0] == 'joint' and len(w) == 21: self.v = np.array(w[1:], dtype=float); self.shift(w[0])

    def update(self, y):
        q, qd, act = y[0:20], y[20:40], y[40:60]
        # Each state branch sets exactly one of (tau, q_ref). Unused channel stays None.
        tau = q_ref = None

        if self.s == 'joint':
            if self.t == 0:
                e_eff = np.linalg.norm(self.v - q) #effective joint error
                duration = int(0.06*self.rate*e_eff) + 1
                self.trj.target(self.v.reshape((1, 20)), [duration], q, self.T)
            if self.has_pd: q_ref = self.trj.generate()
            else: tau = self.pid.update(self.trj.generate(), q, qd)

        elif self.s == 'zero':
            if self.t == 0: self.trj.target(np.zeros((1, 20)), [600], q, self.T)
            if self.has_pd: q_ref = self.trj.generate()
            else: tau = self.pid.update(self.trj.generate(), q, qd)

        elif self.s == 'home':
            if self.t == 0: self.trj.target(self.home.reshape((1, 20)), [600], q, self.T)
            if self.has_pd: q_ref = self.trj.generate()
            else: tau = self.pid.update(self.trj.generate(), q, qd)

        self.one_step_forward()
        return tau, q_ref, None, self.kp, self.kd


class H9Controller:                              # 구 h9.py: Controller
    n_y = 18 #number of outputs
    n_u = 9 #number of control input

    def __init__(self, env, ymlname, prefix='', rate=None, verbose=False):
        self.has_pd = env.has_pd
        self.verbose = verbose

        self.m = tact.Model(ymlname)
        self.env = env
        self.prefix = prefix

        # rate = control loop ticks/sec; runner passes this. Fall back to 240
        # (HW pacing) when None — covers direct construction and CEnv-real.
        self.rate = rate if rate is not None else 240

        self.shift(0)
        self.T = 0

        #self.sk = [0.20, 0.10, 0.00,   0.10, 0.05,   0.10, 0.05,   0.10, 0.05]

        if env.backend == 'real': kp = [2.0, 2.0, 2.0,   2.0, 2.0,   2.0, 2.0,   2.0, 2.0]; kd = [0.0, 0.0, 0.0,   0.0, 0.0,   0.0, 0.0,   0.0, 0.0]
        else: kp = [0.5, 0.5, 0.5,   0.5, 0.5,   0.5, 0.5,   0.5, 0.5]; kd = [0.0, 0.0, 0.0,   0.0, 0.0,   0.0, 0.0,   0.0, 0.0]

        self.pid = tact.PIDController(kp, kd, 0.0, 0.004)
        # Implicit joint-PD gains for the has_pd path (tact backend) — per-decision
        # control outputs, returned from update() as part of the 5-tuple command
        # (tau, q_ref, qd_ref, kp, kd); these attrs are just the gain table.
        # Same values the former YAML `k:` entries carried (gains are control
        # policy, not plant — they moved out of the YAML).
        self.kp = np.full(9, 0.5)   # former h9 yml k: [0.5, 0] (tact sim only;
        self.kd = np.zeros(9)       # real/mujoco ignore the kwargs)
        self.trj = tact.MovingAverageWaypointSmoother(1000)

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1
        self.T += 1

    def msgproc(self, w):
        if w[0] in ['zero', 'home', 'ready']:  self.shift(w[0])
        elif w[0] == 'joint' and len(w) == 10: self.v = np.array(w[1:], dtype=float); self.shift(w[0])
        #elif w[0] == 'jointdeg': self.v = (np.pi/180)*np.array(w[1:], dtype=float); self.shift(w[0])
        #elif w[0] == 'jointalldeg' and 0 <= float(w[1]) <= 50: self.v = (np.pi/180)*float(w[1]); self.shift(w[0])

    def update(self, y):
        q, qd = y[:9], y[9:]
        # Each state branch sets exactly one of (tau, q_ref). Unused channel stays None.
        tau = None
        q_ref = None

        if self.s == 'joint':
            if self.t == 0:
                e_eff = np.linalg.norm(self.v - q) #effective joint error
                duration = int(0.06*self.rate*e_eff) + 1
                self.trj.target(self.v.reshape((1, 9)), [duration], q, self.T)
            if self.has_pd: q_ref = self.trj.generate()
            else:                tau   = self.pid.update(self.trj.generate(), q, qd)

        elif self.s == 'zero':
            if self.t == 0: self.trj.target(np.zeros((1, 9)), [1000], q, self.T)
            if self.has_pd: q_ref = self.trj.generate()
            else:                tau   = self.pid.update(self.trj.generate(), q, qd)

        elif self.s == 'home':
            if self.t == 0: self.trj.target(np.array([[0.7, 0.7, 0.7,   0.3, 0.3,   0.5, 0.5,   0.7, 0.7]]), [1000], q, self.T)
            if self.has_pd: q_ref = self.trj.generate()
            else:                tau   = self.pid.update(self.trj.generate(), q, qd)

        elif self.s == 'ready':
            if self.t == 0: self.trj.target(np.array([[1.2, 0.3, 0.3,   0.8, 0.8,   0.8, 0.8,   0.3, 0.3]]), [1000], q, self.T)
            if self.has_pd: q_ref = self.trj.generate()
            else:                tau   = self.pid.update(self.trj.generate(), q, qd)

        if self.verbose:
            deg = q*180/np.pi
            print('[%8d] %6.2f %6.2f %6.2f   %6.2f %6.2f   %6.2f %6.2f   %6.2f %6.2f' %(self.T, deg[0], deg[1], deg[2], deg[3], deg[4], deg[5], deg[6], deg[7], deg[8]))

        self.one_step_forward()
        return tau, q_ref, None, self.kp, self.kd


# ======================================================================
#  제어 실행기 (구 kida-run / single-run)
# ======================================================================

# Compose multi-controller output. Each returns the 5-tuple command
# (tau, q_ref, qd_ref, kp, kd) with each channel either an array or None.
# combine: all-None → None, else concat with zero-fill for None members so
# length matches n_u. Zero-fill is correct for the GAIN channels too: a module
# whose q_ref was None gets q_ref=0 AND kp=0 on its slots, so its PD term is
# identically zero (gains multiply) — inactive, exactly as intended.
def combine(parts, sizes):
    if all(p is None for p in parts): return None
    return np.concatenate([p if p is not None else np.zeros(s) for p, s in zip(parts, sizes)])


def run_kida(argv):
    """구 kida-run: 듀얼암(14 DOF) + 양손 제어 루프."""
    import argparse, ctypes, atexit
    from collections import deque

    par = argparse.ArgumentParser(prog='kida-app.py kida-run')
    par.add_argument('-g', default=1, type=int, help='gripper type [1: dg5f]')
    par.add_argument('-x', action='store_true', default=False, help='real')
    par.add_argument('-m', action='store_true', default=False, help='mujoco')
    par.add_argument('-v', action='store_true', default=False, help='verbose')
    par.add_argument('-b', action='store_true', default=False, help='use built-in pd controller (real h/w only)')
    par.add_argument('-d', default=None, help='dispatch file: each line "<step_count> <command>"; fires command when cnt matches')
    par.add_argument('-l', action='store_true', default=False, help='headless sim (no render window)')
    arg = par.parse_args(argv)

    try: os.sched_setaffinity(0, {0})
    except Exception as e: print('set affinity failed'); sys.exit()

    if   arg.g == 0: gname = 'h9';   GC = H9Controller
    elif arg.g == 1: gname = 'dg5f'; GC = DG5FController
    else: print('wrong gripper type'); sys.exit()

    #Frameskip: sim runs physics at 1kHz (dt=0.001) but real HW eio paces at ~240Hz.
    #Sim → 4 (control rate 250Hz, matches real). Real → 1 (eio already paces).
    frameskip = 1 if arg.x else 4
    cnt = 0

    if arg.x: #start with real H/W
        cdll = ctypes.CDLL('eio/eio-kida.so')
        cmode = 1 if arg.b else 0
        cdll.init(b'%d 1' %cmode) #control mode, hand-type
        env = tact.CEnv(cdll, n_y=KidaController.n_y + 2*GC.n_y, n_u=KidaController.n_u + 2*GC.n_u, backend='real', has_pd=arg.b)
    #elif arg.m:
    #    # mjenv.so (MuJoCo backend) is not in the pytact wheel; provide a local
    #    # build (extras/mjenv.so, built from tact's mjenv.cpp) before enabling -m.
    #    cdll = ctypes.CDLL('extras/mjenv.so')
    #    cdll.init('/home/ubuntu/uv1/fgx/mujoco/models/kida/mjmodel.xml'.encode(), None, 16)
    #    env = tact.CEnv(cdll, n_y=..., backend='mujoco')
    else: #start with tact simulator
        env = tact.Env('yaml/kida', offset=[0, 0, 1.2, 0, 0, 0], render=not arg.l, redraw=16)
        # offset y=+0.0071 aligns dg5f-left's root origin (ll_dg_palm) with dg5f-right's:
        # the left YAML's 3 central finger bases sit at root-x=-0.0071 vs right's x=0, a 7.1mm
        # origin shift; +0.0071 in tcp1-y compensates it (sign verified on render/HW).
        env.add('yaml/%s-left'  %gname, prefix='hand1.', base='tcp1', offset=[0.02, 0.0071, 0, 0, 90, -90])
        # mirror of the left: dg5f-right's 3 central finger bases also sit at root-x=-0.0071,
        # and with yaw=+90 the tcp2-y correction flips sign -> -0.0071. x matched to the
        # left's 0.02 gap for symmetry.
        env.add('yaml/%s-right' %gname, prefix='hand2.', base='tcp2', offset=[0.02, -0.0071, 0, 0, 90,  90])
        env.add('yaml/desk1')
        env.edit('tcp1', m=0.6, c=[0.05, 0, 0])
        env.edit('tcp2', m=0.6, c=[0.05, 0, 0])
        env.m.view = [0, 0, -0.34+1.2, 1.3, 180, 20]

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind('ipc:///dev/shm/default')       # local clients
    pull.bind('tcp://0.0.0.0:5555')           # remote clients (LAN)

    ppub = ctx.socket(zmq.PUB)
    ppub.setsockopt(zmq.CONFLATE, 1)
    ppub.bind('ipc:///dev/shm/proprio')       # local clients
    ppub.bind('tcp://0.0.0.0:5556')           # remote clients (LAN)

    # Sensor publish setup. Cameras/tactiles are declared in YAML and exposed as
    # env.cameras/env.tactiles. Rate-gating + type dispatch live in env.*_frames();
    # here we only own the sockets. Capability-based (docs/backend-interface.md):
    # CEnv (real) has NO sim sensor attrs (hasattr probes False) → nothing bound.
    proprio_update_cycle = 2 if arg.x else 33  # real HW ~120Hz vs sim ~30Hz
    socks = {}
    for c in list(getattr(env, 'cameras', [])) + list(getattr(env, 'tactiles', [])):
        s = ctx.socket(zmq.PUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.bind('ipc:///dev/shm/%s' %c['name'])                   # local clients
        if c.get('port'): s.bind('tcp://0.0.0.0:%d' %c['port'])  # remote clients (LAN)
        socks[c['name']] = s

    def close_zmq(sockets=list(socks.values()), pub=ppub, cmd=pull, context=ctx):
        if getattr(close_zmq, 'closed', False): return
        close_zmq.closed = True
        for s in sockets + [pub, cmd]:
            try: s.close(linger=0)
            except Exception: pass
        try: context.term()
        except Exception: pass
    close_zmq.closed = False
    atexit.register(close_zmq)

    # Optional dispatch: file lines are "<step_count> <command>" — same format as
    # start's -d. cnt counts physics ticks (post-frameskip), so timings are wall-clock.
    dispatch = deque()
    if arg.d is not None:
        with open(arg.d) as fin:
            for line in fin:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2: dispatch.append([int(parts[0]), parts[1]])

    def msgproc(msg):
        nonlocal cnt
        w = [s.split() for s in msg.decode().split(',')]
        if w[0][0] == 'quit': env.finish(); sys.exit(0)
        elif w[0][0] == 'reset': cnt = 0; return
        if len(w) in (1, 2, 3):
            arms.msgproc(w[0])
            if len(w) >= 2: hand1.msgproc(w[1])
            if len(w) == 3: hand2.msgproc(w[2])
        else: print('wrong command')

    tau = q_ref = qd_ref = kp = kd = None   # held between control ticks (frameskip ZOH)

    while True:
        try: msg = pull.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError: pass
        else: msgproc(msg); continue

        if dispatch and cnt == dispatch[0][0]:
            msgproc(dispatch[0][1].encode()); print(dispatch[0]); dispatch.popleft()

        if cnt == 0:
            # rate = controller ticks/sec, fixed at 240 for both modes (matches real
            # HW eio pacing exactly; sim with frameskip=4 actually runs at 250Hz so
            # trajectories are ~4% faster wall-clock — negligible, not worth the
            # branching). Arms and hands share the loop so they get the same rate.
            arms  = KidaController(env, 'yaml/kida', rate=240, verbose=arg.v)
            hand1 = GC(env, 'yaml/%s-left'  %gname, prefix='hand1.', rate=240, verbose=False)
            hand2 = GC(env, 'yaml/%s-right' %gname, prefix='hand2.', rate=240, verbose=False)
            y = env.reset()

        if cnt % frameskip == 0:
            # 5-tuple command (tau, q_ref, qd_ref, kp, kd) per module — gains are
            # per-decision control outputs (2026-06-07; YAML k: removed). On real
            # the eio accepts kp/kd but ignores them (firmware owns the gains).
            arm_tau,  arm_qr,  arm_qdr,  arm_kp, arm_kd = arms.update(y[:arms.n_y])
            h1_tau,   h1_qr,   h1_qdr,   h1_kp,  h1_kd  = hand1.update(y[arms.n_y:arms.n_y+hand1.n_y])
            h2_tau,   h2_qr,   h2_qdr,   h2_kp,  h2_kd  = hand2.update(y[arms.n_y+hand1.n_y:])
            sizes  = [arms.n_u, hand1.n_u, hand2.n_u]
            tau    = combine([arm_tau, h1_tau, h2_tau], sizes)
            q_ref  = combine([arm_qr,  h1_qr,  h2_qr],  sizes)
            qd_ref = combine([arm_qdr, h1_qdr, h2_qdr], sizes)
            kp     = combine([arm_kp,  h1_kp,  h2_kp],  sizes)
            kd     = combine([arm_kd,  h1_kd,  h2_kd],  sizes)
        y = env.step(tau=tau, q_ref=q_ref, qd_ref=qd_ref, kp=kp, kd=kd)

        #update priprio feedback
        if cnt % proprio_update_cycle == 0:
            proprio = y.astype(np.float32).tobytes()
            ppub.send(proprio)

        #update camera/tactile feedback — capability-based: socket table is empty when
        #the backend declared no sensors (CEnv); rate-gating + dispatch in env.*_frames()
        if socks:
            if hasattr(env, 'camera_frames'):
                for name, buf in env.camera_frames():
                    socks[name].send(buf)
            if hasattr(env, 'tactile_frames'):
                for name, buf in env.tactile_frames():
                    socks[name].send(buf)

        cnt += 1


def run_single(argv):
    """구 single-run: 한쪽 팔(7 DOF) + 한 손 제어 루프."""
    import argparse, ctypes, atexit
    from collections import deque

    par = argparse.ArgumentParser(prog='kida-app.py single-run')
    par.add_argument('-t', default=-1, type=int, help='arm type [0: left, 1: right]')
    par.add_argument('-g', default=-1, type=int, help='gripper type [0: h9, 1: dg5f]')
    par.add_argument('-x', action='store_true', default=False, help='real')
    par.add_argument('-v', action='store_true', default=False, help='verbose')
    par.add_argument('-b', action='store_true', default=False, help='use built-in pd controller (real h/w only)')
    par.add_argument('-d', default=None, help='dispatch file: each line "<step_count> <command>"; fires command when cnt matches')
    par.add_argument('-l', action='store_true', default=False, help='headless sim (no render window)')
    arg = par.parse_args(argv)

    try: os.sched_setaffinity(0, {0})
    except Exception as e: print('set affinity failed'); sys.exit()

    # yoff aligns the dg5f root origin (ll_dg_palm): both hands' 3 central finger bases sit
    # at root-x=-0.0071, so tcp-y is offset by +0.0071. The mount yaw sign (-90 left / +90
    # right) flips it -> +0.0071 left, -0.0071 right (verified on render/HW).
    if   arg.t == 0: ymltail, yaw, yoff = '-left', -90, 0.0071
    elif arg.t == 1: ymltail, yaw, yoff = '-right', 90, -0.0071
    else: print('choose arm type [0: left, 1: right]'); sys.exit(0)
    if   arg.g == 0: gname = 'h9';   GC = H9Controller
    elif arg.g == 1: gname = 'dg5f'; GC = DG5FController
    else: print('wrong gripper type'); sys.exit()

    #Frameskip: sim runs physics at 1kHz (dt=0.001) but real HW eio paces at ~240Hz.
    #Sim → 4 (control rate 250Hz, matches real). Real → 1 (eio already paces).
    frameskip = 1 if arg.x else 4
    cnt = 0

    if arg.x: #start with real H/w
        cdll = ctypes.CDLL('eio/eio-single.so')
        cmode = 1 if arg.b else 0
        cdll.init(b'%d %d 1' %(arg.t, cmode)) #can channel, control-mode, hand-type
        env = tact.CEnv(cdll, n_y=SingleController.n_y + GC.n_y, n_u=SingleController.n_u + GC.n_u, backend='real', has_pd=arg.b)
    else: #start with tact simulator
        env = tact.Env('yaml/kida%s' %ymltail, offset=[0, 0, 1.2, 0, 0, 0], render=not arg.l, redraw=16)
        env.add('yaml/%s' %(gname + ymltail), prefix='hand.', base='tcp', offset=[0.05, yoff, 0, 0, 90, yaw])
        env.add('yaml/desk1')
        env.edit('tcp', m=0.6, c=[0.05, 0, 0])
        env.m.view = [0, 0, -0.34+1.2, 1.3, 180, 20]

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind('ipc:///dev/shm/default')       # local clients
    pull.bind('tcp://0.0.0.0:5555')           # remote clients (LAN)

    ppub = ctx.socket(zmq.PUB)
    ppub.setsockopt(zmq.CONFLATE, 1)
    ppub.bind('ipc:///dev/shm/proprio')       # local clients
    ppub.bind('tcp://0.0.0.0:5556')           # remote clients (LAN)

    # Sensor publish setup (see run_kida for the capability-based rationale).
    proprio_update_cycle = 2 if arg.x else 33  # real HW ~120Hz vs sim ~30Hz
    socks = {}
    for c in list(getattr(env, 'cameras', [])) + list(getattr(env, 'tactiles', [])):
        s = ctx.socket(zmq.PUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.bind('ipc:///dev/shm/%s' %c['name'])                   # local clients
        if c.get('port'): s.bind('tcp://0.0.0.0:%d' %c['port'])  # remote clients (LAN)
        socks[c['name']] = s

    def close_zmq(sockets=list(socks.values()), pub=ppub, cmd=pull, context=ctx):
        if getattr(close_zmq, 'closed', False): return
        close_zmq.closed = True
        for s in sockets + [pub, cmd]:
            try: s.close(linger=0)
            except Exception: pass
        try: context.term()
        except Exception: pass
    close_zmq.closed = False
    atexit.register(close_zmq)

    dispatch = deque()
    if arg.d is not None:
        with open(arg.d) as fin:
            for line in fin:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2: dispatch.append([int(parts[0]), parts[1]])

    def msgproc(msg):
        nonlocal cnt
        w = [s.split() for s in msg.decode().split(',')]
        if w[0][0] == 'quit': env.finish(); sys.exit(0)
        elif w[0][0] == 'reset': cnt = 0; return
        if len(w) in (1, 2):
            arm.msgproc(w[0])
            if len(w) == 2: hand.msgproc(w[1])
        else: print('wrong command')

    tau = q_ref = qd_ref = kp = kd = None   # held between control ticks (frameskip ZOH)

    while True:
        try: msg = pull.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError: pass
        else: msgproc(msg); continue

        if dispatch and cnt == dispatch[0][0]:
            msgproc(dispatch[0][1].encode()); print(dispatch[0]); dispatch.popleft()

        if cnt == 0:
            # rate fixed at 240 for both modes (see run_kida). Arm and hand share
            # the loop so they get the same rate.
            arm  = SingleController(env, 'yaml/kida%s' %ymltail, rate=240, verbose=arg.v)
            hand = GC(env, 'yaml/%s' %(gname + ymltail), rate=240, verbose=False)
            y = env.reset()

        if cnt % frameskip == 0:
            # 5-tuple command (tau, q_ref, qd_ref, kp, kd) per module — gains are
            # per-decision control outputs (2026-06-07; YAML k: removed). On real
            # the eio accepts kp/kd but ignores them (firmware owns the gains).
            arm_tau,  arm_qr,  arm_qdr,  arm_kp,  arm_kd  = arm.update(y[:arm.n_y])
            hand_tau, hand_qr, hand_qdr, hand_kp, hand_kd = hand.update(y[arm.n_y:])
            sizes  = [arm.n_u, hand.n_u]
            tau    = combine([arm_tau,  hand_tau],  sizes)
            q_ref  = combine([arm_qr,   hand_qr],   sizes)
            qd_ref = combine([arm_qdr,  hand_qdr],  sizes)
            kp     = combine([arm_kp,   hand_kp],   sizes)
            kd     = combine([arm_kd,   hand_kd],   sizes)
        y = env.step(tau=tau, q_ref=q_ref, qd_ref=qd_ref, kp=kp, kd=kd)

        #update priprio feedback
        if cnt % proprio_update_cycle == 0:
            proprio = y.astype(np.float32).tobytes()
            ppub.send(proprio)

        #update camera/tactile feedback — capability-based (see run_kida).
        if socks:
            if hasattr(env, 'camera_frames'):
                for name, buf in env.camera_frames():
                    socks[name].send(buf)
            if hasattr(env, 'tactile_frames'):
                for name, buf in env.tactile_frames():
                    socks[name].send(buf)

        cnt += 1


# ======================================================================
#  GUI (구 kida-gui.py)
# ======================================================================

# ------------------------- 백그라운드 헬퍼 -------------------------
class ProcMgr(QObject):
    """subprocess 실행 + stdout을 GUI 콘솔로 전달"""
    line = Signal(str, str)          # (태그, 내용)

    def __init__(self):
        super().__init__()
        self.procs = {}              # name -> Popen
        self.masters = {}            # name -> pty master fd (vmaster용)
        self.screens = {}            # name -> pyte.Screen (ncurses 화면 재현)

    def alive(self, name):
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def spawn(self, name, cmd, use_pty=False):
        if self.alive(name):
            self.line.emit(name, '(이미 실행 중)')
            return
        try:
            if use_pty:
                m, s = pty.openpty()
                # ncurses가 정상 동작하도록 터미널 크기 지정 (30행 100열)
                fcntl.ioctl(s, termios.TIOCSWINSZ,
                            struct.pack('HHHH', 30, 100, 0, 0))
                env = dict(os.environ, TERM='xterm')
                def _child():   # pty를 제어 터미널로 지정 (/dev/tty가 pty를 가리키게)
                    os.setsid()
                    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
                p = subprocess.Popen(cmd, cwd=REPO, stdin=s, stdout=s,
                                     stderr=s, close_fds=True, env=env,
                                     preexec_fn=_child)
                os.close(s)
                fcntl.fcntl(m, fcntl.F_SETFL, os.O_NONBLOCK)
                self.masters[name] = m
                if pyte:
                    scr = pyte.Screen(100, 30)
                    self.screens[name] = (scr, pyte.ByteStream(scr))
                threading.Thread(target=self._read_pty, args=(name, m),
                                 daemon=True).start()
            else:
                p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0)
                threading.Thread(target=self._read_pipe, args=(name, p),
                                 daemon=True).start()
            self.procs[name] = p
            self.line.emit(name, '시작: ' + ' '.join(cmd))
        except Exception as e:
            self.line.emit(name, f'실행 실패: {e}')

    def _read_pipe(self, name, p):
        for raw in iter(p.stdout.readline, b''):
            self.line.emit(name, raw.decode(errors='replace').rstrip())
        self.line.emit(name, f'(종료됨, code={p.wait()})')

    def _read_pty(self, name, m):
        buf = b''
        while True:
            try:
                chunk = os.read(m, 4096)
                if not chunk: break
                if name in self.screens:          # pyte로 ncurses 화면 재현
                    self.screens[name][1].feed(chunk)
                    continue
                buf += chunk                      # fallback: 줄 단위 표시
                while b'\n' in buf or b'\r' in buf:
                    i = min(x for x in (buf.find(b'\n'), buf.find(b'\r')) if x >= 0)
                    ln = ANSI.sub(b'', buf[:i]).strip()
                    buf = buf[i+1:]
                    if ln: self.line.emit(name, ln.decode(errors='replace'))
            except BlockingIOError:
                threading.Event().wait(0.05)
            except OSError:
                break
        self.line.emit(name, '(종료됨)')

    def screen_text(self, name):
        """pyte 화면의 현재 스냅샷 텍스트 (없으면 None)"""
        s = self.screens.get(name)
        if s is None: return None
        return '\n'.join(line.rstrip() for line in s[0].display).rstrip()

    def send_key(self, name, ch):
        """vmaster pty로 단축키 전송 (예: 'a' attach)"""
        m = self.masters.get(name)
        if m is not None:
            try: os.write(m, ch.encode())
            except OSError: pass

    def stop(self, name, sig=signal.SIGINT):
        p = self.procs.get(name)
        if p and p.poll() is None:
            p.send_signal(sig)

    def stop_all(self):
        for n in list(self.procs):
            self.stop(n)


class CamThread(threading.Thread):
    """msender ipc 소켓 구독 → 최신 프레임 보관 (mreceiver의 파이썬판)"""
    def __init__(self, name):
        super().__init__(daemon=True)
        self.name_ = name
        self.frame = None            # np.ndarray(BGR) or None
        self.lock = threading.Lock()
        self.run_flag = True

    def run(self):
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, b'')
        s.setsockopt(zmq.RCVTIMEO, 500)
        s.connect(f'ipc:///dev/shm/{self.name_}')
        while self.run_flag:
            try:
                buf = s.recv()
            except zmq.Again:
                continue
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                with self.lock:
                    self.frame = img
        s.close(0)

# ----------------------------- 창 1 -----------------------------
class Launcher(QWidget):
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr
        self.setWindowTitle('KIDA 런처')
        v = QVBoxLayout(self)

        gm = QGroupBox('실행 모드')
        hm = QHBoxLayout(gm)
        self.rb_real = QRadioButton('실물 (-x)')
        self.rb_sim  = QRadioButton('시뮬레이터')
        self.rb_real.setChecked(True)
        hm.addWidget(self.rb_real); hm.addWidget(self.rb_sim)
        v.addWidget(gm)

        g = QGroupBox('로봇핸드 (DG5F, Modbus)')
        grid = QGridLayout(g)
        self.ip_r = QLineEdit(HAND_IP_RIGHT)
        self.ip_l = QLineEdit(HAND_IP_LEFT)
        self.port = QLineEdit(str(HAND_PORT))
        grid.addWidget(QLabel('오른손 IP'), 0, 0); grid.addWidget(self.ip_r, 0, 1)
        grid.addWidget(QLabel('왼손 IP'),   1, 0); grid.addWidget(self.ip_l, 1, 1)
        grid.addWidget(QLabel('PORT'),      2, 0); grid.addWidget(self.port, 2, 1)
        note = QLabel('* IP/PORT는 연결 체크용입니다. 실제 값은 eio-dg5f에 고정되어 있습니다.')
        note.setStyleSheet('color: gray;')
        v.addWidget(g); v.addWidget(note)

        h = QHBoxLayout()
        for txt, cmd in [('CAN0 UP',  ['./can-up', '0']),
                         ('CAN1 UP',  ['./can-up', '1']),
                         ('CAN0 DOWN', ['./can-down', '0']),
                         ('CAN1 DOWN', ['./can-down', '1'])]:
            b = QPushButton(txt)
            b.clicked.connect(lambda _, c=cmd: self.run_can(c))
            h.addWidget(b)
        v.addLayout(h)

        self.log = QPlainTextEdit(readOnly=True, maximumBlockCount=500)
        self.log.setFont(QFont('monospace', 9))
        v.addWidget(self.log)

        start = QPushButton('Start')
        start.setMinimumHeight(40)
        start.clicked.connect(self.on_start)
        v.addWidget(start)
        self.mgr.line.connect(lambda tag, s: self.log.appendPlainText(f'[{tag}] {s}'))

    def run_can(self, cmd):
        # sudo 비밀번호 없이 실행하려면 sudoers에 NOPASSWD 등록 필요 (README 참고)
        r = subprocess.run(['sudo', '-n'] + cmd, cwd=REPO,
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        self.log.appendPlainText(f'[can] {" ".join(cmd)} → ' +
                                 ('OK' if r.returncode == 0 else f'실패\n{out}'))
        if r.returncode != 0 and 'password' in out.lower():
            QMessageBox.warning(self, 'sudo 필요',
                'sudo 비밀번호 없이 실행할 수 없습니다.\n'
                'sudoers에 다음 줄을 추가하세요:\n\n'
                f'{os.getlogin()} ALL=(ALL) NOPASSWD: /usr/sbin/ip')

    def check(self, ip, port):
        try:
            socket.create_connection((ip, port), timeout=2).close()
            return True
        except OSError:
            return False

    def on_start(self):
        real = self.rb_real.isChecked()
        if real:   # 시뮬레이터 모드에서는 핸드 연결 체크 생략
            port = int(self.port.text())
            fails = [ip for ip in (self.ip_r.text(), self.ip_l.text())
                     if not self.check(ip, port)]
            if fails:
                if QMessageBox.question(self, '연결 실패',
                        f'다음 핸드에 연결할 수 없습니다:\n{", ".join(fails)}\n\n'
                        '그래도 계속할까요?') != QMessageBox.Yes:
                    return
        self.main = MainWindow(self.mgr, real)
        self.main.show()
        self.close()

# ----------------------------- 창 2 -----------------------------
class MainWindow(QWidget):
    def __init__(self, mgr, real=True):
        super().__init__()
        self.mgr = mgr
        self.real = real
        self.kida_cmd = KIDA_BASE + (['-x'] if real else [])
        self.cam_names = CAM_NAMES_REAL if real else CAM_NAMES_SIM
        self.setWindowTitle('KIDA 제어 — ' + ('실물' if real else '시뮬레이터'))
        self.resize(1280, 720)

        # ZMQ: kida-run 명령 전송용 (zmqmsg와 동일 경로)
        self.push = zmq.Context.instance().socket(zmq.PUSH)
        self.push.setsockopt(zmq.IMMEDIATE, 1)
        self.push.setsockopt(zmq.LINGER, 0)
        self.push.connect('ipc:///dev/shm/default')

        root = QHBoxLayout(self)

        # 좌: 카메라 배치 (head 크게 위, left/right 아래 나란히)
        left = QVBoxLayout()
        self.cams, self.views = [], []

        def make_view(n, min_w, min_h):
            lab = QLabel(f'{n}: 대기 중')
            lab.setMinimumSize(min_w, min_h)
            lab.setAlignment(Qt.AlignCenter)
            lab.setFrameShape(QFrame.Box)
            lab.setStyleSheet('background:#111; color:#888;')
            self.views.append(lab)
            t = CamThread(n); t.start(); self.cams.append(t)
            return lab

        left.addWidget(make_view(self.cam_names[0], 640, 360), 2)   # head
        bottom = QHBoxLayout()
        bottom.addWidget(make_view(self.cam_names[1], 300, 200))     # left
        bottom.addWidget(make_view(self.cam_names[2], 300, 200))     # right
        left.addLayout(bottom, 1)
        root.addLayout(left, 3)

        # 우: 콘솔 2개 (위: vmaster, 아래: msender 등) + 버튼
        right = QVBoxLayout()

        def make_console(title):
            box = QVBoxLayout()
            head = QLabel(title)
            head.setStyleSheet('color:#aaa; font-weight:bold;')
            con = QPlainTextEdit(readOnly=True, maximumBlockCount=2000)
            con.setFont(QFont('monospace', 9))
            con.setStyleSheet('background:#111; color:#ddd;')
            con.setFocusPolicy(Qt.NoFocus)   # 키보드는 항상 메인 창이 받도록
            box.addWidget(head); box.addWidget(con)
            return box, con

        vbox, self.console_vm = make_console('vmaster')
        mbox, self.console_ms = make_console('msender')
        kbox, self.console_kd = make_console('kida-run / etc')
        right.addLayout(vbox, 1)
        bottom_con = QHBoxLayout()
        bottom_con.addLayout(mbox, 1)
        bottom_con.addLayout(kbox, 1)
        right.addLayout(bottom_con, 1)

        def btn(text, fn, h=34):
            b = QPushButton(text); b.setMinimumHeight(h); b.clicked.connect(fn)
            b.setFocusPolicy(Qt.NoFocus)     # 버튼 클릭 후에도 키보드는 창으로
            return b
        row1 = QHBoxLayout()
        row1.addWidget(btn('카메라 시작', self.on_camera))
        row1.addWidget(btn('SteamVR 실행', self.on_steamvr))
        row1.addWidget(btn('로봇 연결', self.on_connect))
        row2 = QHBoxLayout()
        row2.addWidget(btn('init', lambda: self.send('init, home, home')))
        row2.addWidget(btn('rest', lambda: self.send('rest')))
        rowv = QHBoxLayout()
        rowv.addWidget(QLabel('vmaster:'))
        for label, key in [('attach (a)', 'a'), ('home (h)', 'h'),
                           ('init (i)', 'i'), ('rest (r)', 'r'),
                           ('jaw- (z)', 'z'), ('jaw+ (x)', 'x')]:
            rowv.addWidget(btn(label, lambda _=None, k=key: self.vm_key(k), 28))
        row3 = QHBoxLayout()
        row3.addWidget(btn('로봇 종료', self.on_robot_quit))
        row3.addWidget(btn('전체 종료', self.on_shutdown))
        right.addLayout(row1); right.addLayout(row2)
        right.addLayout(rowv); right.addLayout(row3)
        root.addLayout(right, 2)

        self.mgr.line.connect(self.on_line)

        self.timer = QTimer(self, interval=33, timeout=self.refresh)  # ~30fps
        self.timer.start()

    def on_line(self, tag, s):
        if   tag == 'vmaster': con = self.console_vm
        elif tag == 'msender': con = self.console_ms
        else:                  con = self.console_kd
        con.appendPlainText(f'[{tag}] {s}')

    def vm_key(self, k):
        if not self.mgr.alive('vmaster'):
            self.console_kd.appendPlainText('[vmaster] 미실행 상태입니다')
            return
        self.mgr.send_key('vmaster', k)
        self.console_kd.appendPlainText(f'[key→vmaster] {k!r}')

    def keyPressEvent(self, e):
        # 메인 창에서 누른 키를 vmaster로 그대로 전달 (한/영 주의: 영문 상태에서)
        t = e.text()
        if e.key() == Qt.Key_Escape: t = '\x1b'
        if t and self.mgr.alive('vmaster'):
            self.vm_key(t)
        else:
            super().keyPressEvent(e)

    # ---- 카메라 / vmaster 화면 갱신 ----
    def refresh(self):
        txt = self.mgr.screen_text('vmaster')
        if txt is not None and txt != getattr(self, '_vm_last', None):
            self._vm_last = txt
            self.console_vm.setPlainText(txt)
        for t, lab in zip(self.cams, self.views):
            with t.lock:
                f = t.frame
            if f is None: continue
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            lab.setPixmap(QPixmap.fromImage(img).scaled(
                lab.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---- 버튼 동작 ----
    def send(self, cmd):
        self.console_kd.appendPlainText(f'[cmd] {cmd}')
        try: self.push.send_string(cmd, zmq.DONTWAIT)
        except zmq.Again:
            self.console_kd.appendPlainText('[cmd] 전송 실패 (kida-run 미실행?)')

    def on_camera(self):
        if not self.real:
            self.console_ms.appendPlainText(
                '[cam] 시뮬레이터 모드: 카메라는 "로봇 연결" 시 kida-run이 직접 송출합니다')
            return
        # 실물: RealSense 스트리머(msender) 실행
        self.mgr.spawn('msender', MSENDER_CMD)

    def on_steamvr(self):
        self.mgr.spawn('steamvr', STEAMVR_CMD)

    def on_connect(self):
        self.mgr.spawn('kida', self.kida_cmd)
        if self.real:
            self.mgr.spawn('msender', MSENDER_CMD)
        self.mgr.spawn('vmaster', VMASTER_CMD, use_pty=True)

    def on_robot_quit(self):
        # ./utils/zmqmsg quit 과 동일: kida-run만 종료 (핸드 프로세스도 함께 정리됨)
        self.send('quit')

    def on_shutdown(self):
        self.send('quit')                      # kida-run 정상 종료
        self.mgr.send_key('vmaster', 'q')      # vmaster 종료키
        self.mgr.stop('msender')
        self.mgr.stop('steamvr')
        QTimer.singleShot(1500, self.close)

    def closeEvent(self, e):
        for t in self.cams: t.run_flag = False
        self.mgr.stop_all()
        e.accept()


def run_gui():
    app = QApplication(sys.argv)
    mgr = ProcMgr()
    win = Launcher(mgr)
    win.resize(520, 420)
    win.show()
    sys.exit(app.exec())


# ----------------------------- main -----------------------------
if __name__ == '__main__':
    role = sys.argv[1] if len(sys.argv) > 1 else None
    if   role == 'kida-run':   run_kida(sys.argv[2:])
    elif role == 'single-run': run_single(sys.argv[2:])
    else:                      run_gui()

#!/usr/bin/env -S uv run python
# -*- coding: utf-8 -*-
"""
KIDA 통합 GUI 런처
- 창 1: can-up/down, 로봇핸드 IP/PORT 연결 체크, Start
- 창 2: 카메라 3분할 + 콘솔, SteamVR/로봇 연결/init/rest/종료 버튼
- 창 3/4: DG-5F-S 택타일 (왼손/오른손). "로봇 연결" 후 자동으로 뜹니다.
- HDF5 에피소드 수집: 작업명을 입력하면 이 폴더의
  data/<작업명>/episode_XXXXX.hdf5 로 저장
  'c' 3단계 사이클 — 1) 녹화 시작  2) 중지 및 저장  3) init 자세로 이동/준비
  → 다음 c부터 다시 1)로. ("저장 후 자동 init"을 켜면 2)에서 3)까지 한 번에.)

이 파일은 kida repo의 devs/taewon_choi_rccl/ 안에 있습니다. 실행 파일(kida-run,
rs2/msender, vive/vmaster, can-up ...)은 repo 루트에서 찾고, 수집 데이터는 이
폴더의 data/ 아래에 쌓입니다.

    ./kida-gui.py                 # 소스 실행 (uv shebang, 이 폴더의 pyproject.toml)
    ./temps/build.sh && ./kida-gui   # nuitka로 컴파일한 실행 파일

필요 패키지는 이 폴더의 pyproject.toml에 있습니다
(pyside6, pyte, pyzmq, opencv-python, h5py, numpy).
"""
import os, sys, re, pty, fcntl, shutil, socket, signal, subprocess, threading, struct, termios, time
from collections import deque

import zmq
try:
    import pyte                      # vmaster ncurses 화면 재현용
except ImportError:
    pyte = None
import numpy as np
import cv2
import h5py
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QLibraryInfo, QRectF

# cv2가 덮어쓴 Qt 플러그인 경로를 PySide6 것으로 복원 (xcb 로드 실패 방지)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = \
    QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout,
    QHBoxLayout, QVBoxLayout, QPlainTextEdit, QMessageBox, QGroupBox, QFrame,
    QRadioButton, QCheckBox, QInputDialog, QComboBox)

import handpose                        # 손 프리셋 자세 표 (deg -> rad)

# ----------------------------- 경로 -----------------------------
# 이 GUI는 kida repo 루트가 아니라 devs/<이름>/ 아래에 있습니다. 그래서 경로를
# 두 개로 나눠 씁니다.
#   APP_DIR   이 파일(또는 nuitka로 컴파일된 실행 파일)이 놓인 디렉터리
#   PROJ_DIR  GUI 프로젝트 폴더 = devs/taewon_choi_rccl (data/ 가 여기 생깁니다)
#   REPO      kida repo 루트 (./kida-run, ./rs2/msender, ./can-up ... 의 기준)
# 컴파일된 실행 파일은 kida-gui.dist/ 안에 놓이므로 APP_DIR != PROJ_DIR 이 됩니다.
# 그래서 두 값 모두 "표지 파일이 있는 상위 디렉터리"를 찾아 올라가며 정합니다.
# 폴더를 통째로 옮겼거나 자동 탐색이 실패하면 환경변수로 덮어쓸 수 있습니다:
#   KIDA_REPO=/path/to/kida  KIDA_GUI_DATA=/path/to/data  ./kida-gui.py


def _app_dir():
    """소스 실행/컴파일 실행 모두에서 '실행 파일이 있는 디렉터리'."""
    if '__compiled__' not in globals():       # 평범한 python 실행
        return os.path.dirname(os.path.abspath(__file__))
    # nuitka onefile은 dist를 /tmp/onefile_*/ 에 풀고 거기서 실행하므로
    # sys.executable도 __file__도 임시 경로를 가리킵니다(NUITKA_ONEFILE_BINARY는
    # 4.x 리눅스에서 설정되지 않음). 원래 위치를 아는 건 argv[0]뿐입니다.
    exe = os.environ.get('NUITKA_ONEFILE_BINARY') or sys.argv[0] or ''
    if os.path.sep not in exe:                # PATH로 실행된 경우
        exe = shutil.which(exe) or sys.executable
    return os.path.dirname(os.path.realpath(exe))


def _find_up(start, marker, depth=8):
    """start에서 위로 올라가며 marker(파일/디렉터리)를 가진 첫 디렉터리."""
    d = start
    for _ in range(depth):
        if os.path.exists(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


APP_DIR  = _app_dir()
PROJ_DIR = _find_up(APP_DIR, 'kida-gui.py') or APP_DIR
REPO     = os.environ.get('KIDA_REPO') or _find_up(APP_DIR, 'kida-run') \
           or os.path.abspath(os.path.join(PROJ_DIR, '..', '..'))

# ----------------------------- 설정 -----------------------------
HAND_IP_RIGHT = '192.168.0.72'
HAND_IP_LEFT  = '192.168.0.73'
HAND_PORT     = 502
CAM_NAMES_REAL = ['head', 'left', 'right']          # msender 토큰 = ipc 소켓명
CAM_NAMES_SIM  = ['headcam', 'leftcam', 'rightcam'] # 시뮬 kida-run이 publish
MSENDER_CMD = ['./rs2/msender'] + CAM_NAMES_REAL
STEAMVR_CMD = ['./vive/steamvr-run']

# 그리퍼 종류. kida-run과 vmaster가 -g 번호를 공유하므로(둘 다 0=h9, 1=dg5f,
# 2=dg5s) 값 하나로 두 프로세스를 동시에 맞출 수 있습니다. 예전에는 KIDA_BASE와
# VMASTER_CMD에 '1'이 따로 박혀 있어서 한쪽만 바꾸면 조용히 어긋났습니다.
#   yaml   yaml/<yaml>-left.yaml 접두사 (kida-run이 -g로 고르는 것과 동일)
#   relay  eio-kida.so가 fork하는 헬퍼 바이너리 (연결 실패 시 안내용)
GRIPPERS = {
    1: dict(yaml='dg5f', label='DG-5F-M', relay='eio-dg5f'),
    2: dict(yaml='dg5s', label='DG-5F-S', relay='eio-dg5s'),
}
DEFAULT_G = 2                                       # 현재 장착: DG-5F-S
TACTILE_G = 2                                       # 택타일 창을 띄우는 그리퍼


def kida_cmd(g, real):
    return ['./kida-run', '-g', str(g)] + (['-x'] if real else [])


# vmaster-gui는 vmaster와 같은 소스를 -DVMASTER_GUI로 빌드한 것입니다(vive/build.sh).
# 차이는 하나뿐: detach('a')할 때 그 시점 자세를 bias에 저장하지 않습니다. 저장하면
# 'i'(init)가 bias를 되돌리지 않기 때문에, init 후 다시 attach했을 때 팔이 init 전
# 자세로 튑니다. 아직 다시 빌드하지 않았다면 예전 vmaster로 넘어갑니다.
VMASTER_BIN = './vive/vmaster-gui'


# 글러브 캘리브레이션 프로필. vmaster의 -pN이 vive/calib/pN/{left,right}.mcal을
# 글러브에 올립니다. 예전에는 vmaster가 무조건 이동혁 박사님 파일을 올렸는데,
# 지금은 -p를 안 주면 아무것도 안 올립니다(= 글러브에 남아있는 직전 값 사용).
# 그래서 GUI가 번호를 명시적으로 넘겨야 합니다.
CALIB_DIR   = os.path.join(REPO, 'vive', 'calib')
DEFAULT_P   = 1                       # 예전 vmaster 기본 동작과 같은 값
PROFILE_MIN = 3                       # p3까지는 폴더가 없어도 목록에 보여 준다
PROFILE_P0  = '선택 안함 (글러브에 남아있는 값)'


def has_profile(n):
    d = os.path.join(CALIB_DIR, 'p%d' % n)
    return all(os.path.isfile(os.path.join(d, f))
               for f in ('left.mcal', 'right.mcal'))


def profiles():
    """[(번호, 표시문자열, 쓸 수 있나)] — 콤보에 넣을 순서 그대로.

    p1..p3은 폴더가 없어도 자리를 보여 줍니다(있다는 걸 알아야 만들 테니).
    없는 건 '(없음)'으로 표시하고 고르지 못하게 막습니다 — vmaster가 -pN을
    받고 폴더가 없으면 exit 64로 죽기 때문에, 고르게 두면 창만 사라집니다.
    p4 이상은 폴더를 만들면 그때 나타납니다.
    """
    nums = set(range(1, PROFILE_MIN + 1))
    try:
        for name in os.listdir(CALIB_DIR):
            m = re.fullmatch(r'p([1-9][0-9]*)', name)
            if m:
                nums.add(int(m.group(1)))
    except OSError:
        pass
    out = []
    for n in sorted(nums):
        ok = has_profile(n)
        out.append((n, 'p%d' % n if ok else 'p%d (없음)' % n, ok))
    out.append((0, PROFILE_P0, True))          # 맨 아래
    return out


def vmaster_cmd(g, p=DEFAULT_P):
    exe = VMASTER_BIN if os.access(os.path.join(REPO, VMASTER_BIN), os.X_OK) \
          else './vive/vmaster'
    return [exe, '-t2', '-g%d' % g, '-p%d' % p]

ANSI = re.compile(rb'\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][0-9A-B]|[\x00-\x08\x0b-\x1f]')

# ------------------------- 택타일 상수 -------------------------
# DG-5F-S 지문 택타일. YAML(yaml/dg5s-*.yaml `tactiles:`)의 손가락 순서와
# 택셀 순서를 그대로 따릅니다. 6x3 격자(손가락 축 방향 6줄 x 가로 3칸)이고
# 행 우선(row-major)인 것은 samples 좌표에서 확인된 배치입니다(utils/taccheck.py).
FINGERS   = ('thumb', 'index', 'middle', 'ring', 'little')
NTAX      = 18                                       # 손가락당 택셀 수
TAC_GRID  = (6, 3)                                   # (행, 열)
TAC_EP    = 'ipc:///dev/shm/tactile'                 # 실물: 180 float32 한 장
# 값 스케일이 sim/실물에서 완전히 다릅니다(TACTILE-ZMQ.md §3). 실물은 ADC raw
# count(무접촉 베이스라인이 0이 아님), sim은 접촉력 물리단위. 그래서 표시는
# "첫 프레임 베이스라인을 뺀 편차"를 오토스케일로 그리고, 스케일 하한만 소스별로
# 다르게 둡니다. 하한이 없으면 무접촉 잡음이 화면 가득 타오릅니다.
TAC_FLOOR = {True: 50.0, False: 0.02}                # {real: floor}
# 이보다 오래된 샘플은 stale로 본다. 실물 폴링은 60Hz, sim은 YAML fps=60이므로
# 0.25초면 4연속 유실에 해당합니다. CONFLATE 구독은 소스가 죽어도 마지막 값을
# 계속 돌려주기 때문에, 이 판정이 없으면 정지한 값이 조용히 기록됩니다.
TAC_STALE_SEC = 0.25

# --------------------- 부분 원격제어 (손가락 마스크) ---------------------
# 손가락 3개만 쓰고 싶을 때: 프리셋 자세로 손을 굳혀 두고, 아래 관절만 글러브를
# 따라가게 합니다. 표기는 사용자가 쓰는 **1-based 관절 번호**(손 벤더 GUI와 같은
# 번호) 그대로 두고, 보낼 때 1을 빼서 dg5.Controller의 0-based 'mask'로 바꿉니다.
#   Finger 1(엄지) 3·4번  = thumb3, thumb4
#   Finger 2(검지) 7·8번  = index3, index4
#   Finger 3(중지) 11·12번 = middle3, middle4
# 즉 세 손가락의 끝쪽 굽힘 2관절씩만 살리고 나머지 14관절은 프리셋에 고정됩니다.
PARTIAL_JOINTS_1BASED = (3, 4, 7, 8, 11, 12)
PARTIAL_POSE = 'pose4'                 # handpose.POSES의 기본 프리셋

RAMP = ((0.0, 18, 26, 20), (0.35, 22, 110, 52), (1.0, 120, 255, 150))


def ramp(t):
    """0..1 → (r, g, b). taccheck.py의 dark 램프와 같은 색."""
    t = 0.0 if t != t else min(max(t, 0.0), 1.0)     # NaN → 0
    for i in range(len(RAMP) - 1):
        p0, p1 = RAMP[i], RAMP[i + 1]
        if t <= p1[0]:
            f = (t - p0[0]) / max(p1[0] - p0[0], 1e-9)
            return tuple(int(round(p0[k] + f * (p1[k] - p0[k]))) for k in (1, 2, 3))
    return RAMP[-1][1:]

# ------------------------- HDF5 수집 상수 -------------------------
LOGGER_EP  = 'ipc:///dev/shm/logger'    # vmaster PUSH → 여기로 log-on/off + 액션
PROPRIO_EP = 'ipc:///dev/shm/proprio'   # kida-run PUB (162 float32)
DATA_ROOT  = os.environ.get('KIDA_GUI_DATA') \
             or os.path.join(PROJ_DIR, 'data')   # repo가 아니라 GUI 폴더 안

# proprio(162) 레이아웃 — usrsample.py / 데이터형식.md
PROPRIO_DIM = 162
ARM_POS, ARM_VEL = slice(0, 14),   slice(14, 28)
LH_POS,  LH_VEL  = slice(42, 62),  slice(62, 82)
RH_POS,  RH_VEL  = slice(102, 122), slice(122, 142)
STATE_DIM = 54                          # 팔14 + 왼손20 + 오른손20

# vmaster -t2 -g{1,2} 가 흘리는 액션 프레임: "task <12>, joint <20>, joint <20>"
ACTION_DIM = 52
LOG_ON, LOG_OFF, QUIT = 'log-on', 'log-off', 'quit'

# HDF5 안의 카메라 키는 실물/시뮬 소켓 이름과 무관하게 head/left/right로 고정합니다
# (학습 파이프라인이 이 이름을 봅니다).
HDF5_CAM_KEYS = ['head', 'left', 'right']

# 'c' 키 3단계 사이클. vmaster의 'c'는 log-on/log-off 2단계 토글뿐이라, 사이클
# 상태는 GUI가 소유하고 녹화 on/off로 넘어가는 순간에만 vmaster로 'c'를 대신
# 눌러 줍니다(init 단계는 vmaster를 건드리지 않음 → log_on 표시가 어긋나지 않음).
EP_READY, EP_REC, EP_SAVED = 0, 1, 2
EP_LABEL = {EP_READY: '녹화 시작 (c)',
            EP_REC:   '녹화 중지·저장 (c)',
            EP_SAVED: 'init 자세로 (c)'}


def proprio_to_qpos(p):
    return np.concatenate([p[ARM_POS], p[LH_POS], p[RH_POS]]).astype(np.float32)


def proprio_to_qvel(p):
    return np.concatenate([p[ARM_VEL], p[LH_VEL], p[RH_VEL]]).astype(np.float32)


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


# ----------------------------- 택타일 -----------------------------
class TactileSource(threading.Thread):
    """택타일 프레임 수집. 실물/시뮬의 채널 차이를 여기서 흡수합니다.

    실물(-x -g 2): kida-run이 'tactile' 채널 하나로 180 float32를 발행.
                   [0:90] 왼손, [90:180] 오른손, 손가락 순 thumb..little.
    시뮬(-g 2)   : YAML tactiles 마다 채널이 따로 있어 hand1./hand2. 각 5개를 구독
                   (채널당 18 float32).
    """
    def __init__(self, real):
        super().__init__(daemon=True)
        self.real = real
        self.data = np.zeros((2, len(FINGERS) * NTAX), np.float32)
        self.count = [0, 0]          # 손별 수신 프레임 수 (0 = 아직 채널 무음)
        self.stamp = [0.0, 0.0]      # 손별 마지막 갱신 시각 (monotonic)
        self.lock = threading.Lock()
        self.run_flag = True

    def _sub(self, ctx, ep):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, b'')
        s.connect(ep)
        return s

    def run(self):
        ctx = zmq.Context.instance()
        socks = {}                   # sock -> None(합본) | (hand, finger)
        if self.real:
            socks[self._sub(ctx, TAC_EP)] = None
        else:
            for h in (0, 1):
                for f, name in enumerate(FINGERS):
                    ep = 'ipc:///dev/shm/hand%d.%s_fingertip_taxel' % (h + 1, name)
                    socks[self._sub(ctx, ep)] = (h, f)
        poller = zmq.Poller()
        for s in socks: poller.register(s, zmq.POLLIN)

        while self.run_flag:
            try: events = dict(poller.poll(timeout=200))
            except zmq.ZMQError: break
            for s, where in socks.items():
                if s not in events: continue
                try: buf = s.recv()
                except zmq.ZMQError: continue
                a = np.frombuffer(buf, '<f4')
                now = time.monotonic()
                with self.lock:
                    if where is None:            # 합본 채널
                        if a.size == 180:
                            self.data[0], self.data[1] = a[:90], a[90:]
                            self.count[0] += 1; self.count[1] += 1
                            self.stamp[0] = self.stamp[1] = now
                        elif a.size == 90:       # single-run은 한 손만 싣는다
                            self.data[0] = a; self.count[0] += 1
                            self.stamp[0] = now
                    elif a.size == NTAX:         # 손가락별 채널
                        h, f = where
                        self.data[h][f*NTAX:(f+1)*NTAX] = a
                        self.count[h] += 1; self.stamp[h] = now
        for s in socks: s.close(0)

    def snapshot(self, hand):
        """(값 90개, 누적 수신 수, 마지막 갱신 후 경과초) — 표시용."""
        with self.lock:
            return (self.data[hand].copy(), self.count[hand],
                    time.monotonic() - self.stamp[hand])

    def snapshot_both(self):
        """(값 180개 = 왼손90 + 오른손90, 양손 모두 신선한가) — 기록용.

        레이아웃은 실물 'tactile' 채널의 와이어 포맷과 정확히 같습니다. 시뮬에서
        손가락별 채널로 받아 조립한 경우도 같은 순서로 맞춰 둡니다.
        """
        now = time.monotonic()
        with self.lock:
            v = self.data.reshape(-1).copy()
            fresh = all(self.count[h] > 0 and now - self.stamp[h] < TAC_STALE_SEC
                        for h in (0, 1))
        return v, fresh


class TactilePad(QWidget):
    """손가락 5개 x 6x3 택셀 격자를 그리는 위젯."""
    def __init__(self):
        super().__init__()
        self.values = np.zeros(len(FINGERS) * NTAX, np.float32)
        self.scale = 1.0
        self.show_raw = False
        self.setMinimumSize(380, 300)

    def set_values(self, v, scale):
        self.values = v
        self.scale = max(float(scale), 1e-9)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(10, 12, 10))
        rows, cols = TAC_GRID
        m, lab_h, gap = 8, 16, 8
        bw = (self.width() - 2*m - gap*(len(FINGERS)-1)) / len(FINGERS)
        bh = self.height() - 2*m - lab_h
        if bw <= 2 or bh <= 2: return
        cw, ch = bw / cols, bh / rows
        f = p.font(); f.setPointSize(8); p.setFont(f)
        for i, name in enumerate(FINGERS):
            x0 = m + i * (bw + gap)
            p.setPen(QColor(170, 170, 170))
            p.drawText(QRectF(x0, m, bw, lab_h), Qt.AlignCenter, name)
            seg = self.values[i*NTAX:(i+1)*NTAX]
            for r in range(rows):
                for c in range(cols):
                    v = float(seg[r*cols + c])
                    rect = QRectF(x0 + c*cw, m + lab_h + r*ch, cw - 1, ch - 1)
                    p.fillRect(rect, QColor(*ramp(abs(v) / self.scale)))
                    if self.show_raw:
                        p.setPen(QColor(235, 235, 235))
                        p.drawText(rect, Qt.AlignCenter, '%.0f' % v)


class TactileWindow(QWidget):
    """한 손의 택타일 창. 무접촉 베이스라인을 빼고 오토스케일로 표시합니다."""
    def __init__(self, src, hand, title, floor):
        super().__init__()
        self.src, self.hand, self.floor = src, hand, floor
        self.base = None             # 무접촉 기준값 (첫 유효 프레임에서 캡처)
        self.peak = 0.0
        self.setWindowTitle(title)
        self.resize(560, 360)

        v = QVBoxLayout(self)
        self.pad = TactilePad()
        v.addWidget(self.pad, 1)

        h = QHBoxLayout()
        b = QPushButton('영점 재설정')
        b.clicked.connect(self.rezero)
        cb = QCheckBox('숫자 표시')
        cb.toggled.connect(lambda on: setattr(self.pad, 'show_raw', on))
        self.status = QLabel('대기 중')
        self.status.setStyleSheet('color:#888;')
        h.addWidget(b); h.addWidget(cb); h.addStretch(1); h.addWidget(self.status)
        v.addLayout(h)

        self.timer = QTimer(self, interval=50, timeout=self.refresh)
        self.timer.start()

    def rezero(self):
        self.base = None             # 다음 프레임을 새 기준으로 잡는다
        self.peak = 0.0

    def refresh(self):
        v, n, age = self.src.snapshot(self.hand)
        if n == 0:
            self.status.setText('대기 중 — 채널 무음')
            return
        if age > TAC_STALE_SEC:
            self.status.setText('⚠ %.1fs 갱신 없음 (frames %d)' % (age, n))
            return
        if self.base is None:
            self.base = v.copy()
        d = v - self.base
        # 피크를 천천히 감쇠시켜 스케일이 튀지 않게 한다. 하한(floor)이 없으면
        # 무접촉 잡음이 최대치로 정규화되어 화면이 전부 밝아진다.
        self.peak = max(float(np.abs(d).max()), self.peak * 0.98)
        scale = max(self.peak, self.floor)
        self.pad.set_values(d, scale)
        self.status.setText('frames %d   scale %.3g' % (n, scale))

    def closeEvent(self, e):
        self.timer.stop()
        e.accept()


# ----------------------------- HDF5 수집 -----------------------------
class Recorder(QObject):
    """vmaster의 log-on/log-off에 맞춰 에피소드를 HDF5로 저장하는 워커.

    소켓 계약은 기존 vive/logger와 같습니다. LOGGER_EP를 PULL로 **bind** 하므로
    vmaster는 반드시 '-l' 없이(= 기본 logger를 띄우지 않고) 실행해야 합니다.
    한 프레임은 "마지막 카메라 도착" 시점에 확정되고, proprio/action/카메라가
    모두 있어야만 저장됩니다(불완전 프레임은 학습 데이터로 못 쓰므로 버립니다).

    저장 레이아웃. 앞의 5개는 act-source가 읽는 것과 동일하고, 택타일은 뒤에
    **추가로만** 붙습니다. 기존 학습 코드는 f['qpos'] / f['action'] /
    f['images/<cam>'] 만 보므로 택타일이 있든 없든 그대로 동작합니다.
        /qpos       (T, 54)  float32   팔14 + 왼손20 + 오른손20 위치
        /qvel       (T, 54)  float32   같은 순서의 속도
        /action     (T, 52)  float32   vmaster task12 + joint20 x 2
        /timestamps (T,)     float64
        /images/{head,left,right}  (T, H, W, 3) uint8, gzip
        --- 아래는 택타일을 켠 경우에만 생성 ---
        /tactile          (T, 180) float32  가공 없는 원값
        /tactile_valid    (T,)     bool     그 프레임의 샘플이 신선했는가
        /tactile_baseline (180,)   float32  에피소드 초반 무접촉 추정치(참고용)
    """
    log  = Signal(str)
    stat = Signal(str)

    def __init__(self, task, cam_names, tac_src=None):
        super().__init__()
        self.task = task
        self.cam_names = list(cam_names)          # 소켓 이름 (실물/시뮬 다름)
        # 택타일은 TactileSource가 실물/시뮬 채널 차이를 이미 흡수하고 있으므로
        # 레코더가 소켓을 따로 열지 않고 그 스냅샷을 가져다 씁니다(창과 공유).
        self.tac = tac_src
        self.dir = os.path.join(DATA_ROOT, task)
        self.cmds = deque()                       # GUI 스레드 → 워커
        self.lock = threading.Lock()
        self.run_flag = True
        self.recording = False                    # 표시용 (워커가 갱신)
        self.n_saved = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    # ---- GUI 스레드에서 호출 ----
    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        self.thread.start()

    def stop(self):
        self.run_flag = False

    def cmd(self, c):
        with self.lock:
            self.cmds.append(c)

    # ---- 워커 스레드 ----
    def _sub(self, ctx, ep):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, b'')
        s.connect(ep)
        return s

    def _next_index(self):
        used = [int(m.group(1)) for m in
                (re.match(r'episode_(\d{5})\.hdf5$', n) for n in os.listdir(self.dir))
                if m]
        return max(used) + 1 if used else 0

    def _run(self):
        ctx = zmq.Context.instance()
        try:
            pull = ctx.socket(zmq.PULL)
            pull.bind(LOGGER_EP)
        except zmq.ZMQError as e:
            self.log.emit('logger 채널 bind 실패: %s — vive/logger가 이미 떠 있지 '
                          '않은지 확인하세요 (vmaster는 -l 없이 실행해야 합니다)' % e)
            self.stat.emit('수집 실패 (logger bind)')
            return

        pro  = self._sub(ctx, PROPRIO_EP)
        cams = {n: self._sub(ctx, 'ipc:///dev/shm/%s' % n) for n in self.cam_names}

        poller = zmq.Poller()
        for s in [pull, pro] + list(cams.values()):
            poller.register(s, zmq.POLLIN)

        # 옛 세션에서 눌린 'c'/'q'가 큐에 남아 빈 에피소드가 연쇄로 지나가는 것을 방지.
        dropped = 0
        while True:
            try: pull.recv(zmq.NOBLOCK); dropped += 1
            except zmq.Again: break
        if dropped:
            self.log.emit('시작 전 큐에 쌓인 옛 메시지 %d개 폐기' % dropped)
        self.log.emit('수집 대기: %s  (c → 녹화 시작 → 저장 → init → 반복)' % self.dir)
        self.log.emit('택타일 %s' % ('기록함' if self.tac is not None else '기록 안 함'))

        last_cam = self.cam_names[-1]
        imgs = {n: None for n in self.cam_names}
        proprio = action = None
        frames, n_drop, last_stat = [], 0, 0.0

        while self.run_flag:
            # 1) GUI 명령
            while True:
                with self.lock:
                    c = self.cmds.popleft() if self.cmds else None
                if c is None: break
                if   c == 'on':      frames, n_drop = self._begin(frames)
                elif c == 'off':     frames = self._end(frames, save=True)
                elif c == 'discard': frames = self._end(frames, save=False)

            try: events = dict(poller.poll(timeout=100))
            except zmq.ZMQError: break

            # 2) vmaster 제어/액션
            if pull in events:
                w = [s.split() for s in pull.recv().decode(errors='replace').split(',')]
                head = w[0][0] if w and w[0] else ''
                if head == LOG_ON:    frames, n_drop = self._begin(frames)
                elif head == LOG_OFF: frames = self._end(frames, save=True)
                elif head == QUIT:    frames = self._end(frames, save=True)
                else:
                    # 액션 프레임: 각 콤마 조각에서 키워드(task/joint/none)를 뺀 숫자만 이어붙인다.
                    tmp = []
                    for part in w: tmp += part[1:]
                    try: t = np.array(tmp, dtype=np.float32)
                    except ValueError: t = np.empty(0, np.float32)
                    # attach가 덜 된 프레임('none' 포함)은 길이가 안 맞으므로 버린다.
                    if t.shape[0] == ACTION_DIM: action = t

            # 3) proprio
            if pro in events:
                p = np.frombuffer(pro.recv(), np.float32)
                if p.shape[0] == PROPRIO_DIM: proprio = p.copy()

            # 4) 카메라 — 마지막 카메라 도착이 한 프레임의 확정 시점
            for name, s in cams.items():
                if s not in events: continue
                imgs[name] = s.recv()
                if name != last_cam or not self.recording: continue
                if proprio is None or action is None or any(
                        imgs[n] is None for n in self.cam_names):
                    n_drop += 1
                    continue
                # 택타일은 "카메라가 확정한 그 순간의 최신값"을 집어옵니다. 채널이
                # 60Hz라 프레임(30Hz)보다 빠르지만, 소스가 멈추면 CONFLATE 구독은
                # 옛 값을 계속 돌려주므로 신선도를 함께 남깁니다.
                tacv, tac_ok = (self.tac.snapshot_both() if self.tac is not None
                                else (None, False))
                frames.append(dict(
                    imgs={n: imgs[n] for n in self.cam_names},
                    qpos=proprio_to_qpos(proprio), qvel=proprio_to_qvel(proprio),
                    action=action.copy(), tactile=tacv, tac_ok=tac_ok,
                    ts=time.time()))

            now = time.time()
            if now - last_stat > 0.3:
                last_stat = now
                self.stat.emit(self._stat_text(frames, n_drop, proprio, action, imgs))

        self._end(frames, save=True)
        for s in [pull, pro] + list(cams.values()):
            try: s.close(linger=0)
            except Exception: pass
        self.stat.emit('수집 종료')

    def _stat_text(self, frames, n_drop, proprio, action, imgs):
        miss = []
        if proprio is None: miss.append('proprio')
        if action  is None: miss.append('action(vmaster attach)')
        if any(imgs[n] is None for n in self.cam_names): miss.append('camera')
        if self.tac is not None and not self.tac.snapshot_both()[1]:
            miss.append('tactile')
        state = '● 녹화중 %d프레임' % len(frames) if self.recording else '○ 대기'
        txt = '[%s%s] %s  저장 %d개' % (self.task,
                                       '+tac' if self.tac is not None else '',
                                       state, self.n_saved)
        if n_drop: txt += '  버림 %d' % n_drop
        if miss:   txt += '  ⚠ 없음: ' + ', '.join(miss)
        return txt

    def _begin(self, frames):
        if self.recording:
            return frames, 0
        self.recording = True
        self.log.emit('에피소드 녹화 시작')
        return [], 0

    def _end(self, frames, save):
        if not self.recording:
            return frames
        self.recording = False
        if not save:
            self.log.emit('에피소드 폐기 (%d프레임)' % len(frames))
            return []
        if not frames:
            self.log.emit('저장할 프레임이 없어 건너뜀 '
                          '(vmaster attach / 카메라 / kida-run 확인)')
            return []
        try:
            path = self._save(frames)
            self.n_saved += 1
            self.log.emit('저장: %s (%d프레임)' % (path, len(frames)))
        except Exception as e:
            self.log.emit('저장 실패: %s' % e)
        return []

    def _save(self, frames):
        path = os.path.join(self.dir, 'episode_%05d.hdf5' % self._next_index())
        T = len(frames)
        with h5py.File(path, 'w') as f:
            f.create_dataset('qpos',   data=np.stack([fr['qpos']   for fr in frames]))
            f.create_dataset('qvel',   data=np.stack([fr['qvel']   for fr in frames]))
            f.create_dataset('action', data=np.stack([fr['action'] for fr in frames]))
            f.create_dataset('timestamps',
                             data=np.array([fr['ts'] for fr in frames], np.float64))
            grp = f.create_group('images')
            for key, name in zip(HDF5_CAM_KEYS, self.cam_names):
                # 저장 시점에 디코드. BGR→RGB는 act-source(시뮬 렌더가 RGB)와 맞춘 것.
                stacked = np.stack([_decode_rgb(fr['imgs'][name]) for fr in frames])
                grp.create_dataset(key, data=stacked, dtype='uint8',
                                   compression='gzip', compression_opts=2,
                                   chunks=(1,) + stacked.shape[1:])
            # --- 여기부터는 부가 데이터. 기존 학습 코드는 위 키만 읽으므로
            #     이 블록이 통째로 없어도(= 택타일 끔) 파일은 그대로 유효합니다.
            if self.tac is not None:
                tac = np.stack([fr['tactile'] for fr in frames])
                ok  = np.array([fr['tac_ok'] for fr in frames], bool)
                f.create_dataset('tactile', data=tac.astype(np.float32))
                f.create_dataset('tactile_valid', data=ok)
                # 무접촉 기준값. 값이 ADC raw count라 baseline이 0이 아니고 개체마다
                # 다르므로, 소비자가 직접 빼 쓸 수 있도록 함께 남깁니다. 에피소드
                # 시작 순간 이미 뭔가를 쥐고 있었다면 이 값은 의미가 없습니다.
                good = tac[ok]
                if len(good):
                    f.create_dataset('tactile_baseline',
                                     data=np.median(good[:30], axis=0).astype(np.float32))
                f.attrs['tactile_dim'] = 180
                f.attrs['tactile_fingers'] = ','.join(FINGERS)
                f.attrs['tactile_grid'] = '%dx%d' % TAC_GRID
                f.attrs['tactile_layout'] = (
                    '[0:90] left, [90:180] right; each hand = '
                    '(thumb,index,middle,ring,little) x 18 taxel, row-major 6x3')
                f.attrs['tactile_units'] = ('raw ADC counts, no baseline removed '
                                            '(see /tactile_baseline)')
                f.attrs['tactile_valid_ratio'] = float(ok.mean())
                if ok.mean() < 0.9:
                    self.log.emit('⚠ 택타일 신선 프레임 %.0f%% — 릴레이/채널을 확인하세요'
                                  % (100 * ok.mean()))
            f.attrs['fps'] = 30
            f.attrs['num_steps'] = T
            f.attrs['cameras'] = ','.join(HDF5_CAM_KEYS)
            f.attrs['task'] = self.task
            f.attrs['sim'] = False
            f.attrs['source'] = 'kida-gui-teleop'
            f.attrs['state_dim'] = STATE_DIM
            f.attrs['action_dim'] = ACTION_DIM
        return path


def _decode_rgb(buf):
    """JPEG bytes → (H, W, 3) uint8 RGB. 실패하면 검은 프레임."""
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)   # BGR
    if img is None:
        return np.zeros((480, 640, 3), np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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

        gg = QGroupBox('그리퍼')
        hg = QHBoxLayout(gg)
        self.rb_grip = {}
        for gnum, info in GRIPPERS.items():
            rb = QRadioButton('%s (-g %d)' % (info['label'], gnum))
            rb.setChecked(gnum == DEFAULT_G)
            rb.toggled.connect(self.on_grip_changed)
            hg.addWidget(rb)
            self.rb_grip[gnum] = rb
        v.addWidget(gg)

        pg = QGroupBox('글러브 캘리브레이션 프로필 (vmaster -p)')
        hp = QHBoxLayout(pg)
        self.cmb_prof = QComboBox()
        for num, label, ok in profiles():
            self.cmb_prof.addItem(label, num)
            if not ok:                          # 폴더 없는 번호는 고를 수 없게
                self.cmb_prof.model().item(self.cmb_prof.count() - 1).setEnabled(False)
        i = self.cmb_prof.findData(DEFAULT_P if has_profile(DEFAULT_P) else 0)
        self.cmb_prof.setCurrentIndex(max(i, 0))
        hp.addWidget(self.cmb_prof)
        hp.addWidget(QLabel('누구 것인지는 vive/calib/README.md'))
        v.addWidget(pg)

        self.hand_box = QGroupBox('로봇핸드 (Modbus)')
        grid = QGridLayout(self.hand_box)
        self.ip_r = QLineEdit(HAND_IP_RIGHT)
        self.ip_l = QLineEdit(HAND_IP_LEFT)
        self.port = QLineEdit(str(HAND_PORT))
        grid.addWidget(QLabel('오른손 IP'), 0, 0); grid.addWidget(self.ip_r, 0, 1)
        grid.addWidget(QLabel('왼손 IP'),   1, 0); grid.addWidget(self.ip_l, 1, 1)
        grid.addWidget(QLabel('PORT'),      2, 0); grid.addWidget(self.port, 2, 1)
        self.note = QLabel()
        self.note.setStyleSheet('color: gray;')
        self.note.setWordWrap(True)
        v.addWidget(self.hand_box); v.addWidget(self.note)
        self.on_grip_changed()

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

    def gripper(self):
        """현재 선택된 -g 번호."""
        for gnum, rb in self.rb_grip.items():
            if rb.isChecked(): return gnum
        return DEFAULT_G

    def on_grip_changed(self):
        info = GRIPPERS[self.gripper()]
        self.hand_box.setTitle('로봇핸드 (%s, Modbus)' % info['label'])
        self.note.setText(
            '* IP/PORT는 연결 체크용입니다. 실제 값은 %s에 고정되어 있습니다.\n'
            '* 이 선택은 kida-run과 vmaster에 같은 -g 값으로 함께 전달됩니다.\n'
            '* 택타일 창은 %s에서만 뜹니다.'
            % (info['relay'], GRIPPERS[TACTILE_G]['label']))

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
        self.main = MainWindow(self.mgr, real, self.gripper(),
                               self.cmb_prof.currentData())
        self.main.show()
        self.close()

# ----------------------------- 창 2 -----------------------------
class MainWindow(QWidget):
    def __init__(self, mgr, real=True, g=DEFAULT_G, p=DEFAULT_P):
        super().__init__()
        self.mgr = mgr
        self.real = real
        self.g = g
        self.p = p
        # 한 번 고른 -g가 kida-run과 vmaster 양쪽으로 나갑니다.
        self.kida_cmd = kida_cmd(g, real)
        self.vmaster_cmd = vmaster_cmd(g, p)
        self.cam_names = CAM_NAMES_REAL if real else CAM_NAMES_SIM
        self.setWindowTitle('KIDA 제어 — %s / %s'
                            % ('실물' if real else '시뮬레이터', GRIPPERS[g]['label'])
                            + (' · 프로필 p%d' % p if p else ' · 프로필 없음'))
        self.resize(1280, 720)

        self.tac_src = None          # TactileSource (dg5s에서만)
        self.tac_wins = []
        self.rec = None              # Recorder (수집 중일 때만)
        self.ep_state = EP_READY     # 'c' 3단계 사이클 위치
        self.partial_on = False      # 손가락 마스크가 걸려 있는가

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
        self.b_tac = btn('택타일 창', self.on_tactile)
        self.b_tac.setEnabled(g == TACTILE_G)
        if g != TACTILE_G:
            self.b_tac.setToolTip('택타일은 %s에만 있습니다' % GRIPPERS[TACTILE_G]['label'])
        row1.addWidget(self.b_tac)
        row2 = QHBoxLayout()
        row2.addWidget(btn('init', lambda: self.send('init, home, home')))
        row2.addWidget(btn('rest', lambda: self.send('rest')))
        rowv = QHBoxLayout()
        rowv.addWidget(QLabel('vmaster:'))
        for label, key in [('attach (a)', 'a'), ('home (h)', 'h'),
                           ('init (i)', 'i'), ('rest (r)', 'r'),
                           ('jaw- (z)', 'z'), ('jaw+ (x)', 'x')]:
            rowv.addWidget(btn(label, lambda _=None, k=key: self.vm_key(k), 28))

        # 부분 원격제어 UI — 버튼 하나로 "프리셋 자세로 이동 + 그 순간부터
        # 지정 관절만 글러브 추종"까지 끝냅니다.
        gf = QGroupBox('부분 원격제어 (손가락 3개)')
        fv = QVBoxLayout(gf)
        frow = QHBoxLayout()
        self.cmb_pose = QComboBox()
        self.cmb_pose.addItems(handpose.names())
        i = self.cmb_pose.findText(PARTIAL_POSE)
        if i >= 0: self.cmb_pose.setCurrentIndex(i)
        self.cmb_pose.setFocusPolicy(Qt.NoFocus)   # 키보드는 계속 vmaster로
        self.b_partial = btn('3손가락 모드 시작', self.on_partial_toggle, 30)
        frow.addWidget(QLabel('프리셋')); frow.addWidget(self.cmb_pose, 1)
        frow.addWidget(self.b_partial, 1)
        self.partial_stat = QLabel('전체 20관절 원격제어 중')
        self.partial_stat.setStyleSheet('color:#888;')
        self.partial_stat.setWordWrap(True)
        fv.addLayout(frow); fv.addWidget(self.partial_stat)

        # HDF5 수집 UI
        gr = QGroupBox('HDF5 에피소드 수집')
        gv = QVBoxLayout(gr)
        rrow = QHBoxLayout()
        self.b_rec = btn('HDF5 에피소드 수집', self.on_record_toggle, 30)
        self.b_ep = btn(EP_LABEL[EP_READY], self.on_episode_toggle, 30)
        self.b_drop = btn('현재 에피소드 폐기', self.on_episode_discard, 30)
        self.b_ep.setEnabled(False); self.b_drop.setEnabled(False)
        rrow.addWidget(self.b_rec); rrow.addWidget(self.b_ep); rrow.addWidget(self.b_drop)
        self.cb_autoinit = QCheckBox('저장 후 자동 init (c 3단계 생략)')
        # 택타일은 /tactile* 로만 추가되고 기존 키는 건드리지 않으므로, 켜 둔 파일도
        # 관절/영상만 쓰는 학습 코드가 그대로 읽습니다. 기본값 on.
        self.cb_tactile = QCheckBox('택타일 함께 기록 (%s)' % GRIPPERS[TACTILE_G]['label'])
        self.cb_tactile.setChecked(g == TACTILE_G)
        self.cb_tactile.setEnabled(g == TACTILE_G)
        self.rec_stat = QLabel('수집 꺼짐')
        self.rec_stat.setStyleSheet('color:#8c8;')
        crow = QHBoxLayout()
        crow.addWidget(self.cb_autoinit); crow.addWidget(self.cb_tactile); crow.addStretch(1)
        gv.addLayout(rrow)
        gv.addLayout(crow)
        gv.addWidget(self.rec_stat)

        row3 = QHBoxLayout()
        row3.addWidget(btn('로봇 종료', self.on_robot_quit))
        row3.addWidget(btn('전체 종료', self.on_shutdown))
        right.addLayout(row1); right.addLayout(row2)
        right.addLayout(rowv); right.addWidget(gf); right.addWidget(gr)
        right.addLayout(row3)
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
        if t == 'c' and self.rec is not None:
            self.on_episode_toggle()     # 3단계 사이클은 GUI가 소유 (vmaster의
            return                       # 'c'는 사이클이 필요할 때만 대신 눌린다)
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
        # vmaster-gui가 없으면 vmaster로 조용히 폴백하는데, 그러면 sticky bias가
        # 살아 있어서 init 후 다시 attach할 때 팔이 init 전 자세로 튑니다.
        # 조용히 넘어가면 원인을 못 찾으니 콘솔에 남깁니다 (vive/build.sh 참고).
        if not self.vmaster_cmd[0].endswith('vmaster-gui'):
            self.console_kd.appendPlainText(
                '[vmaster] 경고: vive/vmaster-gui가 없어 vmaster로 실행합니다. '
                'init 후 재attach 시 팔이 튈 수 있습니다 — cd vive && ./build.sh')
        self.mgr.spawn('vmaster', self.vmaster_cmd, use_pty=True)
        # 택타일 채널은 kida-run이 bind한 뒤에야 생기므로 조금 기다렸다 연다.
        if self.g == TACTILE_G:
            QTimer.singleShot(2000, self.open_tactile)

    # ---- 부분 원격제어 (손가락 마스크) ----
    def hands(self, left, right):
        """팔 자리는 'none'으로 비우고 두 손에만 명령을 보낸다. kida-run이 콤마로
        (arm, hand1, hand2)를 갈라 넘기고, kida.Controller는 모르는 낱말을 그냥
        무시하므로 팔은 하던 동작을 그대로 이어갑니다."""
        self.send('none, %s, %s' % (left, right))

    def on_partial_toggle(self):
        """3손가락 모드 on/off.

        켤 때는 mask -> hold 순서로 보냅니다. 마스크를 먼저 걸어 두어야, 글러브가
        attach 되어 초당 수백 개씩 들어오는 joint 명령이 hold 자세를 즉시 덮어쓰지
        않고 마스크된 14관절을 프리셋으로 끌고 갑니다. detach 상태라면 hold 명령
        자체가 손 전체를 프리셋 자세로 옮깁니다. 어느 쪽이든 버튼을 누른 순간부터
        지정 관절 외에는 움직이지 않습니다.
        """
        if self.partial_on:
            self.hands('mask all', 'mask all')
            self.partial_on = False
            self.b_partial.setText('3손가락 모드 시작')
            self.partial_stat.setText('전체 20관절 원격제어 중')
            return

        name = self.cmb_pose.currentText()
        try:
            pose = {s: handpose.rad(name, s) for s in ('left', 'right')}
        except (KeyError, ValueError) as e:
            self.console_kd.appendPlainText('[partial] 프리셋 %s 오류: %s' % (name, e))
            return
        idx = ' '.join(str(j - 1) for j in PARTIAL_JOINTS_1BASED)   # 1-based -> 0-based
        self.hands('mask ' + idx, 'mask ' + idx)
        self.hands(*['hold ' + ' '.join('%.4f' % v for v in pose[s])
                     for s in ('left', 'right')])
        self.partial_on = True
        self.b_partial.setText('3손가락 모드 해제')
        self.partial_stat.setText(
            '프리셋 %s 유지 — %s번 관절만 글러브 추종 (나머지 14관절 고정)'
            % (name, ', '.join(str(j) for j in PARTIAL_JOINTS_1BASED)))

    # ---- 택타일 ----
    def tactile_source(self):
        """택타일 구독 스레드. 창과 레코더가 하나를 공유하도록 늦게 한 번만 만든다."""
        if self.g != TACTILE_G:
            return None
        if self.tac_src is None:
            self.tac_src = TactileSource(self.real)
            self.tac_src.start()
        return self.tac_src

    def open_tactile(self):
        if self.g != TACTILE_G:
            self.console_kd.appendPlainText(
                '[tactile] %s에는 택타일이 없습니다' % GRIPPERS[self.g]['label'])
            return
        if self.tac_wins:                      # 이미 열려 있으면 앞으로 가져오기만
            for w in self.tac_wins: w.raise_(); w.activateWindow()
            return
        self.tactile_source()
        floor = TAC_FLOOR[self.real]
        geo = self.geometry()
        for hand, title in ((0, '택타일 — 왼손 (DG-5F-S)'), (1, '택타일 — 오른손 (DG-5F-S)')):
            w = TactileWindow(self.tac_src, hand, title, floor)
            w.move(geo.x() + 60 + hand * 580, geo.y() + geo.height() - 300)
            w.show()
            self.tac_wins.append(w)
        self.console_kd.appendPlainText(
            '[tactile] 창 2개 열림 (%s)' % ('실물 tactile 채널' if self.real
                                            else '시뮬 hand1./hand2. 채널'))

    def on_tactile(self):
        if self.tac_wins:                      # 토글: 열려 있으면 닫는다
            for w in self.tac_wins: w.close()
            self.tac_wins = []
            return
        self.open_tactile()

    # ---- HDF5 수집 ----
    def on_record_toggle(self):
        if self.rec is not None:
            self.rec.stop()
            self.rec = None
            self.b_rec.setText('HDF5 에피소드 수집')
            self.b_ep.setEnabled(False); self.b_drop.setEnabled(False)
            self.cb_tactile.setEnabled(self.g == TACTILE_G)
            self.rec_stat.setText('수집 꺼짐')
            self.set_ep_state(EP_READY)
            return
        task, ok = QInputDialog.getText(
            self, 'HDF5 에피소드 수집', '작업명 (%s/<작업명>/ 아래에 저장됩니다):' % DATA_ROOT)
        task = re.sub(r'[^0-9A-Za-z가-힣._-]', '_', (task or '').strip())
        if not ok or not task:
            return
        tac = self.tactile_source() if self.cb_tactile.isChecked() else None
        self.rec = Recorder(task, self.cam_names, tac_src=tac)
        self.rec.log.connect(lambda s: self.console_kd.appendPlainText('[rec] ' + s))
        self.rec.stat.connect(self.rec_stat.setText)
        self.rec.start()
        self.cb_tactile.setEnabled(False)       # 수집 중 변경 금지 (파일마다 달라짐)
        self.b_rec.setText('수집 중지 (%s)' % task)
        self.b_ep.setEnabled(True); self.b_drop.setEnabled(True)
        self.set_ep_state(EP_READY)

    def set_ep_state(self, st):
        self.ep_state = st
        self.b_ep.setText(EP_LABEL[st])

    def ep_log(self, on):
        """녹화 on/off. vmaster가 살아 있으면 'c'를 대신 눌러준다 — log_on 상태를
        vmaster가 소유하므로, GUI가 따로 토글하면 두 상태가 어긋난다."""
        if self.mgr.alive('vmaster'):
            self.vm_key('c')
        else:
            self.rec.cmd('on' if on else 'off')

    def ep_init(self):
        self.send('init, home, home')

    def on_episode_toggle(self):
        """'c' 3단계 사이클: 녹화 시작 → 중지·저장 → init 후 준비 → (반복).

        상태는 GUI가 들고 있다. Recorder.recording은 워커 스레드가 log-on/off를
        받은 뒤에야 바뀌므로, 그걸 보고 판단하면 연타 시 단계가 밀린다.
        """
        if self.rec is None: return
        if self.ep_state == EP_READY:
            self.ep_log(True)
            self.set_ep_state(EP_REC)
        elif self.ep_state == EP_REC:
            self.ep_log(False)
            if self.cb_autoinit.isChecked():   # 3단계를 건너뛰고 바로 준비까지
                QTimer.singleShot(500, self.ep_init)
                self.set_ep_state(EP_READY)
            else:
                self.console_kd.appendPlainText(
                    '[rec] 저장 요청 — 다음 c에서 init 자세로 갑니다')
                self.set_ep_state(EP_SAVED)
        else:                                  # EP_SAVED
            self.ep_init()
            self.console_kd.appendPlainText(
                '[rec] init 이동 — 다음 c에서 녹화를 시작합니다')
            self.set_ep_state(EP_READY)

    def on_episode_discard(self):
        if self.rec is None: return
        if self.ep_state == EP_REC:
            self.ep_log(False)                 # vmaster의 log_on도 꺼준다
        self.rec.cmd('discard')
        self.set_ep_state(EP_SAVED)            # 폐기해도 init 단계는 거친다

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
        if self.tac_src is not None: self.tac_src.run_flag = False
        for w in self.tac_wins: w.close()
        if self.rec is not None: self.rec.stop()
        self.mgr.stop_all()
        e.accept()

# ----------------------------- main -----------------------------
if __name__ == '__main__':
    if '--paths' in sys.argv:      # 배포 후 경로 확인용 (GUI를 띄우지 않음)
        for k in ('APP_DIR', 'PROJ_DIR', 'REPO', 'DATA_ROOT'):
            print('%-9s %s' % (k, globals()[k]))
        sys.exit(0)
    app = QApplication(sys.argv)
    # repo를 못 찾으면 ./kida-run 같은 상대 경로가 전부 조용히 실패합니다.
    # 여기서 한 번만 확인하고, 실패하면 무엇을 고쳐야 하는지 알려 줍니다.
    if not os.access(os.path.join(REPO, 'kida-run'), os.X_OK):
        QMessageBox.critical(None, 'kida repo를 찾지 못했습니다',
            'kida-run을 찾을 수 없습니다:\n  %s\n\n'
            'GUI 폴더를 repo 밖으로 옮겼다면 환경변수로 알려 주세요:\n'
            '  KIDA_REPO=/path/to/kida %s' % (REPO, sys.argv[0]))
        sys.exit(1)
    os.makedirs(DATA_ROOT, exist_ok=True)
    mgr = ProcMgr()
    win = Launcher(mgr)
    win.resize(520, 460)
    win.show()
    sys.exit(app.exec())

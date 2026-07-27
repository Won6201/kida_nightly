#!/usr/bin/env -S uv run python
# -*- coding: utf-8 -*-
"""
KIDA 통합 GUI 런처
- 창 1: can-up/down, 로봇핸드 IP/PORT 연결 체크, Start
- 창 2: 카메라 3분할 + 콘솔, SteamVR/로봇 연결/init/rest/종료 버튼

repo 루트에 이 파일을 두고 실행하세요:
    uv run python kida_gui.py
필요 패키지: pyside6, pyzmq, opencv-python (uv add pyside6)
"""
import os, sys, re, pty, fcntl, socket, signal, subprocess, threading, struct, termios

import zmq
try:
    import pyte                      # vmaster ncurses 화면 재현용
except ImportError:
    pyte = None
import numpy as np
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

# ----------------------------- 설정 -----------------------------
REPO = os.path.dirname(os.path.abspath(__file__))   # repo 루트 기준
HAND_IP_RIGHT = '192.168.0.72'
HAND_IP_LEFT  = '192.168.0.73'
HAND_PORT     = 502
CAM_NAMES_REAL = ['head', 'left', 'right']          # msender 토큰 = ipc 소켓명
CAM_NAMES_SIM  = ['headcam', 'leftcam', 'rightcam'] # 시뮬 kida-run이 publish
KIDA_BASE   = ['./kida-run', '-g', '1']             # DG5F (-x는 실물 모드에서 추가)
MSENDER_CMD = ['./rs2/msender'] + CAM_NAMES_REAL
VMASTER_CMD = ['./vive/vmaster', '-t2', '-g1']
STEAMVR_CMD = ['./vive/steamvr-run']

ANSI = re.compile(rb'\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][0-9A-B]|[\x00-\x08\x0b-\x1f]')

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

# ----------------------------- main -----------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mgr = ProcMgr()
    win = Launcher(mgr)
    win.resize(520, 420)
    win.show()
    sys.exit(app.exec())
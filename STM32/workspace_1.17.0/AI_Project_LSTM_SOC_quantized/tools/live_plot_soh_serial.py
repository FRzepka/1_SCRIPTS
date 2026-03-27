#!/usr/bin/env python3
"""
Live-Plot für STM32 SOH-Stream über UART.

Funktionen
- Liest kontinuierlich Zeilen vom seriellen Port
- Extrahiert die SOH-Vorhersage (robuste Float-Erkennung, nimmt den letzten Wert in [0, 1.2])
- Plottet ein rollendes Fenster (Standard: 2000 Punkte)
- Y-Achse fix auf [0.8, 1.0] (konfigurierbar)
- Optional: Simulation, wenn kein Board angeschlossen ist (--simulate)

Beispiel
python STM32\\workspace_1.17.0\\AI_Project_LSTM_quantized\\tools\\live_plot_soh_serial.py \\
  --port COM7 --baud 115200 --window 2000 --ymin 0.8 --ymax 1.0
"""
import argparse, re, sys, time, threading
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')  # interaktives Backend
import matplotlib.pyplot as plt

try:
    import serial  # pyserial
except Exception as e:
    serial = None


def parse_float(line: str) -> float | None:
    """Extrahiert den letzten Float-Wert in [0, 1.2] aus einer Zeile."""
    # generische Float-Erkennung
    floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    cand = None
    for tok in floats:
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.2:
            cand = v  # nehme den letzten passenden
    return cand


def serial_reader(port: str, baud: int, q: deque, stop_evt: threading.Event, only_prefix: str | None):
    if serial is None:
        print("pyserial nicht verfügbar. Installiere mit: pip install pyserial")
        stop_evt.set(); return
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.2)
        # kurze Wartezeit für Reset
        time.sleep(0.5)
        buf = b""
        while not stop_evt.is_set():
            try:
                chunk = ser.read(256)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode('utf-8', errors='ignore').strip()
                    if only_prefix and not s.startswith(only_prefix):
                        continue
                    v = parse_float(s)
                    if v is not None:
                        q.append(float(v))
            except serial.SerialException:
                break
        ser.close()
    except Exception as e:
        print(f"[serial] Fehler: {e}")
        stop_evt.set()


def simulate_reader(q: deque, stop_evt: threading.Event):
    """Erzeugt eine leicht fallende Kurve mit Rauschen, Start bei 1.0"""
    import random, math
    t = 0
    y = 1.0
    while not stop_evt.is_set():
        # kleiner Drift nach unten + Rauschen
        y = max(0.7, y - 1e-5 + random.uniform(-2e-4, 2e-4))
        q.append(y)
        t += 1
        time.sleep(0.001)


def serial_driver_demo(port: str, baud: int, rate_hz: float, stop_evt: threading.Event):
    """Sendet Demo-Featurezeilen (6 Floats) zyklisch an das Board."""
    if serial is None:
        print("pyserial nicht verfügbar. Installiere mit: pip install pyserial")
        stop_evt.set(); return
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.2)
        time.sleep(0.3)
        period = 1.0 / max(1.0, float(rate_hz))
        t = 0
        vbat = 3.35
        cur = 0.60
        temp = 26.0
        cyc = 1000.0
        dv = 1.0e-4
        di = 0.8e-4
        while not stop_evt.is_set():
            # einfache Variation um plausible Werte
            vbat += 0.0005 * ((t % 200) - 100) / 100.0
            cur  += 0.001  * ((t % 150) - 75)  / 75.0
            temp += 0.01   * ((t % 500) - 250) / 250.0
            cyc  += 1.0
            dv    = 1.0e-4 + 0.2e-4 * ((t % 300) - 150) / 150.0
            di    = 0.8e-4 + 0.2e-4 * ((t % 400) - 200) / 200.0
            line = f"{vbat:.6f} {cur:.6f} {temp:.6f} {cyc:.6f} {dv:.6f} {di:.6f}\n"
            ser.write(line.encode('utf-8'))
            ser.flush()
            t += 1
            time.sleep(period)
        ser.close()
    except Exception as e:
        print(f"[driver] Fehler: {e}")
        stop_evt.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM7')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--window', type=int, default=2000, help='Anzahl Punkte im Plot-Fenster')
    ap.add_argument('--ymin', type=float, default=0.8)
    ap.add_argument('--ymax', type=float, default=1.0)
    ap.add_argument('--title', default='STM32 SOH live')
    ap.add_argument('--prefix', default='SOC:', help='Nur Zeilen mit diesem Prefix auswerten (z.B. "SOC:"); leer = alle')
    ap.add_argument('--drive', choices=['none','demo'], default='none', help='Host sendet Featurezeilen an das Board (demo)')
    ap.add_argument('--rate', type=float, default=200.0, help='Sende-Rate für --drive (Hz)')
    ap.add_argument('--eol', choices=['lf','crlf'], default='crlf', help='Zeilenende beim Senden (demo): lf=\n, crlf=\r\n')
    ap.add_argument('--simulate', action='store_true', help='Ohne serielles Board eine SOH-Kurve simulieren')
    args = ap.parse_args()

    data = deque(maxlen=args.window)
    stop_evt = threading.Event()

    if args.simulate or serial is None:
        th = threading.Thread(target=simulate_reader, args=(data, stop_evt), daemon=True)
    else:
        pref = None if (args.prefix is None or len(args.prefix)==0) else args.prefix
        th = threading.Thread(target=serial_reader, args=(args.port, args.baud, data, stop_evt, pref), daemon=True)
        if args.drive == 'demo':
            # paralleler Sender, der das Board füttert
            # wrap demo sender to honor chosen EOL
            def _drv():
                if serial is None:
                    return
                try:
                    ser = serial.Serial(args.port, baudrate=args.baud, timeout=0.2)
                    time.sleep(0.3)
                    period = 1.0 / max(1.0, float(args.rate))
                    t = 0
                    vbat = 3.35; cur = 0.60; temp = 26.0; cyc = 1000.0; dv = 1.0e-4; di = 0.8e-4
                    eol = '\r\n' if args.eol == 'crlf' else '\n'
                    while not stop_evt.is_set():
                        vbat += 0.0005 * ((t % 200) - 100) / 100.0
                        cur  += 0.001  * ((t % 150) - 75)  / 75.0
                        temp += 0.01   * ((t % 500) - 250) / 250.0
                        cyc  += 1.0
                        dv    = 1.0e-4 + 0.2e-4 * ((t % 300) - 150) / 150.0
                        di    = 0.8e-4 + 0.2e-4 * ((t % 400) - 200) / 200.0
                        line = f"{vbat:.6f} {cur:.6f} {temp:.6f} {cyc:.6f} {dv:.6f} {di:.6f}" + eol
                        ser.write(line.encode('utf-8')); ser.flush()
                        t += 1; time.sleep(period)
                    ser.close()
                except Exception as e:
                    print(f"[driver] Fehler: {e}")
                    stop_evt.set()
            drv = threading.Thread(target=_drv, daemon=True)
            drv.start()
    th.start()

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.set_title(args.title)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlim(0, args.window - 1)
    ax.grid(True, alpha=0.2)
    line, = ax.plot([], [], lw=1.0, label='SOH')
    ax.legend(loc='upper right')

    paused = {'v': False}

    def on_key(event):
        if event.key in ('q', 'Q'):
            stop_evt.set(); plt.close('all')
        elif event.key == ' ':
            paused['v'] = not paused['v']

    plt.gcf().canvas.mpl_connect('key_press_event', on_key)

    xs = list(range(args.window))
    ys = [None] * args.window

    def redraw():
        if not paused['v']:
            n = len(data)
            if n:
                # rechtsbündig ins Fenster schreiben
                ys[:] = [None] * args.window
                d = list(data)
                take = min(args.window, n)
                ys[-take:] = d[-take:]
                # Matplotlib kann keine None; ersetze fehlende Werte mit NaN
                import numpy as np
                yplot = np.array([float('nan') if v is None else v for v in ys], dtype=float)
                line.set_data(xs, yplot)
        plt.pause(0.01)

    try:
        while not stop_evt.is_set():
            redraw()
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        th.join(timeout=1.0)


if __name__ == '__main__':
    main()

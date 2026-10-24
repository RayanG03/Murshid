#!/usr/bin/env python3
"""One-command launcher: starts the FastAPI backend AND the Angular frontend.

Run from the repo root (the folder that contains backend/ and frontend/):

    python main.py                 # both servers + opens the browser
    python main.py --backend-only  # just the API on :8000
    python main.py --frontend-only # just ng serve on :4200
    python main.py --dry-run       # print the resolved commands and exit
    python main.py --reload        # backend autoreload during development

What it does:
  1. Prefers the project virtualenv (backend/.venv) for the backend process;
     falls back to the interpreter running this script.
  2. Starts uvicorn (src.main:app) on http://127.0.0.1:8000 and waits until
     GET /health returns {"status": "ok"} (model load can take a while on
     the first run, so the wait is patient and shows elapsed time).
  3. Starts `npm start` (ng serve) in frontend/ and waits until the dev
     server accepts TCP connections on :4200. The probe tries every address
     family `localhost` resolves to (IPv4 127.0.0.1 AND IPv6 ::1), because
     ng serve/Vite may bind either one depending on the OS; the dev server's
     own "Local:" banner in its output is accepted as a second signal. A
     readiness timeout is NON-FATAL: ng serve keeps running and the browser
     still opens — a slow check must never kill healthy servers.
  4. Opens the browser at http://localhost:4200 (unless --no-browser).
  5. Both children are killed on Ctrl+C / when either exits early, so no
     orphaned node/uvicorn processes are left behind.

Console output is line-tagged [backend] / [frontend]; both children run
with UTF-8 forced (PYTHONUTF8/PYTHONIOENCODING) so the Arabic log lines
never crash Windows console encodings. Standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
BACKEND_ENTRY = BACKEND_DIR / "src" / "main.py"


def backend_python() -> Path:
    """Project venv python if present, else the running interpreter."""
    candidates = [
        BACKEND_DIR / ".venv" / "Scripts" / "python.exe",  # Windows
        BACKEND_DIR / ".venv" / "bin" / "python",          # macOS / Linux
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def frontend_npm_command(port: int) -> list[str]:
    """npm start (ng serve); explicit cmd /c wrapper on Windows because
    npm is a .cmd shim and raw .cmd execution varies across Python
    versions. `npm start` already maps to `ng serve` in package.json."""
    if os.name == "nt":
        return ["cmd.exe", "/c", "npm", "start", "--", "--port", str(port)]
    npm = shutil.which("npm") or "npm"
    return [npm, "start", "--", "--port", str(port)]


def backend_command(port: int, reload_dev: bool) -> list[str]:
    cmd = [
        str(backend_python()), "-m", "uvicorn", "src.main:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if reload_dev:
        cmd.append("--reload")
    return cmd


def child_env() -> dict[str, str]:
    """Inherit the environment; force UTF-8 for the children so Arabic
    logs/answers survive Windows console code pages."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


# ng serve (Vite) prints its URL banner when the dev server starts listening,
# e.g. "  ➜  Local:   http://localhost:4200/".
_FRONTEND_READY_RE = re.compile(r"\bLocal:\s*https?://", re.IGNORECASE)


def _pump(pipe, tag: str, on_line=None) -> None:
    """Stream a child's output line-by-line with a [tag] prefix."""
    for raw_line in iter(pipe.readline, ""):
        line = raw_line.rstrip("\r\n")
        if line:
            print(f"[{tag}] {line}", flush=True)
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:
                    pass  # a watcher must never break the output pump
    pipe.close()


def spawn(tag: str, cmd: list[str], cwd: Path, on_line=None) -> subprocess.Popen:
    print(f"[launcher] starting {tag}: {' '.join(cmd)}  (cwd={cwd})", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    threading.Thread(target=_pump, args=(proc.stdout, tag, on_line),
                     daemon=True).start()
    return proc


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a child and everything it spawned (npm -> ng -> esbuild...)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except OSError:
        pass


def wait_for_health(port: int, timeout_s: float) -> bool:
    """Poll GET /health until it reports ok (or give up)."""
    url = f"http://127.0.0.1:{port}/health"
    print(f"[launcher] waiting for backend health at {url} "
          f"(first start can take a few minutes: models are loading)",
          flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if time.monotonic() - t0 > 1.0:  # show progress after the first try
            print(f"[launcher]   ...waiting for backend "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200 and b"ok" in resp.read():
                    print(f"[launcher] backend healthy after "
                          f"{time.monotonic() - t0:.1f}s", flush=True)
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    print(f"[launcher] ERROR: backend not healthy after {timeout_s:.0f}s",
          flush=True)
    return False


def _tcp_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def port_open(port: int) -> bool:
    """Probe the dev server on EVERY address family localhost may use.

    ng serve/Vite binds to 'localhost', which on Windows Node often resolves
    to IPv6 ::1 only — probing just 127.0.0.1 then never connects even though
    the app is up and the browser works. Try 127.0.0.1, ::1, and whatever
    getaddrinfo('localhost') returns.
    """
    hosts = ["127.0.0.1", "::1"]
    try:
        infos = socket.getaddrinfo("localhost", port, type=socket.SOCK_STREAM)
        for info in infos:
            addr = info[4][0]
            if addr not in hosts:
                hosts.append(addr)
    except socket.gaierror:
        pass
    return any(_tcp_ok(host, port) for host in hosts)


def wait_for_port(port: int, timeout_s: float,
                  ready_event: threading.Event | None = None) -> bool:
    """Wait until the Angular dev server accepts TCP connections.

    Two readiness signals: an actual TCP connect on any address family, or
    ng serve's own 'Local: http://localhost:<port>/' banner in its output
    (ready_event, set by the output pump). On timeout this only WARNS —
    the caller keeps the servers running.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if port_open(port):
            print(f"[launcher] frontend ready after "
                  f"{time.monotonic() - t0:.1f}s", flush=True)
            return True
        if ready_event is not None and ready_event.is_set():
            # ng serve printed its URL; give the listener a moment to bind,
            # then trust the banner even if our probes still fail (a probe
            # can miss what a browser reaches, never the other way around).
            for _ in range(6):
                if port_open(port):
                    break
                time.sleep(0.5)
            print(f"[launcher] frontend ready (dev-server banner) after "
                  f"{time.monotonic() - t0:.1f}s", flush=True)
            return True
        time.sleep(1.0)
    print(f"[launcher] WARNING: frontend port {port} not ready after "
          f"{timeout_s:.0f}s (check [frontend] logs above) — "
          f"continuing anyway; open http://localhost:{port} manually if "
          f"the [frontend] logs look healthy", flush=True)
    return False


def dry_run(args: argparse.Namespace) -> int:
    problems: list[str] = []
    if not BACKEND_ENTRY.exists():
        problems.append(f"missing {BACKEND_ENTRY}")
    if not (FRONTEND_DIR / "package.json").exists():
        problems.append(f"missing {FRONTEND_DIR / 'package.json'}")
    py = backend_python()
    print(f"backend python : {py}"
          + ("" if py.exists() else "  (falls back to running interpreter)"))
    print(f"backend cmd    : {' '.join(backend_command(args.backend_port, args.reload))}")
    print(f"backend cwd    : {BACKEND_DIR}")
    print(f"frontend cmd   : {' '.join(frontend_npm_command(args.frontend_port))}")
    print(f"frontend cwd   : {FRONTEND_DIR}")
    print(f"browser        : {'off' if args.no_browser else f'http://localhost:{args.frontend_port}'}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("dry-run OK" if not problems else "dry-run FAILED")
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the backend (uvicorn :8000) and frontend "
                    "(ng serve :4200) together.",
    )
    parser.add_argument("--backend-port", type=int, default=8000,
                        help="backend port (default 8000; the frontend "
                             "hardcodes http://localhost:8000)")
    parser.add_argument("--frontend-port", type=int, default=4200,
                        help="Angular dev server port (default 4200)")
    parser.add_argument("--backend-only", action="store_true",
                        help="do not start the frontend")
    parser.add_argument("--frontend-only", action="store_true",
                        help="do not start the backend")
    parser.add_argument("--reload", action="store_true",
                        help="uvicorn --reload (backend autoreload)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser when both are up")
    parser.add_argument("--health-timeout", type=float, default=300.0,
                        help="seconds to wait for backend /health "
                             "(default 300; model load is slow on first run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved commands and exit")
    args = parser.parse_args()

    if args.dry_run:
        return dry_run(args)

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        if not args.frontend_only:
            procs.append(("backend", spawn(
                "backend", backend_command(args.backend_port, args.reload),
                BACKEND_DIR,
            )))
            if not wait_for_health(args.backend_port, args.health_timeout):
                # Non-fatal: uvicorn may simply be slow (first-run model
                # download). If the process actually died, the monitor loop
                # below tears everything down.
                print("[launcher] continuing without a healthy backend "
                      "check — watch [backend] logs", flush=True)
        if not args.backend_only:
            frontend_ready = threading.Event()
            procs.append(("frontend", spawn(
                "frontend", frontend_npm_command(args.frontend_port),
                FRONTEND_DIR,
                on_line=lambda line: (
                    frontend_ready.set() if _FRONTEND_READY_RE.search(line)
                    else None
                ),
            )))
            wait_for_port(args.frontend_port, timeout_s=240.0,
                          ready_event=frontend_ready)
            if not args.no_browser:
                webbrowser.open(f"http://localhost:{args.frontend_port}")
        print(f"[launcher] both up: http://localhost:{args.frontend_port} "
              f"(Ctrl+C to stop everything)", flush=True)
        # Block until a child dies; report which one and stop the other.
        while True:
            for tag, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[launcher] {tag} exited (code={code}); "
                          f"shutting everything down", flush=True)
                    return code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C: stopping...", flush=True)
        return 0
    finally:
        for _tag, proc in procs:
            kill_tree(proc)


if __name__ == "__main__":
    sys.exit(main())

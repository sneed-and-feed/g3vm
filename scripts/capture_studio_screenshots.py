#!/usr/bin/env python3
"""
scripts/capture_studio_screenshots.py — Capture 1920x1080 headless Chrome/Edge screenshots
of G3VM Studio v0.6.0 (`studio/g3vm_portable_studio.html`) and regenerate Matplotlib showcase figures.
"""

from __future__ import annotations

import http.server
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g3vm.viewer import (
    export_portable_html_studio,
    render_leapfrog_showcase_figure,
    render_well_log_beltrami_figure,
)


def find_browser() -> str:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Could not find Chrome or Edge executable.")


def main() -> int:
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    studio_dir = ROOT / "studio"
    studio_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export latest portable HTML5 Studio bundle & Matplotlib showcase figures
    export_portable_html_studio(studio_dir / "g3vm_portable_studio.html", grid_shape=(24, 24, 24))
    if "--skip-matplotlib" not in sys.argv:
        render_leapfrog_showcase_figure(out_dir / "g3vm_leapfrog_showcase.png", grid_shape=(32, 24, 24))
        render_well_log_beltrami_figure(out_dir / "g3vm_well_log_beltrami.png", grid_shape=(48, 48, 48))
        print("[+] Regenerated g3vm_leapfrog_showcase.png and g3vm_well_log_beltrami.png")

    # 2. Capture headless browser screenshots from local HTTP server
    browser = find_browser()
    print(f"[*] Using headless browser: {browser}")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def log_message(self, format: str, *args) -> None:
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), QuietHandler)
    port = httpd.server_address[1]
    srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv_thread.start()

    base_url = f"http://127.0.0.1:{port}/studio/g3vm_portable_studio.html"
    shots = [
        (
            "g3vm_studio_v060_lunar_isru.png",
            None,
            f"{base_url}?scenario=lunar_lava_tube_isru&field=g3vm_eps_r&well=6&sliceZ=13&sliceY=12&sliceX=12&zoom=1.08",
        ),
        (
            "g3vm_studio_v060_utah_forge.png",
            None,
            f"{base_url}?preset=utah_forge_egs_doublet&field=g3vm_vp_tti&well=6&sliceZ=16&sliceY=16&sliceX=16&zoom=1.08",
        ),
        (
            "g3vm_studio_v060_vp_tti.png",
            None,
            f"{base_url}?scenario=thrust_anticline&field=g3vm_vp_tti&well=6&sliceZ=11&sliceY=12&sliceX=12&zoom=1.08",
        ),
        (
            "g3vm_studio_v060_zp.png",
            None,
            f"{base_url}?scenario=subsalt_graben&field=g3vm_zp&well=7&sliceZ=11&sliceY=12&sliceX=12&zoom=1.08&chair=fence_only",
        ),
        (
            "g3vm_studio_v060_blind_mode_b.png",
            None,
            f"{base_url}?scenario=subsalt_graben&field=blind_gain&well=7&sliceZ=11&sliceY=12&sliceX=12&zoom=1.08&scoreMode=modeB&tab=tabBlind&chair=nw_upper&ve=1.2",
        ),
    ]

    import shutil
    for fname, legacy_fname, url in shots:
        dest = out_dir / fname
        prof = tempfile.mkdtemp(prefix="g3vm_chrome_")
        try:
            cmd = [
                browser,
                "--headless=new",
                f"--user-data-dir={prof}",
                "--disable-gpu",
                "--no-sandbox",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-crash-reporter",
                "--disable-breakpad",
                "--disable-background-networking",
                "--hide-scrollbars",
                "--window-size=1920,1080",
                "--virtual-time-budget=2500",
                f"--screenshot={dest}",
                url,
            ]
            subprocess.run(
                cmd,
                check=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if legacy_fname:
                shutil.copy2(dest, out_dir / legacy_fname)
            print(f"[+] Captured {dest.name} ({dest.stat().st_size:,} bytes)", flush=True)
        finally:
            shutil.rmtree(prof, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

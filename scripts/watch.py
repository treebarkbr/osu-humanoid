"""Real-time GUI viewer (browser/MJPEG).

Rolls the nets out on a map, judges the play, and streams it to your browser
in real time -- no native GUI toolkit, so it can't take down the session
(the OpenCV/Qt6 window did).

    .venv/bin/python scripts/watch.py --ckpt checkpoints_fix --maps 8
    # then open http://127.0.0.1:8760/  (auto-opened unless --no-open)

Controls in the page: pause/resume, next map, regenerate, speed.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import re
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.beatmap import parse_beatmap  # noqa: E402
from pipeline.hitwindows import approach_time, object_radius  # noqa: E402
from pipeline.humanize import from_profile  # noqa: E402
import scripts.live_sim as LS  # noqa: E402
import scripts.render as R  # noqa: E402
import scripts.rollout as RO  # noqa: E402

DT = 16.0
BOUND = "frame"


_FONT_PATHS = ("/usr/share/fonts/noto/NotoSans-Regular.ttf",
               "/usr/share/fonts/TTF/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/home/nocone/.local/share/fonts/SF-Pro.ttf")


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default(size)


_BM_DIR = Path(__file__).resolve().parent.parent / "data" / "beatmaps"
_UA = "osu-humanoid/0.1 (research project; local ML pipeline) by nocone"
_MIRROR = "https://mirror.hinamizawa.ai/api/osu/{bid}"
_OSU = "https://osu.ppy.sh/osu/{bid}"


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _set_difficulty(sid: str) -> int:
    page = _http_get(f"https://osu.ppy.sh/beatmapsets/{sid}").decode("utf-8", "ignore")
    m = re.search(r'id="json-beatmapset"[^>]*>(.*?)</script>', page, re.S)
    if not m:
        raise ValueError("beatmapset link has no difficulty id; paste a difficulty link")
    data = json.loads(html.unescape(m.group(1)))
    diffs = [b for b in data.get("beatmaps", []) if b.get("mode") == "osu"]
    diffs.sort(key=lambda b: b.get("difficulty_rating", 0))
    if not diffs:
        raise ValueError("no osu!standard difficulty in that beatmapset")
    return int(diffs[0]["id"])


def resolve_beatmap(text: str) -> int:
    text = text.strip()
    m = re.search(r"#osu/(\d+)", text) or re.search(r"/beatmaps/(\d+)", text) \
        or re.search(r"/b/(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"/beatmapsets/(\d+)", text)
    if m:
        return _set_difficulty(m.group(1))
    m = re.search(r"\d+", text)
    if not m:
        raise ValueError("no beatmap id found in that input")
    return int(m.group(0))


def fetch_beatmap(bid: int) -> Path:
    data = b""
    for url in (_MIRROR.format(bid=bid), _OSU.format(bid=bid)):
        try:
            data = _http_get(url)
            if b"osu file format" in data[:200]:
                break
        except Exception:
            data = b""
    if b"osu file format" not in data[:200]:
        raise ValueError(f"could not download beatmap {bid}")
    _BM_DIR.mkdir(parents=True, exist_ok=True)
    dest = _BM_DIR / f"{bid}.osu"
    dest.write_bytes(data)
    return dest


def build_play(map_path: Path, ckpt: str, profile: str, seed: int) -> dict:
    bm = parse_beatmap(map_path)
    aim, tap = RO._load(ckpt)
    sim = LS.PlaySim(bm, aim, tap, od=bm.od, ar=bm.ar,
                     rng=np.random.default_rng(seed), params=from_profile(profile))
    objs = sim.objs
    paths = [R._slider_pts(o) if o.is_slider and o.path is not None else None
             for o in objs]
    ticks = [R._slider_ticks(bm, o) if o.is_slider and o.path is not None else None
             for o in objs]
    return {
        "bm": bm,
        "sim": sim,
        "judge": LS.LiveJudge(bm),
        "map_path": Path(map_path),
        "objs": objs,
        "paths": paths,
        "ticks": ticks,
        "radius": object_radius(bm.cs),
        "ap": approach_time(bm.ar),
    }


class Player:
    """Holds the current play + playback state and renders frames."""

    def __init__(self, maps: list[Path], ckpts: list[str], profile: str, seed: int, scale: int):
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.maps = maps
        self.ckpts = ckpts
        self.ci = 0
        self.profile = profile
        self.seed = seed
        self.scale = scale
        self.i = 0
        self.paused = False
        self.speed = 1.0
        self.latest: bytes | None = None
        self._pending: str | None = None
        self._rebuild_until = 0.0
        self._load()
        self.playhead = float(self.play["sim"].t0)
        self.last = time.perf_counter()
        self._cache = None

    def _load(self) -> None:
        for _ in range(max(1, len(self.maps))):
            try:
                self.play = build_play(self.maps[self.i], self.ckpts[self.ci],
                                       self.profile, self.seed)
                break
            except Exception as e:
                print(f"load failed ({self.maps[self.i]}): {e}", flush=True)
                self.i = (self.i + 1) % len(self.maps)
        else:
            raise RuntimeError("no loadable beatmap")
        self.names = (f"[{Path(self.ckpts[self.ci]).name}] "
                      f"{self.play['bm'].title[:30]} [{self.play['bm'].version[:16]}]")
        self.playhead = float(self.play["sim"].t0)
        self.last = time.perf_counter()
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.counts = {"300": 0, "100": 0, "50": 0, "miss": 0}
        self.combo = 0
        self.best_combo = 0
        self._nres = 0
        self._errs = []
        self._hud = None
        self._hud_tick = 0

    # -- controls (called from HTTP threads) --------------------------------
    # next/regen only set a flag; the render thread performs (and coalesces)
    # the expensive rebuild, so spamming the buttons can't pile up work.
    def next_map(self) -> None:
        with self.lock:
            self._pending = "next"

    def regen(self) -> None:
        with self.lock:
            self._pending = "regen"

    def set_ckpt(self, name: str) -> None:
        with self.lock:
            for j, c in enumerate(self.ckpts):
                if Path(c).name == name or c == name:
                    if j != self.ci:
                        self.ci = j
                        self._pending = "ckpt"
                    return

    def play_path(self, path) -> None:
        with self.lock:
            p = Path(path)
            self.maps.append(p)
            self.i = len(self.maps) - 1
            self._pending = "reload"

    def toggle_pause(self) -> None:
        with self.lock:
            self.paused = not self.paused
            self.last = time.perf_counter()

    def set_speed(self, v: float) -> None:
        with self.lock:
            self.speed = max(0.1, min(4.0, v))

    def export_osr(self) -> tuple[str, bytes]:
        with self.lock:
            p = build_play(self.maps[self.i], self.ckpts[self.ci], self.profile, self.seed)
            sim = p["sim"]
            while not sim.done():
                sim.step()
            bm = p["bm"]
            out = Path("/tmp/opencode") / f"osu-humanoid_{bm.beatmap_id}.osr"
            RO.write_osr(bm, sim.finalize(), out, map_path=p["map_path"])
            name = f"{bm.artist} - {bm.title} [{bm.version}].osr"
            name = "".join(ch for ch in name if ch not in '/\\:*?"<>|').strip() or "play.osr"
            return name, out.read_bytes()

    def state(self) -> dict:
        with self.lock:
            sim = self.play["sim"]
            return {"map": self.names, "index": self.i, "playhead": self.playhead,
                    "t0": sim.t0, "t1": sim.t1, "paused": self.paused,
                    "speed": self.speed, "counts": self.counts, "obj_i": self.play["judge"].obj_i}

    def _apply_pending(self) -> None:
        if not self._pending:
            return
        now = time.perf_counter()
        if now < self._rebuild_until:
            return
        act = self._pending
        self._pending = None
        if act == "next":
            self.i = (self.i + 1) % len(self.maps)
        elif act == "regen":
            self.seed += 1
        self._load()
        self.last = now
        self._cache = None
        self._rebuild_until = now + 0.25

    # -- rendering ----------------------------------------------------------
    def _render_next(self) -> bytes:
        with self.lock:
            self._apply_pending()
            if self.paused and self._cache is not None:
                return self._cache
            now = time.perf_counter()
            dt_ms = min((now - self.last) * 1000.0, 120.0)
            self.last = now
            if not self.paused:
                self.playhead += dt_ms * self.speed

            play = self.play
            sim = play["sim"]
            if self.playhead >= sim.t1:
                self._pending = "next"
                self._apply_pending()
                play = self.play
                sim = play["sim"]

            sim.advance_to(self.playhead)
            x, y, held, idx = sim.sample(self.playhead)
            t = float(sim.times[idx])

            j = play["judge"]
            j.update(sim)
            res = j.results
            while self._nres < len(res):
                r = res[self._nres]
                if r is not None:
                    k = r.kind
                    self.counts[k] = self.counts.get(k, 0) + 1
                    if k == "miss":
                        self.combo = 0
                    else:
                        self.combo += 1
                        self.best_combo = max(self.best_combo, self.combo)
                    if r.hit_error is not None:
                        self._errs.append(r.hit_error)
                self._nres += 1

            bm = play["bm"]
            img = R.frame_image(bm, play["objs"], play["paths"], t,
                                x, y, held, play["radius"], play["ap"],
                                ticks=play["ticks"])

            errs = self._errs
            ur = 10.0 * float(np.std(errs)) if len(errs) >= 2 else 0.0
            total = sum(self.counts.values())
            acc = (300 * self.counts["300"] + 100 * self.counts["100"]
                   + 50 * self.counts["50"]) / max(1, total * 300)
            c = self.counts
            lines = [
                self.names,
                f"300 {c['300']}  100 {c['100']}  50 {c['50']}  miss {c['miss']}",
                f"combo {self.combo} (best {self.best_combo})  acc {acc * 100:5.1f}%  UR {ur:4.0f}",
                f"t {t / 1000:6.1f}s  speed {self.speed:.2f}x"
                f"{'  [PAUSED]' if self.paused else ''}",
            ]
            self._hud_tick += 1
            if self._hud is None or self._hud_tick % 6 == 0:
                ov = Image.new("RGBA", (R.RW, R.RH), (0, 0, 0, 0))
                ImageDraw.Draw(ov).multiline_text(
                    (8, 6), "\n".join(lines), fill=(235, 235, 245, 255),
                    font=_font(15), spacing=5)
                self._hud = ov
            img.paste(self._hud, (0, 0), self._hud)

            if self.scale != 1:
                img = img.resize((R.RW * self.scale, R.RH * self.scale), Image.NEAREST)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            jpg = buf.getvalue()
            self._cache = jpg
            return jpg

    def run(self) -> None:
        while True:
            try:
                jpg = self._render_next()
            except Exception as e:
                print(f"render error: {e}", flush=True)
                with self.lock:
                    self.i = (self.i + 1) % max(1, len(self.maps))
                    self._pending = "reload"
                time.sleep(0.3)
                continue
            with self.cond:
                self.latest = jpg
                self.cond.notify_all()
            time.sleep(0.004)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>osu-humanoid viewer</title>
<style>
 body{margin:0;background:#0b0b10;color:#e6e6ee;font:14px system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
 img{background:#000;border:1px solid #222;image-rendering:pixelated;max-width:96vw}
 .bar{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
 button{background:#1c1c28;color:#e6e6ee;border:1px solid #333;border-radius:8px;
        padding:8px 14px;font-size:14px;cursor:pointer}
 button:hover{background:#26263a}
 .upl{background:#1c1c28;border:1px solid #333;border-radius:8px;padding:8px 14px;cursor:pointer}
 .upl:hover{background:#26263a}
 input{background:#141420;color:#e6e6ee;border:1px solid #333;border-radius:8px;
       padding:8px 12px;font:14px system-ui,sans-serif}
 .s{opacity:.7}
</style></head><body>
<h3 style="margin:4px">osu-humanoid</h3>
<div class="bar" id="ckbar"></div>
<img src="/stream.mjpg">
<div class="bar">
 <button onclick="c('/api/pause')">pause/resume <span class=s>(space)</span></button>
 <button onclick="c('/api/next')">next map <span class=s>(n)</span></button>
 <button onclick="c('/api/regen')">regenerate <span class=s>(r)</span></button>
 <button onclick="spd(0.5)">0.5x</button>
 <button onclick="spd(1)">1x</button>
 <button onclick="spd(1.5)">1.5x</button>
 <button onclick="spd(2)">2x</button>
 <button onclick="location='/api/export'"
         title="downloads the full replay; drag it onto your osu!lazer window to watch">
  export .osr <span class=s>(e)</span></button>
</div>
<div class="bar">
 <input id="bm" placeholder="beatmap link or id  (e.g. osu.ppy.sh/beatmapsets/1#osu/75)"
        style="width:360px" onkeydown="if(event.key==='Enter')addMap()">
 <button onclick="addMap()">play beatmap</button>
 <label class="upl">upload .osu<input type="file" accept=".osu" style="display:none"
        onchange="upMap(this)"></label>
 <span id="msg" class="s"></span>
</div>
<script>
 function c(u){fetch(u)}
 function spd(x){fetch('/api/speed?x='+x)}
 function st(m){document.getElementById('msg').textContent=m}
 async function addMap(){
   const v=document.getElementById('bm').value.trim(); if(!v)return;
   st('loading '+v+' ...');
   try{const r=await fetch('/api/add',{method:'POST',body:v});
       const j=await r.json();
       st(j.ok?('playing '+j.name):('error: '+(j.error||r.status)));}
   catch(e){st('error: '+e)}
 }
 async function upMap(inp){
   const f=inp.files[0]; if(!f)return;
   st('uploading '+f.name+' ...');
   try{const t=await f.text();
       const r=await fetch('/api/upload',{method:'POST',
               headers:{'X-Filename':f.name},body:t});
       const j=await r.json();
       st(j.ok?('playing '+j.name):('error: '+(j.error||r.status)));}
   catch(e){st('error: '+e)}
 }
 const CKPTS = __CKPTS__;
 const cb = document.getElementById('ckbar');
 CKPTS.forEach((n,i)=>{
   const b=document.createElement('button');
   b.textContent=n; b.dataset.i=i;
   b.onclick=()=>{fetch('/api/ckpt?name='+encodeURIComponent(n));
                   [...cb.children].forEach(x=>x.style.borderColor='#333');
                   b.style.borderColor='#5af';};
   if(i===0)b.style.borderColor='#5af';
   cb.appendChild(b);
 });
 addEventListener('keydown',e=>{
   if(e.code==='Space'){c('/api/pause');e.preventDefault()}
   if(e.key==='n')c('/api/next');
   if(e.key==='r')c('/api/regen');
   if(e.key==='e')location='/api/export';
 });
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _ok(self):
        self._json({"ok": True})

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        p = urlparse(self.path).path
        body = self._read_body()
        if p == "/api/add":
            try:
                bid = resolve_beatmap(bm := body.decode("utf-8", "ignore").strip())
                dest = fetch_beatmap(bid)
                parse_beatmap(dest)
                PLAYER.play_path(dest)
                self._json({"ok": True, "id": bid, "name": str(dest)})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
        elif p == "/api/upload":
            try:
                text = body.decode("utf-8", "ignore")
                if "osu file format" not in text[:200]:
                    raise ValueError("that file is not an .osu beatmap")
                name = re.sub(r"[^A-Za-z0-9_. -]", "_",
                              self.headers.get("X-Filename", "upload.osu"))
                if not name.lower().endswith(".osu"):
                    name += ".osu"
                _BM_DIR.mkdir(parents=True, exist_ok=True)
                dest = _BM_DIR / name
                dest.write_text(text, encoding="utf-8")
                parse_beatmap(dest)
                PLAYER.play_path(dest)
                self._json({"ok": True, "name": str(dest)})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
        else:
            self.send_error(404)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/":
            names = [Path(c).name for c in PLAYER.ckpts]
            body = PAGE.replace("__CKPTS__", json.dumps(names)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/stream.mjpg":
            self._stream()
        elif p == "/api/state":
            body = json.dumps(PLAYER.state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/next":
            PLAYER.next_map(); self._ok()
        elif p == "/api/regen":
            PLAYER.regen(); self._ok()
        elif p == "/api/ckpt":
            q = parse_qs(u.query)
            PLAYER.set_ckpt(q.get("name", [""])[0]); self._ok()
        elif p == "/api/pause":
            PLAYER.toggle_pause(); self._ok()
        elif p == "/api/speed":
            q = parse_qs(u.query)
            PLAYER.set_speed(float(q.get("x", ["1"])[0])); self._ok()
        elif p == "/api/export":
            try:
                name, data = PLAYER.export_osr()
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUND}")
        self.end_headers()
        last = None
        try:
            while True:
                with PLAYER.cond:
                    while PLAYER.latest is last or PLAYER.latest is None:
                        PLAYER.cond.wait(1.0)
                    jpg = PLAYER.latest
                last = jpg
                self.wfile.write(f"--{BOUND}\r\nContent-Type: image/jpeg\r\n"
                                 f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


PLAYER: Player = None  # type: ignore


def main() -> int:
    global PLAYER
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    ap.add_argument("--maps", type=int, default=8)
    ap.add_argument("--ckpt", default="checkpoints",
                    help="checkpoint dir, or comma-separated list of dirs to switch between")
    ap.add_argument("--profile", default="average")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if args.map:
        maps = [Path(args.map)]
    else:
        maps = sorted(Path("data/beatmaps").glob("*.osu"))[: args.maps]
    if not maps:
        print("no maps found")
        return 1

    ckpts = [c.strip() for c in args.ckpt.split(",") if c.strip()]
    PLAYER = Player(maps, ckpts, args.profile, args.seed, args.scale)
    threading.Thread(target=PLAYER.run, daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {url}  (Ctrl-C to quit)", flush=True)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

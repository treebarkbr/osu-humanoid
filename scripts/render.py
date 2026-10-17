"""Render a generated (or real) play to MP4 / GIF so we can *watch* it.

    # render a freshly generated play
    .venv/bin/python scripts/render.py --map data/beatmaps/1932649.osu --out /tmp/play.mp4

    # or render an existing replay next to it for comparison
    .venv/bin/python scripts/render.py --replay data/replays/1932649/4787612321.osr \
        --map data/beatmaps/1932649.osu --out /tmp/human.mp4
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.beatmap import parse_beatmap, curve_position, slider_events  # noqa: E402
from pipeline.hitwindows import approach_time, object_radius  # noqa: E402
from pipeline.humanize import from_profile  # noqa: E402
from pipeline.judge import judge_replay  # noqa: E402
from pipeline.replay import ReplayFrames, load_replay  # noqa: E402
import scripts.rollout as R  # noqa: E402

PW, PH = 512, 384
MARGIN = 96
RW, RH = PW + 2 * MARGIN, PH + 2 * MARGIN
BG = (10, 10, 14)
FIELD = (20, 20, 30)
C_HIT = (235, 235, 240)
C_APP = (255, 90, 90)
C_SLIDER = (120, 190, 255)
C_TICK = (90, 150, 210)
C_BALL = (255, 230, 120)
C_CURSOR = (120, 255, 180)
C_DONE = (90, 90, 110)
C_TEXT = (205, 205, 220)


_FONT_PATHS = ("/usr/share/fonts/noto/NotoSans-Regular.ttf",
               "/usr/share/fonts/TTF/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/home/nocone/.local/share/fonts/SF-Pro.ttf")


@lru_cache(maxsize=32)
def _font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default(size)



def _slider_pts(obj, n=32):
    return [curve_position(obj, i / n) for i in range(n + 1)]


def _slider_ticks(bm, obj):
    if not obj.is_slider or obj.path is None:
        return []
    pts = []
    for ev in slider_events(bm, obj):
        if ev.type != "tick":
            continue
        prog = ev.path_progress
        if ev.span_index % 2 == 1:
            prog = 1.0 - prog
        p = curve_position(obj, prog)
        pts.append((p.x, p.y))
    return pts


def _ball_xy(obj, t):
    if obj.path is None or obj.span_duration <= 0:
        return obj.x, obj.y
    u = (t - obj.time) / obj.span_duration
    span = int(np.floor(u))
    local = u - span
    span = max(0, min(span, obj.span_count - 1))
    if span % 2 == 1:
        local = 1.0 - local
    local = min(max(local, 0.0), 1.0)
    p = obj.path.position_at(local)
    return obj.x + p.x, obj.y + p.y


def _base_image(bm):
    img = Image.new("RGB", (RW, RH), BG)
    dr = ImageDraw.Draw(img)
    dr.rectangle([MARGIN, MARGIN, MARGIN + PW, MARGIN + PH],
                 fill=FIELD, outline=(44, 44, 60), width=1)
    return img


_META_CACHE: dict = {}


def _meta_banner(bm):
    key = (bm.artist, bm.title, bm.version, bm.ar, bm.hp, bm.od, bm.cs)
    ov = _META_CACHE.get(key)
    if ov is None:
        ov = Image.new("RGBA", (RW, 42), (0, 0, 0, 0))
        ImageDraw.Draw(ov).multiline_text(
            (10, 2),
            f"{bm.artist} - {bm.title} [{bm.version}]\n"
            f"AR {bm.ar:.1f}   HP {bm.hp:.1f}   OD {bm.od:.1f}   CS {bm.cs:.1f}",
            fill=C_TEXT, font=_font(15), spacing=3)
        _META_CACHE[key] = ov
    return ov


def frame_image(bm, objs, paths, t, cx, cy, held, radius, ap, ticks=None):
    img = _base_image(bm)
    dr = ImageDraw.Draw(img)
    M = MARGIN
    if ticks is None:
        ticks = [_slider_ticks(bm, o) if (o.is_slider and o.path is not None) else None
                 for o in objs]

    for j, o in enumerate(objs):
        ot = float(o.time)
        oe = float(o.end_time)
        end = oe if oe > ot else ot
        if ot - ap > t or end + 80 < t:
            continue
        done = t > ot + 30
        col = C_DONE if done else C_HIT
        r = radius if not done else radius * 0.8

        if o.is_slider and paths[j] is not None:
            pts = [(p.x + M, p.y + M) for p in paths[j]]
            dr.line(pts, fill=(60, 90, 120) if done else C_SLIDER, width=3)
            if ticks[j]:
                for (tx, ty) in ticks[j]:
                    dr.ellipse([M + tx - 3, M + ty - 3, M + tx + 3, M + ty + 3], fill=C_TICK)

        dr.ellipse([M + o.x - r, M + o.y - r, M + o.x + r, M + o.y + r], outline=col, width=2)

        if not done and t >= ot - ap:
            prog = max(0.0, (ot - t) / ap)
            ar_ = radius * (1.0 + 3.5 * prog)
            dr.ellipse([M + o.x - ar_, M + o.y - ar_, M + o.x + ar_, M + o.y + ar_],
                       outline=C_APP, width=2)

        if o.is_slider and o.path is not None and ot - 20 <= t <= end + 20:
            bx, by = _ball_xy(o, t)
            dr.ellipse([M + bx - 6, M + by - 6, M + bx + 6, M + by + 6], fill=C_BALL)

        if o.is_spinner and t >= ot - ap:
            mx, my = M + PW / 2, M + PH / 2
            dr.ellipse([mx - 60, my - 60, mx + 60, my + 60], outline=C_APP, width=2)
            if t >= ot:
                ang = np.arctan2(cy - PH / 2, cx - PW / 2)
                px, py = mx + 52 * np.cos(ang), my + 50 * np.sin(ang)
                dr.line([mx, my, px, py], fill=C_BALL, width=2)
                dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=C_BALL)

    cr = 6.0
    if held:
        dr.ellipse([M + cx - cr - 4, M + cy - cr - 4, M + cx + cr + 4, M + cy + cr + 4],
                   outline=C_CURSOR, width=2)
    dr.ellipse([M + cx - cr, M + cy - cr, M + cx + cr, M + cy + cr], fill=C_CURSOR)

    fnt = _font(15)
    banner = _meta_banner(bm)
    img.paste(banner, (0, RH - 42), banner)
    return img


def render_from_frames(bm, times, xs, ys, keys, objs, out_path, *, fps=60, scale=2, gif=False):
    import cv2

    paths = [_slider_pts(o) if o.is_slider and o.path is not None else None for o in objs]
    radius = object_radius(bm.cs)
    ap = approach_time(bm.ar)

    writer = None
    if not gif:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (RW * scale, RH * scale))
    gif_frames = []
    ticks = [_slider_ticks(bm, o) if (o.is_slider and o.path is not None) else None
             for o in objs]

    for i in range(len(times)):
        held = bool(keys[i] & 0b1111)
        img = frame_image(bm, objs, paths, float(times[i]), float(xs[i]), float(ys[i]),
                          held, radius, ap, ticks=ticks)
        if scale != 1:
            img = img.resize((RW * scale, RH * scale), Image.NEAREST)
        arr = np.asarray(img)[:, :, ::-1]  # RGB->BGR
        if gif:
            gif_frames.append(img)
        else:
            writer.write(np.ascontiguousarray(arr))

    if gif:
        gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / fps), loop=0)
    else:
        writer.release()
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay", default=None, help="render a real .osr instead")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--profile", default="average")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--step", type=int, default=2, help="render every Nth 16ms frame")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bm = parse_beatmap(args.map)
    if args.replay:
        fr = load_replay(args.replay)
        objs = [o for o in bm.hit_objects if o.is_clickable or o.is_spinner]
        times, xs, ys, keys = fr.times, fr.x, fr.y, fr.keys
    else:
        aim, tap = R._load(args.ckpt)
        gen = R.generate(bm, aim, tap, od=bm.od, ar=bm.ar,
                         rng=np.random.default_rng(args.seed),
                         params=from_profile(args.profile))
        times, xs, ys, keys = gen["grid"], gen["x"], gen["y"], gen["keys"]
        objs = gen["objs"]
        rep = R.judge_generated(bm, gen)
        print("generated:", rep.summary())

    sl = slice(None, None, max(1, args.step))
    render_from_frames(bm, times[sl], xs[sl], ys[sl], keys[sl], objs, args.out,
                       fps=args.fps // max(1, args.step), scale=args.scale, gif=args.gif)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

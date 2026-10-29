"""Check that no text inside the axes touches a tick mark, and that no two texts overlap.
This runner patches figstyle.export so that, just before each figure is saved, the window extents of all texts drawn
in the axes (ax.texts, annotations and legend entries) are compared with the extents of the visible major and minor
tick marks on all four sides (tick length and direction from the tick objects). Tick labels and axis labels are
outside the axes and are not compared.
Usage: python check_tick_text.py <script.py> [script args ...]  (run from this directory)"""
import json, runpy, sys
from matplotlib.transforms import Bbox
import figstyle as S

ORIG = S.export
FOUND = []
TEXTS = []                                      # pairs of texts (any kind) whose extents overlap


def tick_boxes(ax, renderer):
    boxes = []
    for axis, name in ((ax.xaxis, 'x'), (ax.yaxis, 'y')):
        lo, hi = sorted(axis.get_view_interval())
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            loc = tick.get_loc()
            if loc is None or not (lo <= loc <= hi):
                continue
            size = renderer.points_to_pixels(tick._size)
            inward = tick._tickdir in ('in', 'inout')
            for line, side in ((tick.tick1line, 0), (tick.tick2line, 1)):
                if not line.get_visible():
                    continue
                if name == 'x':
                    px = ax.transData.transform((loc, 0))[0]
                    y_sp = ax.bbox.y0 if side == 0 else ax.bbox.y1
                    y_in = y_sp + (size if side == 0 else -size) if inward else y_sp
                    boxes.append((f'x={loc:g} {"bottom" if side == 0 else "top"}',
                                  Bbox.from_extents(px - 1, min(y_sp, y_in), px + 1, max(y_sp, y_in))))
                else:
                    py = ax.transData.transform((0, loc))[1]
                    x_sp = ax.bbox.x0 if side == 0 else ax.bbox.x1
                    x_in = x_sp + (size if side == 0 else -size) if inward else x_sp
                    boxes.append((f'y={loc:g} {"left" if side == 0 else "right"}',
                                  Bbox.from_extents(min(x_sp, x_in), py - 1, max(x_sp, x_in), py + 1)))
    return boxes


def all_texts(fig):
    out = []
    for ax in fig.axes:
        out += [ax.xaxis.label, ax.yaxis.label, *ax.texts]
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            out += [t.label1 for t in axis.get_major_ticks() if t.get_loc() is not None and lo <= t.get_loc() <= hi and t.label1.get_visible()]
        legs = [a for a in ax.artists if hasattr(a, 'get_texts')] + ([ax.get_legend()] if ax.get_legend() else [])
        out += [t for lg in legs for t in lg.get_texts()]
    return [t for t in out if t.get_visible() and t.get_text()]


def export(fig, path):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    ts = all_texts(fig)
    ext = [t.get_window_extent(r) for t in ts]
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if ext[i].overlaps(ext[j]):
                TEXTS.append({'figure': str(path), 'a': ts[i].get_text(), 'b': ts[j].get_text()})
    for ax in fig.axes:
        texts = [t for t in ax.texts if t.get_visible() and t.get_text()]
        legs = [a for a in ax.artists if hasattr(a, 'get_texts')] + ([ax.get_legend()] if ax.get_legend() else [])
        texts += [t for lg in legs for t in lg.get_texts()]
        for name, tb in tick_boxes(ax, r):
            for t in texts:
                if t.get_window_extent(r).overlaps(tb):
                    FOUND.append({'figure': str(path), 'tick': name, 'text': t.get_text()})
    return ORIG(fig, path)


S.export = export
script, args = sys.argv[1], sys.argv[2:]
sys.argv = [script] + args
try:
    runpy.run_path(script, run_name='__main__')
finally:
    print('TICK-TEXT', json.dumps({'script': script, 'overlaps': FOUND}, ensure_ascii=False))
    print('TEXT-TEXT', json.dumps({'script': script, 'overlaps': TEXTS}, ensure_ascii=False))

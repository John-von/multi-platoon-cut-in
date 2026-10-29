"""Common style of the data figures (Figs. 4-10).
* One font format in all figures: text in TeX Gyre Termes (the text font of the manuscript), mathematics in
  Computer Modern (the math font of the manuscript: cmmi10 letters, cmr10 digits); every text element 8 pt (tick
  labels, axis labels, legends, annotations), panel letters 8 pt bold; no other bold or italic text.
* The three cut-in vehicles are told apart by colour and line style together (c1 orange solid, c2 red dashed,
  c3 purple dotted) in every figure that shows them.
* Four-sided frame with inward ticks, panel letters inside the axes, fixed canvas size, and export checks.
The manuscript schematics (Figs. 1-3) use the same fonts and sizes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

# TeX Gyre Termes with TrueType outlines (made by otf2ttf.py from the OpenType files of the TeX distribution), so that
# matplotlib embeds clean Type 42 fonts; the OpenType files themselves are not registered.
GYRE = Path(__file__).resolve().parent / 'fonts'
_fm = font_manager.fontManager
_fm.ttflist = [f for f in _fm.ttflist if not ('termes' in f.fname.lower() and f.fname.lower().endswith('.otf'))]
for f in ('texgyretermes-regular.ttf', 'texgyretermes-bold.ttf', 'texgyretermes-italic.ttf', 'texgyretermes-bolditalic.ttf'):
    _fm.addfont(str(GYRE / f))
font_manager.FontManager._findfont_cached.cache_clear()
FONT = 'TeX Gyre Termes'
assert font_manager.findfont(FONT, fallback_to_default=False).endswith('texgyretermes-regular.ttf')
FS = 8                                   # the single figure font size (pt)

SINGLE_IN = 252 / 72.27                  # IEEEtran column width (TeX pt)
DOUBLE_IN = 516 / 72.27                  # IEEEtran text width
METHOD = {'A-H': '#0077BB', 'A-S': '#33BBEE', 'F': '#EE7733', 'D-DRPC': '#009988', 'N': '#CC3311', 'P': '#EE3377',
          'B1': '#AA3377', 'B2': '#555555'}
QUEUE = ['#4477AA', '#66CCEE', '#228833']
QUEUE_LS = ['-', (0, (4, 2)), (0, (1, 1.5))]
CUT = ['#EE7733', '#CC3311', '#AA3377']  # c1, c2, c3: colour ...
CUT_LS = ['-', (0, (4, 2)), (0, (1, 1.5))]  # ... and line style
CUT_COLOR = CUT[0]                       # a single cut-in (c1 only) keeps the c1 colour
FRAME = '#30343B'
GRID = '#E3E7EC'
REF = '#4D4D4D'


def apply():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [FONT],
        # mathematics as in the manuscript (Computer Modern): letters cmmi10, upright symbols cmr10, operators cmsy10;
        # digits, '=' and units in math are written with \\mathrm{} so that they come from cmr10 as in LaTeX
        'mathtext.fontset': 'custom', 'mathtext.rm': 'cmr10', 'mathtext.it': 'cmmi10', 'mathtext.bf': 'cmb10',
        'mathtext.cal': 'cmsy10', 'mathtext.sf': 'cmss10', 'mathtext.tt': 'cmtt10', 'mathtext.default': 'it',
        'mathtext.fallback': 'cm',
        'font.size': FS, 'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
        'axes.titlesize': FS, 'figure.titlesize': FS,
        'axes.linewidth': .65, 'axes.edgecolor': FRAME, 'lines.linewidth': 1.1,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.minor.visible': False, 'ytick.minor.visible': False, 'xtick.color': FRAME, 'ytick.color': FRAME,
        'legend.frameon': False, 'legend.handlelength': 1.9, 'legend.handletextpad': .4, 'legend.labelspacing': .25,
        'legend.columnspacing': .9, 'legend.borderaxespad': .3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': True,
    })


def axes(ax, letter=None, pos=(.025, .965), ha='left', va='top', grid=True, ticks=True):
    """Four-sided frame, inward ticks on all sides (labels bottom/left), letter inside the axes."""
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(.65); s.set_edgecolor(FRAME)
    if ticks:
        ax.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True,
                       labeltop=False, labelright=False, labelbottom=True, labelleft=True, pad=2.5)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis='y', color=GRID, linewidth=.35)
    if letter:
        ax.text(pos[0], pos[1], f'({letter})', transform=ax.transAxes, ha=ha, va=va, fontsize=FS, fontweight='bold',
                zorder=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .3, 'alpha': .85})


def export(fig, path):
    """Fixed physical canvas (no tight cropping); report texts that leave the canvas and font sizes in use."""
    path = Path(path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, sizes = [], set()
    for ax in fig.axes:
        ticks = []
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            ticks.extend(t.label1 for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi)
        legend = ax.get_legend()
        legs = [a for a in ax.artists if hasattr(a, 'get_texts')] + ([legend] if legend else [])
        texts = [ax.xaxis.label, ax.yaxis.label, *ax.texts, *ticks, *[t for lg in legs for t in lg.get_texts()]]
        for a in texts:
            if not a.get_visible() or not a.get_text():
                continue
            sizes.add(round(a.get_fontsize(), 2))
            b = a.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1 + 1 or b.y1 > fig.bbox.y1 + 1:
                outside.append(a.get_text())
    for lg in fig.legends:
        sizes.update(round(t.get_fontsize(), 2) for t in lg.get_texts())
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=300)
    with Image.open(path.with_suffix('.png')) as im:
        im.convert('L').save(path.with_name(path.name + '_gray').with_suffix('.png'))
    size = [float(x) for x in fig.get_size_inches()]
    plt.close(fig)
    return {'name': path.name, 'size_in': size, 'texts_outside_canvas': outside, 'font_sizes_pt': sorted(sizes),
            'png_dpi': 300, 'cropping': 'fixed canvas; no bbox_inches=tight'}


def overlaps(fig, pairs):
    """Return the pairs of artists whose window extents overlap (for letter/legend/label checks)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for a, b in pairs:
        if a.get_window_extent(r).overlaps(b.get_window_extent(r)):
            bad.append((getattr(a, 'get_text', lambda: str(a))(), getattr(b, 'get_text', lambda: str(b))()))
    return bad


def bound(ax, y, label, x=.975, above=True, color=REF):
    """Dashed threshold line; label (if any) at the font size of all figure text."""
    ax.axhline(y, color=color, lw=.6, ls=(0, (4, 3)), zorder=1.5)
    if label:
        ax.annotate(label, xy=(x, y), xycoords=('axes fraction', 'data'), xytext=(0, 2 if above else -2),
                    textcoords='offset points', ha='right', va='bottom' if above else 'top', color='black')

"""Convert the TeX Gyre Termes OpenType (CFF) fonts to TrueType outlines for the figures.
matplotlib embeds OpenType-CFF fonts in a way that some PDF readers report as a mismatch between font type and
embedded font file; TrueType outlines are embedded as clean Type 42 fonts. The glyph shapes are the same font (TeX Gyre
Termes, the text font of the manuscript); cubic curves are approximated by quadratic ones (cu2qu, max error 1 unit).
Usage: python otf2ttf.py <out_dir> [<dir with texgyretermes-*.otf>]
Without the second argument the OpenType files are looked up with kpsewhich (TeX Live or MiKTeX), then in the
Debian TeX path. The converted files are used only to draw the figures of this manuscript."""
import shutil, subprocess, sys
from pathlib import Path
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen



def source_dir():
    if len(sys.argv) > 2:
        return Path(sys.argv[2])
    if shutil.which('kpsewhich'):
        hit = subprocess.run(['kpsewhich', 'texgyretermes-regular.otf'], capture_output=True, text=True).stdout.strip()
        if hit:
            return Path(hit).parent
    return Path('/usr/share/texmf/fonts/opentype/public/tex-gyre')


SRC = source_dir()
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
MAX_ERR = 1.0                                   # font units (1000 per em)


def convert(src, dst):
    font = TTFont(str(src))
    assert font.sfntVersion == 'OTTO'
    gs = font.getGlyphSet()
    glyf = newTable('glyf'); glyf.glyphOrder = font.getGlyphOrder(); glyf.glyphs = {}
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, MAX_ERR, reverse_direction=True))
        glyf[name] = pen.glyph()
    font['loca'] = newTable('loca'); font['glyf'] = glyf
    shapes = {}
    del font['CFF ']
    if 'VORG' in font:
        del font['VORG']
    font['maxp'] = maxp = newTable('maxp'); maxp.tableVersion = 0x00010000
    for k in ('maxZones', 'maxTwilightPoints', 'maxStorage', 'maxFunctionDefs', 'maxInstructionDefs', 'maxStackElements',
              'maxSizeOfInstructions', 'maxComponentElements'):
        setattr(maxp, k, 0)
    maxp.maxZones = 1
    maxp.maxComponentElements = max((len(g.components if hasattr(g, 'components') else []) for g in glyf.glyphs.values()),
                                    default=0)
    font['post'].formatType = 2.0; font['post'].extraNames = []; font['post'].mapping = {}
    font['post'].glyphOrder = font.getGlyphOrder()
    font['head'].glyphDataFormat = 0
    font.sfntVersion = '\x00\x01\x00\x00'
    font.save(str(dst))
    TTFont(str(dst))['glyf']                  # reload check
    return dst


for f in ('texgyretermes-regular', 'texgyretermes-bold', 'texgyretermes-italic', 'texgyretermes-bolditalic'):
    print(convert(SRC / f'{f}.otf', OUT / f'{f}.ttf'))

"""Load the frozen scenario() verbatim from ../sim/sources/simulation.py without importing
that module (it needs CVXOPT). The function source is extracted by AST and executed
in a namespace providing ROOT, loadmat and numpy only; the file hash is checked."""
import ast, hashlib, os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
# V20_SRC: directory holding the frozen simulation.py (package: code/sim/sources)
# V20_ROOT: directory holding baseline/stage1_plan.mat (package: code/sim); default parent of V20_SRC,
#           which is exactly the frozen code's own ROOT = Path(robust_core.__file__).parents[1].
FROZEN = Path(os.environ.get('V20_SRC', str(Path(__file__).resolve().parents[1] / 'sim' / 'sources')))
ROOT = Path(os.environ.get('V20_ROOT', str(FROZEN.parent)))
SIM_SHA = '97e0b11140e42bc65c787806046fcaab4adce565392beecc579029d4b688f69e'
MAT_SHA = '3d66d27ec0939a44c9371267b0a51f9bf7524f44db3cb164c8938be680690fa7'

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_scenario():
    src = FROZEN / 'simulation.py'
    assert _sha(src) == SIM_SHA, 'frozen simulation.py hash mismatch'
    assert _sha(ROOT / 'baseline/stage1_plan.mat') == MAT_SHA, 'stage1_plan.mat hash mismatch'
    tree = ast.parse(src.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scenario')
    ns = {'np': np, 'loadmat': loadmat, 'ROOT': ROOT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ns)
    return ns['scenario']

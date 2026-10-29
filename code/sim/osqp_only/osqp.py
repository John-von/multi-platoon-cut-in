"""Import-only OSQP stand-in. No numerical solver is implemented here.

The registered runner requires real CVXOPT 1.3.2 and asserts that the
selected lateral solver is CVXOPT. lateral_fast imports osqp at import
time; only that import is served by this module. Any OSQP call fails.
"""
STANDIN = True
__version__ = 'import-only stand-in'

class OSQP:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('OSQP is import-only in this registered CVXOPT run')

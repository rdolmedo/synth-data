"""Answer types (fitted to the CritPt templates' class-level mix) and their code-template conventions.

The skeletons are written for this pipeline from the conventions of the CritPt template format (docstring with
Inputs/Outputs, a FILL IN block with `...` placeholders, sympy symbols declared at module level for symbolic
inputs). None is copied from a test item.
"""

ANSWER_TYPES = {
    'numeric_scalar': {
        'desc': 'a single number (float or integer) computed for fixed parameter values; the function takes no '
                'arguments. State the required precision (e.g. "to three decimal places" or "to 4 significant digits") '
                'in the main problem when the value is not exact.',
        'skeleton': '''def answer():
    r"""
    Return <one-line description>.

    Inputs
    ----------
    None

    Outputs
    ----------
    <name>: float, <description with LaTeX symbol>
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    <name> = ...
    # ---------------------------------------------------------------

    return <name>
'''},
    'numeric_multi': {
        'desc': 'several numbers for fixed parameter values, returned as a list/tuple in a specified order or as '
                'several return values (e.g. a set of coefficients, the lowest few eigenvalues, a value at several '
                'parameter points); the function takes no arguments. Specify order and precision.',
        'skeleton': '''def answer():
    r"""
    Return <one-line description>.

    Inputs
    ----------
    None

    Outputs
    ----------
    <name>: list[float], <description, including the order of the entries>
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    <name> = ...
    # ---------------------------------------------------------------

    return <name>
'''},
    'discrete': {
        'desc': 'a discrete or categorical result: an integer count or invariant, a tuple of integers (e.g. Chern '
                'numbers, quantum numbers), a set of allowed labels or selection-rule tuples, a yes/no classification, '
                'possibly mixed with a few numbers; the function takes no arguments. Spell out the exact allowed '
                'strings/encodings of every label in the docstring.',
        'skeleton': '''def answer():
    r"""
    Return <one-line description>.

    Inputs
    ----------
    None

    Outputs
    ----------
    <name>: set[tuple[int, int]], <description; exact encoding of each entry>
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    <name> = ...
    # ---------------------------------------------------------------

    return <name>
'''},
    'symbolic_expr': {
        'desc': 'one closed-form SymPy expression in terms of the problem\'s symbols, which are the function\'s '
                'arguments (declared with sp.symbols at module level, with assumptions such as positive=True where '
                'the setup implies them). State any order of expansion or limit explicitly.',
        'skeleton': '''import sympy as sp

<a>, <b> = sp.symbols('<a> <b>', positive=True)

def answer(<a>, <b>):
    r"""
    Return <one-line description>.

    Inputs
    ----------
    <a>: sympy.Symbol, <meaning>, $<latex>$
    <b>: sympy.Symbol, <meaning>, $<latex>$

    Outputs
    ----------
    <name>: sympy.Expr, <description>
    """

    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    <name> = ...   # a SymPy expression of inputs
    # ---------------------------------------------------------------

    return <name>
'''},
    'symbolic_multi': {
        'desc': 'two to five closely related SymPy expressions (e.g. several coefficients, both branches of a '
                'dispersion relation, a set of allowed expressions) in terms of the problem\'s symbols, which are the '
                'function\'s arguments (declared with sp.symbols at module level). Returned as a tuple in a specified '
                'order, or as a set when order is meaningless.',
        'skeleton': '''import sympy as sp

<a>, <b> = sp.symbols('<a> <b>')

def answer(<a>, <b>):
    r"""
    Return <one-line description>.

    Inputs
    ----------
    <a>: sympy.Symbol, <meaning>, $<latex>$
    <b>: sympy.Symbol, <meaning>, $<latex>$

    Outputs
    ----------
    <name1>: sympy.Expr, <description>
    <name2>: sympy.Expr, <description>
    """

    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    <name1> = ...   # a SymPy expression of inputs
    <name2> = ...   # a SymPy expression of inputs
    # ---------------------------------------------------------------

    return <name1>, <name2>
'''},
    'numeric_function': {
        'desc': 'a numerical function of numeric inputs (floats), returning a float (or a bool for a criterion); '
                'the solver implements the closed form or procedure with numpy (and scipy.constants if physical '
                'constants are needed).',
        'skeleton': '''import numpy as np

def answer(<x>, <y>):
    r"""
    Return <one-line description>.

    Inputs
    ----------
    <x>: float, <meaning and units>
    <y>: float, <meaning and units>

    Outputs
    ----------
    <name>: float, <description and units>
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    <name> = ...
    # ---------------------------------------------------------------

    return <name>
'''},
}

ALLOWED_IMPORTS = {'sympy', 'numpy', 'scipy.constants', 'math', 'cmath', 'sympy.physics.quantum',
                   'sympy.physics.units', 'sympy.physics.quantum.constants', 'sympy.physics.wigner',
                   'sympy.physics.matrices', 'scipy', 'fractions'}
ARGS_TYPES = {'symbolic_expr', 'symbolic_multi', 'numeric_function'}

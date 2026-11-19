"""Generator and problem-check prompts (DeepSeek). Only class-level CritPt statistics appear here; no test item.

The style example is the CritPt authors' public example challenge (`quantum_error_correction_main`, not one of the
70 test challenges), loaded from the pinned authors' repo at run time.
"""
from __future__ import annotations

from .templates import ANSWER_TYPES

WORK_MODES = {
    'analytic derivation': 'the answer follows from a pen-and-paper derivation (the sandbox is only for checking '
                           'algebra)',
    'analytic + light numerics': 'a derivation that ends in a light numerical step: a root, an integral, a small '
                                 'matrix diagonalization, a series sum, evaluated in Python at stated parameter values',
    'heavy numerical computation': 'the solver must set up and run a nontrivial numerical computation (exact '
                                   'diagonalization of a modest lattice, transfer matrix, ODE/PDE integration, '
                                   'Monte Carlo or optimization) that finishes in minutes on a laptop',
    'combinatorial/enumeration': 'the answer is obtained by counting or enumerating discrete structures '
                                 '(configurations, states, group elements, diagrams, allowed transitions)',
    'symbolic computation': 'a lengthy but mechanical symbolic calculation (series expansion, tensor algebra, '
                            'commutators) best done with SymPy',
}

GROUNDING = {
    'intermediate': 'an intermediate or derived result of the paper: a quantity the paper obtains along the way (a '
                    'coefficient, a limit, an effective parameter, a special case evaluated explicitly, a step of '
                    'its main derivation), not necessarily its headline result',
    'main': 'one of the paper\'s main results, restricted to a concrete, self-contained special case so that it can '
            'be derived in a focused session',
    'extension': 'a direct variant of a result derived in the paper (a different parameter value, dimension, '
                 'boundary condition, or a next order the paper\'s method gives straightforwardly); use it only '
                 'when you can derive the variant yourself with certainty from the paper\'s method',
}

GEN = """You are writing ONE research-level theoretical physics challenge for a training set modelled on the CritPt
benchmark ("Complex Research using Integrated Thinking - Physics Test"). CritPt challenges are written by physicists
from their own research: a self-contained setup that defines a model or system, and a main problem asking for one
precise result, to be returned by filling in a Python code template. Solvers are strong physicists (or AI agents)
with an offline Python sandbox (numpy, scipy, sympy, mpmath, matplotlib) but no access to any paper.

Your challenge must be grounded in the arXiv paper given below (LaTeX source, preamble stripped). Read it carefully.

## Assignment
- Field: {field}
- Subtopic: {subtopic} ({subtopic_scope})
- Technique the solution should mainly use: {technique}
- Work mode: {work_mode}: {work_mode_desc}
- Grounding: {grounding_desc}
- Answer type: {answer_type}: {answer_type_desc}

The paper was retrieved for this subtopic and may fit it loosely. Stay within the field and the answer type. Use the
subtopic and technique as guidance: pick the result of the paper that best matches them; if the paper does not offer a
fitting result, follow the paper's own technique instead. If the paper offers no result that can become a
well-posed, self-contained challenge of this answer type (e.g. it is purely experimental, a review without
derivations, or its results are only qualitative or only in figures), output a skip (see Output).

## Requirements
1. Grounded and correct: the target quantity is derived or computed in the paper (or, for a variant, follows with
   certainty from the paper's method). Never invent results. The problem must have a unique correct answer that
   follows from the setup alone. Check every definition, sign, factor of 2 and normalization against the paper.
2. Self-contained: define every symbol, model, convention, normalization, unit system and sign convention the solver
   needs. Standard notation may be used without definition only when it is unambiguous. The solver never sees the
   paper: never mention "the paper", "the authors", figures, equation or section numbers, or references.
3. The answer must not be readable in the problem: state the model and definitions, never the result, and do not
   spell out the key intermediate steps that would turn the problem into a lookup.
4. One precise target. Specify exactly what to return: which variable, which limit or order of expansion, which
   normalization, which units, and for numerical answers the precision required (e.g. "to three decimal places").
   For several outputs, specify their order.
5. Research-level difficulty: CritPt challenges are hard even for strong researchers: answering takes hours of
   real work, combining several nontrivial ideas (choosing the right formalism, a non-obvious transformation or
   symmetry argument, a careful limit or expansion, a numerical computation the solver has to design and converge).
   Build the challenge on one of the paper's own nontrivial derived results, or a variant that requires redoing the
   paper's key derivation under changed conditions. Not acceptable: textbook exercises, standard closed forms (known
   spectra, standard integrals, classic bound states), quantities that follow from one substitution into a formula
   given in the setup, or setups whose definitions already do most of the work. Before writing, ask yourself: could a
   strong graduate student answer this in fifteen minutes from standard knowledge? If so, pick a deeper result of
   the paper, or skip. Still keep it finishable: one precise target reachable in one long working session, not a
   whole paper's research program. The example challenge below shows the expected depth.
6. Length and shape: the description has a "# Problem setup:" part of about 80 to 250 words (definitions, model,
   conventions) and a "# Main problem:" part of about 30 to 150 words (the question). Use LaTeX ($...$, \\begin{{equation}}).
   Do not copy sentences from the paper; write it in your own words as a research physicist would pose it.
7. Code template: follow the skeleton for the answer type below. Fill in the description, names and types; keep the
   docstring layout (Inputs / Outputs with "----------" rules), the FILL IN block with `...` placeholders and the
   return statement. Every function argument must be a symbol defined in the problem; for symbolic answers declare
   them at module level with sp.symbols (add assumptions such as positive=True when the setup implies them). Import
   only what the answer needs (sympy as sp, numpy as np, or names from scipy.constants); no answer, no hints, no
   helper code in the template.

Skeleton for this answer type:
```python
{skeleton}
```

## Format example
Here is a public example challenge from the benchmark's authors, to show the FORMAT only. Its topic, wording and
structure must not be reused: your challenge comes from your paper and your assignment.

<example>
{example_description}

Code template:
```python
{example_template}
```
</example>

## Output
Write your answer in exactly this form (no reference answer anywhere). In the template, every line of the FILL IN
block stays `name = ...` with a literal `...`: never write the value, an expression for it, or a hint in a comment
there, even if you derived it while reading the paper.

<status>OK</status>
<source>where in the paper the result comes from (section / equation), for our records; one line</source>
<target>the requested quantity in words (not its value); one line</target>
<problem>
# Problem setup:
...

# Main problem:
...
</problem>
<template>
```python
...
```
</template>

or, if the paper cannot support a fitting challenge:

<status>SKIP: one-line reason</status>

## The paper (arXiv {arxiv_id}, "{title}")
<paper>
{paper}
</paper>
"""

CHECK = """You are checking a research-level physics challenge written for a training set. The solver sees only the
problem and the code template below, works in an offline Python sandbox, and fills in the template.

<problem>
{problem}
</problem>
<template>
```python
{template}
```
</template>

Check ONLY these well-posedness and format properties (not difficulty, not whether you can solve it):
1. defined: every symbol, parameter, convention and normalization needed is defined in the problem or is standard
   unambiguous notation.
2. self_contained: it does not refer to an external paper, figure, equation number, reference or "the authors".
3. no_leak: the problem statement does not state the requested answer itself (giving the model and definitions is
   fine).
4. single_target: it asks for a precisely specified result (with precision for numerical values, order for several
   outputs), and the template's outputs match what the main problem asks for.
5. template_ok: the template's function arguments are exactly the symbols/inputs the answer depends on, all defined in
   the problem, and the docstring's outputs say what to return.

Return ONLY a JSON object: {{"defined": true|false, "self_contained": true|false, "no_leak": true|false,
"single_target": true|false, "template_ok": true|false, "problems": "one short sentence per failed item, else empty"}}
"""


def gen_prompt(leaf: dict, paper: dict, paper_text: str, example) -> str:
    at = ANSWER_TYPES[leaf['answer_type']]
    return GEN.format(field=leaf['field'], subtopic=leaf['subtopic'], subtopic_scope=leaf['subtopic_scope'],
                      technique=leaf['technique'], work_mode=leaf['work_mode'],
                      work_mode_desc=WORK_MODES[leaf['work_mode']], grounding_desc=GROUNDING[leaf['grounding']],
                      answer_type=leaf['answer_type'], answer_type_desc=at['desc'], skeleton=at['skeleton'].strip(),
                      example_description=example.description.strip(), example_template=example.code_template.strip(),
                      arxiv_id=paper['arxiv_id'], title=(paper.get('title') or '').replace('\n', ' ').strip(),
                      paper=paper_text)


def check_prompt(problem: str, template: str) -> str:
    return CHECK.format(problem=problem, template=template)

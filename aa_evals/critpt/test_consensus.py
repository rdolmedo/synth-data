"""Unit tests of the answer equivalence on synthetic fingerprints (no CritPt data):
/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python -m unittest aa_evals.critpt.test_consensus
(the BraKet cases run the probe and need sympy; they are skipped without it)

Regression for audit 2 (#32): a fixed absolute tolerance of 1e-12 made any two SI-scale answers equal, and
expressions whose terms underflow to 0 at every probe point matched each other."""
import subprocess
import sys
import unittest

from aa_evals.critpt.consensus import ATOL, RTOL, _eq, cluster, comparable, equivalent, problem_rules


def num(x, im=0.0):
    return {'t': 'num', 'v': [x, im]}


def seq(*xs):
    return {'t': 'seq', 'v': [num(x) for x in xs]}


def expr(*xs):
    return {'t': 'expr', 'v': [None if x is None else [x, 0.0] for x in xs]}


class TinyMagnitudes(unittest.TestCase):
    def test_no_default_absolute_tolerance(self):
        self.assertEqual(ATOL, 0.0)
        self.assertEqual(problem_rules('nothing stated')['atol'], 0.0)

    def test_different_tiny_numbers_differ(self):
        for a, b in [(3.0e-21, 2.0e-22), (4.0e-67, 1.5e-27), (1.0e-13, 1.0e-30), (2.0e-40, -2.0e-40)]:
            self.assertFalse(equivalent(num(a), num(b)), (a, b))

    def test_equal_tiny_numbers_match(self):
        self.assertTrue(equivalent(num(3.0e-21), num(3.0e-21 * (1 + RTOL / 10))))
        self.assertTrue(equivalent(num(0.0), num(0.0)))

    def test_tiny_sequences_elementwise(self):
        self.assertFalse(equivalent(seq(3.0e-23, 5.0e-24), seq(3.0e-23, 4.4e-24)))
        self.assertTrue(equivalent(seq(3.0e-23, 5.0e-24), seq(3.0e-23, 5.0e-24)))

    def test_significant_figures_stay_relative(self):
        rules = problem_rules('to two significant figures')
        self.assertEqual(rules['atol'], 0.0)
        self.assertTrue(_eq(num(9.3e-23), num(9.28e-23), rules, RTOL))
        self.assertFalse(_eq(num(9.3e-23), num(5.2e-23), rules, RTOL))

    def test_stated_decimal_places_still_give_an_absolute_tolerance(self):
        rules = problem_rules('to three decimal places')
        self.assertEqual(rules['atol'], 1e-3)
        self.assertTrue(_eq(num(0.0), num(0.0004), rules, RTOL))
        self.assertFalse(_eq(num(0.0), num(0.071), rules, RTOL))

    def test_distinct_tiny_answers_form_distinct_classes(self):
        fps = [seq(1.3e-27, 1.6e-26), seq(1.2e-13, 1.5e-12), seq(1.6e-27, 2.0e-26), seq(9.7e-69, 1.2e-67)]
        self.assertEqual(cluster(fps), [0, 1, 2, 3])
        self.assertEqual(cluster([num(2.0e-22), num(2.0e-22), num(7.0e-23)]), [0, 0, 1])


class Underflow(unittest.TestCase):
    def test_all_zero_expressions_do_not_match(self):
        zeros = expr(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertTrue(comparable(zeros))
        self.assertFalse(equivalent(zeros, expr(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))

    def test_zero_against_tiny_nonzero_differs(self):
        self.assertFalse(equivalent(expr(0.0, 0.0, 0.0, 0.0), expr(1e-40, 3e-41, 2e-45, 5e-44)))

    def test_one_informative_point_is_not_enough(self):
        self.assertFalse(equivalent(expr(0.0, 0.0, 2.5e-30, 0.0), expr(0.0, 0.0, 2.5e-30, 0.0)))

    def test_two_informative_points_suffice(self):
        self.assertTrue(equivalent(expr(0.0, 1.5e-30, 2.5e-30, 0.0), expr(0.0, 1.5e-30, 2.5e-30, 0.0)))
        self.assertFalse(equivalent(expr(0.0, 1.5e-30, 2.5e-30, 0.0), expr(0.0, 1.5e-30, 2.4e-30, 0.0)))

    def test_underflowing_component_blocks_a_composite_match(self):
        # an answer whose first component underflows everywhere cannot be verified, even if the rest agrees
        a = {'t': 'seq', 'v': [expr(0.0, 0.0, 0.0, 0.0), expr(1.0e-17, 7.0e-17, 1.3e-16, 1.8e-16)]}
        self.assertFalse(equivalent(a, a))

    def test_ordinary_expressions_unaffected(self):
        self.assertTrue(equivalent(expr(1.0, 2.5, None, 3.7), expr(1.0, 2.5, 4.0, 3.7)))
        self.assertFalse(equivalent(expr(1.0, 2.5, 3.7), expr(1.0, 2.5, 3.8)))


try:
    import sympy  # noqa: F401
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False

BRAKET_HEAD = """import sympy as sp
from sympy.physics.quantum import Ket, Bra, Dagger
phi = Ket('phi'); psi = Ket('psi'); phi_star = Ket('phi*'); psi_star = Ket('psi*')
d_P, d_B = sp.symbols('d_P d_B')
d = d_P * d_B
def answer(phi, psi, phi_star, psi_star, d_P, d_B, d):
    expr = %s
    return expr
"""


def braket_fp(expr_code: str) -> dict:
    from aa_evals.critpt.consensus import probe
    run_py = lambda prog, timeout, stdin=None: subprocess.run(  # noqa: E731
        [sys.executable, '-I', '-c', prog], capture_output=True, text=True, timeout=timeout, input=stdin)
    return probe(BRAKET_HEAD % expr_code, run_py)['fp']


@unittest.skipUnless(HAVE_SYMPY, 'needs sympy (run with the critpt venv)')
class BraKet(unittest.TestCase):
    """Probe v3 (#185): Challenge_30's bra-ket answers were compared as expression trees, so a factor distributed over
    a sum did not match (qwen-full-xhigh, audit 2 minor f)."""
    FACTORED = '(d_P**2*d_B/(d*(d-1)*(d+2)))*(((d+1)*d_B-2)*(Dagger(phi)*psi)*(Dagger(psi)*phi) + (d-d_B)*((Dagger(phi)*phi)*(Dagger(psi)*psi) + (Dagger(phi_star)*psi)*(Dagger(psi)*phi_star)))'
    DISTRIBUTED = '(d_P**2/(d*(d-1)*(d+2)))*(d_B*(d_B*(d+1)-2)*sp.Abs(Dagger(phi)*psi)**2 + d_B*(d-d_B)*(Dagger(phi)*phi)*(Dagger(psi)*psi) + d_B*(d-d_B)*sp.Abs(Dagger(phi_star)*psi)**2)'

    def test_numeric_fingerprint(self):
        fp = braket_fp(self.FACTORED)
        self.assertEqual(fp['t'], 'expr')
        self.assertTrue(comparable(fp))

    def test_distributed_factor_matches(self):
        self.assertTrue(equivalent(braket_fp(self.FACTORED), braket_fp(self.DISTRIBUTED)))

    def test_conjugate_states(self):
        # |x*> is the complex conjugate of |x>: <phi*|psi><psi|phi*> = <phi|psi*><phi*|psi>
        a = braket_fp('(Dagger(phi_star)*psi)*(Dagger(psi)*phi_star) + d_B')
        b = braket_fp('(Dagger(phi)*psi_star)*(Dagger(phi_star)*psi) + d_B')
        self.assertTrue(equivalent(a, b))

    def test_norms_are_not_one(self):
        # the states are not stated normalized: <phi|phi><psi|psi> is not 1
        self.assertFalse(equivalent(braket_fp('(Dagger(phi)*phi)*(Dagger(psi)*psi) + d_B'), braket_fp('1 + d_B')))

    def test_different_answers_differ(self):
        self.assertFalse(equivalent(braket_fp(self.FACTORED), braket_fp(self.FACTORED + ' + d_B*(Dagger(phi)*psi)')))
        self.assertFalse(equivalent(braket_fp('(Dagger(phi)*psi)*(Dagger(psi)*phi)'),
                                    braket_fp('(Dagger(phi)*psi)*(Dagger(phi)*psi)')))

    def test_bra_of_a_ket_is_another_state(self):
        # Bra(phi) with phi = Ket('phi') is Bra(Ket(phi)): a different SymPy object, kept apart (qwen-full-medium)
        self.assertFalse(equivalent(braket_fp('(Dagger(phi)*phi)*(Dagger(psi)*psi) + d_B'),
                                    braket_fp('(Bra(phi)*phi)*(Bra(psi)*psi) + d_B')))


if __name__ == '__main__':
    unittest.main()

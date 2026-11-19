"""Issuer + series exclusions: documents from the same issuer and series as a GDP.pdf test PDF (another year's filing
by the same company, the same agency's report series, the same insurer's policy forms, ...). Reusing a test task's
scaffold counts as contamination (#357), so these never enter the cache's selection.

Written by reading the test documents' first pages (class-level identification only). Matched case-insensitively
against the classifier's issuer/series/title, the source URL and the first 1,500 characters of the snippet. Broad
series that recur in the test set (MMWR, HICPAC guidelines, SEC filings of the named companies, Supreme Court slip
opinions) are excluded as whole series. File-hash and MinHash checks against the 100 test documents run separately
(decontam.py).
"""
import re

PATTERNS = [
    # Healthcare
    r'ehlers.danlos', r'american journal of medical genetics', r'pulmonary embolism.*guideline|guideline.*pulmonary embolism',
    r'aha/acc', r'american heart association.*american college of cardiology', r'global report on diabetes',
    r'morbidity and mortality weekly report|\bmmwr\b', r'\bhicpac\b|healthcare infection control practices advisory',
    r'routine practices and additional precautions', r'hand hygiene in health.care settings',
    r'catheter.associated urinary tract', r'nurturing care', r'health at a glance', r'apollo hospitals?',
    # STEM / research
    r'nesdis', r'regional climate trends and scenarios', r'national climate assessment',
    r'naca report 1135|equations, tables,? and charts for compressible flow', r'wolbachia',
    r'lightweight cryptography', r'nist\.ir\.8107|smart manufacturing systems', r'manufacturing usa',
    r'nist ams 600|ams600', r'chronic wasting disease', r'sulforaphane', r'birinapant', r'sierra leone.*(anemia|micronutrient)', r'palindromic units|isKpn|is1397', r'diffuse correlation spectroscopy',
    r'european journal of psychotraumatology', r'coronary wall atherosclerosis', 
    # Engineering / manufacturing / construction
    r'henny penny', r'pressure fryer', r'mil-r-6855|mil.r.6855', r'everyspec', r'hydra.cell|wanner engineering',
    r'rohde\s*&\s*schwarz|r&s.?rtp', r'nasa tn d-21|near.field and far.field noise',
    r'nrel/tp-510|biomass to hydrogen', r'angelus', r'seamer', r'drilling technical guide',
    r'em 1110-2-2906|design of pile foundations', r'import refusal', r'fda.*refus', r'greenbergfarrow',
    r'freefarmhouse|jay osborne', r'meridian industrial supply',
    # Real estate
    r'milford heights', r'tuttle armfield wagner', r'cushman\s*&\s*wakefield|marketbeat',
    r'national property inspections', r'costello real estate', r'windeed|lexis.*automated valuation',
    r'metropolitan nashville', r'smith douglas',
    # Insurance
    r'\bniac\b|nonprofits.? insurance alliance', r'wellmark', r'anthem dental', r'community insurance company',
    r'american freedom insurance', r'young america insurance', r'acadia insurance', r'farmers insurance',
    r'make model pricing', r'hail damage claims',
    # HR
    r'university of minnesota', r'office of personnel management|opm\.gov', r'work scheduling flexibilities',
    r'university of southern mississippi', r'harry caray', r'university of california.*ppsm|\bppsm\b',
    r'university of texas system', r'employee annual enrollment', r'leading on leave', r'human resource management and organizational effectiveness',
    # Finance
    r'comcast|cmcsa', r'airbnb|abnb', r'walt disney|disney', r'tripadvisor', r'oracle corp', r'edgar/data/1341439',
    r'dave\s*&\s*buster|play-20', r'fdic quarterly', r'quarterly banking profile',
    # Legal
    r'navient', r'public service loan forgiveness', r'civil cases commenced.*nature of suit', r'judicial business',
    r'federal rules of bankruptcy procedure', r'public access opinion', r'attorney general.*illinois|illinois.*attorney general',
    r'slip opinion', r'supreme court of the united states', r'ssa-16|application for disability insurance benefits',
    r'great ideas creative group',
]
_RE = re.compile('|'.join(f'(?:{p})' for p in PATTERNS), re.I)


def excluded(*texts):
    """The first matching pattern, or None."""
    for t in texts:
        if t:
            m = _RE.search(str(t))
            if m:
                return m.group(0)
    return None

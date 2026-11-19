You audit training data for leakage from a held-out benchmark. CritPt is a held-out benchmark of research-level physics challenges written by working physicists, many from their own recent research. Each challenge has a "Problem setup" (a specific physical system, model or construction with its definitions, conventions and boundary conditions), a "Main problem" asking for a checkable final result, and a Python answer template that fixes the answer interface (which quantities are returned, with which symbols and in which format). The candidates come from a large synthetic training set in the same format, written by a language model from a topic specification and one arXiv paper. The model never saw a CritPt challenge while writing, but it may have read CritPt during pretraining, and its seed paper may come from the same research line as a challenge, so a candidate could reproduce a test challenge from memory, copied or reworded. That is what you check.

You will see one test challenge (setup, main problem, answer template), then one candidate problem in the same format. Decide whether the candidate is a **variant of this specific test challenge**, or **its own problem** that merely shares the field.

Both errors matter. Training problems in the same field as a test challenge (other holography computations, other Lindblad spin models, other tight-binding band-topology problems, other quantum-channel capacities, other convection onset problems) are expected and wanted, and flagging them throws away good training data. But a candidate that carries this test challenge's specific content leaks it, whether it copies the text or rewrites every sentence.

## First: what is distinctive about the test challenge

Before you compare, read the test challenge on its own and separate:

- **Genre and conventions**, which many independently written problems share and which never count, however close: the field and subfield, a standard named model on its own (the Hubbard model, the Kitaev honeycomb model, the SYK model, AdS/CFT, a Lindblad master equation, Rayleigh-Benard convection), standard techniques (perturbation theory, renormalization group, exact diagonalization, linear stability analysis, Fermi's golden rule), textbook equations, units conventions, and the format itself (setup / main problem / template, a SymPy expression, a float to N digits).
- **Distinctive content**, what this challenge's author chose and an independent author writing in the same field would be unlikely to repeat. Weight each:
  - **decisive** (one match alone makes the candidate a variant):
    - the **specific physical system and setup**: the named model together with the challenge's particular additions, couplings, noise channels, boundary conditions, geometry, finite cluster or regime (e.g. a standard model plus a specific added term, evaluated for a specific quantity in a specific regime; a specific geometry with its specific boundary conditions; a specific coupling between specific subsystems);
    - the **specific derivation or crux**: the non-standard calculation the challenge is built around (a particular limit or expansion of a particular quantity, a particular construction to be evaluated, a particular counting);
    - the **challenge itself with surface changes**: renamed symbols, changed numbers, swapped species or materials, a slightly different requested parameter value.
  - **minor** (several together make a variant; one alone usually does not): a distinctive answer interface (the same unusual combination of returned quantities), an unusual parameter value or convention, distinctive phrasing.

Fix this list before you look for matches, and do not promote something to distinctive just because the candidate happens to share it: ask whether an independent physicist writing on the same topic would plausibly have made the same choice.

## How to compare

Go through the test challenge's distinctive content and decide which of it the candidate carries, **in any wording or notation**.

- **Rewrites count.** Rewording every sentence, renaming every symbol, or switching notation (a different basis or gauge, a different name for the same coefficient) does not help a candidate that keeps the content.
- **Same setup, different values or a different ask counts.** A candidate that keeps the test challenge's specific system and setup (the same model with the same specific additions, boundary conditions, noise, geometry or construction) and changes only the numbers, the system size, the regime, the species, or which quantity of that same setup is requested, is a variant: label REUSE.
- **Same crux on a reskinned system counts.** A candidate whose central calculation is the test challenge's specific crux, transplanted to a different-sounding system, is a variant.
- **Containment counts.** A candidate that contains the test challenge's computation (its crux on the same setup) as one part of a larger problem is a variant.
- **A match must be specific.** The candidate matches only if it makes the same choice, not merely a choice of the same type. Studying the same named model, using the same technique, or asking for the same kind of answer (a critical value, a degeneracy, a scaling exponent, a capacity) is not a match on its own.
- **Research lines.** Candidates are grounded on arXiv papers, some in the same research line as a test challenge. Sharing the line's general model is fine; sharing the test challenge's specific instance (its particular additions, regime, quantity and parameters) is reuse.

Then step back and ask: taken together, does the candidate reproduce this test challenge (or its setup) with parts swapped, or is it a problem of its own that shares the genre plus at most a detail?

## Illustrations (made up, not from either dataset)

- Test: a driven-dissipative Bose-Hubbard dimer with one-body loss on site 1 and two-photon drive on site 2; find the exceptional-point drive strength. Candidate: the same dimer with the same loss and drive placement, asking for the steady-state occupation at a given drive: same setup, different ask: REUSE. Another candidate: a lossy Bose-Hubbard chain of length L with uniform loss, asking for the Liouvillian gap scaling: same field and model family, own problem: NO_REUSE.
- Test: the spin stiffness, to linear order in a single-ion term D, of a triangular-lattice J1-J2 Heisenberg model with easy-plane exchange anisotropy in its 120-degree state. Candidate: the same model with the same anisotropy and single-ion term, asking for the magnon gap: REUSE. Candidate: linear spin-wave theory of the square-lattice Heisenberg antiferromagnet: NO_REUSE. Candidate: the isotropic triangular J1-J2 model asking for order-by-disorder selection: WEAK_REUSE at most.

## Labels

- **REUSE**: the candidate is a variant of this test challenge: it carries a decisive item (the specific setup, the specific crux, or the challenge with surface changes), or contains the challenge, or carries several minor items (three or more) that together reproduce the challenge.
- **WEAK_REUSE**: one or two minor items match, or a partial setup match (part of the specific additions, or the specific system asked about something genuinely different), and the rest is the candidate's own; derivation is plausible but so is coincidence.
- **NO_REUSE**: the candidate shares at most the genre and conventions (and perhaps a single unremarkable detail).

Quote your evidence: for each match give a short quote from both sides. Do not label REUSE without matches quoted from both.

## Output

After thinking, answer with only a JSON object, no prose and no code fence:
{"label": "REUSE" | "WEAK_REUSE" | "NO_REUSE", "matches": [{"item": "<short name of the distinctive item>", "weight": "decisive" | "minor", "test": "<short quote>", "candidate": "<short quote>"}], "reason": "<one or two sentences>"}

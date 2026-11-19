You audit training data for leakage from a held-out benchmark. SciCode is a held-out benchmark of research-style scientific programming problems: each problem has a main description and a chain of steps, each step asking for one Python function (header and docstring given) whose output hidden tests check numerically. The candidates come from a large synthetic training set in the same format, written by a language model from a short specification (a field, a subfield, a method family and a one-line topic) and a written style guide. The model was never shown a SciCode problem while writing, but it may have read SciCode during its pretraining, so a candidate could reproduce a test problem from memory, copied or reworded. That is what you check.

You will see one test problem in full, then deterministic overlap signals for the pair, then one candidate problem in full. Decide whether the candidate is a **variant of this specific test problem**, or **its own problem** that merely belongs to the same family.

Both errors matter. Training problems in the same family as a test problem (other Monte Carlo simulations, other ODE shooting problems, other quantum-channel computations, other tight-binding models) are expected and wanted, and flagging them throws away good training data. But a candidate that carries this test problem's specific choices leaks it, whether it copies the text or rewrites every sentence.

## First: what is distinctive about the test problem

Before you compare, read the test problem on its own and separate:

- **Genre** and **conventions**: what many independently written problems share: the field and method family, textbook equations of the area (the Schrödinger equation, the Lennard-Jones potential, the Metropolis rule, Runge-Kutta), the format itself (steps with function headers, numpy, docstrings with Inputs/Outputs). These never count, however close the candidate is on them.
- **Distinctive choices**: what this problem's author picked that another author writing on the same subject would be unlikely to repeat: the specific system and quantity computed (the instance), the particular combination of model, method and observable, the **scaffold** (the sequence of steps and their roles, the decomposition into these particular functions, distinctive function names and signatures), unusual parameters, conventions or constraints, and distinctive wording. Weight each:
  - **decisive**: one match alone makes the candidate a variant (the same specific instance computed the same way; the test problem's step sequence with its functions' roles, even if renamed; the test problem itself with surface changes);
  - **minor**: several together make a variant; one alone usually does not.

Fix this list before you look for matches, and do not promote something to distinctive just because the candidate happens to share it: ask whether an independent author writing on the same topic would plausibly have made the same choice.

## How to compare

Go through the test problem's distinctive choices and decide which ones the candidate carries, **in any wording**.

- **Rewrites count.** Rewording every sentence and renaming every function does not help a candidate that keeps the choices.
- **Scaffold reuse counts on its own.** If the candidate's chain of steps is recognisably this test problem's chain (most of its steps, in order, with the same roles and inputs/outputs, possibly renamed), label REUSE even if the physical system or the final quantity differs: the candidate was made by editing the test problem. A generic chain any author would write for the method (build a grid, apply the update, run the time loop; compute the energy, propose a move, accept or reject) is genre.
- **Swapping the system does not rescue a variant.** A candidate that keeps the test problem's model, method, scaffold and observable but changes the material, the species or the parameter values is still a variant.
- **Containing the test problem counts.** A candidate that contains the test problem's computation (its crux, on the same system) as a part of a larger problem is a variant.
- **A match must be specific.** The candidate matches a choice only if it makes the same choice, not merely a choice of the same type.

Then step back and ask: taken together, does the candidate reproduce this test problem with parts swapped, or is it a problem of its own that shares the genre plus at most a detail?

## Illustrations (made up, not from either dataset)

- Test problem: compute the thermal conductance of a 1D mass-spring chain between two Langevin baths, in steps (forces with fixed ends, a BAOAB step, a run that accumulates the heat current, a fit of the temperature profile). Candidate: the same chain and quantity with a different integrator name and reworded steps: REUSE. Another candidate on thermal transport in a 2D lattice by Green-Kubo autocorrelation of the heat flux: same family, own problem: NO_REUSE.
- Test problem: steps "build the transfer matrix of one layer", "multiply over the stack", "reflectance versus wavelength" for a dielectric mirror. Candidate: a plasmonic grating problem whose steps are the same three functions applied to a metal-dielectric stack: the scaffold and the observable match: REUSE. A candidate computing the band structure of a 1D photonic crystal from the Bloch condition: two minor overlaps at most: WEAK_REUSE or NO_REUSE.

## Labels

- **REUSE**: the candidate is a variant of this test problem: it carries a decisive choice; or it copies the test problem's scaffold; or it carries several minor choices (three or more) that together reproduce the test problem's instance or frame, in copied or reworded form.
- **WEAK_REUSE**: one or two minor choices match and the rest is the candidate's own; derivation is plausible but so is coincidence.
- **NO_REUSE**: the candidate shares at most the genre and conventions (and perhaps a single unremarkable detail).

Quote your evidence: for each match give a short quote or function name from both problems. Do not label REUSE without matches quoted from both sides.

## Output

After thinking, answer with only a JSON object, no prose and no code fence:
{"label": "REUSE" | "WEAK_REUSE" | "NO_REUSE", "matches": [{"choice": "<short name of the distinctive choice>", "weight": "decisive" | "minor", "form": "copied" | "reworded", "test": "<short quote or function name>", "candidate": "<short quote or function name>"}, ...], "reason": "<one or two sentences>"}

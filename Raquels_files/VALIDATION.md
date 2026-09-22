# Validation of the handoff

Executed September 22, 2026 using the versions in `tested_environment.txt`.

- Both portable render commands completed from a working directory outside the repository, using only the package's plotting inputs and bundled style. No original workstation paths were needed.
- The four real-space SVD NPZ files are byte-identical to their saved source files. Each retained spectrum/transition array is exactly equal to the corresponding array in its larger original NPZ; the provenance manifest records both source and delivered hashes.
- All four SVD probability distributions sum to one. Orbital-to-atom aggregation and both endpoint marginals agree. The selected zero-based indices are 0, 0, 1, and 0 for NaCl, diamond, CsI3, and GeTe; the corresponding 90-percent radii are 2.847184, 3.880232, 5.919573, and 16.373891 angstroms.
- The spectral line sums are zero for benzene, coronene, and the open flake and +1 for periodic bulk, within the recorded numerical tolerance. Broadening preserves the unbroadened integrals, including the separate conversion to the plotted per-electron units.
- The Figure 4 loop receipt agrees exactly with the original r14 receipt. PDFs rendered to the same 1600-pixel-wide PNG have identical pixels. The exported small-system projectors pass Hermiticity, idempotency, and the independent finite-system imaginary Qxy trace check.
- Figure 3 was visually compared with the current manuscript raster: all six panels, highlighted modes, range curves, and glyphs agree. Its raster backend differs from the original PDF-to-PNG route, so pixel identity is not claimed. The generated PNG is 4260 by 2876 pixels, corresponding to 600 dpi at 7.1 inches wide; the reference is 4260 by 2877.

These checks establish that the handoff reproduces the current figure inputs and conventions. They do not add a range-convergence study, certify unresolved material basis provenance, or equate loop weights with physical electric currents. Original numerical diagnostics remain in each figure's `provenance/` and `data/` metadata.

`verify.py` checks the delivered file hashes and numerical conservation rules. It reads the supplied tables; rerendering writes fresh tables under `figure_4/output/tables/` and leaves the delivered tables unchanged. If you intentionally edit an input or script, its original hash check will fail by design.

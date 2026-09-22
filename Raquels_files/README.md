# Raquel's files

Files for Fatmagul to remake Figures **3 (SVD)** and **4 (loops and transverse response)** of Raquel's v15 manuscript. This folder is self-contained and separate from the repository's other calculations. The figure numbering follows `bonding_draft_v15_Raquel_working_copy.tex` as refreshed on September 22, 2026.

| Figure | Start here | Existing manuscript artwork |
|---|---|---|
| 3: SVD spectrum, spatial extent, and material glyphs | [Figure 3 instructions](figure_3/README.md) | [600-dpi PNG](figure_3/reference/figure_3_v15.png) |
| 4: molecular/flake loops and transverse spectra | [Figure 4 instructions](figure_4/README.md) | [PDF](figure_4/reference/figure_4_v15.pdf) |

## Reproduce the figures

From the repository root, using Python 3.10 or newer:

```bash
python -m venv Raquels_files/.venv
source Raquels_files/.venv/bin/activate
python -m pip install -r Raquels_files/requirements.txt
python Raquels_files/figure_3/scripts/render.py
python Raquels_files/figure_4/scripts/render.py
python Raquels_files/verify.py
```

Each render writes PNG, PDF, and numerical checks into that figure's ignored `output/` directory. Script paths resolve relative to this folder, so neither Raquel's filesystem nor a particular working directory is required. There is no LaTeX, DFT, Wannier, Git LFS, or large-system diagonalization requirement for these commands. Figure 4 recomputes only the small 6-, 24-, and 96-site drawing projectors.

To redesign the artwork, edit `scripts/` or import the supplied `tables/` and `data/`. Keep `reference/` as the visual comparison. The Figure 3 output is rasterized at 600 dpi when printed 7.1 inches wide, avoiding the heavy original vector glyphs.

## What is included

The saved numerical arrays needed to redraw every panel, the plotting code, readable tables, the common plotting style, and original producer scripts for provenance. Full raw DFT/Wannier inputs and the large unused SVD eigenvector arrays are not needed to remake the figures and are not bundled. This is a reproduction package for the existing figures, not a fresh calculation or a claim of converged material results.

`provenance_manifest.json` records original filenames, SHA-256 hashes, and every array subset or script adaptation. `file_manifest.json` checks the delivered inputs, scripts, tables, and references. Historical original scripts end in `.py.txt`: they document the calculations and may contain old local paths; use the portable `scripts/render.py` entry points. Numerical producer dependencies outside this package are recorded in the original metadata, so the historical producers are not advertised as standalone full-calculation commands.

See [VALIDATION.md](VALIDATION.md) for the executed checks and [tested_environment.txt](tested_environment.txt) for the actual software versions. The original producer filenames for Figure 4 say `fig3`; they predate the current manuscript numbering.

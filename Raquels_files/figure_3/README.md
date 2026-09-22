# Figure 3: SVD and real-space mode patterns

Run `python Raquels_files/figure_3/scripts/render.py` from the repository root. Outputs are `figure_3/output/figure_3.png` and `figure_3.pdf`. The existing main-text figure is in `reference/figure_3_v15.png`.

## Panels and the data they use

| Panel | Inputs | Plotting prescription |
|---|---|---|
| (a), mode spectrum | `data/bonding_modes_20260919_v1_*_mesh*.npz`; `tables/*_spectrum.csv` | Plot eigenvalue divided by the sum of represented eigenvalues against one-based rank, both on logarithmic axes. Circles identify the selected mode block. |
| (b), spatial extent | `data/bonding_mode_realspace_20260919_v2_*.npz`; `tables/*_range_cdf.csv` | Exact step cumulative probability versus occupied-empty endpoint separation in angstroms. Use a post-step curve; circle the saved 90-percent radius. |
| (c-f), material patterns | The same real-space NPZ files | Draw the saved periodic atom/image pairs using `bonding_full_mode_glyphs_20260919_r10.py`. Colors, camera, arrow construction, and per-material display bounds are explicit there. |

Materials are NaCl, diamond, CsI3, and GeTe. The saved unshifted meshes are 24 cubed, 24 cubed, 12 cubed, and 24 cubed, respectively. Atomic fractions multiply the lattice matrix on the right to give Cartesian positions in angstroms.

## Meaning and normalization

The original calculation coherently sums orbital processes into directed atomic-pair/image channels before decomposing the response. It keeps a principal channel Gram matrix selected by a 99-percent trace target, completed for reversal partners and numerical cutoff ties. The eigenvalues are squared singular values. Their plotted fractions sum to one over this represented Gram matrix; they are not fractions of the full metric.

The highlighted block maximizes the restricted uniform metric contribution, `lambda_s * abs(sum_p u_ps)**2`, summed across a numerically unresolved eigenvalue block. It is not necessarily the block with the largest eigenvalue. Each selected block in the delivered data contains one mode. The original JSON metadata records the retained trace and metric fractions separately.

The mode is reconstructed in the orbital occupied-empty kernel, transformed from k to relative image R, then summed in absolute square over orbital pairs and Cartesian directions. The saved probability sums to one across the entire periodic relative-coordinate grid. This produces both the endpoint glyph and the range CDF. It describes relative endpoint separation of a uniform zero-wavevector mode, not localization of the mode's center.

The CDF includes all saved probabilities, including separations beyond the visible 40-angstrom axis. Do not renormalize the visible window. The four 90-percent radii are approximately 2.847, 3.880, 5.920, and 16.374 angstroms in material order.

Glyph arrow/halo sizes are scaled within each material. They show nonnegative mode probability, not signed bond interference or electric current. Same-site halos include all orbital pairs on the same atom at zero relative image. Periodic reverse pairs include the Nyquist identification. The full probability is retained before clipping to the chosen display window; the halo's area is a weight, not a physical cloud radius.

## NPZ array guide

Spectrum files contain only `eigenvalues` and `mode_metric_weights`, copied exactly from the original saved results. Unused full eigenvectors are omitted; the complete real-space probability required by the figure is already supplied.

| Real-space array | Meaning |
|---|---|
| `process_keys` | Rows `(A, B, Rx, Ry, Rz)`; zero-based atom labels, integer relative images |
| `probabilities` | Normalized probability for each process row |
| `nuclear_displacement_A` | Corresponding Cartesian endpoint displacement in angstroms |
| `atom_frac`, `lattice`, `atom_symbols` | Structure and species used by the drawing |
| `orbital_pair_probability`, `orbital_to_atom` | Orbital weights and their mapping to atomic marginals |
| `occupied_marginal`, `empty_marginal` | Summed endpoint probabilities |
| `mode_indices`, `mode_eigenvalues` | Zero-based selected block indices and eigenvalues |
| `r90_A` | Saved radius containing 90 percent of the full probability |

## How the source calculation was made

`provenance/bonding_modes_20260919.py.txt` constructs the restricted Gram matrix, normalizes by the k-point count times the active occupied-band count, diagonalizes it, and saves the spectrum and metric weights. `bonding_mode_realspace_20260919_v2.py.txt` selects the block, reconstructs its occupied-empty right kernel, applies the unitary Fourier transform, and saves the probability arrays. The original per-material JSON files contain input hashes, operator conventions, and diagnostics.

The portable plotting code preserves the main figure's numerical and drawing operations. Its changes are local paths, a bundled style, omission of an unused supplemental plot, and raster export. This package makes no new claim about basis provenance, completeness of the retained manifold, or convergence of the mode range. The older orbital-pair SVD and chemical class percentages are distinct decompositions and must not be substituted for these inputs.

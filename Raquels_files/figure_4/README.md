# Figure 4: loops and transverse geometric response

Run `python Raquels_files/figure_4/scripts/render.py`. The current manuscript reference is `reference/figure_4_v15.pdf`. Historical producer names contain `fig3`, but these files generate **Figure 4** in the current v15.

## Data for remaking the panels

| Panels | Files | Contents |
|---|---|---|
| (a-c), sites and loops | `tables/*_sites.csv`, `tables/drawing_data.npz` | Site indices, positions, exact small-system occupied projectors, and site markers for benzene, coronene, and the 96-site open flake |
| (a-c), loop overlays | `tables/loop_drawings.json`; `scripts/loops.py`, `scripts/render.py` | Selected triangles, coordinates, signed weights, arrow orientation, and display shifts |
| (d-f), transverse spectra | `data/spectra.npz` | Original unbroadened transition energies and signed line weights for benzene, coronene, the 864-site open flake, and periodic bulk |
| (d-f), ready-to-plot curves | `tables/spectra_display.csv` | Energy grid and the four displayed broadened curves |

The 96-site upper drawing and the 864-site lower spectrum are intentional. Do not label them as the same finite sample. The benzene geometry is rotated 30 degrees and displayed twice with vertical shifts to separate examples; the underlying projector is the same. The site CSV coordinates are the unshifted physical geometry; `loop_drawings.json` records the displayed shifts.

## Model and signs

All panels use nearest-neighbor hopping magnitude `t1=1`, next-nearest-neighbor magnitude `t2=1`, phase `phi=pi/2` with `phase_sign=-1`, zero sublattice mass, and half filling. The nearest-neighbor distance is `a=1`. Preserve the hopping/phase convention in `lattice.py` and the archived original producer; flipping it reverses the chirality and bulk Chern number.

The node color is the site marker `M_i = 4*pi*Im(P*x*P*y*P)_ii`, in units of `a^2`. All panels share the same coolwarm scale clipped at -1 and +1, with colorbar extensions. The complete site sum vanishes for each finite open system. For benzene the marker also vanishes site by site, although individual loops can be nonzero.

The triangle weight is `8*pi*A_xy*Im(P_ab*P_bc*P_ca)`, where `A_xy` is the signed triangle area. Red/blue encodes the sign of this geometric contribution. The arrow traversal is chosen with positive imaginary projector product; triangle area also enters the sign. These are selected examples, not a complete cancellation group. These arrows are not a calculation of physical equilibrium bond currents.

## Spectral normalization and broadening

In `spectra.npz`, each `*_de` array contains transition energies in units of t, and each `*_w` array contains the matching dimensionless Chern-normalized line weights. `Egrid` runs from 0 to 12 in steps of 0.001. The plotted quantity is `-2 Im Qxy(E)` per electron, in `a^2/t`.

The renderer multiplies the broadened line-weight spectrum by `A_cell/(2*pi)`, with `A_cell=3*sqrt(3)/2`. Consequently, the integrals of the stored line weights are zero for finite open samples and +1 for bulk; the integral of the plotted bulk curve is `A_cell/(2*pi)`, **not 1**. The selected 480-by-480 bulk mesh is recorded in the original checks.

For display, each line is deposited linearly on the grid, then convolved with a reflected Gaussian of standard deviation 0.075 t (`mode='mirror'`). Exact sum rules use the unbroadened weights. Broadening does not establish the optical gap, and the low-energy open-flake feature is distinct from the periodic bulk spectrum.

## How it was produced

The archived transverse producer contains the finite-system Hamiltonian, occupied-empty transitions, projector/operator cross-checks, periodic Bloch calculation, and the final r14 layout. `tbcore.py.txt` contains its numerical helpers. The archived lattice and loop scripts supply the small geometries and selected loop conventions. The original checks and r14 layout/loop receipts are retained in `provenance/`.

The portable renderer uses the saved spectral lines and recalculates only 6-, 24-, and 96-site drawing projectors. It preserves the original panel layout and checks the finite sums, positive bulk chirality, and loop signs. It also exports the drawing data and ready-to-plot spectral table for editing in other software.

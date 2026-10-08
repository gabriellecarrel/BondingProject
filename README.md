# Code and data for the figures and tables

Every notebook reads its inputs from `data/` and runs from this folder.
Requires Python 3 with numpy, scipy, pandas, matplotlib, tqdm, pymatgen and pythtb 2.0.

## Where each calculation is

| Paper item | Notebook / file |
|---|---|
| Fig. 1, Fig. 2 A–D, Figs. S1–S2 | Schematics (no calculation) |
| Fig. 2 I, Tables S5–S6 | `Fig2_minimal.ipynb` (method); paper values in `cluster_outputs/position_resolved_QM_trace_<material>.xlsx` |
| Fig. 2 E–H, Table S4 | `Fig2_minimal.ipynb` §10–12 (method); paper values in `cluster_outputs/table_S4_fluctuations.csv` |
| Fig. 3 A–B, Table S7 | `fig3_code/fig3.ipynb` (with `fig3_inputs.json`) |
| Fig. 4, Fig. S3 | `Fig4_minimal.ipynb` (uses `TBLR2.py`); tables written to `fig4_minimal_output/` |
| Table S3 (LOBSTER and Wannier columns) | `LOBSTER_calc.ipynb` |
| Tables S1–S2 | Read from `data/<material>/POSCAR`, `wannier90.win` and `OUTCAR` |

## Inputs (`data/<material>/`, for NaCl, Diamond, CsI3, GeTe)

- `wannier90.win`, `wannier90_hr.dat`, `wannier90_centres.xyz`: Wannier90 models
- `OUTCAR`, `POSCAR`: VASP static calculation (Fermi level, structure)
- `QM_orb_res_qgt{xx,yy,zz}.npy`: orbital-resolved quantum-metric tensors (2×2×2 supercell; CsI3 primitive cell), used by Fig. 3
- `LOBSTER/COBICAR.lobster`, `LOBSTER/ICOBILIST.lobster`: LOBSTER bond indices, used by Table S3

## Notes

- The paper's Fig. 2 and Tables S4–S6 used `min_hopping_norm = 1e-4` eV and `nk` = 48 (NaCl), 46 (diamond), 15 (CsI3), 42 (GeTe), run on a cluster with the method of `Fig2_minimal.ipynb`. The notebook's defaults (0.01 eV, `nk` = 30) run locally in about a minute.
- Drawings of individual fluctuations (Fig. 2 E–H, Fig. S3 A–C, G–I) are not included; the values they show are in Table S4 and `fig4_minimal_output/`.

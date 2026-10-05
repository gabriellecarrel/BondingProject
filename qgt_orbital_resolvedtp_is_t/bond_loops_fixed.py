"""Three-centre bond-index pipeline, parameterised by model.

Same physics as the notebook cells (hopping table -> ham_func -> TBLinearResponse
-> QGT_from_bonding), but taking the model as an argument so every flake and the
periodic Haldane bulk go through one code path.  Molecule vs. bulk is detected
from the hoppings rather than set by hand.  Nothing here reads a cached table:
run_models() always computes, and save_tables() writes only when you ask it to.

The one substantive change is that `bond_element` is memoised on R: the R1/R2
double loop asks for only ~(2*nR-1)^2 distinct R's but calls it 3*nR^4 times, so
caching turns the Haldane run from minutes into seconds.  Results are unchanged.
"""
import json
import os

import numpy as np
import pandas as pd

import TBLR2 as TBLR

try:
    from tqdm import tqdm
except ImportError:                                   # progress bar is optional
    def tqdm(it, **kw):
        return it

def is_molecule(model):
    """True when no hopping leaves the home cell, i.e. a finite cluster."""
    return not any(np.any(np.asarray(h.get("lattice_vector", [0, 0])) != 0)
                   for h in model.hoppings)


def hop_table(model):
    """H[i][j] -> list of [t, dx, dy] in the natural (cartesian) embedding."""
    norb = model.norb
    H = [[[] for _ in range(norb)] for _ in range(norb)]
    orb = np.asarray(model.orb_vecs)
    A = np.asarray(model.get_lat_vecs())
    bad = {o for h in model.hoppings for o in (h["from_orbital"], h["to_orbital"])
           if o >= norb}
    if bad:
        raise ValueError(f"model has {norb} orbitals but its hoppings still reference "
                         f"orbital(s) {sorted(bad)} -- rebuild it (remove_orb reindexes, "
                         f"so the removals must be re-run from a fresh supercell)")
    if len(orb) != norb:
        raise ValueError(f"model.norb = {norb} but orb_vecs has {len(orb)} rows")
    for h in model.hoppings:
        i, j = h["from_orbital"], h["to_orbital"]
        R = np.asarray(h.get("lattice_vector", [0, 0]))
        d = ((orb[j] + R) - orb[i]) @ A
        H[i][j].append([complex(h["amplitude"]), d[0], d[1]])
    return H


def make_ham(model):
    """H(k) built from the cartesian bond vectors (needed for the QGT)."""
    H = hop_table(model)
    onsite = np.round(np.asarray(model._site_energies, float), 10)
    norb = model.norb

    def ham(k):
        kx, ky = k
        Hk = np.zeros((norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                for t, dx, dy in H[i][j]:
                    v = np.round(t, 10) * np.exp(1j * (kx * dx + ky * dy))
                    Hk[i, j] += v
                    Hk[j, i] += np.conj(v)          # h.c. partner (also fixes i==j, +-R)
        Hk[np.diag_indices(norb)] += onsite
        return Hk

    return ham


def make_tb(model, nk=100, name="model"):
    """TBLinearResponse for `model`.  A molecule's H is k-independent -> 1 k-point."""
    nk = 1 if is_molecule(model) else nk
    return TBLR.TBLinearResponse(make_ham(model),
                                 Lar=np.asarray(model.get_lat_vecs()),
                                 Nkp=[nk, nk],
                                 prestr=f"loops_{name}",     # per-model, never collides
                                 save_all=False,
                                 kpath=["G", "X", "M", "G", "Y", "M"])


def default_R_list(model, rmax=4):
    if is_molecule(model):
        return [[0, 0]]
    r = np.arange(-rmax, rmax + 1)
    return [[i, j] for i in r for j in r]


def bond_df(model, name="model", R_list=None, rmax=4, nk=100, mu=0.0,
            threshold=0.0, progress=True):
    """DataFrame of every oriented three-centre loop a->b->c and its C_imag."""
    mol = is_molecule(model)
    tb = make_tb(model, nk=nk, name=name)

    # Occupation.  The molecule branch below builds rho from H(k=0), so count there.
    # (The notebook counted at tb.enUM[0] i.e. kSpan[0] = -q1/2 - q2/2; in the natural
    # embedding H(k) is k-dependent even for a cluster, so the two are not the same
    # matrix.  They agree for every model here -- half filling -- but only by luck.)
    if mol:
        en0, _ = TBLR.eigenstates(tb.ham(k=[0, 0]))
        occ = int(np.sum(en0.real <= mu))
    else:
        occ = int(np.sum(tb.enUM[0, :, 0].real <= mu))

    tau_lat = np.asarray(model.orb_vecs) @ tb.Lar
    r0 = tau_lat[np.newaxis, :] - tau_lat[:, np.newaxis]
    pref = (2 * np.pi * occ) / np.linalg.det(tb.Lar)
    if R_list is None:
        R_list = default_R_list(model, rmax)

    cache = {}

    def bond_element(R):
        """(rho, r) for lattice vector R.  Memoised: the R1/R2 loop repeats these."""
        key = (int(R[0]), int(R[1]))
        if key in cache:
            return cache[key]
        if mol:
            en, um = TBLR.eigenstates(tb.ham(k=[0, 0]))
            u = um[:, :occ]
            out = (u @ u.conj().T, r0)
        else:
            r = r0 + np.asarray(R) @ tb.Lar
            um = tb.enUM[:, :, 1:occ + 1]
            val = 0.0 + 0.0j
            for ik, k in enumerate(tb.kSpan):
                u = um[ik]
                val = val + (u @ u.conj().T) * np.exp(-1j * np.einsum("k,bdk->bd", k, r))
            out = (val / len(tb.kSpan), r)
        cache[key] = out
        return out

    total, chunks = 0.0, []
    for R1 in tqdm(R_list, disable=not progress, desc=name):
        for R2 in R_list:
            R1 = np.asarray(R1)
            R2 = np.asarray(R2)
            rho_ab, dist_ab = bond_element(R1)
            rho_bc, _ = bond_element(R2)
            rho_ca, dist_ca = bond_element(-R1 - R2)

            BI = np.einsum("ab,bc,ca->abc", rho_ab, rho_bc, rho_ca)
            x_ba, y_ba = -dist_ab[:, :, 0], -dist_ab[:, :, 1]
            x_ca, y_ca = dist_ca[:, :, 0], dist_ca[:, :, 1]
            dist_part_1 = np.einsum("ab,ca->abc", x_ba, y_ca)
            dist_part_2 = -np.einsum("ab,ca->abc", y_ba, x_ca)
            dist_part = dist_part_1 + dist_part_2
            total += pref * dist_part * BI

            mask = np.abs(dist_part * BI) > threshold
            if not mask.any():
                continue
            ai, bi, ci = np.where(mask)
            vals = BI[mask]
            dist_tot = dist_part[mask]
            chunks.append(pd.DataFrame({
                "a": ai, "b": bi, "c": ci,
                "R1x": R1[0], "R1y": R1[1], "R2x": R2[0], "R2y": R2[1],
                "dist_ab_x": x_ba[ai, bi], "dist_ab_y": y_ba[ai, bi],
                "dist_ca_x": x_ca[ci, ai], "dist_ca_y": y_ca[ci, ai],
                "dist_part_1": dist_part_1[mask],
                "dist_part_2": dist_part_2[mask],
                "dist_part": dist_tot,
                "BI_real": vals.real, "BI_imag": vals.imag, "BI_abs": np.abs(vals),
                "C_real": pref * dist_tot * vals.real,
                "C_imag": pref * dist_tot * vals.imag,
            }))

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    df.attrs.update(name=name, norb=int(model.norb), occ=occ, molecule=bool(mol),
                    nk=(1 if mol else nk), rmax=(0 if mol else rmax),
                    n_R=len(R_list), pref=float(pref), threshold=float(threshold),
                    mu=float(mu), n_loops=len(df),
                    sum_C_imag=float(df["C_imag"].sum()) if len(df) else 0.0,
                    sum_abs_C_imag=float(df["C_imag"].abs().sum()) if len(df) else 0.0)
    return df


def run_models(models, nk=100, rmax=4, mu=0.0, threshold=0.0, progress=False,
               stop_on_error=False):
    """{name: model} -> {name: df}, every one COMPUTED, never read from disk.

    There is deliberately no cache.  With bond_element memoised the whole set costs
    about ten seconds, and an on-disk cache can only ever hand back a table built
    from a model or parameters that no longer match the ones in front of you.
    Use save_tables() when you actually want files.
    """
    out, failed = {}, {}
    for name, model in models.items():
        try:
            df = bond_df(model, name=name, rmax=rmax, nk=nk, mu=mu,
                         threshold=threshold, progress=progress)
        except Exception as exc:                    # keep going, but say so loudly
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"{name:12s} *** FAILED *** {failed[name]}")
            if stop_on_error:
                raise
            continue
        out[name] = df
        a = df.attrs
        kind = "molecule" if a["molecule"] else f"bulk nk={a['nk']} rmax={a['rmax']}"
        print(f"{name:12s} norb={a['norb']:3d} occ={a['occ']:3d} {kind:22s} "
              f"{a['n_loops']:7d} loops   sum C_imag = {a['sum_C_imag']:+.6e}   "
              f"sum |C_imag| = {a['sum_abs_C_imag']:.4f}")
    if failed:
        print("\nFAILED: " + ", ".join(failed))
    return out


def save_tables(dfs, out_dir, overwrite=False):
    """Write <name>.csv plus a provenance sidecar.  Refuses to clobber by default."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    for name, df in dfs.items():
        path = os.path.join(out_dir, f"{name}.csv")
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        df.to_csv(path, index=False)
        meta[name] = dict(df.attrs)
        print(f"wrote {path}  ({len(df)} loops, {os.path.getsize(path)/1e6:.1f} MB)")
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta

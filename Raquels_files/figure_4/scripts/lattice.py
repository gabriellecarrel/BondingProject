#!/usr/bin/env python3
"""fig2_loop_chemistry.py -- Fig. 2 of the bonding concept paper (loop chemistry).

QUESTION (what the figure argues, before any equation is read):
    Berry curvature is carried by oriented multicenter loops -- loop chemistry.
    (a) BENZENE   -- a molecular Haldane model carries a strong oriented ring
                     loop (|Im b| = 0.024) yet, by C6 symmetry, the per-site
                     curvature Omega^{xy}(d) is exactly zero on every site.
    (b) CORONENE  -- larger molecule: per-site Omega^{xy}(d) develops (node
                     color) but the full molecular sum still vanishes.
    (c) HALDANE FLAKE (OBC) -- the same lattice with open boundaries: the bulk
                     marker is positive, the edge counter-weight negative, the
                     open-boundary sum exactly zero; with PBC the bulk value
                     survives and Omega^{xy}/2pi = +1 exactly.
    Molecule -> crystal at the level of concept. STYLE = GABI'S manuscript
    figures (benzene_coronene.png, Bond_Index_Part.png,
    cut_haldane_local_chern.png), factored into the lab-style named style
    'bonding' (Raquel 2026-07-03, rounds 2-4): sans-serif, flat rimless node
    disks, 'a)' regular-weight panel labels, coolwarm node coloring with ONE
    shared colorbar. Round-1 review also baked in: loop arrows STRAIGHT /
    THIN / BLACK, loops labeled with their actual site triples, every
    molecular site numerated, notes on common baselines.

CHIRALITY (pinned, uniform across ALL panels):
    phi -> PHI_SIGN * phi with PHI_SIGN = -1. why: at the molecular table's
    +phi the flake bulk marker is NEGATIVE (central -0.41 at 54 sites,
    checked 2026-07-03), i.e. PBC would give -1, contradicting the draft's
    quoted +1. The 2026-07-02 Studio run pinned the POSITIVE-marker chirality
    (HAL_CHIRALITY = -1); this figure uses it everywhere so molecules and
    crystal share one circulation sense. CONSEQUENCE: the benzene Im b > 0
    traversal is 2->4->0 (the draft text's "Im b_{204} = 0.024" is the +phi
    orientation -- flagged to Raquel, one of the two needs an orientation
    note in the text).

INPUTS:
    None from disk. Contains drawing-level exact diagonalizations (n = 6, 24,
    and 96 dense eigh; the 96-site flake marker costs ~0.5 s) ONLY to place
    loop orientations and node-color values truthfully. why not Studio: this
    is figure logic, not a numerical run producing data (README precedent);
    every number it relies on is GATED against the Studio-verified table run
    2026-07-02_local_chern_marker_table.py (host-stamped npz in this folder).

MODEL (matches 2026-07-02_local_chern_marker_table.py, chirality pinned):
    NN hopping -t1 (t1 = 1, energy unit); complex NNN -t2 exp(i*PHI_SIGN*phi*nu),
    phi = pi/2, nu = sign((v1 x v2)_z); t2 = 1.0 t1 everywhere (above the
    benzene level crossing at ~0.6 t1). Drawing unit: NN distance d = 1.

CONVENTIONS:
    Loop weight  Im b_{d,a,c} = Im(rho_da rho_ac rho_cd), traversal d->a->c->d;
    arrows drawn in the traversal direction with Im b > 0. Per-site curvature
    (pinned Bianco--Resta marker, the manuscript's Omega^{xy}(d)/2pi):
        C(d) = 2*pi * sum_{a,c} (x_a y_c - y_a x_c) Im(rho_da rho_ac rho_cd)
    Node color: coolwarm (red > 0, blue < 0, white = 0) on ONE shared
    symmetric scale across all panels, colorbar at right labeled Omega^{xy}(d).

GATES (all asserted before saving; known limits from the Studio table run):
    G1  benzene |Im b(2,0,4)| = 0.024 (plateau constant, |err| < 2e-3)
    G2  benzene sum_d C(d) = 0 (exact) AND each C(d) = 0 (C6 symmetry)
    G3  coronene sum_d C(d) = 0 (trace cyclicity, |sum| < 1e-9)
    G4  flake central marker > 0 (pinned-positive chirality sanity) AND
        flake sum_d C(d) = 0 (OBC sum rule)
    Panel (c) quotes the PBC value Omega^{xy}/2pi = +1 (exact Chern number,
    pinned convention; Studio run G1), NOT the finite-flake bulk value.

OUTPUTS (overwritten each run):
    RQLAB/Projects/Bonding/attachments/loop_chemistry_fig2.pdf   (vector)
    RQLAB/Projects/Bonding/attachments/loop_chemistry_fig2.png   (300 dpi)
    stdout: per-panel Omega^{xy}(d)/2pi extremes (for the caption).
    PDF is then copied (with Raquel's per-action approval -- Dropbox guard) to
    Dropbox/Apps/Overleaf/quantum_metric/images/ and replaces the TikZ block
    of fig:loop_chemistry_summary in bonding_draft_v3.tex.

DEPENDENCIES: numpy, matplotlib; style = lab-style 'bonding' (sans-serif, Gabi
grammar, usetex pinned off). Run: ~/venvs/sci/bin/python fig2_loop_chemistry.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # why: headless render; MUST precede lab_style (it imports pyplot)

# --- style: the named "bonding" style from the lab-style library -------------
# why a named style, not inline rcParams (Raquel 2026-07-03, round 3): every
# figure of the manuscript (Fig. 1, this one, and future panels) must share one
# grammar -- fonts, colors, coolwarm Omega map -- defined ONCE in
# Drive/Code/lab-style/styles/bonding.mplstyle + BONDING_* constants.
LAB_STYLE = str(Path(__file__).resolve().parents[2]/'style')
sys.path.insert(0, LAB_STYLE)
from lab_style import apply_style, BONDING_INK, BONDING_CELL  # noqa: E402

plt = apply_style("bonding")  # Gabi grammar: sans (DejaVu), 9 pt, fonttype 42, coolwarm
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1]/'output'

# --- palette (single source: lab_style BONDING_* constants) ------------------
C_INK = BONDING_INK    # skeleton bonds, loop arrows, text
C_CELL = BONDING_CELL  # shaded unit cell (flake panel)
# why coolwarm: matches the manuscript's established local-Chern figure
# (cut_haldane_local_chern.png); red = positive, blue = negative, white = 0
CMAP = plt.get_cmap("coolwarm")

FS_LABEL = 11     # a) b) c) panel labels (Gabi reference size)
FS_NOTE = 9.0     # one/two-line bottom annotations
FS_IDX_A = 6.5    # benzene site indices
FS_IDX_B = 5.2    # coronene site indices (two digits)
FS_LOOP = 8.0     # b_{i,j,k} loop labels

# --- model constants (match 2026-07-02_local_chern_marker_table.py) ---------
T1 = 1.0          # NN hopping, energy unit
T2 = 1.0          # NNN magnitude, all panels; ABOVE benzene crossing at ~0.6 t1
PHI = np.pi / 2.0
PHI_SIGN = -1.0   # pinned-positive-marker chirality (see header CHIRALITY note)
D_NN = 1.0        # drawing unit: NN carbon-carbon distance
D_NNN = np.sqrt(3.0) * D_NN
FLAKE_RCUT = 5.3 * D_NN   # hexagon-center cutoff -> 96-site flake (why: central
                          # marker +0.49 and a clearly resolved edge at ~0.5 s)

# Studio-verified anchors (2026-07-02 table run, host-stamped npz)
IM_B204_REF = 0.024   # benzene plateau loop-weight magnitude, shell (2,0,4)


# ---------------------------------------------------------------------------
# Geometry (indexing matches images/Second_Draft_Images/benzene_coronene.png)
# ---------------------------------------------------------------------------
def benzene_positions() -> np.ndarray:
    """6 carbons on a regular hexagon, radius D_NN; site n at 60*n deg CCW."""
    ang = np.deg2rad(60.0 * np.arange(6))
    return D_NN * np.column_stack([np.cos(ang), np.sin(ang)])


def coronene_positions() -> np.ndarray:
    """24 carbons of coronene, indexed exactly as the manuscript figure."""
    d = D_NN
    pos = {}
    central = {7: 210, 8: 150, 9: 90, 16: 30, 15: 330, 14: 270}
    for i, dg in central.items():
        a = np.deg2rad(dg)
        pos[i] = d * np.array([np.cos(a), np.sin(a)])
    bridge = {6: 210, 2: 150, 10: 90, 17: 30, 21: 330, 13: 270}
    for i, dg in bridge.items():
        a = np.deg2rad(dg)
        pos[i] = 2.0 * d * np.array([np.cos(a), np.sin(a)])
    rim_dir = {3: 90, 1: 210, 4: 150, 11: 30, 18: 90, 23: 330,
               22: 30, 20: 270, 19: 330, 12: 210, 5: 270, 0: 150}
    bof = {3: 2, 1: 2, 4: 10, 11: 10, 18: 17, 23: 17,
           22: 21, 20: 21, 19: 13, 12: 13, 5: 6, 0: 6}
    for i, dg in rim_dir.items():
        a = np.deg2rad(dg)
        pos[i] = pos[bof[i]] + d * np.array([np.cos(a), np.sin(a)])
    return np.array([pos[i] for i in range(24)])


def hexagonal_flake(rcut: float):
    """Hexagonal honeycomb flake: all vertices of hexagons with center < rcut.

    Returns (pos, hexes) where hexes = list of (center, [6 vertex indices,
    CCW from 90 deg]). rcut = 3.5 d gives circumcoronene (54 sites); the
    figure uses FLAKE_RCUT (96 sites). Same construction family as coronene,
    so panel (c) reads as "the molecule, grown".
    """
    a1 = D_NNN * np.array([1.0, 0.0])
    a2 = D_NNN * np.array([0.5, np.sqrt(3.0) / 2.0])
    centers = [m * a1 + n * a2 for m in range(-6, 7) for n in range(-6, 7)]
    centers = [c for c in centers if np.linalg.norm(c) < rcut]

    verts, vid = [], {}

    def vindex(p):
        key = (round(p[0], 4), round(p[1], 4))
        if key not in vid:
            vid[key] = len(verts)
            verts.append(np.asarray(p))
        return vid[key]

    hexes = []
    for c in centers:
        ring = [vindex(c + D_NN * np.array([np.cos(a), np.sin(a)]))
                for a in np.deg2rad(90 + 60.0 * np.arange(6))]
        hexes.append((c, ring))
    return np.array(verts), hexes


# ---------------------------------------------------------------------------
# Drawing-level ED: half-filled projector of the (molecular) Haldane model
# ---------------------------------------------------------------------------
def haldane_rho(pos: np.ndarray, t2: float) -> np.ndarray:
    """Half-filled density matrix rho = sum_occ |n><n| (dimensionless), OBC.

    NN: -t1; NNN (same shell d = sqrt(3) d_NN): -t2 exp(i PHI_SIGN phi nu),
    nu = sign((v1 x v2)_z) over the unique two-NN-hop path i->m->j (unique
    because same-sublattice NNN pairs on a honeycomb share exactly one NN).
    Fails loudly if the gap at half filling is closed (wrong t2 regime).
    """
    n = len(pos)
    dm = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    nn = np.abs(dm - D_NN) < 1e-3 * D_NN
    nnn = np.abs(dm - D_NNN) < 1e-3 * D_NN
    H = np.zeros((n, n), complex)
    H[nn] = -T1
    for i in range(n):
        for j in range(n):
            if not nnn[i, j]:
                continue
            mids = np.where(nn[i] & nn[j])[0]
            if len(mids) == 0:
                continue
            m = mids[0]
            v1, v2 = pos[m] - pos[i], pos[j] - pos[m]
            nu = np.sign(v1[0] * v2[1] - v1[1] * v2[0])
            H[i, j] = -t2 * np.exp(1j * PHI_SIGN * PHI * nu)
    w, v = np.linalg.eigh(H)
    nocc = n // 2
    assert w[nocc] - w[nocc - 1] > 1e-6, "gap closed at half filling -- wrong t2 regime"
    occ = v[:, :nocc]
    return occ @ occ.conj().T


def im_loop(rho: np.ndarray, d: int, a: int, c: int) -> float:
    """Signed loop weight Im b_{d,a,c} for the traversal d -> a -> c -> d."""
    return float(np.imag(rho[d, a] * rho[a, c] * rho[c, d]))


def omega_per_site(pos: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Per-site curvature Omega^{xy}(d)/2pi (pinned 2026-07-02 marker form).

    C(d) = 2*pi * sum_{a,c} (x_a y_c - y_a x_c) Im(rho_da rho_ac rho_cd).
    Dimensionless; sums to zero exactly on ANY open-boundary system and
    converges to the Chern number per site in a periodic bulk. This is the
    reading rule: 'sum over all three-point loops through a site = its color'.
    """
    x, y = pos[:, 0], pos[:, 1]
    n = len(pos)
    C = np.zeros(n)
    for d in range(n):
        # why explicit O(n^2) loop, not broadcasting: clarity beats micro-speed
        # (n = 96 worst case, ~0.5 s total), and the index order (d,a,c) maps
        # 1:1 onto the formula.
        acc = 0.0
        for a in range(n):
            for c in range(n):
                acc += (x[a] * y[c] - y[a] * x[c]) * np.imag(rho[d, a] * rho[a, c] * rho[c, d])
        C[d] = 2.0 * np.pi * acc
    return C


def oriented(tri, rho):
    """Return the traversal order of `tri` with Im b > 0 (physical circulation)."""
    d, a, c = tri
    return (d, a, c) if im_loop(rho, d, a, c) > 0 else (d, c, a)


# ---------------------------------------------------------------------------
# Drawing helpers (Fig. 1 visual language)
# ---------------------------------------------------------------------------
def draw_bonds(ax, pos, pairs, lw=1.6):
    for i, j in pairs:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=C_INK, lw=lw, zorder=1, solid_capstyle="round")


def _text_color_for(fill):
    """Ink on light fills, white on dark fills (indices must stay legible on
    the deep ends of the coolwarm scale)."""
    r, g, b = matplotlib.colors.to_rgb(fill)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return C_INK if lum > 0.55 else "white"


def draw_nodes(ax, pos, fills, ms, labels=None, fs=FS_IDX_A):
    # why mec='none' (flat disks): Gabi's reference figures draw nodes as
    # flat colored circles WITHOUT dark rims (benzene_coronene.png,
    # Bond_Index_Part.png) -- the rimmed nodes of earlier rounds were off-style
    for i, p in enumerate(pos):
        ax.plot(p[0], p[1], "o", ms=ms, mfc=fills[i], mec="none", zorder=4)
        if labels is not None:
            ax.text(p[0], p[1], labels[i], ha="center", va="center",
                    fontsize=fs, color=_text_color_for(fills[i]), zorder=5)


def draw_loop(ax, pos, order, r_trim, lw=1.1, label=None):
    """Oriented three-center loop: STRAIGHT, thin, black arrows leg by leg.

    why straight/thin/black (Raquel 2026-07-03): the loop must read as the
    computational object b_{i,j,k} -- three directed coherences between
    numbered sites -- not as a decorative swirl. Optional b_{i,j,k} label at
    the loop centroid (chords of a hexagon pass 0.5 d from every other node,
    so straight legs never graze a node at the trims used here).
    """
    k = len(order)
    for s in range(k):
        p = np.asarray(pos[order[s]], float)
        q = np.asarray(pos[order[(s + 1) % k]], float)
        u = (q - p) / np.hypot(*(q - p))
        p2, q2 = p + u * r_trim, q - u * r_trim
        ax.add_patch(FancyArrowPatch(
            p2, q2, arrowstyle="-|>", mutation_scale=9, color=C_INK, lw=lw,
            shrinkA=0, shrinkB=0, zorder=3))
    if label is not None:
        cen = pos[list(order)].mean(axis=0)
        # why the white backing: in the dense coronene panel the label is
        # wider than the triangle inradius and grazes the legs (sixth render)
        ax.text(cen[0], cen[1], label, ha="center", va="center",
                fontsize=FS_LOOP, color=C_INK, zorder=3.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))


def frame(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


VSCALE = 1.0  # shared color scale saturates at +-1 (why: the natural unit is
              # the Chern number; matches the reference figure's -1..+1 bar.
              # Extreme flake-edge sites (|C| up to 1.74) clip -- letting them
              # set the scale washes the +0.49 bulk to near-white, 8th render)


def omega_fills(C):
    """Node colors from per-site Omega^{xy} on the SHARED symmetric scale.

    why one scale for all panels (Raquel round 2): the colorbar at right is
    the same bar for benzene, coronene, and the flake -- colors are directly
    comparable across panels (benzene = neutral, exactly zero).
    """
    return [CMAP(0.5 + 0.5 * float(np.clip(c, -VSCALE, VSCALE)) / VSCALE)
            for c in C]


# ---------------------------------------------------------------------------
# Panels (draw only -- all ED/marker data computed once in build_figure)
# ---------------------------------------------------------------------------
def panel_benzene(ax, pos, rho, C):
    frame(ax, (-1.85, 1.85), (-2.55, 2.15))
    nn = [(i, (i + 1) % 6) for i in range(6)]
    draw_bonds(ax, pos, nn, lw=1.8)
    order = oriented((2, 0, 4), rho)
    draw_loop(ax, pos, order, r_trim=0.24, label=r"$b_{%d,%d,%d}$" % order)
    draw_nodes(ax, pos, omega_fills(C), ms=13,
               labels=[str(i) for i in range(6)], fs=FS_IDX_A)
    return order


def panel_coronene(ax, pos, rho, C):
    frame(ax, (-3.45, 3.45), (-4.45, 3.80))

    dm = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    nn = [(i, j) for i in range(24) for j in range(i + 1, 24)
          if abs(dm[i, j] - D_NN) < 1e-3]
    draw_bonds(ax, pos, nn, lw=1.5)

    # elementary curvature triangles = mutually-NNN triples (equilateral,
    # side sqrt(3) d -> each circumscribes one hexagon center); pick the
    # strongest hub loop and the strongest rim loop as representatives
    tris = [(i, j, k) for i in range(24) for j in range(i + 1, 24)
            for k in range(j + 1, 24)
            if abs(dm[i, j] - D_NNN) < 1e-3 and abs(dm[j, k] - D_NNN) < 1e-3
            and abs(dm[i, k] - D_NNN) < 1e-3]
    w = {t: abs(im_loop(rho, *t)) for t in tris}
    cen = {t: np.linalg.norm(pos[list(t)].mean(axis=0)) for t in tris}
    # why the |cen - sqrt(3)d| filter: coronene's outer hexagon centers sit at
    # sqrt(3) d; without it the "rim" pick can circumscribe an EMPTY hexagon
    # position outside the molecule and its legs tangle across the rim bonds
    hub = max((t for t in tris if cen[t] < 0.5 * D_NN), key=w.get)
    rim = max((t for t in tris if abs(cen[t] - D_NNN) < 0.1 * D_NN), key=w.get)
    for t in (hub, rim):
        o = oriented(t, rho)
        draw_loop(ax, pos, o, r_trim=0.24, label=r"$b_{%d,%d,%d}$" % o)

    draw_nodes(ax, pos, omega_fills(C), ms=11,
               labels=[str(i) for i in range(24)], fs=FS_IDX_B)


def panel_flake(ax, pos, hexes, rho, C):
    """Haldane flake, OPEN boundaries (Raquel round 2: 'panel (c) was better
    with the open boundaries'): bulk positive, edge negative, sum exactly 0.
    The bottom note carries the periodic statement (PBC: Omega/2pi = +1)."""
    frame(ax, (-6.8, 6.8), (-8.7, 7.0))

    n = len(pos)
    dm = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    nn = [(i, j) for i in range(n) for j in range(i + 1, n)
          if abs(dm[i, j] - D_NN) < 1e-3]
    draw_bonds(ax, pos, nn, lw=1.0)

    # shade ONE hexagonal cell (the per-cell restriction the caption names)
    ring0 = min(hexes, key=lambda h: np.linalg.norm(h[0]))[1]
    ax.add_patch(Polygon(pos[ring0], closed=True, facecolor=C_CELL,
                         edgecolor="none", zorder=0))

    # oriented loops on the sqrt(3) x sqrt(3) hexagon subset near the center:
    # why not every hexagon: adjacent loops share NNN sites and the arrows
    # pile up into clutter (first render); orientation is taken from the
    # ACTUAL flake projector, so the drawn sense is the computed one.
    for c, ring in hexes:
        # invert c = m*a1 + n*a2 (a1 = D_NNN(1,0), a2 = D_NNN(1/2, sqrt3/2))
        n_idx = round(2.0 * c[1] / (np.sqrt(3.0) * D_NNN))
        m_idx = round(c[0] / D_NNN - n_idx / 2.0)
        if np.linalg.norm(c) > 3.2 * D_NN or (m_idx - n_idx) % 3 != 0:
            continue
        tri = oriented((ring[0], ring[2], ring[4]), rho)
        draw_loop(ax, pos, tri, r_trim=0.22, lw=0.9)

    draw_nodes(ax, pos, omega_fills(C), ms=4.6)


# ---------------------------------------------------------------------------
def build_figure():
    # --- data first: shared color normalization needs all three panels ------
    b_pos = benzene_positions()
    b_rho = haldane_rho(b_pos, T2)
    b_C = omega_per_site(b_pos, b_rho)

    c_pos = coronene_positions()
    c_rho = haldane_rho(c_pos, T2)
    c_C = omega_per_site(c_pos, c_rho)

    f_pos, f_hexes = hexagonal_flake(FLAKE_RCUT)
    f_rho = haldane_rho(f_pos, T2)
    f_C = omega_per_site(f_pos, f_rho)
    f_r = np.linalg.norm(f_pos, axis=1)

    # --- gates (Studio-anchored known limits) -------------------------------
    b204 = im_loop(b_rho, 2, 0, 4)
    assert abs(abs(b204) - IM_B204_REF) < 2e-3, f"G1 fail: |Im b_204| = {abs(b204):.4f}"
    assert abs(b_C.sum()) < 1e-9 and np.all(np.abs(b_C) < 1e-9), "G2 fail: benzene Omega(d) != 0"
    assert abs(c_C.sum()) < 1e-9, f"G3 fail: coronene sum = {c_C.sum():+.2e}"
    central = f_C[f_r < 1.2].mean()
    assert central > 0 and abs(f_C.sum()) < 1e-8, \
        f"G4 fail: flake central {central:+.3f}, sum {f_C.sum():+.2e}"

    # --- layout: full-width triptych + one shared colorbar at right ---------
    fig = plt.figure(figsize=(7.6, 3.2))
    gs = fig.add_gridspec(1, 3, left=0.005, right=0.925, top=0.99, bottom=0.01,
                          wspace=0.04, width_ratios=[0.9, 1.05, 1.15])
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    order = panel_benzene(axs[0], b_pos, b_rho, b_C)
    panel_coronene(axs[1], c_pos, c_rho, c_C)
    panel_flake(axs[2], f_pos, f_hexes, f_rho, f_C)

    # shared colorbar: SAME bar for all three panels (Raquel round 2)
    # why cax at x=0.925: the rotated label needs ~0.05 fig-width to its
    # right -- at 0.945 it clipped off the figure edge (8th render)
    cax = fig.add_axes([0.925, 0.20, 0.015, 0.62])
    sm = plt.cm.ScalarMappable(cmap=CMAP,
                               norm=matplotlib.colors.Normalize(-VSCALE, VSCALE))
    cb = fig.colorbar(sm, cax=cax, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=8)
    cb.set_label(r"$\Omega^{xy}(d)/2\pi$", fontsize=9)

    # panel labels + notes in FIGURE coordinates on common baselines:
    # equal-aspect box shrinkage differs per panel, so axes-coordinate text
    # ends up at visibly different heights (fourth render). No panel titles
    # (Raquel round 1) -- molecule names live in the caption.
    notes = [
        [r"$\mathrm{Im}\,b_{%d,%d,%d}=%+.3f$" % (*order, im_loop(b_rho, *order)),
         r"$\Omega^{xy}(d)=0$ on every site"],
        [r"local $\Omega^{xy}(d)\neq 0$",
         r"molecular sum $=0$"],
        [r"open boundaries: sum $=0$",
         r"PBC: $\Omega^{xy}/2\pi=+1$"],
    ]
    # panel labels "a)" -- single parenthesis, regular weight, matching Gabi's
    # benzene_coronene.png / Bond_Index_Part.png (not bold "(a)")
    for ax, lab, lines in zip(axs, ["a)", "b)", "c)"], notes):
        ax.apply_aspect()  # why: get_position() is stale until aspect applied
        bb = ax.get_position()
        fig.text(bb.x0 + 0.012, 0.965, lab, ha="left", va="top",
                 fontsize=FS_LABEL, color=C_INK)
        fig.text(0.5 * (bb.x0 + bb.x1), 0.025, "\n".join(lines), ha="center",
                 va="bottom", fontsize=FS_NOTE, color=C_INK, linespacing=1.35)

    # for the caption: extremes on the printed shared color scale
    print("coronene Omega^xy(d)/2pi: min %+.3f max %+.3f" % (c_C.min(), c_C.max()))
    print("flake    Omega^xy(d)/2pi: central %+.3f edge-min %+.3f (sum %+.1e)"
          % (central, f_C.min(), f_C.sum()))
    return fig

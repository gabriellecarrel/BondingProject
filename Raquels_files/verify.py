"""Verify delivered files, probability conservation, and spectral sum rules."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT / 'file_manifest.json').read_text())
    for filename, expected in manifest.items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'File changed relative to delivered package: {filename}')
    for mat, mesh in [('NaCl', 24), ('Diamond', 24), ('CsI3', 12), ('GeTe', 24)]:
        z = np.load(ROOT / f'figure_3/data/bonding_mode_realspace_20260919_v2_{mat}.npz')
        g = np.load(ROOT / f'figure_3/data/bonding_modes_20260919_v1_{mat}_mesh{mesh}.npz')
        p, keys = z['probabilities'], z['process_keys']
        assert p.min() >= 0
        np.testing.assert_allclose(p.sum(), 1, atol=3e-12)
        np.testing.assert_allclose(np.bincount(keys[:, 0], weights=p), z['occupied_marginal'], atol=3e-12)
        np.testing.assert_allclose(np.bincount(keys[:, 1], weights=p), z['empty_marginal'], atol=3e-12)
        distances = np.linalg.norm(z['nuclear_displacement_A'], axis=1)
        order = np.argsort(distances)
        r90 = distances[order[np.searchsorted(np.cumsum(p[order]), .9)]]
        np.testing.assert_allclose(r90, z['r90_A'], atol=1e-9)
        spectrum = np.loadtxt(ROOT / f'figure_3/tables/{mat}_spectrum.csv', delimiter=',', skiprows=1)
        ev = np.maximum(g['eigenvalues'], 0)
        np.testing.assert_allclose(spectrum[:, 2], ev / ev.sum(), atol=1e-15)
        print(f'{mat}: probability=1, r90={r90:.6f} angstrom')

    data = np.load(ROOT / 'figure_4/data/spectra.npz')
    curves = np.loadtxt(ROOT / 'figure_4/tables/spectra_display.csv', delimiter=',', skiprows=1)
    scale = 3 * np.sqrt(3) / (4 * np.pi)
    for j, case in enumerate(('benzene', 'coronene', 'flake864', 'bulk'), 1):
        weights = data[case + '_w']
        expected = 1 if case == 'bulk' else 0
        np.testing.assert_allclose(weights.sum(), expected, atol=1e-6)
        np.testing.assert_allclose(np.trapezoid(curves[:, j], curves[:, 0]), scale * weights.sum(), atol=1e-10)
        print(f'{case}: unbroadened C-normalized sum={weights.sum():.12g}')
    drawing = np.load(ROOT / 'figure_4/tables/drawing_data.npz')
    for case in ('benzene', 'coronene', 'flake96'):
        P = drawing[case + '_projector']
        xy = drawing[case + '_positions']
        np.testing.assert_allclose(P, P.conj().T, atol=1e-12)
        np.testing.assert_allclose(P @ P, P, atol=1e-12)
        # Independent operator trace from the exported projector and positions.
        X, Y = np.diag(xy[:, 0]), np.diag(xy[:, 1])
        qxy = np.trace(P @ X @ (np.eye(len(P)) - P) @ Y @ P)
        assert abs(qxy.imag) < 1e-10
        assert abs(drawing[case + '_marker'].sum()) < 1e-9
    print(f'PASS: {len(manifest)} file hashes and all numerical checks')

if __name__ == '__main__':
    verify()

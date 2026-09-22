"""Portable subset of the original lab style needed by these two figures."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
BONDING_INK = "#2b2b2b"
BONDING_CELL = "#e4e9ee"
def apply_style(name="bonding", usetex=False):
    if name != "bonding": raise ValueError(name)
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(Path(__file__).with_name("bonding.mplstyle"))
    mpl.rcParams["text.usetex"] = False
    return plt

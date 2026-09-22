"""Linked panels for the same atom-pair response modes; candidate, not production."""
from pathlib import Path
import hashlib
import json
import socket
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'
ATT=ROOT/'output'
STYLE=ROOT.parent/'style'
sys.path.insert(0,str(STYLE))
from lab_style import apply_style
apply_style('bonding',usetex=False)
FONT=20
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':FONT,
    'axes.labelsize':FONT,'axes.titlesize':FONT,'xtick.labelsize':FONT,
    'ytick.labelsize':FONT,'legend.fontsize':FONT,'mathtext.fontset':'cm',
    'text.usetex':False,'svg.fonttype':'none','axes.spines.top':False,
    'axes.spines.right':False})
MATS=('NaCl','Diamond','CsI3','GeTe')
LABEL={'NaCl':'NaCl','Diamond':'Diamond','CsI3':r'CsI$_3$','GeTe':'GeTe'}
COLOR={'NaCl':'#0077bb','Diamond':'#009988','CsI3':'#ee7733','GeTe':'#cc3311'}
STEM='bonding_modes_20260919_r1'
MESH={'NaCl':24,'Diamond':24,'CsI3':12,'GeTe':24}


def input_path(material):
    return OUT/f'bonding_modes_20260919_v1_{material}_mesh{MESH[material]}.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

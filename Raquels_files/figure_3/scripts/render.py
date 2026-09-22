"""Matched main-text and supplementary figures from the same saved SVD modes."""
from pathlib import Path
import json
import socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullLocator
from bonding_modes_figure_20260919 import OUT, ATT, MATS, MESH, LABEL, COLOR, FONT, sha, input_path
from bonding_full_mode_glyphs_20260919_r10 import draw

STEM='bonding_matched_mode_figures_20260919_r3'


def inputs():
    result={}
    for m in MATS:
        p=OUT/f'bonding_mode_realspace_20260919_v2_{m}.npz'
        z=dict(np.load(p));g=dict(np.load(input_path(m)))
        meta=json.loads(input_path(m).with_suffix('.json').read_text())
        assert np.all(g['eigenvalues']>=-1e-10*g['eigenvalues'][0])
        np.testing.assert_allclose(z['probabilities'].sum(),1,atol=3e-12)
        W=z['orbital_pair_probability'];oa=z['orbital_to_atom']
        A=np.zeros((len(z['atom_symbols']),)*2)
        np.add.at(A,(oa[:,None],oa[None,:]),W)
        B=np.zeros_like(A)
        np.add.at(B,(z['process_keys'][:,0],z['process_keys'][:,1]),z['probabilities'])
        np.testing.assert_allclose(A,B,atol=3e-12)
        np.testing.assert_allclose(A.sum(axis=1),z['occupied_marginal'],atol=3e-12)
        np.testing.assert_allclose(A.sum(axis=0),z['empty_marginal'],atol=3e-12)
        np.testing.assert_allclose(A.sum(),1,atol=3e-12)
        result[m]=(z,g,meta,A,p)
    return result


def export(fig,name):
    from PIL import Image
    output=Path(__file__).resolve().parents[1]/'output'
    output.mkdir(exist_ok=True)
    png=output/'figure_3.png'
    # 600 dpi at 7.1 inches in the manuscript; avoid the 67 MB vector glyphs.
    fig.savefig(png,dpi=600*7.1/15.4,facecolor='white')
    pdf=output/'figure_3.pdf'
    with Image.open(png) as im: im.convert('RGB').save(pdf,resolution=600.0)
    plt.close(fig)
    return [str(png),str(pdf)]


def main_figure(data,records):
    fig=plt.figure(figsize=(15.4,10.4))
    grid=fig.add_gridspec(2,3,left=.078,right=.978,bottom=.095,top=.93,
                         width_ratios=[1,1.12,1.12],hspace=.38,wspace=.28)
    ax=fig.add_subplot(grid[0,0]);bx=fig.add_subplot(grid[1,0])
    for m in MATS:
        z,g,meta,A,path=data[m]
        ev=np.maximum(g['eigenvalues'],0);fraction=ev/ev.sum()
        rank=np.arange(1,len(ev)+1);positive=fraction>0
        ax.plot(rank[positive],fraction[positive],color=COLOR[m],lw=2.5,label=LABEL[m])
        ids=z['mode_indices']
        ax.scatter(ids+1,fraction[ids],s=70,facecolors='white',edgecolors=COLOR[m],lw=2,zorder=5)
        # Aggregate equal separations for an exact step CDF; all weights set its normalization.
        r=np.linalg.norm(z['nuclear_displacement_A'],axis=1)
        rr,inv=np.unique(np.round(r,10),return_inverse=True)
        mass=np.bincount(inv,weights=z['probabilities'])
        cumulative=np.cumsum(mass)
        bx.step(np.r_[0,rr],100*np.r_[0,cumulative],where='post',color=COLOR[m],lw=2.5)
        r90=float(z['r90_A'])
        bx.scatter(r90,90,s=65,facecolors='white',edgecolors=COLOR[m],lw=2,zorder=5)
        n90=int(np.searchsorted(np.cumsum(fraction),.9)+1)
        records[m]={'mode_indices_zero_based':ids.tolist(),
                    'selected_mode_metric_fraction':float(g['mode_metric_weights'][ids].sum()/g['mode_metric_weights'].sum()),
                    'spectrum_normalization':'sum of eigenvalues of represented restricted Gram matrix',
                    'modes_for_90_percent_represented_trace':n90,
                    'r90_A':r90,'range_probability_within_40_A':float(z['probabilities'][r<=40].sum()),
                    'atom_pair_probability':A.tolist(),'atom_labels':z['atom_symbols'].astype(str).tolist(),
                    'realspace_source':str(path),'realspace_sha256':sha(path),
                    'spectrum_source':str(input_path(m)),'spectrum_sha256':sha(input_path(m)),
                    'restricted_over_full_metric':meta['selected_metric_fraction'],
                    'full_trace':meta['full_trace'],'represented_trace':float(ev.sum())}
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.9,4000);ax.set_ylim(1e-8,1)
    ax.set_xticks([1,10,100,1000]);ax.set_xticklabels(['1','10','100','1000'])
    ax.set_yticks([1,1e-2,1e-4,1e-6,1e-8]);ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Mode index');ax.set_ylabel('Eigenvalue fraction')
    ax.set_title('(a)  Mode spectrum',loc='left',fontsize=FONT,fontweight='normal',pad=16)
    ax.legend(frameon=False,loc='lower left',fontsize=FONT,labelspacing=.18,handlelength=1.1)
    bx.set_xlim(0,40);bx.set_ylim(0,103);bx.set_yticks([0,50,90,100]);bx.set_xticks([0,20,40])
    bx.set_xlabel('Pair separation (Å)');bx.set_ylabel('Cumulative mode weight (%)')
    bx.set_title('(b)  Spatial extent',loc='left',fontsize=FONT,fontweight='normal',pad=16)
    for i,m in enumerate(MATS):
        axis=fig.add_subplot(grid[i//2,1+i%2])
        records[m]['glyph']=draw(axis,data[m][0],m,zoom=True)
        axis.set_title(f'({chr(99+i)})  '+LABEL[m],loc='left',fontsize=FONT,fontweight='normal',pad=16)
    return export(fig,'main')


if __name__ == '__main__':
    data=inputs(); records={}; paths=main_figure(data,records)
    output=Path(__file__).resolve().parents[1]/'output'
    (output/'render_checks.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Figure 3 written to', output)

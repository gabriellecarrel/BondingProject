"""Replot manuscript Figure 4 from saved lines and small drawing projectors."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import loops
m=loops.m
ns=vars(loops)
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'output'
ROOT.mkdir(exist_ok=True)
FIGSTEM='figure_4'
AC=3*np.sqrt(3)/2
Q_SCALE=AC/(2*np.pi)
EGRID=np.linspace(0,12,12001)
ETA=.075

def panel_flake_grouped(ax,pos,hexes,P,C):
    """Six sign-verified finite-projector examples, grouped by visible hexagon."""
    m.frame(ax,(-6.8,6.8),(-8.7,7.0));ns['skeleton'](ax,pos,1.0)
    groups=[('positive',[-np.sqrt(3)/2,1.5],[[26,25,24],[38,39,40]],
             [1,1,np.sqrt(3)],1),
            ('negative',[3*np.sqrt(3)/2,-1.5],[[70,69,80],[68,81,82]],
             [1,np.sqrt(3),2],-1),
            ('star',[-np.sqrt(3)/2,-1.5],[[37,35,52],[36,51,53]],
             [np.sqrt(3)]*3,1)]
    records=[]
    for name,center,triangles,sides,sign in groups:
        center=np.asarray(center)
        matching=[ring for c,ring in hexes if np.linalg.norm(c-center)<1e-8]
        assert len(matching)==1
        for ids in triangles:
            assert set(ids)<=set(matching[0])
            q=pos[ids];u=q[1]-q[0];v=q[2]-q[0]
            lengths=np.sort(np.linalg.norm(q-np.roll(q,1,axis=0),axis=1))
            assert np.max(abs(lengths-sides))<1e-8
            rec=ns['loop'](ax,pos,P,ids,.20,lw=1.0)
            assert sign*rec['weight_raw']>1e-8
            order=list(map(int,m.oriented(ids,P)))
            a,b,c=ids;product=P[a,b]*P[b,c]*P[c,a]
            rec.update(group=name,hexagon_center=center.tolist(),
                       hexagon_ring=list(map(int,matching[0])),
                       side_lengths=lengths.tolist(),
                       signed_area=float((u[0]*v[1]-u[1]*v[0])/2),
                       projector_product=[float(product.real),float(product.imag)],
                       arrow_order=order)
            records.append(rec)
    m.draw_nodes(ax,pos,m.omega_fills(C),ms=4.6)
    audit=dict(N=len(pos),boundary='open',triangles=records,
               normalization='w=8 pi Axy Im(Pab Pbc Pca), in a^2; a is nearest-neighbor bond length',
               parameters=dict(t1=1,t2=1,phi=float(np.pi/2),phase_sign=-1,mass=0,half_filling=True),
               interpretation='Selected examples, not a complete decomposition or cancellation group. Same cell means a visible hexagon.',
               site_sum=float(C.sum()))
    ns['records']['haldane']=audit
    (ROOT/(FIGSTEM+'_loops.json')).write_text(json.dumps(audit,indent=2))

def broaden_grid(de,w,grid,eta):
    dx=grid[1]-grid[0];p=de/dx;lo=np.floor(p).astype(int);f=p-lo
    assert lo.min()>=0 and lo.max()+1<len(grid)
    hist=(np.bincount(lo,weights=w*(1-f),minlength=len(grid))+
          np.bincount(lo+1,weights=w*f,minlength=len(grid)))/dx
    return gaussian_filter1d(hist,eta/dx,mode='mirror')

def broaden(de,w,eta=ETA):
    # Gaussian display only: linear deposit then reflected kernel at E=0.
    # Reflection preserves area on E>=0; exact sums use unbroadened lines.
    return broaden_grid(de,w,EGRID,eta)

def broadening_checks(data):
    result={}
    for name in data:
        if not name.endswith('_de'):continue
        key=name[:-3]
        for eta in (.025,.055,.10):
            curve=broaden(data[key+'_de'],data[key+'_w'],eta)
            val=float(np.trapezoid(curve,EGRID));exact=float(data[key+'_w'].sum())
            assert abs(val-exact)<1e-10
            result[f'{key}_eta{eta}']=dict(integral=val,exact=exact)
    return result

def sign_fill(ax,y,alpha=.18):
    ax.fill_between(EGRID,y,0,where=y>=0,interpolate=True,color=ns['RED'],alpha=alpha,lw=0,rasterized=True,zorder=.7)
    ax.fill_between(EGRID,y,0,where=y<=0,interpolate=True,color=ns['BLUE'],alpha=alpha,lw=0,rasterized=True,zorder=.7)

def qcurve(data,key):
    """-2 Im Qxy(E), Q=T/Ne. Units dNN^2/t; Q_SCALE*C(E)."""
    return Q_SCALE*broaden(data[key+'_de'],data[key+'_w'])

def render(data,report,layout_only=False):
    m.panel_benzene=ns['panel_benzene'];m.panel_coronene=ns['panel_coronene'];m.panel_flake=panel_flake_grouped
    fig=m.build_figure()
    # Keep the existing top-row dimensions exactly, adding space below.
    old_height=3.2;new_height=5.5;fig.set_size_inches(7.6,new_height)
    shift=(new_height-old_height)/new_height;scale=old_height/new_height
    for ax in fig.axes:
        p=ax.get_position(original=True);ax.set_position([p.x0,shift+scale*p.y0,p.width,scale*p.height])
        for t in list(ax.texts):
            if t.get_text().lower() in ['positive','negative']: t.remove()
    for t in fig.texts:
        x,y=t.get_position();t.set_position((x,shift+scale*y))
        s=t.get_text()
        if s.startswith('$\\mathrm{Im'):t.set_text('Full site sum $=0$')
        elif s.startswith('local '):t.set_text('Local $M_i\\neq0$\nMolecular sum $=0$')
        elif s.startswith('open boundaries:'):t.set_text('Open boundaries: sum $=0$\nPeriodic bulk: $C=+1$')
    cax=fig.axes[-1];cax.clear()
    sm=m.plt.cm.ScalarMappable(cmap=m.CMAP,norm=matplotlib.colors.Normalize(-m.VSCALE,m.VSCALE))
    cb=fig.colorbar(sm,cax=cax,ticks=[-1,0,1],extend='both');cb.ax.tick_params(labelsize=8)
    cb.set_label(r'$M_i/a^2$',fontsize=9)
    gs=fig.add_gridspec(1,3,left=.080,right=.920,bottom=.093,top=.343,wspace=.22)
    colors=['#0077bb','#cc3311','#009988']
    for j,key in enumerate(['benzene','coronene','flake864']):
        ax=fig.add_subplot(gs[j]);ax.axhline(0,color='.65',lw=.65)
        curve=qcurve(data,key);sign_fill(ax,curve,.12 if j==2 else .24)
        ax.plot(EGRID,curve,color=colors[j],lw=1.35,label=r'$N=864,\ C=0$' if j==2 else None)
        ax.set_xlim(0,11);ax.set_xticks([0,5,10]);ax.tick_params(labelsize=8)
        ax.set_xlabel(r'$E/t$',fontsize=9,labelpad=1)
        ax.set_title(['d) Benzene','e) Coronene','f) Haldane'][j],loc='left',fontsize=10,pad=7)
        if j==0:ax.set_ylabel(r'$-2\,\mathrm{Im}\,\mathcal{Q}^{xy}(E)\ (a^2/t)$',fontsize=9,labelpad=3)
        if j==2:
            bulk_curve=qcurve(data,'bulk')
            sign_fill(ax,bulk_curve,.14)
            ax.plot(EGRID,bulk_curve,color='black',lw=1.5,ls='-',label=r'Bulk, $C=+1$')
            ax.set_xlim(-.45,11)
            ax.set_ylim(1.12*min(curve.min(),bulk_curve.min()),1.18*max(curve.max(),bulk_curve.max())+.03)
            ax.legend(fontsize=8,frameon=False,loc='lower right',bbox_to_anchor=(1,.06),
                      handlelength=1.7,labelspacing=.6,handletextpad=.6)
            # Point to the upper half of the peak's right flank, leaving labels below.
            trough=np.flatnonzero(EGRID<.75)[np.argmin(curve[EGRID<.75])]
            flank=np.flatnonzero((EGRID>=EGRID[trough])&(EGRID<.75))
            edge_idx=flank[np.argmin(abs(curve[flank]-.45*curve[trough]))]
            ax.annotate('edge states',xy=(EGRID[edge_idx],curve[edge_idx]),
                        xytext=(.25,.59),textcoords='axes fraction',ha='left',va='center',fontsize=7,
                        arrowprops=dict(arrowstyle='->',lw=.8,color='#333333',shrinkB=4))
        else:
            ax.text(.98,.07,r'$C=0$',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
    for ext in ['png','pdf']:fig.savefig(ROOT/(FIGSTEM+'.'+ext),dpi=600,facecolor='white')
    plt.close(fig)

def export_tables(data):
    tables=ROOT/'tables'
    tables.mkdir(exist_ok=True)
    curves=[qcurve(data,k) for k in ('benzene','coronene','flake864','bulk')]
    np.savetxt(tables/'spectra_display.csv',np.column_stack([EGRID]+curves),delimiter=',',comments='',
        header='E_over_t,benzene_minus2ImQxy,coronene_minus2ImQxy,flake864_minus2ImQxy,bulk_minus2ImQxy')
    drawing={}
    for key,pos in [('benzene',m.benzene_positions()),('coronene',m.coronene_positions()),('flake96',m.hexagonal_flake(m.FLAKE_RCUT)[0])]:
        P=m.haldane_rho(pos,m.T2);marker=m.omega_per_site(pos,P)
        drawing.update({key+'_positions':pos,key+'_projector':P,key+'_marker':marker})
        np.savetxt(tables/(key+'_sites.csv'),np.c_[np.arange(len(pos)),pos,marker],delimiter=',',comments='',header='site_index_0based,x_over_a,y_over_a,M_i_over_a_squared')
    np.savez_compressed(tables/'drawing_data.npz',**drawing)

if __name__ == '__main__':
    data=dict(np.load(BASE/'data/spectra.npz'))
    np.testing.assert_array_equal(EGRID,data['Egrid'])
    checks=broadening_checks(data)
    for key in ('benzene','coronene','flake864','bulk'):
        target=1 if key=='bulk' else 0
        np.testing.assert_allclose(data[key+'_w'].sum(),target,atol=1e-6)
        np.testing.assert_allclose(np.trapezoid(qcurve(data,key),EGRID),Q_SCALE*data[key+'_w'].sum(),atol=1e-10)
    render(data,{},layout_only=True)
    export_tables(data)
    (ROOT/'render_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (ROOT/'loop_drawings.json').write_text(json.dumps(ns['records'],indent=2)+'\n')
    (ROOT/'tables/loop_drawings.json').write_text(json.dumps(ns['records'],indent=2)+'\n')
    print('Figure 4 written to',ROOT)

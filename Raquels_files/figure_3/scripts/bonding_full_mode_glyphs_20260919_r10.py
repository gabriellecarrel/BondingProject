"""Full periodic mode glyphs and magnified views; no probability cutoff."""
from pathlib import Path
import json, hashlib, socket, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba, LogNorm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'
sys.path.insert(0,str(Path(__file__).parent))
from bonding_modes_figure_20260919 import COLOR, LABEL, MATS, MESH, FONT, sha

STEM='bonding_full_periodic_modes_20260919_r10'
# Exact flat colors and legend-disc diameters measured from the v14 PNG.
DISC_DIAMETER_PX={'Na':195.,'Cl':106.,'C':232.,'Cs':197.,'I':91.,'Ge':144.,'Te':165.}
ELEMENT_COLOR={'Na':'#e9cd35','Cl':'#27b14a','C':'#007f92','Cs':'#2aedb5',
               'I':'#792677','Ge':'#7d6d9e','Te':'#a1984c'}
STYLE_SOURCE='SVD_image_transitions.png'
STYLE_SHA256='ddc10c88615ac527244111f6d7ac312d124cf896e820925326e499c5a9a0dcc6'



def pairs(z,mesh):
    keys=z['process_keys'];p=z['probabilities']
    lookup={tuple(k):i for i,k in enumerate(keys)}
    neg=(-keys[:,2:]+mesh//2)%mesh-mesh//2
    rev=np.array([lookup[(int(B),int(A),*map(int,R))]
                  for (A,B),R in zip(keys[:,:2],neg)])
    np.testing.assert_array_equal(rev[rev],np.arange(len(rev)))
    reps=np.flatnonzero(np.arange(len(rev))<=rev)
    weights=p[reps]+np.where(reps==rev[reps],0,p[rev[reps]])
    np.testing.assert_allclose(weights.sum(),p.sum(),atol=2e-12)
    np.testing.assert_allclose(p.sum(),1,atol=2e-9)
    a=z['atom_frac'][keys[reps,0]]@z['lattice']
    b=(z['atom_frac'][keys[reps,1]]+keys[reps,2:])@z['lattice']
    return reps,rev,weights,a,b


def draw(ax,z,mat,zoom=False):
    reps,rev,weights,a,b=pairs(z,MESH[mat])
    keys=z['process_keys'];p=z['probabilities'];symbols=z['atom_symbols'].astype(str)
    az=np.deg2rad(-66 if mat=='CsI3' else (4 if mat=='Diamond' else 34))
    el=np.deg2rad(3 if mat=='Diamond' else 20)
    basis=np.array([[-np.sin(az),np.cos(az),0.],
                   [-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)]])
    aa=a@basis.T;bb=b@basis.T
    center=(z['atom_frac'][np.unique(keys[:,0])]@z['lattice']@basis.T).mean(axis=0)
    aa-=center;bb-=center
    span=max(np.ptp(np.r_[aa,bb],axis=0))
    bound=.56*span
    if zoom:
        # Magnify the central physical structure, never change the selected data.
        bound={'NaCl':6.,'Diamond':5.,'CsI3':12.,'GeTe':12.}[mat]
    # Compute marker edges in screen points, so arrows meet the circle rim
    # at any figure size or viewing scale. Restore the original endpoint-weight sizing.
    ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound)
    ax.set_aspect('equal');ax.apply_aspect()
    nodekeys=np.r_[np.c_[keys[reps,0],np.zeros((len(reps),3),int)],keys[reps,1:]]
    unique,inverse=np.unique(nodekeys,axis=0,return_inverse=True)
    nw=np.bincount(inverse,weights=np.r_[weights,weights])
    pos=(z['atom_frac'][unique[:,0]]+unique[:,1:])@z['lattice']@basis.T-center
    marker_sizes=4+70*np.sqrt(nw/nw.max())
    data_per_pt=(2*bound)/ax.bbox.width*ax.figure.dpi/72
    radii=(np.sqrt(marker_sizes)/2+.4)*data_per_pt
    ra=radii[inverse[:len(reps)]];rb=radii[inverse[len(reps):]]
    delta=bb-aa;length=np.linalg.norm(delta,axis=1)
    direction=np.divide(delta,length[:,None],out=np.zeros_like(delta),where=length[:,None]>1e-12)
    # Overlapping projected circles already cover the unresolved short shaft.
    available=np.maximum(length-ra-rb,0.)
    half=(aa+bb)/2
    start=np.where((available>0)[:,None],aa+ra[:,None]*direction,half)
    end=np.where((available>0)[:,None],bb-rb[:,None]*direction,half)
    rel=weights/weights.max();colors=np.tile(to_rgba(COLOR[mat]),(len(weights),1))
    colors[:,3]=np.sqrt(rel);order=np.argsort(weights)
    distinct=(reps!=rev[reps])&(available>1e-12)
    forward_head=np.where(distinct,np.minimum(.090*bound*(p[reps]/p.max())**.5,.36*available),0.)
    backward_head=np.where(distinct,np.minimum(.090*bound*(p[rev[reps]]/p.max())**.5,.36*available),0.)
    # One shared shaft joins the two head bases; it never runs under the tips.
    shaft_start=start+backward_head[:,None]*direction
    shaft_end=end-forward_head[:,None]*direction
    assert np.all(forward_head+backward_head<=available+1e-12)
    ax.add_collection(LineCollection(np.stack([shaft_start,shaft_end],axis=1)[order],
                      colors=colors[order],linewidths=(.1+3.2*np.sqrt(rel))[order],
                      capstyle='butt',zorder=1))
    arrow_start=np.r_[start[distinct],end[distinct]]
    arrow_end=np.r_[end[distinct],start[distinct]]
    headweight=np.r_[p[reps[distinct]],p[rev[reps[distinct]]]]
    delta=arrow_end-arrow_start;length=np.linalg.norm(delta,axis=1)
    unit=delta/length[:,None];normal=np.c_[-unit[:,1],unit[:,0]]
    size=np.r_[forward_head[distinct],backward_head[distinct]]
    vertices=np.stack([arrow_end,arrow_end-size[:,None]*unit+.60*size[:,None]*normal,
                                arrow_end-size[:,None]*unit-.60*size[:,None]*normal],axis=1)
    hc=np.tile(to_rgba(COLOR[mat]),(len(headweight),1));hc[:,3]=np.sqrt(headweight/p.max())
    ho=np.argsort(headweight)
    ax.add_collection(PolyCollection(vertices[ho],facecolors=hc[ho],edgecolors='none',zorder=5))
    elements=sorted(set(symbols[unique[:,0]]))
    for element in elements:
        mask=symbols[unique[:,0]]==element
        c=np.tile(to_rgba(ELEMENT_COLOR[element]),(mask.sum(),1))
        c[:,3]=np.sqrt(nw[mask]/nw.max())
        ec=np.tile(to_rgba('#333333'),(mask.sum(),1));ec[:,3]=c[:,3]
        ax.scatter(pos[mask,0],pos[mask,1],s=marker_sizes[mask],
                   facecolors=c,edgecolors='none',linewidths=0,zorder=4)
    # Same-site probability is retained explicitly as an area-weighted halo.
    onsite=(keys[reps,0]==keys[reps,1])&np.all(keys[reps,2:]==0,axis=1)
    ax.scatter(aa[onsite,0],aa[onsite,1],s=800*weights[onsite]/weights.max(),
               c=COLOR[mat],alpha=.25,edgecolors='none',zorder=2)
    if mat=='CsI3':
        cs=np.flatnonzero(symbols=='Cs')
        cspos=z['atom_frac'][cs]@z['lattice']@basis.T-center
        ax.scatter(cspos[:,0],cspos[:,1],s=50,facecolors=ELEMENT_COLOR['Cs'],edgecolors='none',linewidths=0,zorder=4)
    ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound)
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(f'({chr(97+MATS.index(mat))})  '+LABEL[mat],loc='left',pad=16,y=1.0,fontsize=FONT,fontweight='normal')
    # Element keys stay above the drawing, alongside the panel title.
    legend_elements={'NaCl':['Na','Cl'],'Diamond':['C'],'CsI3':['Cs','I'],'GeTe':['Ge','Te']}[mat]
    handles=[Line2D([],[],linestyle='none',marker='o',
                    markersize=np.sqrt(74),
                    markerfacecolor=ELEMENT_COLOR[e],markeredgecolor='none',
                    markeredgewidth=0,label=e) for e in legend_elements]
    ax.legend(handles=handles,loc='lower right',bbox_to_anchor=(1,1.015),
              borderaxespad=0,frameon=False,ncol=len(handles),fontsize=FONT,
              handlelength=.8,handletextpad=.35,columnspacing=.85)
    # A physical scale bar makes zoom and relative-coordinate extent explicit.
    bar=bound*.5;rounded=10**np.floor(np.log10(bar));bar=max(rounded,np.floor(bar/rounded)*rounded)
    x0=-.88*bound;y0=-.86*bound
    ax.plot([x0,x0+bar],[y0,y0],color='#333333',lw=2)
    ax.text(x0+bar/2,y0-.05*bound,f'{bar:g} Å',ha='center',va='top',fontsize=FONT)
    return {'saved_probability':float(p.sum()),'periodic_pair_orbits':len(reps),
            'rendered_probability':float(weights.sum()),'mode_indices':z['mode_indices'].tolist(),
            'onsite_probability':float(weights[onsite].sum()),'halo_definition':'A=B and R=0 right-mode probability, summed over Cartesian components and all orbital pairs on that atom',
            'halo_area_pt2':'800 same-site probability / maximum pair-or-onsite probability within material','display_window_A':[-bound,bound],
            'zoom_only':zoom,'probability_cutoff':None,'transverse_arrow_offset':0,'arrowhead_halfwidth_over_length':0.60,'arrowhead_length_factor':0.090,'arrowhead_max_gap_fraction':0.36,
            'arrow_shafts':'head base to head base, butt caps; point tips end at circle rims with 0.4 pt clearance',
            'boundary_condition':'periodic; Nyquist images identified modulo mesh',
            'circle_radius_convention':'Original weight-scaled dots: area in pt^2 = 4 + 70 sqrt(endpoint marginal / largest endpoint marginal); not depth or atomic radius',
            'atom_area_pt2':'4 + 70 sqrt(endpoint marginal / largest endpoint marginal)',
            'cesium_context_area_pt2':50,
            'disc_diameters_source_pixels':DISC_DIAMETER_PX,'style_source':str(STYLE_SOURCE),
            'style_source_sha256':STYLE_SHA256,'element_colors':ELEMENT_COLOR,
            'atom_outlines':False,'view_azimuth_degrees':float(np.rad2deg(az)),'view_elevation_degrees':float(np.rad2deg(el)),
            'shaft_width_pt':'0.1 + 3.2 sqrt(pair weight / maximum pair-or-onsite weight within material)',
            'shaft_opacity':'sqrt(pair weight / maximum pair-or-onsite weight within material)',
            'head_opacity':'sqrt(directional weight / maximum directional-or-onsite weight within material)',
            'atom_opacity':'sqrt(endpoint marginal / largest endpoint marginal)',
            'maximum_pair_or_onsite_probability':float(weights.max()),
            'source_sha256':sha(OUT/f'bonding_mode_realspace_20260919_v2_{mat}.npz')}

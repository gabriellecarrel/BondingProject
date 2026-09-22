#!/usr/bin/env python3
"""Preserve the original lattice/site-map composition; change only loop overlays.
Run on Studio. Original source, manuscript, and image assets are never overwritten.
"""
import importlib.util,itertools,json,hashlib,socket
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyArrowPatch,Polygon
import lattice as m
RED='#C73E1D';BLUE='#2E86AB';records={}

def weight(pos,P,ids):
 a,b,c=ids;u=pos[b]-pos[a];v=pos[c]-pos[a]
 return float(4*np.pi*(u[0]*v[1]-u[1]*v[0])*(P[a,b]*P[b,c]*P[c,a]).imag)

def lookup(pos,q):
 d=np.linalg.norm(np.asarray(q)[:,None,:]-pos[None,:,:],axis=-1)
 ids=d.argmin(1);assert d.min(1).max()<1e-9
 return ids.tolist()

def loop(ax,pos,P,ids,trim,lw=1.15):
 w=weight(pos,P,ids);color=RED if w>0 else BLUE
 order=m.oriented(ids,P)
 q=pos[list(order)]
 ax.add_patch(Polygon(q,facecolor=color,alpha=.055,edgecolor='none',zorder=1.5))
 for p,r in zip(q,np.roll(q,-1,axis=0)):
  u=(r-p)/np.linalg.norm(r-p)
  ax.add_patch(FancyArrowPatch(p+trim*u,r-trim*u,arrowstyle='->',mutation_scale=8,color=color,lw=lw,shrinkA=0,shrinkB=0,zorder=3))
 return dict(ids=list(map(int,ids)),weight_raw=w,positions=pos[list(ids)].tolist(),color=color)

def skeleton(ax,pos,lw):
 d=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
 m.draw_bonds(ax,pos,[(i,j) for i in range(len(pos)) for j in range(i+1,len(pos)) if abs(d[i,j]-1)<1e-8],lw=lw)

def panel_benzene(ax,pos,P,C):
 m.frame(ax,(-1.85,1.85),(-3.05,2.70))
 theta=np.pi/6;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 base=pos@rot.T
 upper=base+np.array([0.,1.30]);lower=base+np.array([0.,-1.30])
 pure_triangles=[[0,2,4],[1,3,5]];negative_triangle=[0,2,5]
 rr=[]
 for drawing,triangles,dy in [(upper,pure_triangles,1.30),(lower,[negative_triangle],-1.30)]:
  skeleton(ax,drawing,1.5)
  for t in triangles:
   item=loop(ax,drawing,P,t,.19,lw=1.1)
   assert abs(item['weight_raw']-weight(base,P,t))<1e-12
   item['display_shift_y']=dy;rr.append(item)
  m.draw_nodes(ax,drawing,m.omega_fills(C),ms=10.5,labels=[str(i) for i in range(6)],fs=6.0)
 assert rr[0]['weight_raw']>0 and abs(rr[0]['weight_raw']-rr[1]['weight_raw'])<1e-12
 assert rr[2]['weight_raw']<0
 ax.text(0,.08,'Positive',ha='center',va='top',fontsize=8,color=RED)
 ax.text(0,-2.54,'Negative',ha='center',va='top',fontsize=8,color=BLUE)
 total=sum(weight(base,P,t) for t in itertools.combinations(range(6),3))
 pure=sum(x['weight_raw'] for x in rr[:2]);assert abs(total)<1e-12
 records['benzene']=dict(triangles=rr,pure_geometric_sum=pure,mixed_geometric_sum=total-pure,total=total,rotation_degrees=30,drawings='Same six-site projector shown twice; upper positive, lower negative. The selected negative example does not alone cancel the positive pair.')
 return m.oriented((2,0,4),P)

def panel_coronene(ax,pos,P,C):
 m.frame(ax,(-3.45,3.45),(-4.45,3.80));skeleton(ax,pos,1.5)
 positive=[[2,4,9],[3,8,10]];negative=[[14,20,21]]
 centers=[np.array([-np.sqrt(3)/2,1.5]),np.array([np.sqrt(3)/2,-1.5])]
 rr=[]
 for triangles,center in zip([positive,negative],centers):
  for t in triangles:
   assert np.max(abs(np.linalg.norm(pos[t]-center,axis=1)-1))<1e-9
   item=loop(ax,pos,P,t,.24);item['hexagon_center']=center.tolist();rr.append(item)
 assert all(x['weight_raw']>0 for x in rr[:2]) and rr[2]['weight_raw']<0
 m.draw_nodes(ax,pos,m.omega_fills(C),ms=11,labels=[str(i) for i in range(24)],fs=m.FS_IDX_B)
 records['coronene']=dict(triangles=rr,site_sum=float(C.sum()),selected_are_not_a_complete_cancellation_group=True,layout='Positive examples in upper-left hexagon; negative example in lower-right hexagon.')

def panel_flake(ax,pos,hexes,P,C):
 m.frame(ax,(-6.8,6.8),(-8.7,7.0));skeleton(ax,pos,1.0)
 ring0=min(hexes,key=lambda h:np.linalg.norm(h[0]))[1]
 ax.add_patch(Polygon(pos[ring0],facecolor=m.C_CELL,edgecolor='none',zorder=0))
 # One fixed anchor, two opposite-position pure triangles, well inside the flake.
 anchor=39;pure0=[anchor,24,26];pure1=lookup(pos,2*pos[anchor]-pos[pure0])
 pure=[pure0,pure1]
 assert weight(pos,P,pure0)*weight(pos,P,pure1)<0
 # Keep the mixed pair separate from the pure pair in the same lattice drawing.
 candidates=[]
 for ids in itertools.combinations(range(len(pos)),3):
  q=pos[list(ids)];ds=np.sort(np.linalg.norm(q-np.roll(q,1,axis=0),axis=1))
  if np.max(abs(ds-[1,1,np.sqrt(3)]))>1e-8 or np.linalg.norm(q,axis=1).max()>3.7:continue
  w=weight(pos,P,ids)
  if w<=0:continue
  inv=lookup(pos,-q)
  assert abs(w-weight(pos,P,inv))<1e-12
  if tuple(ids)>tuple(sorted(inv)):continue
  q2=pos[inv];pure_vertices=set(pure0+pure1)
  shared=len(pure_vertices & set(ids+tuple(inv)))
  sep=min(np.linalg.norm(c-d) for c in [q.mean(0),q2.mean(0)] for d in [pos[pure0].mean(0),pos[pure1].mean(0)])
  centre_radius=np.linalg.norm(q.mean(0))
  score=-20*shared+sep-.35*centre_radius
  candidates.append((score,list(ids),inv))
 _,mix0,mix1=max(candidates,key=lambda x:x[0])
 rr=[loop(ax,pos,P,t,.20,lw=1.0) for t in pure+[mix0,mix1]]
 m.draw_nodes(ax,pos,m.omega_fills(C),ms=4.6)
 ax.plot(*pos[anchor],marker='o',ms=6.1,mfc='none',mec='#1B1B1E',mew=.8,zorder=6)
 records['haldane']=dict(triangles=rr,same_sublattice_anchor=anchor,pure_pair_residual=rr[0]['weight_raw']+rr[1]['weight_raw'],mixed_pair_sum=rr[2]['weight_raw']+rr[3]['weight_raw'],site_sum=float(C.sum()),marker_min=float(C.min()),marker_max=float(C.max()),boundary='open',pure_pair_exact_cancellation_only_in_periodic_bulk=True)

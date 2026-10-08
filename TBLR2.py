# Python class for calculating conductivity and other linear response coefficients

# designed to only generate data
# use minimal modules - numpy, scipy

import numpy as np
from numpy.linalg import eigh
from numpy.linalg import inv
import time
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

###################################################################################
###########					S0: Utilities								###########
###################################################################################


# diagonalization
def eigenstates(Hk):
	evals, evecs = eigh(Hk)
	idx = evals.argsort()[:]  
	en = evals[idx]
	um = np.array(evecs[:,idx])

	return en, um

# string manipulation for filename
def rp(txt):
	txt = str(txt)
	return (txt.replace('.','p')).replace('-','m')

# filename given a set of parameters
def get_fname(begin, params, end):
	# given a dictionary of parameters,
	# create a string that makes a good file name

	keys = list( params.keys() )
	fname = begin
	for prm in keys:
		prm_val = params[prm]
		prm_name = prm
		if type(prm_val)==str:
			fname += '_' + rp( prm_name + prm_val)
		elif int(prm_val) == prm_val:
			fname += '_' + rp( prm_name+'%d'%prm_val)
		else:
			fname += '_' + rp( prm_name+'%3.2f'%prm_val)
	fname += end
	return fname

# Fermi function
def Fermi(beta, x):
	return 1.0/(1.0 + np.exp(beta*x))

def DFermi(beta, x):
	f = Fermi(beta, x)
	return -beta * f * (1.0 - f)


###################################################################################
###########					S1: Class begins							###########
###################################################################################

class TBLinearResponse():

	#initializing
	def __init__(self,
		ham, 			    # given hamiltonian, a function of vector k
		Lar,			    # array of Lattice vectors, such that a1 = L1[:,0] 
		Nkp = [40, 40],   # number of points in high-symmetry path, number of points in k-mesh
		save_all = True,	# whether to save/load things or not
		kpath = ['K', 'G', 'M', 'Kp'],	# High Symmetry Path for plotting band structure
		prestr = '',				# prefix string for saving files
		endstr = '',	# suffix string for saving files
        
		):

		self.ham = ham
		self.Lar = np.array(Lar)
		self.Nkp = Nkp
		self.save_all = save_all
		self.kpath = kpath
		self.prestr = prestr
		self.endstr = endstr
		# lattice vectors
		Lar = np.array(Lar)
		q =  (2*np.pi*inv(Lar).T)   # reciprocal lattice: q_i . a_j = 2pi delta_ij (needs transpose for non-symmetric Lar)
		self.q = q

		dim = np.shape(Lar)[0]
		self.dim = dim

		# let us create kSpan
		if dim == 2:
			#not sure if these are in the right order
			nkx = Nkp[0]
			nky = Nkp[1]
			#Gabby changed this so that the quadrants would be conventional

			kSpan = np.array([i*q[0] + j*q[1] for i in np.linspace(-0.5,0.5,nkx,endpoint=False) for j in np.linspace(-0.5,0.5,nky,endpoint=False)])
			kSpan = np.array([i*q[0] + j*q[1] for i in np.linspace(-0.5,0.5,nkx,endpoint=False) for j in np.linspace(-0.5,0.5,nky,endpoint=False)]) 
			norb = np.shape(ham(kSpan[0]))[0]
		elif dim == 1:
			nkx = Nkp[0]
			kSpan = np.array([i*q[0] for i in np.linspace(-0.5,0.5,nkx,endpoint=False)])
			norb = np.shape(ham(kSpan[0]))[0]
						
		elif dim == 3:
			nkx = Nkp[0]
			nky = Nkp[1]
			nkz = Nkp[2]
			#Not multiplying by unit vector because pythTB ham input is in reduced coords...
			kSpan = np.array([i*q[0] + j*q[1]+k*q[2] for i in np.linspace(-0.5,0.5,nkx,endpoint=False) for j in np.linspace(-0.5,0.5,nky,endpoint=False) for k in np.linspace(-0.5,0.5,nkz,endpoint=False)])
			norb = np.shape(ham(kSpan[0]))[0]
			
		else:
			print('error in spatial dimensions\n')

		self.kSpan = kSpan
		self.norb = norb

		# fname for all files
		begin = 'dataEnUm/' + prestr
		params = {
		"Nkp": Nkp[0]
		}
		end = endstr
		datastr = get_fname(begin, params, end)
		self.datastr = datastr

		# next we save/load eigen -- set to true for computations, set to false for band structure
		try:
			enUM = np.load(datastr+'_enUM.npy')
			self.enUM = enUM
		except:
			enUM = []

			for k in kSpan:
				Hk = ham(k)
				en, um = eigenstates(Hk)
				matr = np.zeros([norb, norb+1], dtype=complex)
				matr[:,0] = en
				matr[:,1:] = um
				enUM.append(matr)
			
			enUM = np.array(enUM)
			if save_all:			
				np.save(datastr+'_enUM', enUM)
			self.enUM = enUM

		
	def solve_dos(self, muar = np.arange(-5,5,0.01), sigma = 5):
		# code to solve for full density of states

		fname =  self.datastr + '_hist'
		try:
			hist0 = np.load(fname+'.npy')
		except:
			# collect all energies
			enar = []
			for enum in self.enUM:
				enar.append(enum[:,0])
			enar = np.array(enar).ravel()

			hist = np.histogram(enar, bins=muar)
			hist0 = hist[0]
			hist2 = np.zeros([len(hist0)+1])		# add one element to last entry to match size
			hist2[0:-1] = hist0
			hist0 = hist2
			if self.save_all:
				np.save(fname, hist0)
			

		dos = gaussian_filter(hist0, sigma=sigma)

		return dos
	

	def solve_pdos(self, muar = np.arange(-5,5,0.01), beta = 10):
		# code to solve for orbital resolved density of states
  
		fname =  self.datastr + '_pdosPP'

		try:
			hist0 = np.load(fname+'.npy')
		except:
			# collect all energies and 'colors'
			enUM = self.enUM
			hist0 = np.zeros_like(enUM)
			for i, enum in enumerate(self.enUM):
				hist0[i,:,0] = enum[:,0]
				um = enum[:,1:]
				hist0[i, :, 1:self.norb+1] = (np.abs(um)**2).T
			if self.save_all:
				np.save(fname, hist0)

		# given hist0, calculate pDOS
		pDOS = np.zeros([len(muar), self.norb])
		enar = hist0[:,:,0].ravel()
		for i in range(self.norb):
			color = hist0[:,:,i+1].ravel()
			pDOS[:,i] = [ np.dot(-DFermi(beta, enar-mu), color ) for mu in muar]

		return pDOS

	def solve_sigma(self, omar = np.arange(-5,5,0.01), 
				 mu = 0.0, beta = 10.0,
				 tau = 1.0, dkx = np.array([1,0]), k_step = 1.e-3):
		
		sigma_ar = np.zeros_like(omar)

		def FNM(em, en):
			delta = 1.e-6  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn - fm

			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / emn)

			if np.abs( np.sum( result[result<0.0] ) ) > 0.0:
				print('error: fnm < 0')
				
			return result
			
		fname =  self.datastr + '_jop'	

		t1 = time.time()
		print('begin current load')

		# save/load current operator on grid
		try:
			Jop = np.load(self.datastr+'_jop.npy')
			self.Jop = Jop
			print('loaded, done')
		except:
			Jop = []
			for i,k in enumerate( self.kSpan ):
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				dk = k_step*dkx
				Hk_dk = self.ham(k + dk)
				dH = (Hk_dk - Hk)/k_step
				um = np.asmatrix(um)
				Jv = np.dot( um.H, np.dot(dH, um) )
				#Gabby added this just to see what happens
				#np.fill_diagonal(Jv,0)
				Jop.append(Jv)

			Jop = np.array(Jop)
			print('computed, done')
			if self.save_all:
				np.save(fname, Jop)
			self.Jop = Jop

		t2 = time.time()
		print('current operator load, done', t2-t1)

		# Precompute 'em' and 'en' since they do not depend on 'om'
		em = self.enUM[:, :, 0] - mu
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)

		# Compute jx outside the loop, assuming Jop[:,:,0] holds the relevant matrix for the operation
		jx = np.abs(Jop[:, :, :])**2  # Shape (num_i, norb, norb)

		# Initialize sigma_ar array
		sigma_ar = np.zeros(len(omar))

		t3 = time.time()
		print('ready for loop over omega', t3-t2)


		# Loop over each omega in the omega array
		for o, om in tqdm(enumerate(omar)):
			# Update pref, den, and sg calculation
			den = tau**2 + (om + en - em)**2  # Shape (num_i, norb, norb)
			pref = FNM(en, em)  # Assuming FNM can be vectorized or accept arrays
			sg = tau * np.sum(pref * jx / den, axis=(1, 2))  # Sum over 'i', 'm', and 'n'
			
			sigma_ar[o] = np.sum(sg)  # Sum all sg contributions

		if self.save_all:
			np.save(fname+'_sigma', sigma_ar)

		return sigma_ar

	def solve_sigma_BondResolved(self,orb1, orb2, omar = np.arange(-5,5,0.01), 
				 mu = 0.0, beta = 10.0,
				 tau = 1.0, dkx = np.array([1,0]),FS='False', k_step = 1.e-3):
		print("orbitals n,m",orb1,orb2)
		sigma_ar = np.zeros_like(omar)

		def FNM(em, en):
			delta = 1.e-6  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn - fm

			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			if FS:
				result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / emn**2)
			else:
				result = np.where(np.abs(emn) < delta, 0.0, fnm / emn**2)

			if np.abs( np.sum( result[result<0.0] ) ) > 0.0:
				print('error: fnm < 0')
				
			return result
			
		fname =  self.datastr + '_jop'	

		t1 = time.time()
		print('begin current load')

		# save/load current operator on grid
		try:
			Jop = np.load(self.datastr+'_jop.npy')
			self.Jop = Jop
			print('loaded, done')
		except:
			Jop = []
			Hk = self.ham(self.kSpan[0])
			U_alpha = np.zeros_like(Hk)
			U_alpha[orb1, orb1] = 1  # |alpha⟩⟨alpha|

			U_beta = np.zeros_like(Hk)
			U_beta[orb2, orb2] = 1  # |beta⟩⟨beta|
			print("you are using the right version")
			for i,k in enumerate( self.kSpan ):
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				dk = k_step*dkx
				Hk_dk = self.ham(k + dk)
				dH_full = (Hk_dk - Hk)/k_step 
				dH = U_alpha @ dH_full @ U_beta
				um = np.asmatrix(um)
				Jv = np.dot( um.H, np.dot(dH, um) )
				#Gabby added this 
				#np.fill_diagonal(Jv,0)
				Jop.append(Jv)

			Jop = np.array(Jop)
			print('computed, done')
			if self.save_all:
				np.save(fname, Jop)
			self.Jop = Jop

		t2 = time.time()
		print('current operator load, done', t2-t1)

		# Precompute 'em' and 'en' since they do not depend on 'om'
		em = self.enUM[:, :, 0] - mu
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)

		# Compute jx outside the loop, assuming Jop[:,:,0] holds the relevant matrix for the operation
		jx = np.abs(Jop[:, :, :])**2  # Shape (num_i, norb, norb)

		# Initialize sigma_ar array
		sigma_ar = np.zeros(len(omar))

		t3 = time.time()
		print('ready for loop over omega', t3-t2)


		# Loop over each omega in the omega array
		for o, om in tqdm(enumerate(omar)):
			# Update pref, den, and sg calculation
			den = tau**2 + (om + en - em)**2  # Shape (num_i, norb, norb)
			pref = FNM(en, em)  # Assuming FNM can be vectorized or accept arrays
			sg = tau * np.sum(pref * jx / den, axis=(1, 2))  # Sum over 'i', 'm', and 'n'
			
			sigma_ar[o] = np.sum(sg)  # Sum all sg contributions #MULTIPLY BY TAU??

		if self.save_all:
			np.save(fname+'_sigma', sigma_ar)

		return sigma_ar

	def solve_FS(self, mu, dmu = 1.e-3,orb_res=False,set1=[],set2=[]):
		# gives collection of k-points that form the Fermi surface

		kSpan = self.kSpan
		kspan_FS = []
		alpha_list=[]
		for i, enum in enumerate(self.enUM):
			enar = enum[:,0] - mu
			umar = enum[:,1:]
			if np.min( np.abs(enar) ) < dmu:
				fermi_idx = np.abs(enar-mu)<dmu
				if orb_res == False:
					kspan_FS.append( kSpan[i] )	
				else:
					kspan_FS.append(kSpan[i])
					alpha1 = np.average(np.abs(umar[[0,2],fermi_idx])**2)/2
					alpha2 = np.average(np.abs(umar[[1,3],fermi_idx])**2)/2
					kspan_FS.append(kSpan[i])
					alpha_list.append([alpha1,alpha2])
		return np.array(kspan_FS),np.array(alpha_list)
	
	def solve_cohp_energy_resolved(self):		
		COHP = np.zeros([ self.norb, self.norb,self.norb,np.size(self.kSpan[:,0])], dtype=complex)
		for i, k in enumerate(self.kSpan):
			ham_k = self.ham(k=k)
			um = self.enUM[i, :, 1:]  # shape: (norb, norb) #order is k,orb,band
			for j in range(self.norb):
				COHP[:,:,j,i] = np.abs(um[:,j] @ np.conj(um[:,j]).T)*ham_k
		#to plot as historgram: 
		# hist,bin_edges = np.histogram(tb.enUM[:, :, 0].ravel(),weights =COHP[i,j].ravel(),bins=100)
		return COHP

	def solve_wiberg_kresolved(self,mu):
		OpRDM = np.zeros([len(self.kSpan), self.norb, self.norb], dtype=complex)		
		for i, k in enumerate(self.kSpan):
			en = self.enUM[i, :, 0]
			um = self.enUM[i, :, 1:]  # shape: (norb, norb) #order is k,orb,band
			for band1 in range(self.norb):
				for band2 in range(self.norb):
					enband1 = en[band1]
					enband2 = en[band2]
					for a in range(self.norb):
						for b in range(self.norb):
							if enband1 < mu and enband2 < mu:
							#bandWF1 = um[:,band1][:,np.newaxis]
							#bandWF2 = um[b,band2][:,np.newaxis]
								OpRDM[i, a, b] += um[a,band1]*np.conj(um[b,band1])*np.conj(um[a,band2])*um[b,band2]
		return OpRDM
	

	def solve_multicenter_kresolved(self,mu):
		OpRDM = np.zeros([len(self.kSpan), self.norb, self.norb,self.norb], dtype=complex)

		# i refers to which k-point - so maybe 1000 of those
		for i, k in enumerate(self.kSpan):
			en = self.enUM[i, :, 0]
			um = self.enUM[i, :, 1:]  # shape: (norb, norb) #order is k,orb,band
			# um is k index, orbital index, eigenfunction index.
			# um[orbital, band] -> this represents the inner product of a given orbital and band


			# norb = numbre of orbitals

			# We are calculating the overlap of a and b, through n
			# and overlap of b and c thru m
			# and c and a thru p

			# Band 1, 2, and 3 indices represent the possible values of n,m and p


			# We calculate this sum for each of a,b, and c. This is where a,b,c come from


			for band1 in range(self.norb):
				for band2 in range(self.norb):
					for band3 in range(self.norb):
						enband1 = en[band1]
						enband2 = en[band2]
						enband3 = en[band3]
						for a in range(self.norb):
							for b in range(self.norb):
								for c in range(self.norb):
									if enband1 < mu and enband2 < mu and enband3 < mu:
										OpRDM[i, a, b,c] += um[a,band1]*np.conj(um[b,band1]) * um[b,band2]*np.conj(um[c,band2]) * um[c,band3]*np.conj(um[a,band3])
		return OpRDM

	def solve_jdos(self, muar = np.arange(-5,5,0.01), sigma = 5):
		# code to solve for joint density of states

		fname =  self.datastr + '_jdos_hist'
		try:
			hist0 = np.load(fname+'.npy')
		except:
			enar_diff = []
			# go to each k
			for enum in self.enUM:
				enar = enum[:,0]
				enar_d = np.abs( enar[:, np.newaxis] - enar )
				enar_diff.append( np.triu(enar_d,k=1).ravel() )

			enar_diff = np.array( enar_diff ).ravel()

			hist = np.histogram(enar_diff, bins=5+muar)
			hist0 = hist[0]
			hist2 = np.zeros([len(hist0)+1])		# add one element to last entry to match size
			hist2[0:-1] = hist0
			hist0 = hist2
			if self.save_all:
				np.save(fname, hist0)
			
		jdos = gaussian_filter(hist0, sigma=sigma)

		return jdos

	def solve_k(self, kpt):
		Hk = self.ham(kpt)
		val, vec = eigenstates(Hk)
		return val, vec

	def solve_alongpath(self,proj=False,proj_orbs=[0,0,0,0]):
		#band structure along a high symmetry path
  
		kpt = lambda m,n: m*self.q[0] + n*self.q[1]
#path=[[0.5,0.5,0.5],[0.0,0.0, 0.0],[0.5,-0.5,0.0], [0.375,-0.375,0.0], [0.0, 0.0, 0.0]]
		
		K = kpt(1/3.0 , 1/3.0)
		Kp = kpt(2/3.0, 2/3.0)
		Gamma = kpt(0, 0)
		M1 = kpt(1/2.0, 0)
		M2 = kpt(0, 1/2.0)
		M3 = kpt(1/2.0, 1/2.0)

		kdict = {"G": Gamma, "K": K, "M": M3, "Kp": Kp, "M1": M1, "M2": M2, "M3": M3, "X": M1, "Y":M2}
		#want distance to be related to distance in k-space. 

		kPath_desired = self.kpath
		kpath = []
		k_node=[0]
		knode_n=0
		NK = len(kPath_desired)-1
		for i in range(NK):
			k0 = kdict[ kPath_desired[i] ]
			k1 = kdict[ kPath_desired[i+1] ]
			k_dist = np.linalg.norm(k1-k0)
			#print("k0,k1",k0,k1)
			#print("k_dist?/",k_dist)
			n0kp=int(self.Nkp[0]*k_dist)
			knode_n+=n0kp
			k_node.append(knode_n)
			kpath += [(1-i)*k0 + i*k1 for i in np.linspace(0,1, n0kp) ]

		evals = []
		evecs = []
		for kpt in kpath: #for each kpt along the path
			Hk = self.ham(kpt)

			val, vec = eigenstates(Hk)

			#print("h* vec",Hk@vec[:,0])
			#print("e *vec",val[0]*vec[:,0])
			#NOW - symmetrizing the evec. 
			evals.append(val)
			evecs.append(vec)

		evals = np.array(evals)
		evals = np.array(evals)

		return evals, evecs,k_node
	
	def solve_alongpath3D(self,path):
		kpt = lambda m,n,p: m*self.q[0] + n*self.q[1] + p*self.q[2]
		kpath_to_take = [kpt(m,n,p) for m,n,p in path]
		kPath_desired = kpath_to_take
		kpath = []
		k_node=[0]
		knode_n=0
		NK = len(kPath_desired)-1
		for i in range(NK):
			k0 = kPath_desired[i]
			k1 = kPath_desired[i+1] 
			k_dist = np.linalg.norm(k1-k0)
			n0kp=int(self.Nkp[0]*k_dist)
			knode_n+=n0kp
			k_node.append(knode_n)
			kpath += [(1-i)*k0 + i*k1 for i in np.linspace(0,1, n0kp) ]

		evals = []
		evecs = []
		for kpt in kpath: #for each kpt along the path
			Hk = self.ham(kpt)
			val, vec = eigenstates(Hk)
			evals.append(val)
			evecs.append(vec)

		evals = np.array(evals)
		evals = np.array(evals)

		return evals, evecs,k_node


	def solve_Drude(self, mu, 
				dmu = 1.e-3,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# for taking k derivative
				beta = 1.e3,
				k_step = 1.e-3
				):
		
		# solve d2en 

		fname =  self.datastr + '_d2en'	

		# save/load d2en on grid
		try:
			D2en = np.load(self.datastr+'_d2en.npy')
			self.D2en = D2en
		except:
			D2en = []
			for i,k in enumerate( self.kSpan ):
				en = self.enUM[i,:,0]
				dk = k_step*dkx
				Hk_dk = self.ham(k + dk)
				Hk_mdk = self.ham(k - dk)
				en_dk, um = eigenstates(Hk_dk)
				en_mdk, um = eigenstates(Hk_mdk)
				d2en = (en_dk + en_mdk - 2*en)/k_step**2
				D2en.append(d2en)
			D2en = np.array(D2en)

			if self.save_all:
				np.save(fname, D2en)
			self.D2en = D2en
		
		fermi = Fermi(beta, self.enUM[:, :, 0] - mu)
		drude = np.sum(fermi * self.D2en)
		vol = np.linalg.det(self.Lar)*(self.Nkp[1])**2
	
		return drude/vol

	def solve_tQGT_offD(self, time_ar, mu, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# for taking k derivative
				k_step = 1.e-3
				):

		# solve for tQGT for a given chemical potential

		tqgt_ar = np.zeros_like(time_ar, dtype=complex)

		def FNM(em, en):
			delta = 1.e-3  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn*(1 - fm)

			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			result = np.where(np.abs(emn) < delta, 0.0, fnm / emn**2 )

			return result

		fname =  self.datastr + '_jop'	
		# save/load current operator on grid
		try:
			Jop = np.load(self.datastr+'_j_op.npy')
			self.Jop = Jop
		except:
			Jop = []
			for i,k in enumerate( self.kSpan ):
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				dk = k_step*dkx
				Hk_dk = self.ham(k + dk)
				dH = (Hk_dk - Hk)/k_step
				um = np.asmatrix(um)
				Jv = np.dot( um.H, np.dot(dH, um) )
				Jop.append(Jv)
			Jop = np.array(Jop)
			if self.save_all:
				np.save(fname, Jop)
			self.Jop = Jop


		# Precompute 'em' and 'en' since they do not depend on 'time'
		em = self.enUM[:, :, 0] - mu
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)

		# Compute jx outside the loop, assuming Jop[:,:,0] holds the relevant matrix for the operation
		jx = np.abs(Jop[:, :, :])**2  # Shape (num_i, norb, norb)

		for t, time in enumerate(time_ar):
			# Update pref, den, and sg calculation
			den = np.exp(1j*time*(en - em) )  # Shape (num_i, norb, norb)
			pref = FNM(en, em)  # Assuming FNM can be vectorized or accept arrays
			gt = np.sum( pref * jx * den, axis=(1, 2))
			tqgt_ar[t] = np.sum(gt)

		return tqgt_ar
	

	def solve_complex_sigma(self, omar = np.arange(-5,5,0.01), 
				 mu = 3.2, beta = 10.0,
				 tau = 1.e-3, dkx = np.array([1,0]), k_step = 1.e-3):

		def FNM(em, en):
			delta = 1.e-6  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn - fm

			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / emn)

			if np.abs( np.sum( result[result<0.0] ) ) > 0.0:
				print('error: fnm < 0')
				
			return result


		fname =  self.datastr + '_jop'	

		t1 = time.time()
		print('begin current load')

		# save/load current operator on grid
		try:
			Jop = np.load(self.datastr+'_jop.npy')
			self.Jop = Jop
			print('loaded, done')
		except:
			Jop = []
			for i,k in enumerate( self.kSpan ):
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				dk = k_step*dkx
				Hk_dk = self.ham(k + dk)
				dH = (Hk_dk - Hk)/k_step
				um = np.asmatrix(um)
				Jv = np.dot( um.H, np.dot(dH, um) )
				Jop.append(Jv)
			Jop = np.array(Jop)
			print('computed, done')
			if self.save_all:
				np.save(fname, Jop)
			self.Jop = Jop

		t2 = time.time()
		print('current operator load, done', t2-t1)

		# Precompute 'em' and 'en' since they do not depend on 'om'
		em = self.enUM[:, :, 0] - mu
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)

		# Compute jx outside the loop, assuming Jop[:,:,0] holds the relevant matrix for the operation
		jx = np.abs(Jop[:, :, :])**2  # Shape (num_i, norb, norb)

		# Initialize sigma_ar array
		re_sigma_ar = np.zeros(len(omar))
		im_sigma_ar = np.zeros(len(omar))

		t3 = time.time()
		print('ready for loop over omega', t3-t2)

		for o, om in tqdm(enumerate(omar)):
			# Update pref, den, and sg calculation
			den = tau**2 + (om + en - em)**2  # Shape (num_i, norb, norb)
			pref = FNM(en, em)  # Assuming FNM can be vectorized or accept arrays
			sg = tau * np.sum(pref * jx / den, axis=(1, 2))  # Sum over 'i', 'm', and 'n'
			
			re_sigma_ar[o] = np.sum(sg)  # Sum all sg contributions

			# Calculate num and den using broadcasting
			om_plus_en_minus_em = om + en - em

			# Accumulate re_sigma_ar and im_sigma_ar using vectorized operations
			sg = np.sum(om_plus_en_minus_em * pref * jx / den, axis=(1,2))

			im_sigma_ar[o] = np.sum(sg)

		complex_sigma_ar = np.zeros([len(omar), 2])
		complex_sigma_ar[:,0] = re_sigma_ar 
		complex_sigma_ar[:,1] = im_sigma_ar

		self.complex_sigma_ar = complex_sigma_ar

		return complex_sigma_ar/len(self.kSpan)


	def solve_complex_epsilon(self, omar = np.arange(-5,5,0.01), mu=3.2, beta = 10
						   ):
		
		complex_sigma_ar = self.solve_complex_sigma(omar = omar, mu=mu, beta=beta)
		complex_eps_ar = np.zeros_like(complex_sigma_ar, dtype=complex)

		for i,om in enumerate(omar):	 
			if np.abs(om) > 0:
				complex_eps_ar[i,0] = 1.0 + 4*np.pi*complex_sigma_ar[i,1]/om
				complex_eps_ar[i,1] = 4*np.pi*complex_sigma_ar[i,0]/om
			
		self.complex_eps_ar = complex_eps_ar

		return complex_eps_ar


	def solve_reflectivity(self, omar = np.arange(-5,5,0.01), mu = 3.2, beta = 10,
						   ):
		
		complex_eps_ar = self.solve_complex_epsilon(omar=omar, mu=mu, beta = beta)
		eta_ar = np.zeros_like(omar)

		nonzero_mask = np.abs(omar) > 0
		eps = complex_eps_ar[:, 0] + 1j * complex_eps_ar[:, 1]
		sq_eps = np.sqrt(eps)
		num = sq_eps - 1.0
		den = sq_eps + 1.0
		eta_ar[nonzero_mask] = np.abs(num[nonzero_mask] / den[nonzero_mask])
		
		return eta_ar
	

	def solve_QGT_kresolved(self, mu, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# direction for taking k derivative
				k_step = 1.e-3,FS=False,delta=1.e-3							# infinitesimal step
				):

		# solve for tQGT for a given chemical potential

		def FNM(em, en):
			#delta = 1.e-3  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn*(1.0 - fm)
			#print("this is the energy difference",np.where(fnm < delta, 100000, emn))
			if FS:
				result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / (emn**2))
			else:
				result = np.where(np.abs(emn) < delta, 0, fnm / (emn**2))

			return result

		fname =  self.datastr + '_jop'	
		# save/load current operator on grid
		try:
			Jop = np.load(self.datastr+'_j_op.npy')
			self.Jop = Jop
		except:
			Jop = []
			for i,k in enumerate( self.kSpan ):
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				Hk_dk = self.ham(k + k_step*dkx)
				dH = (Hk_dk - Hk)/k_step
				um = np.asmatrix(um)
				Jv = np.dot( um.H, np.dot(dH, um) )
				#if not FS:
					#np.fill_diagonal(Jv,0) #trying only interband terms???
				Jop.append(Jv)
			Jop = np.array(Jop)
			if self.save_all:
				np.save(fname, Jop)
			self.Jop = Jop


		# Precompute 'em' and 'en' since they do not depend on 'time'
		em = self.enUM[:, :, 0] - mu
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)

		# Compute jx outside the loop, assuming Jop[:,:,0] holds the relevant matrix for the operation
		jx = Jop[:, :, :]*Jop[:, :, :].conj()#np.abs(Jop[:, :, :])**2  # Shape (num_i, norb, norb)

		den = np.exp(1j * 0 * (en - em))		# it is an identity matrix
		pref = FNM(en, em)  # vectorized FNM
		qgt = np.sum( pref * jx * den, axis=(1, 2))

		return qgt
	
	def solve_QGT_kresolved_offD(self, mu, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# direction for taking k derivative
				dky = np.array([0,1]),				#second direction for taking k derivative
				k_step = 1.e-3,
				delta=1.e-3,
				FS=False							# infinitesimal step
				):
		#solves QGT gxy component.
		# solve for tQGT for a given chemical potential

		def FNM(em, en):
			#delta = 1.e-3  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn*(1.0 - fm)
			if FS:
				result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / emn**2)
			else:
				result = np.where(np.abs(emn) < delta, 0.0, fnm / emn**2)
			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			#result = np.where(np.abs(emn) < delta, 0.0, fnm / emn**2 )

			return result
		fname =  self.datastr + '_jop'	
		# save/load current operator on grid
		try:
			Jop = np.load(self.datastr+'_j_op.npy')
			self.Jopx = Jop
		except:
			Jopx = []
			Jopy = []
			for i,k in enumerate( self.kSpan ):
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				Hk_dkx = self.ham(k + k_step*dkx)
				Hk_dky = self.ham(k + k_step*dky)
				dHx = (Hk_dkx - Hk)/k_step
				dHy = (Hk_dky - Hk)/k_step
				um = np.asmatrix(um)
				Jvx = np.dot( um.H, np.dot(dHx, um) )
				Jvy = np.dot( um.H, np.dot(dHy, um) )
				Jopx.append(Jvx)
				Jopy.append(Jvy)
			Jopx = np.array(Jopx)
			Jopy = np.array(Jopy)
			if self.save_all:
				np.save(fname, Jop)
			self.Jopx = Jopx
			self.Jopy = Jopy


		# Precompute 'em' and 'en' since they do not depend on 'time'
		em = self.enUM[:, :, 0] - mu
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)

		# Compute jx outside the loop, assuming Jop[:,:,0] holds the relevant matrix for the operation
		jx = np.abs(Jopx[:, :, :])*np.abs(Jopy[:, :, :])  # Shape (num_i, norb, norb)

		den = np.exp(1j * 0 * (en - em))		# it is an identity matrix
		pref = FNM(en, em)  # vectorized FNM
		qgt = np.sum( pref * jx * den, axis=(1, 2))

		return qgt,pref
	def find_mu(self,occ,crit = 1e-3):
		en = self.enUM[:,:,0]
		filled_en = np.ones_like(en)
		nkpt = len(self.kSpan)
		mu_low= np.real(np.min(en))
		mu_high = np.real(np.max(en))
		
		en_mask_low = [en<mu_low][0]
		en_mask_high = [en<mu_high][0]
		
		mu_occ_low=np.sum(filled_en[en_mask_low])/nkpt
		mu_occ_high=np.sum(filled_en[en_mask_high])/nkpt
		m=0
		mu_new = mu_low
		mu_occ_new = mu_occ_low
		while np.abs(mu_occ_new - occ) > crit:
			mu_new = 0.5*(mu_low+mu_high)
			en_mask_new = [en<mu_new][0]
			mu_occ_new=np.sum(filled_en[en_mask_new])/nkpt
			if mu_occ_new < occ: 
				mu_low = mu_new
			elif mu_occ_new > occ:
				mu_high = mu_new
			#print(f"{mu_occ_new=}",f"{occ=}")
		return mu_new
	
	def solve_QGT_BondResolved_kresolved(self, mu, orb1, orb2, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# direction for taking k derivative
				dky = np.array([0,1]),
				k_step = 1.e-3,							# infinitesimal step
				#H_tb = np.array([0]), #hamiltonian that corresponds to tight binding-have the atomic distances here. NEEDS TO BE OF A SUPERCELL (where there is only one entry per term. )
				FS = False
				):

		# solve for tQGT for a given chemical potential

		def FNM(em, en):
			delta = 1.e-3  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn*(1.0 - fm)

			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			if FS:
				result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / (emn**2))
			else:
				result = np.where(np.abs(emn) < delta, 0.0, fnm / (emn**2))
			return result
		k_phase = np.zeros(np.size(self.kSpan[:,0]))
		try:
			Jopx_orb = np.load(self.datastr+'_j_opx_%d_%d.npy'%(orb1, orb2))
			Jopy_orb = np.load(self.datastr+'_j_opy_%d_%d.npy'%(orb1, orb2))
		except:
			
			Jopx_orb = []
			Jopy_orb = []
			Hk = self.ham( self.kSpan[0] )
			U_alpha = np.zeros_like(Hk)
			U_alpha[orb1, orb1] = 1  # |alpha⟩⟨alpha|
			#H_tb_orb = H_tb[orb1][orb2][0]
			#print("H_tb_orb",H_tb_orb)
			#xdist = complex(H_tb_orb[1])
			#ydist = complex(H_tb_orb[2])
			U_beta = np.zeros_like(Hk)
			U_beta[orb2, orb2] = 1  # |beta⟩⟨beta|
			print("you are using the right version")
			for i,k in enumerate( self.kSpan ):
				#H_tb_orb #0 is magnitude hopping, 1 is xdist, 2 is ydist
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				#Hk_dk = self.ham(k + k_step*dkx)
				Hk_dkx = self.ham(k + k_step*dkx)
				Hk_dky = self.ham(k + k_step*dky)
				dHx = (Hk_dkx - Hk)/k_step #element-wise multiplication
				dHy = (Hk_dky - Hk)/k_step 
				um = np.asmatrix(um)
				#Gabby edited 5/21/25 (is the same output as what was previously there)
				#making matrix of current operators instead of one at a time
				Jvx_mat = np.zeros_like(Hk,dtype = 'object')
				Jvy_mat = np.zeros_like(Hk,dtype = 'object')
				for alpha in range(self.norb):
					U_alpha = np.zeros_like(Hk)
					U_alpha[alpha, alpha] = 1  # |alpha⟩⟨alpha|
					for beta in range(self.norb):
						U_beta = np.zeros_like(Hk)
						U_beta[beta, beta] = 1  # |beta⟩⟨beta|
						Jvx_mat[alpha,beta] = um.H @ U_alpha @ dHx @ U_beta@ um
						Jvy_mat[alpha,beta] = um.H @ U_alpha @ dHy @ U_beta@ um

				Jopx_orb.append(Jvx_mat[0,0])
				Jopy_orb.append(Jvy_mat[0,0])
				#k_phase[i]=np.exp(1j*k[0]*xdist+1j*k[1]*ydist)
			#Jop_orb = np.array(Jop_orb)
			Jopx_orb = np.array(Jopx_orb)#*(np.exp(1j*k[0]*xdist+1j*k[1]*ydist)) #Might need to add back??
			Jopy_orb = np.array(Jopy_orb)#*(np.exp(1j*k[0]*xdist+1j*k[1]*ydist))
			if self.save_all:
				np.save(self.datastr+'_j_opx_%d_%d', Jopx_orb)
				np.save(self.datastr+'_j_opy_%d_%d', Jopy_orb)
			self.Jopx_orb = Jopx_orb
			self.Jopy_orb= Jopy_orb

		# Precompute 'em' and 'en' since they do not depend on 'time'
		em = self.enUM[:, :, 0] - mu #This is the energy value
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)
		#should we do element-wise multiplication or matrix multiplication????
		Jvx2 = np.multiply(Jvx_mat,Jvx_mat.conj())
		Jvy2 = np.multiply(Jvy_mat,Jvy_mat.conj())
		Jvxy = np.multiply(Jvx_mat,Jvy_mat.conj())

		#print("check if the same",jxy==jyx,jxy,jyx)
		den = np.exp(1j * 0 * (en - em))		# it is an identity matrix
		pref = FNM(en, em) #Gabby added the square   # vectorized FNM

		qgt_xx = np.sum( pref * Jvx2 * den, axis=(1, 2))#*k_phase 
		qgt_yy = np.sum( pref * Jvy2 * den, axis=(1, 2))#*k_phase 
		qgt_xy = np.sum( pref * Jvxy * den, axis=(1, 2))#*k_phase 

		#qgt_yx = np.sum( pref * jyx * den, axis=(1, 2))*k_phase 
		qgt = qgt_xx+qgt_yy
		#print("check if the same",qgt_xy==qgt_yx,qgt_xy,qgt_yx)
		return qgt,qgt_xx,qgt_yy,qgt_xy
	


	def solve_QGT_BondResolved_kresolved_GABBY(self, mu, orb1, orb2, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# direction for taking k derivative
				dky = np.array([0,1]),
				k_step = 1.e-3,							# infinitesimal step
				#H_tb = np.array([0]), #hamiltonian that corresponds to tight binding-have the atomic distances here. NEEDS TO BE OF A SUPERCELL (where there is only one entry per term. )
				FS = False
				):

		# solve for tQGT for a given chemical potential

		def FNM(em, en):
			delta = 1.e-3  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn*(1.0 - fm)

			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			if FS:
				result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / (emn**2))
			else:
				result = np.where(np.abs(emn) < delta, 0.0, fnm / (emn**2))
			return result
		k_phase = np.zeros(np.size(self.kSpan[:,0]))
		try:
			Jopx_orb = np.load(self.datastr+'_j_opx_%d_%d.npy'%(orb1, orb2))
			Jopy_orb = np.load(self.datastr+'_j_opy_%d_%d.npy'%(orb1, orb2))
		except:
			Jopx_orb = []
			Jopy_orb = []
			Hk = self.ham( self.kSpan[0] )
			U_alpha = np.zeros_like(Hk)
			U_alpha[orb1, orb1] = 1  # |alpha⟩⟨alpha|
			U_beta = np.zeros_like(Hk)
			U_beta[orb2, orb2] = 1  # |beta⟩⟨beta|
			print("you are using the right version")
			for i,k in enumerate( self.kSpan ):
				#H_tb_orb #0 is magnitude hopping, 1 is xdist, 2 is ydist
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				#Hk_dk = self.ham(k + k_step*dkx)
				Hk_dkx = self.ham(k + k_step*dkx)
				Hk_dky = self.ham(k + k_step*dky)
				dHx = (Hk_dkx - Hk)/k_step #element-wise multiplication
				dHy = (Hk_dky - Hk)/k_step 
				um = np.asmatrix(um)
				#Gabby edited 5/21/25 (is the same output as what was previously there)
				#making matrix of current operators instead of one at a time
				Jvx_mat = np.zeros_like(Hk,dtype = 'object')
				Jvy_mat = np.zeros_like(Hk,dtype = 'object')
				for alpha in range(self.norb):
					U_alpha = np.zeros_like(Hk)
					U_alpha[alpha, alpha] = 1  # |alpha⟩⟨alpha|
					for beta in range(self.norb):
						U_beta = np.zeros_like(Hk)
						U_beta[beta, beta] = 1  # |beta⟩⟨beta|
						Jvx_mat[alpha,beta] = um.H @ U_alpha @ dHx @ U_beta@ um
						Jvy_mat[alpha,beta] = um.H @ U_alpha @ dHy @ U_beta@ um

				Jopx_orb.append(Jvx_mat[0,0])
				Jopy_orb.append(Jvy_mat[0,0])
				#k_phase[i]=np.exp(1j*k[0]*xdist+1j*k[1]*ydist)
			#Jop_orb = np.array(Jop_orb)
			Jopx_orb = np.array(Jopx_orb)#*(np.exp(1j*k[0]*xdist+1j*k[1]*ydist)) #Might need to add back??
			Jopy_orb = np.array(Jopy_orb)#*(np.exp(1j*k[0]*xdist+1j*k[1]*ydist))
			if self.save_all:
				np.save(self.datastr+'_j_opx_%d_%d', Jopx_orb)
				np.save(self.datastr+'_j_opy_%d_%d', Jopy_orb)
			self.Jopx_orb = Jopx_orb
			self.Jopy_orb= Jopy_orb

		# Precompute 'em' and 'en' since they do not depend on 'time'
		em = self.enUM[:, :, 0] - mu #This is the energy value
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)
		#should we do element-wise multiplication or matrix multiplication????
		Jvx2 = np.multiply(Jvx_mat,Jvx_mat.conj())
		Jvy2 = np.multiply(Jvy_mat,Jvy_mat.conj())
		Jvxy = np.multiply(Jvx_mat,Jvy_mat.conj())

		#print("check if the same",jxy==jyx,jxy,jyx)
		den = np.exp(1j * 0 * (en - em))		# it is an identity matrix
		pref = FNM(en, em) #Gabby added the square   # vectorized FNM

		qgt_xx = np.sum( pref * Jvx2 * den, axis=(1, 2))#*k_phase 
		qgt_yy = np.sum( pref * Jvy2 * den, axis=(1, 2))#*k_phase 
		qgt_xy = np.sum( pref * Jvxy * den, axis=(1, 2))#*k_phase 

		#qgt_yx = np.sum( pref * jyx * den, axis=(1, 2))*k_phase 
		qgt = qgt_xx+qgt_yy
		#print("check if the same",qgt_xy==qgt_yx,qgt_xy,qgt_yx)
		return qgt,qgt_xx,qgt_yy,qgt_xy


	def solve_QGT_ksummed(self, mu, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# direction for taking k derivative
				k_step = 1.e-3,
				FS=False,
				delta=1.e-3,
				occ=0											# infinitesimal step
				):

		# solve for tQGT for a given chemical potential

		qgt = self.solve_QGT_kresolved(mu, 
				beta = beta,			# smearing for Fermi factors
				dkx = dkx,				# direction for taking k derivative
				k_step = k_step,							# infinitesimal step
				FS=FS,
				delta=delta,
				)

		qgt_sum = np.sum(qgt)/len(qgt)
		#Lar = self.Lar
		#Energy at random point (here we assume gap spans BZ)
		if occ==0:
			occ = sum(1 for en in self.enUM[0, :, 0] if en-mu <= 0) # Sum over "occupied bnads"
		#vol = np.abs( Lar[0,0]*Lar[1,1] - Lar[0,1]*Lar[1,0] )
		print("occ",occ)
		return qgt_sum/occ
	
	def solve_QGT_ksummed_offD(self, mu, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),
				dky = np.array([0,1]),			# direction for taking k derivative
				k_step = 1.e-3							# infinitesimal step
				):

		# solve for tQGT for a given chemical potential

		qgt = self.solve_QGT_kresolved_offD(mu, 
				beta = beta,			# smearing for Fermi factors
				dkx = dkx,				# direction for taking k derivative
				dky = dky,
				k_step = k_step							# infinitesimal step
				)

		qgt_sum = np.sum(qgt)/len(qgt)
		Lar = self.Lar
		vol = np.abs( Lar[0,0]*Lar[1,1] - Lar[0,1]*Lar[1,0] )

		return qgt_sum/vol
	

	def solve_QGT_BondResolved_ksummed(self, mu, orb1, orb2,
				beta = 1.e2,			# smearing for Fermi factors
				k_step = 1.e-3,				# infinitesimal step
				FS=False,
				occ=0
				#H_tb=np.array([0])
				):

		# solve for tQGT for a given chemical potential
		if occ==0:
			occ = sum(1 for en in self.enUM[0, :, 0] if en-mu <= 0) # Sum over "occupied bnads"
		qgt_kres = self.solve_QGT_BondResolved_kresolved(mu, orb1, orb2,
				beta = beta,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# direction for taking k derivative
				k_step = k_step,							# infinitesimal step
				FS=FS
				#H_tb=H_tb
				)
		qgt_sum = np.sum(qgt_kres[0])/(len(qgt_kres[0])*occ)
		qgt_xx_sum=np.sum(qgt_kres[1])/(len(qgt_kres[1])*occ)
		qgt_yy_sum = np.sum(qgt_kres[2])/(len(qgt_kres[2])*occ)
		qgt_xy_sum = np.sum(qgt_kres[3])/(len(qgt_kres[3])*occ)
		
		return [qgt_sum,qgt_xx_sum,qgt_yy_sum,qgt_xy_sum]
	
	def solve_1pRDM_kresolved(self, mu):
		OpRDM = np.zeros([len(self.kSpan), self.norb, self.norb], dtype=complex)

		for i, k in enumerate(self.kSpan):
			en = self.enUM[i, :, 0]
			um = self.enUM[i, :, 1:]  # shape: (norb, norb)
			for band in range(self.norb):
				enband = en[band]
				if enband < mu:
					bandWF = um[:, band].reshape(-1, 1)  # ensure column vector
					OpRDM[i, :, :] += bandWF @ bandWF.conj().T
					#print("OpRDM[i, :, :]",OpRDM[i, :, :])

		return OpRDM
	def mulliken_charge(self,mu,beta=1.e2):
		overlap = np.zeros([self.norb,self.norb],dtype=complex)
		#density = np.zeros([self.norb,self.norb],dtype=complex)
		for i,k in enumerate(self.kSpan):
			en = self.enUM[i,:,0] - mu
			um = self.enUM[i,:,1:]
			density = Fermi(beta, en)*um.H@um
			print(density)

					

	def solve_1pRDM(self, mu):
		OpRDM = self.solve_1pRDM_kresolved(mu)
		return np.sum(OpRDM, axis=0)/self.vol
	def solve_QGT_BondResolved_kresolved_test(self, mu, orb1, orb2, 
				beta = 1.e2,			# smearing for Fermi factors
				dkx = np.array([1,0]),				# direction for taking k derivative
				dky = np.array([0,1]),
				k_step = 1.e-3,							# infinitesimal step
				#H_tb = np.array([0]), #hamiltonian that corresponds to tight binding-have the atomic distances here. NEEDS TO BE OF A SUPERCELL (where there is only one entry per term. )
				FS = False
				):

		# solve for tQGT for a given chemical potential

		def FNM(em, en):
			delta = 1.e-3  # Threshold for considering energies equivalent
			emn = em - en
			fn = Fermi(beta, en)
			fm = Fermi(beta, em)
			fnm = fn*(1.0 - fm)

			# Use np.where to handle the conditional logic
			# np.where(condition, [value if true], [value if false])
			if FS:
				result = np.where(np.abs(emn) < delta, -DFermi(beta, em), fnm / (emn**2))
			else:
				result = np.where(np.abs(emn) < delta, 0.0, fnm / (emn**2))
			return result
		k_phase = np.zeros(np.size(self.kSpan[:,0]))
		try:
			Jopx_orb = np.load(self.datastr+'_j_opx_%d_%d.npy'%(orb1, orb2))
			Jopy_orb = np.load(self.datastr+'_j_opy_%d_%d.npy'%(orb1, orb2))
		except:
			
			Jopx_orb = []
			Jopy_orb = []
			Hk = self.ham( self.kSpan[0] )
			U_alpha = np.zeros_like(Hk)
			U_alpha[orb1, orb1] = 1  # |alpha⟩⟨alpha|
			U_beta = np.zeros_like(Hk)
			U_beta[orb2, orb2] = 1  # |beta⟩⟨beta|
			print("you are using the right version")
			for i,k in enumerate( self.kSpan ):
				#H_tb_orb #0 is magnitude hopping, 1 is xdist, 2 is ydist
				Hk = self.ham(k)
				um = self.enUM[i,:,1:]
				#Hk_dk = self.ham(k + k_step*dkx)
				Hk_dkx = self.ham(k + k_step*dkx)
				Hk_dky = self.ham(k + k_step*dky)
				dHx = (Hk_dkx - Hk)/k_step #element-wise multiplication
				dHy = (Hk_dky - Hk)/k_step 
				um = np.asmatrix(um)
				#Gabby edited 5/21/25 (is the same output as what was previously there)
				#making matrix of current operators instead of one at a time
				Jvx_mat = np.zeros_like(Hk,dtype = 'object')
				Jvy_mat = np.zeros_like(Hk,dtype = 'object')
				for alpha in range(self.norb):
					for beta in range(self.norb):
						#alpha multiply rows
						alpha_mult = np.ones_like(Hk, dtype='object')* um[alpha,:]
						beta_mult = np.ones_like(Hk, dtype='object')* um[beta,:][:,None]
						#beta multiply columns
						Jvx_mat[alpha,beta] = um.H @ dHx @ um *alpha_mult * beta_mult
						Jvy_mat[alpha,beta] = um.H @ dHy @ um *alpha_mult * beta_mult

				Jopx_orb.append(Jvx_mat[0,0])
				Jopy_orb.append(Jvy_mat[0,0])
				#k_phase[i]=np.exp(1j*k[0]*xdist+1j*k[1]*ydist)
			#Jop_orb = np.array(Jop_orb)
			Jopx_orb = np.array(Jopx_orb)#*(np.exp(1j*k[0]*xdist+1j*k[1]*ydist)) #Might need to add back??
			Jopy_orb = np.array(Jopy_orb)#*(np.exp(1j*k[0]*xdist+1j*k[1]*ydist))
			if self.save_all:
				np.save(self.datastr+'_j_opx_%d_%d', Jopx_orb)
				np.save(self.datastr+'_j_opy_%d_%d', Jopy_orb)
			self.Jopx_orb = Jopx_orb
			self.Jopy_orb= Jopy_orb

		# Precompute 'em' and 'en' since they do not depend on 'time'
		em = self.enUM[:, :, 0] - mu #This is the energy value
		en = self.enUM[:, :, 0] - mu

		# Reshape for broadcasting to compute differences
		em = em[:, :, np.newaxis]  # Shape (num_i, norb, 1)
		en = en[:, np.newaxis, :]  # Shape (num_i, 1, norb)
		#should we do element-wise multiplication or matrix multiplication????
		Jvx2 = np.multiply(Jvx_mat,Jvx_mat.conj())
		Jvy2 = np.multiply(Jvy_mat,Jvy_mat.conj())
		Jvxy = np.multiply(Jvx_mat,Jvy_mat.conj())

		#print("check if the same",jxy==jyx,jxy,jyx)
		den = np.exp(1j * 0 * (en - em))		# it is an identity matrix
		pref = FNM(en, em) #Gabby added the square   # vectorized FNM

		qgt_xx = np.sum( pref * Jvx2 * den, axis=(1, 2))#*k_phase 
		qgt_yy = np.sum( pref * Jvy2 * den, axis=(1, 2))#*k_phase 
		qgt_xy = np.sum( pref * Jvxy * den, axis=(1, 2))#*k_phase 

		#qgt_yx = np.sum( pref * jyx * den, axis=(1, 2))*k_phase 
		qgt = qgt_xx+qgt_yy
		#print("check if the same",qgt_xy==qgt_yx,qgt_xy,qgt_yx)
		return qgt,qgt_xx,qgt_yy,qgt_xy
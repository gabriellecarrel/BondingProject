exec(open('nb_defs.py').read())
import time, pickle
h = build_models()["haldane"]
res = {}
for (n1, n2) in [(2, 2), (3, 3), (4, 5)]:
    t0 = time.time()
    m = h.make_supercell(np.array([[n1, 0], [0, n2]]))
    H, lat, pos = hopping_table(m); ham = make_ham_func(H, m)
    nk = [round(NK/n1), round(NK/n2)]
    tb = TBLR.TBLinearResponse(ham, Lar=np.array(lat), Nkp=nk, prestr=f"sc{n1}x{n2}", save_all=False, kpath=['G','X','M','G','Y','M'])
    occ = int(np.sum(tb.enUM[0, :, 0].real <= MU))
    qgt = calc_QM(tb, occ)
    atoms = [[i] for i in range(tb.norb)]
    dq, tq, _ = fluctuation_table(qgt["xx"]+qgt["yy"], atoms, "QM")
    db, tbc, _ = fluctuation_table(1j*(qgt["xy"]-qgt["yx"]), atoms, "BC")
    d = os.path.join(OUT, f"haldane_{n1}x{n2}"); os.makedirs(d, exist_ok=True)
    for c in qgt: np.save(os.path.join(d, f"qgt_{c}.npy"), qgt[c])
    dq.to_csv(os.path.join(d, "fluctuation_quantum_metric.csv"), index=False)
    db.to_csv(os.path.join(d, "fluctuation_berry_curvature.csv"), index=False)
    print(f"{n1}x{n2}: norb={tb.norb} k={nk} TrG={tq:.6f} Omega={tbc:.6f} C={2*np.pi*occ/abs(np.linalg.det(lat))*tbc:.5f}  {time.time()-t0:.0f}s", flush=True)

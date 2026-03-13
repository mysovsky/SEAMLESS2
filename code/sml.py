import qpp
import psi4
import numpy as np
from smlinp import strip_atom

bohr2angstroms = 0.529177208
au2ev = 27.211324570273

def common_elems(list1, list2):
    for l in list1:
        if l in list2:
            return True
    return False

# -----------------------------------------------------------------------------

def gdict_add(k1,g1,k2,g2):
    aa = [a for a in g1] + [a for a in g2 if not a in g1]
    g = {}
    v0 = np.array([0e0,0e0,0e0])
    for a in aa:
        g[a] = k1*g1.get(a,v0) + k2*g2.get(a,v0)
    return g

# ------------------------------------------------------------------------------

def reg_custom_basis(basdefn,ecpdefn = {}):

    basname = 'custom'+str(hash(basdefn.__repr__())).replace('-','_')
    intname = 'int_' + basname
    intbas  = False 

    block = []
    for label in basdefn:
        if type(basdefn[label]) == str:
            if basdefn[label] == 'none':
                continue
            block.append('assign ' + label + ' ' + basdefn[label] + '\n')
        else:
            block.append('assign ' + label + ' ' + intname + '\n')
            intbas = True

    if intbas:
        block.append('[' + intname + ']\n')
        for label in basdefn:
            if type(basdefn[label]) == list:
                block.append(label + ' 0\n')
                for l in basdefn[label]:
                    block.append(l)
        for label in ecpdefn:
            if type(ecpdefn[label]) == list:
                block.append(label + ' 0\n')
                for l in ecpdefn[label]:
                    if '****' in l:
                        break
                    block.append(l)
    print(''.join(block))
                
    psi4.basis_helper(''.join(block), name = basname )
    print()
    print(''.join(block))
    print()
    return basname

# ------------------------------------------------------------------------

def psi4_electric_field(wfn,points, do_potential = True, do_field = False):
    # wfn(in) : psi4.core.Wavefunction
    # points(in) : List[List[float]]
    # return value (out) : one or two np.array containing electric field and/or
    #                      electrostatic potential values
    # Calculate electric field at points created by
    # charge distribution of wfn
    # points are in angstroms!
    pt_mtr = psi4.core.Matrix.from_list(points)
    fcalc = psi4.core.ESPPropCalc(wfn)
    if do_potential:
        esp_mtr = fcalc.compute_esp_over_grid_in_memory(pt_mtr)
    if do_field:
        fld_mtr = fcalc.compute_field_over_grid_in_memory(pt_mtr)
    if do_potential and do_field:
        return fld_mtr.np, esp_mtr.np
    elif do_potential:
        return esp_mtr.np
    elif do_field:
        return fld_mtr.np

# ------------------------------------------------------------------------

def xgeom2mmcharges(geom,regions, return_dict = False, core_shell=False):
    chrg = {}
    for i in range(len(geom)):
        yes = False
        q = 0e0
        for j in range(len(geom.reg[i])):
            if geom.reg[i][j] in regions:
                yes = True
                q += geom.qmm[i][j]
        if yes:
            chrg[i] = [q, geom.x[i]/bohr2angstroms, geom.y[i]/bohr2angstroms, geom.z[i]/bohr2angstroms]
    if core_shell:
        snum = [a for a in chrg]
        cs=[-1 for i in snum]
        for i in range(len(snum)):
            k=geom.core_shells[snum[i]]
            if k>-1 and k in snum:
                cs[i] = snum.index(k)
#    chrg['coreshell'] = cs

    if return_dict:
        if core_shell:
            return chrg, cs
        else:
            return chrg
    else:
        if core_shell:
            return [chrg[a] for a in chrg],cs
        else:
            return [chrg[a] for a in chrg]

# -----------------------------------------------------------------------

def lbl2symbol(atom):
    dgs = [str(i) for i in range(10)]+['_']
    i = 0
    while i < len(atom):
        if  atom[i] in dgs:
            break
        i += 1
    return atom[:i]

# -----------------------------------------------------------------------
    
def xgeom2lbl(g,i, regions = []):
    if not isinstance(regions, dict):
        regions = {r:True for r in regions}
    rl = [ (r,l) for r,l in zip(g.reg[i],g.lbl[i]) if r in regions and regions[r] ]
    if len(rl) == 1:
        return rl[0][1]
    return g.atom[i]
    
# -----------------------------------------------------------------------

def xgeom2mol(geom, regions = [], return_dict = False, return_chrg = False, return_atoms = False):
    noregions = False
    try:
        geom.reg[0]
    except AttributeError:
        #print('Noregion via Attribute error')
        noregions = True        
    if len(regions) == 0:
        noregions = True

    if not isinstance(regions,dict):
        regions = { r:True for r in regions }
        
    if noregions:
        atoms = list(range(geom.nat()))
    else:
        atoms = [i for i in range(geom.nat()) if [r for r in geom.reg[i] if r in regions]!=[] ]
                
    # filter out cores
    atoms = [ i for i in atoms if not 'cor' in geom.atom[i] ]
        
    coord = [[geom.x[i], geom.y[i], geom.z[i]] for i in atoms]
    elbl  = [xgeom2lbl(geom, i, regions) for i in atoms]
    #print(elbl)
    elez  = [psi4.qcel.periodictable.to_atomic_number(lbl2symbol(a)) for a in elbl]

    partial = {}
    
    if not noregions:
        partial = {at:
                   sum([q for r,q in zip(geom.reg[at],geom.q[at])
                        if regions.get(r,False)])
                   for at in atoms}

    #print('------------------Partial charges------------------',partial)
    not_ghost = [True for at in atoms]
    chrg = {}
    for i in range(len(atoms)):
        at = atoms[i]
        Z  = elez[i]
        if abs(partial[at] - Z) > 1e-6:
            not_ghost[i] = False
            elbl[i] = 'Gh(' + elbl[i] + ')'
            
        if not not_ghost[i] and abs(partial[at]) > 1e-6:
            chrg[at] = [partial[at],
                        geom.x[at]/bohr2angstroms,
                        geom.y[at]/bohr2angstroms,
                        geom.z[at]/bohr2angstroms]
        
    moldict = {'geom':coord, 'elez':elez, 'elbl':elbl, 'real':not_ghost, 'fix_com':True,
               'fix_orientation':True, 'units':'angstrom'}
    
    if return_dict:
        mol = [moldict]
    else:        
        mol = [psi4.core.Molecule.from_dict(psi4.qcel.molparse.from_arrays(**moldict))]

    res = mol
    if return_chrg:        
        res = res + [chrg]
    if return_atoms:
        res = res + [atoms]
    return tuple(res)
        
    
# -----------------------------------------------------------------------
# Calculate part of molecule by SCF in electrostatic field of another part

def fragment_qm( geom,
                 fragdefn,
                 basdefn,
                 chargdefn = [],
                 frozen    = [],
                 do_ecp    = False,
                 do_d1e    = False,
                 do_d2e    = False,
                 dft       = None,
                 guess     = None,
                 projector = None,
                 charge    = 0,
                 mult      = 1
                ):
    if not isinstance(fragdefn, dict):
        fragdefn = {r:True for r in fragdefn}
    mol, cmol, atmol = xgeom2mol(geom,fragdefn, return_chrg=True, return_atoms = True)
    atactive = [i for i in range(geom.nat()) if [r for r in geom.reg[i] if r in frozen]==[] ]

    ghost_charge  = sum([cmol[a][0] for a in cmol])
    ##print(cmol)
    #print('mol=',atmol)
    #print('molcharge =', round(charge - ghost_charge), ghost_charge)

    mol.set_molecular_charge(round(charge - ghost_charge))
    mol.set_multiplicity(mult)
    mol.reset_point_group('c1')

    cext = xgeom2mmcharges(geom, chargdefn, return_dict = True)

    # find external charges that in fact belong to trick charges
    for a in cmol:
        if a in cext:
            cmol[a][0]+=cext[a][0]
            del cext[a]

    chrg = [cmol[a] for a in cmol]
    chrg2 = [cext[a] for a in cext]

    if isinstance(dft,dict):
        from smlinp import sml_dft_nested_DFTbuilder
        dft = sml_dft_nested_DFTbuilder(dft)

    if mult>1: psi4.core.set_global_option('REFERENCE','UHF')
    psi4.core.set_global_option('PRINT',0)

    if do_d1e:
        G, wfn = psi4.gradient('SCF', molecule = mol, basis = basdefn, dft_functional = dft,
                               return_wfn = True, external_potentials = chrg+chrg2)
        E = wfn.energy()
        
    else:
        E, wfn = psi4.energy('SCF', molecule = mol, basis = basdefn, dft_functional = dft,
                             return_wfn = True, external_potentials = chrg+chrg2)

    if mult>1:psi4.core.set_global_option('REFERENCE','RHF')
    #    psi4.core.set_global_option('PRINT',1)
    psi4.oeprop(wfn,'MULLIKEN_CHARGES')

    chrg_field  = qpp.coulomb_point_charges_d(chrg)
    chrg2_field  = qpp.coulomb_point_charges_d(chrg2)

    DE = 0.5*chrg_field.interaction_energy(chrg_field) +\
        chrg_field.interaction_energy(chrg2_field) 

    if do_d1e:
        grad = {a:g for a,g in zip(atmol,G.np)}

        ptmol = [[geom.x[a], geom.y[a], geom.z[a]] for a in cmol]
        ptext = [[geom.x[a], geom.y[a], geom.z[a]] for a in cext if a in atactive]

        if ptmol != []:
            efield_mol = psi4_electric_field(wfn, ptmol, do_field=True, do_potential = False)

            for i in range(len(cmol)):
                efield_mol[i] *= -chrg[i][0]                       

            efield_mol += chrg_field.interaction_gradients(chrg_field) + \
                chrg_field.interaction_gradients(chrg2_field)

            i = 0
            for a in cmol:
                if not a in grad:
                    grad[a] = np.array([0e0,0e0,0e0])
                grad[a] += efield_mol[i]
                i += 1

        if ptext != []:
            efield_ext = psi4_electric_field(wfn, ptext, do_field=True, do_potential = False)
            for i in range(len(ptext)):
                efield_ext[i] *= -chrg2[i][0]
            
            chrg2_active = qpp.coulomb_point_charges_d([cext[a] for a in cext if a in atactive])
            efield_ext += chrg2_active.interaction_gradients(chrg_field)

            i = 0
            for a in cext:
                if not a in atactive:
                    continue
                if not a in grad:
                    grad[a] = np.array([0e0,0e0,0e0])
                grad[a] = efield_ext[i]
                i += 1
            
    if do_d1e:
        return E+DE, wfn, grad
    else:
        return E+DE, wfn

# ----------------------------------------------------------
    
def fragment_coulomb( geom,
                      active_regions,
                      field_regions,
                      do_d1e = False,
                      do_d2e = False,
                      too_close = 1e-6
                     ):
    shell_model = (geom.core_shells!=[])
    r1=xgeom2mmcharges(geom, active_regions, return_dict = True, core_shell=shell_model)
    print(r1)
    if shell_model:
        active_chrg, active_cs = r1
    else:
        active_chrg = r1
        active_cs = []
    if shell_model:
        all_chrg,all_cs  = xgeom2mmcharges(geom, field_regions,  return_dict = True, core_shell=shell_model)
    else:
       all_chrg  = xgeom2mmcharges(geom, field_regions,  return_dict = True, core_shell=shell_model)
       all_cs = []
    active_fld = qpp.coulomb_point_charges_d([active_chrg[a] for a in active_chrg], active_cs)
    active_fld.too_close = too_close
    all_fld  = qpp.coulomb_point_charges_d([all_chrg[a] for a in all_chrg], all_cs)
    
    E = active_fld.interaction_energy(all_fld)
    if shell_model:
        isect_chrg, isect_cs = xgeom2mmcharges(geom, [r for r in active_regions if r in field_regions], core_shell=shell_model)
    else:
        isect_chrg  = xgeom2mmcharges(geom, [r for r in active_regions if r in field_regions], core_shell=shell_model)
        isect_cs = []
        
    isect_fld = qpp.coulomb_point_charges_d(isect_chrg, isect_cs)
    isect_fld.too_close = too_close

    E -= .5*isect_fld.interaction_energy(isect_fld)
    
    if do_d1e:
        G = active_fld.interaction_gradients(all_fld)
        g = {}
        i=0
        for a in active_chrg:
            g[a] = G[i]
            i += 1
        return E, g
    else:
        return E
    
def fragment_mm( geom,
                 potentials,
                 active_regions,
                 field_regions = [],
                 do_d1e    = False,
                 do_d2e    = False
                ):        
    mm_calc = qpp.mm_calculator_d()
    for p in potentials:
        mm_calc.add_potential(p)
    if field_regions==[]:
        field_regions = active_regions
    frozen_regions = [r for r in field_regions if not r in active_regions]
    mm_calc.set_active_regions(active_regions)    
    mm_calc.set_frozen_regions(frozen_regions)
    mm_calc.set_core_shells(geom.core_shells)

    if do_d1e:
        E, G = mm_calc.energy(geom, do_d1e = True)
        E /= au2ev
        G *= bohr2angstroms/au2ev
        g = {}
        for a in range(geom.nat()):
            if common_elems(geom.reg[a],active_regions):
                g[a] = G[a]
        return E, g#, G
    else:
        E = mm_calc.energy(geom)/au2ev
        return E
    
def interaction_mm(geom, potentials,
                   regions1, regions2,
                   do_d1e    = False,
                   do_d2e    = False):
    mm_calc = qpp.mm_calculator_d()
    for p in potentials:
        mm_calc.add_potential(p)
    mm_calc.set_active_regions(regions1)    
    mm_calc.set_frozen_regions(regions2)
    mm_calc.set_core_shells(geom.core_shells)
    if do_d1e:
        E, G = mm_calc.energy(geom, do_d1e = True)
        E /= au2ev
        G *= bohr2angstroms/au2ev
    else:      
        E = mm_calc.energy(geom)/au2ev
    mm_calc.set_active_regions(regions1)
    mm_calc.set_frozen_regions([])
    if do_d1e:
        E1, G1 = mm_calc.energy(geom, do_d1e = True)
        E1 /= au2ev
        G1 *= bohr2angstroms/au2ev
    else:      
        E1 = mm_calc.energy(geom)/au2ev

    if do_d1e:
        g={}        
        for a in range(geom.nat()):
            if common_elems(geom.reg[a],regions1 + regions2):
                g[a] = G[a] - G1[a]
        return E - E1, g#, G, G1 
    else:
        return E - E1

# ---------------------------------------------------------------------
    
def trisection_MM(inp, basdefn, do_d1e = False, do_d2e = False):
    geom        = inp['geometry']
    potentials  = inp['mm']
    projection  = inp.get('projection', False)
    dft         = inp.get('dft', None)
    frag_charge = inp.get('charge',[0,0,0])
    frag_mult   = inp.get('mult', [1,1,1])

    res13m = interaction_mm(geom, potentials, [1],[3], do_d1e = do_d1e, do_d2e = do_d2e)
    res23m = fragment_mm(geom, potentials, [2,3],field_regions=[2,3,4],do_d1e = do_d1e, do_d2e = do_d2e)
    res23c = fragment_coulomb( geom, [2,3], [2,3,4], do_d1e = do_d1e, do_d2e = do_d2e, too_close = 5e-1)
    
    charge12 = frag_charge[0] + frag_charge[1]
    res12q  = fragment_qm(geom, {1:True, 2:True}, basdefn,
                          chargdefn = [3,4],frozen = [4],
                          do_d1e = do_d1e, do_d2e = do_d2e,
                          dft = dft, charge  = charge12, mult = frag_mult[0]*frag_mult[1] )
    wfn12 = res12q[1]

    charge2 = frag_charge[1]
    if projection:
        fragdefn = {1:False, 2:True}
        proj = 1 # todo
    else:
        fragdefn = {2:True}
        proj = None
            
    res2q = fragment_qm(geom, fragdefn, basdefn,
                        chargdefn = [3,4],frozen = [4],
                        do_d1e = do_d1e, do_d2e = do_d2e,
                        projector = proj,
                        dft = dft, charge  = charge2, mult = frag_mult[1] )
    wfn2 = res2q[1]
    if do_d1e:
        E13m, G13m = res13m
        E23m, G23m = res23m
        E23c, G23c = res23c

        E12q = res12q[0]
        G12q = res12q[2]
        E2q  = res2q[0]
        G2q  = res2q[2]
        
    else:
        E13m = res13m
        E23m = res23m
        E23c = res23c

        E12q = res12q[0]
        E2q  = res2q[0]

    E = E12q - E2q + E23m + E23c + E13m
    if do_d1e:
        G = gdict_add(1, gdict_add(1, G12q, -1, G2q), 1, gdict_add(1, gdict_add(1, G23m, 1, G23c), 1, G13m))
        return E, G, {'w12':wfn12, 'w2':wfn2, 'g12q':G12q, 'g2q':G2q, 'g23m':G23m, 'g23c':G23c}
    else:
        return E, {'w12':wfn12, 'w2':wfn2}


def additive_QMMM(inp, basdefn, do_d1e = False, do_d2e = False):
    g          = inp['geometry']
    geom = qpp.xgeometry('double',qpp.periodic_cell_d(0),atom='str',
                      x='r', y='r', z='r', reg1 = 'i', reg2 = 'i', qmm1 = 'r', qmm2 = 'r',
                      q1 = 'r', q2 = 'r', lbl1 = 'str', lbl2 = 'str')                                            
    geom.__setattr__('reg',[r.copy() for r in g.reg])
    geom.q   = [q.copy() for q in g.q]
    geom.qmm = [q.copy() for q in g.qmm]
    geom.lbl = [l.copy() for l in g.lbl]
    for i in range(len(g)):
        geom.add(g[i])
 #       geom.reg.append(g.reg[i])
 #       geom.q.append(g.q[i])
 #       geom.qmm.append(g.qmm[i])
 #       geom.lbl.append(g.lbl[i])
    geom.core_shells = g.core_shells.copy()
    geom.build_types()
    
    potentials  = inp['mm']
#    projection  = inp.get('projection', False)
    dft         = inp.get('dft', None)
    frag_charge = inp.get('charge',[0,0])
    frag_mult   = inp.get('mult', [1,1])

    regecp=[]
    ecprad = inp.get('ecprad', 0e0)
    ecp_types = []
    ecp_charges = []
    ecps = []
    reg1 = [ i for i in range(geom.nat()) if (1 in geom.reg[i]) and not '_cor' in geom.atom[i] ]
    reg3 = [ i for i in range(geom.nat()) if (3 in geom.reg[i]) ]
    if ecprad > 1e-6:
        dist31 = [min([(geom.pos(i) - geom.pos(j)).norm() for j in reg1]) for i in reg3]
        reg3ecp = [reg3[i] for i in range(len(reg3)) if dist31[i] < ecprad]
        for i in reg3ecp:
            if '_shl' in geom.atom[i].lower():
                continue
            ecp_type = None
            elem = strip_atom(geom.atom[i]).lower()
            for k in inp['ecp']:
                if  elem == strip_atom(k).lower():
                    ecp_type = k
                    break
            if ecp_type:
                ecps.append(i)
                ecp_types.append(ecp_type)
                z_core = int(inp['ecp'][ecp_type][0].split()[2])
                ecp_charges.append(psi4.qcel.periodictable.to_atomic_number(elem) - z_core)

    #print(ecps)
    #print(ecp_types)
    #print(ecp_charges)
    ecpcharge = sum(ecp_charges)
    for i in range(len(ecps)):
        a = ecps[i]
        geom.reg[a] = [2]
        geom.reg1[a] = 2
        geom.lbl1[a] = ecp_types[i]
        geom.lbl[a] = [ecp_types[i]]
    #print('Alive after geom modify')
    res13m = interaction_mm(geom, potentials, [1,2],[1,2,3], do_d1e = do_d1e, do_d2e = do_d2e)
    #print('Alive after geom modify1')
    
    res3m = fragment_mm(geom, potentials, [3],field_regions=[3,4],do_d1e = do_d1e, do_d2e = do_d2e)
    #print('Alive after geom modify2')
    res3c = fragment_coulomb( geom, [3], [3,4], do_d1e = do_d1e, do_d2e = do_d2e, too_close = 5e-1)
    #print('Alive after geom modify3')
    charge1 = frag_charge[0]

        
    res1q  = fragment_qm(geom, [1,2], basdefn,
                          chargdefn = [3,4],frozen = [4],
                          do_d1e = do_d1e, do_d2e = do_d2e,
                          dft = dft, charge  = charge1+ecpcharge, mult = frag_mult[0] )
    wfn1 = res1q[1]

    if do_d1e:
        E13m, G13m = res13m
        print('',G13m[35])
        E3m, G3m = res3m
        E3c, G3c = res3c

        E1q = res1q[0]
        G1q = res1q[2]
    else:
        E13m = res13m
        E3m = res3m
        E3c = res3c

        E1q = res1q[0]

    E = E1q + E3m + E3c + E13m
    if do_d1e:
        G = gdict_add(1, gdict_add(1, G1q, 1, G13m), 1, gdict_add(1, G3m, 1, G3c))
        return E, G, {'w1':wfn1, 'g1q':G1q, 'g3m':G3m, 'g3c':G3c, 'g13m':G13m}
    else:
        return E, {'w1':wfn1}

def pure_MM(inp, data, mmregs=[1], do_d1e = False, do_d2e = False):
    geom     = inp['geometry']
    potentials  = inp['mm']
#    projection  = inp.get('projection', False)

    resmm = fragment_mm(geom, potentials, mmregs,field_regions=mmregs,do_d1e = do_d1e, do_d2e = do_d2e)
    resc = fragment_coulomb( geom, mmregs, mmregs, do_d1e = do_d1e, do_d2e = do_d2e, too_close = 5e-1)
    if do_d1e:
        E, G = resmm
        print(resc)
        Eq = resc[0]
        Gq = resc[1]
    else:
        E = resmm
        Eq = resc


    E = E + Eq
    if do_d1e:
        G = gdict_add(1, G, 1, Gq)
        return E, G, {}
    else:
        return E, {}
    

def additive(inp, basdefn, do_d1e = False, do_d2e = False):
    m1 = inp['methods'][0]
    m2 = inp['methods'][1]
    print('Entering additive m1=', m1, ' m2= ', m2)
    if m1 in ['scf', 'rhf', 'uhf', 'rks']:
        m1 = 'qm'
    if m1=='qm' and m2=='mm':
        #QM/MM
        return additive_QMMM(inp, basdefn, do_d1e = do_d1e, do_d2e = do_d2e)
    elif m1=='qm' and m2 =='cp2k':
        #QM/PDFT
        pass
    elif m1 == 'cp2k' and m2=='mm':
        #PDFT/MM region 1 is the region of PDFT atoms, 2 and 3 are iface and MM atoms
        print("PDFT+MM called")
        charge = inp.get('charge',[0,0])[0]
        mult = inp.get('mult',[1,1])[0]
        return pdft_MM(inp,basdefn, charge = charge, mult=mult, do_d1e = do_d1e, do_d2e = do_d2e)
    

def periodic_dft(inp,basdefn,regions,charge=0, mult=1,do_d1e = False, do_d2e = False):
    geom = inp['geometry']
    g = qpp.xgeometry('double',qpp.periodic_cell_d(3))
    cll = inp['cell']
    for j in [0,1,2]:
        for i in [0,1,2]:
            g.cell[i,j] = cll[3*i+j]
    for i in range(len(geom)):
        if geom.reg1[i] in regions or  geom.reg2[i] in regions:
            g.add(geom.atom[i],geom.pos(i))
    prog = inp.get('periodic_dft_program',None)
    if not prog:
        i = inp['methods'].index('pdft')
        prog = inp['programs'][i]

    if prog=='cp2k':
        from smlcp2k import cp2k_calculation
        eng,frc,rest = cp2k_calculation(g,inp,basdefn,charge=charge, mult=mult,do_d1e = do_d1e, do_d2e = do_d2e)
        if do_d1e:
            return eng,frc,rest
        else:
            return eng,rest
    elif prog=='vasp':
        raise NotImplementedError('VASP interface not implemented yet')
    else:
        raise NotImplementedError('Seamless itself is not capable of periodic DFT calculations, but rather relies on external program, e.g.CP2K. Provide correct choice of periodic_dft_program')
        
            
def pdft_MM(inp,basdefn, charge=0, mult=1, do_d1e = False, do_d2e = False):
    print('PDFT/MM called')
    print('regions: 1 - PDFT atoms, 2 - iface between PDFT and MM, 3- MM atoms')
    geom = inp['geometry']
    potentials = inp['mm']
    res13m = interaction_mm(geom, potentials, [1],[3], do_d1e = do_d1e, do_d2e = do_d2e)
    res23m = fragment_mm(geom, potentials, [2,3],field_regions=[2,3,4],do_d1e = do_d1e, do_d2e = do_d2e)
    res1m  = periodic_dft(inp,basdefn,[1],charge=charge, mult=mult, do_d1e = do_d1e, do_d2e = do_d2e)
    if do_d1e:
        E13m, G13m = res13m
        E23m, G23m = res23m
        E1q = res1m[0]
        G1q = res1m[1]
    else:
        E13m = res13m
        E23m = res23m
        E1q = res1m[0]
    if do_d1e:
        return E1q+E13m+E23m, gdict_add(1,G1q,1,gdict_add(1,G13m,1,G23m)), 'something else'
    else:
        return E1q+E13m+E23m, 'something else'
    
from math import tanh, cosh

def read_zfe(inp,data):
    geom = inp['geometry']
    n = len(geom)
    zfe = {}
    if 'zfe' in inp:
        zfe = inp['zfe']
        kx0 = inp['zfe'].get('kx0',0.3)
        zfile =open(zfe['filename'])
        g0 = qpp.xgeometry('d',qpp.periodic_cell_d(0))
        F0 = np.zeros((n,3))
        line=zfile.readline()
        print(line.split())
        line=zfile.readline()
        print(line.split())
        x1 = float(line.split()[0])
        i=0
        for line in zfile:
            if '#' in line: continue
            print(line)
            s = line.split()
            g0.add(s[0],*[float(ss) for ss in s[1:4]])
            F0[i,0] = float(s[4])/bohr2angstroms
            F0[i,1] = float(s[5])/bohr2angstroms
            F0[i,2] = float(s[6])/bohr2angstroms
            i+=1
        zfile.close()
        if inp.get('cp2k_force',False):
            for i in range(2,n):
                F0[i]-=F0[0]
            F0[0]-=F0[0]
        F0 = -F0
        zfe['F0'] = F0
        zfe['g0']=g0
        zfe['x1']=x1
        if 'bonds' in inp['zfe']:
            zfe['bonds'] = inp['zfe']['bonds']
            zfe['alpha_bond'] = inp['zfe'].get('alpha',0e0)
        data['zfe']=zfe
    else:
        data['zfe']=None
            
def zfe_add(inp, data):
    geom = inp['geometry']
    n = len(geom)
    if not 'zfe' in data:
        read_zfe(inp,data)
        
    if data['zfe'] is None:
        return 0e0, np.zeros((n,3))

    F0 = data['zfe']['F0']
    g0 = data['zfe']['g0']
    x1 = data['zfe']['x1']
    DR = np.zeros((n,3))
            
    for i in range(n):
        dr = geom.pos(i)-g0.pos(i)
        for j in [0,1,2]:
            DR[i,j] = dr[j]
    x = np.sum(F0*DR)
    f0 = np.linalg.norm(F0)
    x/=f0
 #       kx0=.3
    kx = 1./x1
    A = f0*x1
    Ub=0e0
    Gb = np.zeros((n,3))
    if 'bonds' in data['zfe']:
        alpha = data['zfe']['alpha_bond']
        for i,j in data['zfe']['bonds']:
            rij = geom.pos(i) - geom.pos(j)
            r = rij.norm()
            r0 = (g0.pos(i) - g0.pos(j)).norm()
            Ub += alpha*(r-r0)**2
            gij = 2*alpha*(r-r0)*rij/r
            print(i,j,Ub,gij,r,r0)
            #print(g0[i],g0[j],geom[i],geom[j])
            for k in [0,1,2]:
                Gb[i,k] += gij[k]
                Gb[j,k] -= gij[k]
    print('=======================Zero Force Elimination================')
    print('data=',data)
    print('f0 absolute value = ',f0)
    print('x coordinate is ',x)
    print('k=',kx, ' th= ', tanh(kx*x), ' A= ', A)
    return A*tanh(kx*x)+Ub, bohr2angstroms*(F0/cosh(kx*x)**2+Gb)
#    return A*tanh(kx*x), bohr2angstroms*F0/cosh(kx*x)**2


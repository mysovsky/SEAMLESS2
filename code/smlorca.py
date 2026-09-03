import subprocess
import os
import numpy as np
import sml
from sml import qpp

def run_orca(inp,ifname,ofname):
    orcainp = inp['orca']
    mpi_run = orcainp.get('mpirun','mpirun')
    mpi_opt = orcainp.get('mpiopt',[])
    orcarun = orcainp.get('orca','orca')
    orca_opt = orcainp.get('orcaopt',[])
#    if mpi_run != '':
#        comlist = [mpi_run]
#        for x in mpi_opt:
#            if x[0]=='$':
#                k = x[1:]
#                comlist.append(os.environ[k])
#            elif x=='threads':
#                threads = inp.get('threads',1)
#                comlist.append(str(threads))
#            else:
#                comlist.append(stripquotes(x))
#        comlist.append(orcarun)
#        comlist.append(orcaif)
#    else:
    comlist = [orcarun]
    for x in orca_opt:
        if x[0]=='$':
            k = x[1:]
            comlist.append(os.environ[k])
        elif x=='threads':
            threads = inp.get('threads',1)
            comlist.append(str(threads))
    comlist.append(ifname)
        
    with open(ofname, 'w') as ofile:
        print(comlist)
        subprocess.run(comlist, stdout=ofile)

def write_orca_inp_block(orcinp,f):
    kwdlineopts = []
    for k in orcinp:
        if orcinp[k] in [None, True]:
            kwdlineopts.append(k)
    print(('!'+'{} '*len(kwdlineopts)).format(*kwdlineopts),file=f)
    for k in orcinp:
        if k=="pointcharges":
            print('%pointcharges "{}"'.format(orcinp[k]),file=f)
        elif  k not in kwdlineopts and k!='replace_atom':
            print('%{}'.format(str(k)),file=f)
            for parm in orcinp[k]:
                print('  ',parm,orcinp[k][parm],file=f)
            print('end',file=f)

            
def orca_automethod(inp,m):
    methods = inp['methods']
    mtd = methods[m]
    orc_mtd=inp.get(mtd,'HF')
    nc = inp.get('threads',1) 
    if 'pal' not in orc_mtd:
        if nc>1:
            orc_mtd['pal']={'nprocs':nc}
        
    return orc_mtd

def write_orca_geom(geom,mol,ecps,atoms,replace,charge,mult,f):
    print('*xyz', int(charge),mult,file=f)
    for i in atoms:
        if i not in ecps:
            smb = geom.atom[i]
            if smb in replace:
                smb = replace[smb]
            else:
                smb = sml.lbl2symbol(smb)
            ghost = not mol['real'][atoms.index(i)]
            if ghost: smb=smb+':'
            print(smb,*geom.pos(i),file=f)
        
def write_orca_mol(mol,replace,charge,mult,f):
    print('*xyz', int(charge),mult,file=f)
    for i in range(len(mol['elbl'])):
        smb = mol['elbl'][i]
        ghost = False
        if 'Gh(' in smb:
            #ghost atom
            ghost=True
            smb =  smb.split('(')[1].split(')')[0]
        if '_ecp' in smb: continue
        if smb in replace:
            smb = replace[smb]
        if ghost:
            smb = smb + ':'
        else:
            smb  =smb.split('_')[0]
        r = mol['geom'][i]
        print(smb,*r,file=f)

def write_orca_basis_by_elem(geom,fragdefn,basdefn,ecpdefn,ecps,f):
    print('%basis',file=f)
    print('  GhostECP true', file=f)
    for k in basdefn:
        bas=basdefn[k]
        if isinstance(bas,str):
            print('  NewGTO {} \"{}\" end'.format(sml.lbl2symbol(k),bas),file=f)
        else:
            pass
    #bare ECPs
    #print(mol,ecps)
    #lbls = mol['elbl']
    ecplabels=[geom.atom[i] for i in ecps]
    print(ecplabels)
    ecplabels  =set(ecplabels)
        
    for k in ecpdefn:
        if len(k)>2: continue
        ecp=ecpdefn[k]
        if isinstance(ecp,str):
            print('  NewECP {} \"{}\" end'.format(sml.lbl2symbol(k),ecp),file=f)
        else:
            ss = orca_ecpdefn(ecpdefn,k)
            for l in ss:
                print('    ',l,file=f)
            print('  end', file=f)
    print('end',file=f)
        
        
def orca_ecpdefn(ecpdefn,k):
    res=[]
    spd=['s','p','d','f','g','h','i']

    elem = sml.lbl2symbol(k)
    s = ecpdefn[k][0]
    lmax = int(s.split()[1])
    nel = int(s.split()[2])
    res.append('NewECP {} N_core {} lmax {}'.format(elem,nel,spd[lmax]))
    i=0
    for l in [lmax]+list(range(lmax)):
        i+=2
        print(l,i,ecpdefn[k][i])
        s=ecpdefn[k][i]
        ncomp = int(s.split()[0])
        res.append('{} {}'.format(spd[l],ncomp))
        print(res[-1])
        for j in range(ncomp):
            i+=1
            s=ecpdefn[k][i].split()
            res.append('{} {} {} {}'.format(j+1,s[1],s[2],s[0]))
            #print(j,i,ecpdefn[k][i])
       

    return res

def write_orca_ecp(geom,data,fragdefn,ecps,ecpdefn,f):
    replace = data['replace']
    for i in ecps:
        smb = sml.xgeom2lbl(geom,i,fragdefn)
        if smb in replace:
            smb = replace[smb]
        else:
            smb = sml.lbl2symbol(smb)
        ecp = ecpdefn[sml.xgeom2lbl(geom,i,regions={2:True})]
        newecp = True
        if not isinstance(ecp,str):
            newecp=False
            ecp = smb
        q  = geom.qmm[i][0]
        if newecp:
            print('{}> {:15.10f} {:10.6f} {:10.6f} {:10.6f} NewECP \"{}\" end'.format(smb,q,*geom.pos(i),ecp),file=f)
        else:
            print('{}> {:15.10f} {:10.6f} {:10.6f} {:10.6f}'.format(smb,q,*geom.pos(i)),file=f)

def write_orca_charges(chrg,fname):
    b=sml.bohr2angstroms
    if chrg==[]: return
    f = open(fname,'w')
    print(len(chrg),file=f)
    print('',file=f)
    for c in chrg:
        print('{:13.8f} {:10.6f} {:10.6f} {:10.6f}'.format(c[0],c[1]*b, c[2]*b, c[3]*b),file=f)
    f.close()

def write_orca_inp(inp,data,orcinp,geom,fragdefn,basdefn,ecpdefn,charge,mult,fname,charges=[],recp={}):
    if not isinstance(fragdefn, dict):
        fragdefn = {r:True for r in fragdefn}
    mol, cmol, atmol = sml.xgeom2mol(geom,fragdefn, return_chrg=True, return_atoms = True, return_dict=True)
    if len(charges)>0:
        orcinp['pointcharges']='pointcharges.pc'
        write_orca_charges(charges,'pointcharges.pc')
    nc=qpp.globals.ncores
    replace = data['replace']
    if nc>1:
        orcinp['pal']={'nprocs':nc}
    f=open(fname+'.inp','w')
    write_orca_inp_block(orcinp,f)
    if isinstance(basdefn,str) and 'custom' in basdefn:
        basdefn = inp['basis']
    lbls = mol['elbl']
    ecps = data['reg3ecps']
    #ecps=[atmol[i] for i in range(len(atmol)) if lbls[i] in ecpdefn and lbls[i] not in basdefn]
    #print(recp)
    #print(ecps, atmol,lbls)
    #for i in ecps+atmol:
    #    j = atmol.index(i)
    #    if lbls[j] in recp:
            #print(j,i,lbls[j], recp[lbls[j]])
    #        gls = geom[i]
    #        gls[0] = recp[lbls[j]]
    #        geom[i] = gls
    write_orca_basis_by_elem(geom,fragdefn, basdefn,ecpdefn,ecps,f)
    write_orca_geom(geom,mol,ecps,atmol,replace,charge,mult,f)
    #write_orca_mol(mol,replace,charge,mult,f)
    print('ecps=',ecps)
    for i in range(100):
        print(i,geom[i])
    write_orca_ecp(geom,data,fragdefn,ecps,ecpdefn,f)
    print('*', file=f)
    f.close()
    
def orca_dft_spec(dft):
    pass

def read_orca_ed1e(ofname, do_d1e=False):
    of = open(ofname+'.out')
    while True:
        l = of.readline()
        if 'Total Energy' in l:break
    E = float(l.split()[3])
    of.close()
    if do_d1e:
        of = open(ofname+'.engrad')
        while 'Number of atoms' not in l:
            l = of.readline()
        l = of.readline()
        l = of.readline()
        N = int(l)
        while 'The current gradient'not in l:
            l = of.readline()
        l = of.readline()
        G = []
        for i in range(N):
            gx = float(of.readline())
            gy = float(of.readline())
            gz = float(of.readline())
            G.append(np.array([gx,gy,gz]))
        of.close()
        of = open(ofname+'.pcgrad')
        N =int(of.readline())
        Gq = []
        for i in range(N):
            l=of.readline()
            Gq.append(np.array([float(s) for s in l.split()]))
        return E,G,Gq
    else:
        return E
    
        
def fragment_orca( inp,
                   data,
                   geom,
                   fragdefn,
                   basdefn,
                   ecpdefn,                  
                   chargdefn = [],
                   frozen    = [],
                   do_ecp    = False,
                   do_d1e    = False,
                   do_d2e    = False,
                   method    = 'SCF',
                   dft       = None,
                   guess     = None,
                   projector = None,
                   charge    = 0,
                   ecpcharge = 0,
                   mult      = 1
                  ):
  
    atactive = [i for i in range(geom.nat()) if [r for r in geom.reg[i] if r in frozen]==[] ]

    #ghost_charge  = sum([cmol[a][0] for a in cmol])

    orcdict = method.copy()
    replace = orcdict.get('replace_atom',{})
    if 'replace' not in data:
        data['replace'] = replace
    print(replace)
    if isinstance(basdefn,str):
        basdefn = inp['basis']
    for k in replace:
        if k in ecpdefn:
            ecpdefn[replace[k]] = ecpdefn[k]        
    for k in replace:
        if k in basdefn:
            basdefn[replace[k]] = basdefn[k]
    #if 'replace_ecp' in orcdict:
    #    del orcdict['replace_ecp']
    ##print(cmol)
    #print('mol=',atmol)
    #print('molcharge =', round(charge - ghost_charge), ghost_charge)

    #mol.set_molecular_charge(round(charge - ghost_charge))
    #mol.set_multiplicity(mult)
    #mol.reset_point_group('c1')
    print(basdefn,ecpdefn)
   
    mol, cmol, atmol = sml.xgeom2mol(geom,fragdefn, return_chrg=True, return_atoms = True, return_dict=True)

    print(mol,atmol, cmol)
    
    cext = sml.xgeom2mmcharges(geom, chargdefn, return_dict = True)   

    # find external charges that in fact belong to trick charges
    for a in cmol:
        if a in cext:
            cmol[a][0]+=cext[a][0]
            del cext[a]

    chrg = [cmol[a] for a in cmol]
    chrg2 = [cext[a] for a in cext]

    orcdict['pointcharges']='pointcharges.pc'
    
    
    if isinstance(dft,dict):
        #from smlinp import sml_dft_nested_DFTbuilder
        #dft = sml_dft_nested_DFTbuilder(dft)
        dft = orca_dft_spec(dft)

#    if mult>1: psi4.core.set_global_option('REFERENCE','UHF')
#    psi4.core.set_global_option('PRINT',0)

    if do_d1e:
        #G, wfn = psi4.gradient(method, molecule = mol, basis = basdefn, dft_functional = dft,
        #                       return_wfn = True, external_potentials = chrg+chrg2)
        #E = wfn.energy()
        orcdict['EnGrad']=True
        
    #else:
        #E, wfn = psi4.energy(method, molecule = mol, basis = basdefn, dft_functional = dft,
        #                     return_wfn = True, external_potentials = chrg+chrg2)

    #if mult>1:psi4.core.set_global_option('REFERENCE','UHF')
    #    psi4.core.set_global_option('PRINT',1)
    #psi4.oeprop(wfn,'MULLIKEN_CHARGES')

    fname=inp.get('orca',None)
    if fname:
        fname = fname.get('inp','inp.inp')
        fname = '.'.join(fname.split('.')[:-1])
    else:
        fname = 'orcsml'
    fname = fname + ''.join([str(r) for r in fragdefn])
    ofname = inp.get('orca',None)
    if ofname:
        ofname = ofname.get('out','out.out')
        ofname = '.'.join(ofname.split('.')[:-1])
    else:
        ofname = 'orcsml'
    ofname = ofname + ''.join([str(r) for r in fragdefn])
    print(ofname,fragdefn)
    print(orcdict)
    write_orca_inp(inp,data,orcdict,geom,fragdefn,basdefn,ecpdefn,charge-ecpcharge,mult,fname,charges=chrg+chrg2, recp = replace)

    run_orca(inp,fname+'.inp',ofname+'.out')

    if do_d1e:
        E,g,gq = read_orca_ed1e(fname,do_d1e=True)
        grad = {a:g for a,g in zip(atmol,g)}
        i=0
        for a in cmol:
            grad[a] = gq[i]
            i+=1            
        for a in cext:
            grad[a] = gq[i]
            i+=1
        return E,None,grad
            
    else:
        E = read_orca_ed1e(fname)
        return E,None


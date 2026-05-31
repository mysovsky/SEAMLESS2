#!/usr/bin/python3 -i
#  #!/home/amysovsky/envr/bin/python3 -i

import sys
import numpy as np
#import qpp
#import psi4
import sml
from sml import psi4
from sml import qpp
import smlinp
from bfgs import bfgs
from smldrv import single_point, nactive_atoms, coords_to_optimize, \
    energy_to_optimize, gradients_to_optimize

import faulthandler
faulthandler.enable()

f = open(sys.argv[1])
inp = smlinp.read_sml(f)
data = {}

psi4.core.set_output_file('/dev/stdout', False)

memory = inp.get('memory','1Gb')
psi4.set_memory(memory)

threads = inp.get('threads',1)
psi4.core.set_num_threads(threads)
qpp.globals.ncores = threads

programs = inp.get('programs','psi4')

basdefn = inp['basis']
ecpdefn = inp.get('ecp',{})
if programs == 'psi4' or programs[0] == 'psi4':
    if isinstance(basdefn, dict):
        basdefn = sml.reg_custom_basis(basdefn,ecpdefn)

geom = inp['geometry']
geom.core_shells = qpp.find_core_shells(geom, .5)
n = nactive_atoms(geom)
    

runtype = inp.get('runtype','energy')

if runtype == 'check':
    pass
elif runtype in ['energy','gradients','genzfe']:
    
    do_d1e = (runtype in ['gradients', 'genzfe'])
    
    results = single_point(inp, data, basdefn, do_d1e = do_d1e, do_d2e = False)

    if do_d1e:
        FSCF_E = results[0]
        FSCF_G = results[1]
        results = results[2:]
    else:
        FSCF_E = results[0]
        results = results[1:]
        
    print("FINAL SML TOTAL ENERGY = ", FSCF_E)
    if do_d1e:
        gg1 = max([FSCF_G[a].max() for a in FSCF_G])
        gg2 = min([FSCF_G[a].min() for a in FSCF_G])
        print("SML MAXIMUM GRADIENT", gg1 if gg1>-gg2 else -gg2)
    if runtype == 'genzfe':
        f=open('sml.zfe','w')
        print(n,file=f)
        print('Erlx',file=f)
        for i in range(n):
            print(geom.atom[i],*geom.pos(i), *FSCF_G[i], file=f)
        f.close()

elif runtype == 'numgrad':
    numgrad = inp.get('numgrad',{})
    step  = numgrad.get('step',0.01)
    atoms = numgrad.get('atoms',list(range(n)))
    G = np.zeros((n,3))
    ZG = np.zeros((n,3))
    results = {}
    orig_coord = [[geom.coord[i,j] for j in [0,1,2]] for i in range(n)]
    for i in atoms:
        for j in [0,1,2]:
            for i1 in range(n):
                for j1 in [0,1,2]:
                    geom.coord[i1,j1] = orig_coord[i1][j1]
            geom.coord[i,j] += step
            FSCF_E1, results[i,j, 1] = single_point(inp, data, basdefn, do_d1e=False, do_d2e = False)
            ZE1,__ = sml.zfe_add(inp,data)
            geom.coord[i,j] -= 2*step
            FSCF_E2, results[i,j,-1] = single_point(inp, data, basdefn, do_d1e=False, do_d2e = False)
            ZE2,__ = sml.zfe_add(inp,data)

            G[i,j] = 0.5*sml.bohr2angstroms*(FSCF_E1 - FSCF_E2)/step
            ZG[i,j] = 0.5*sml.bohr2angstroms*(ZE1 - ZE2)/step
        print('Final numerical gradients\n')
        print(G, ZG)
        
elif runtype == 'opt':
    opt = inp.get('opt',{})
    alpha    = opt.get('alpha',1)
    gtol     = opt.get('gtol', 5e-4)
    minstep  = opt.get('minstep', 1e-3)
    maxstep  = opt.get('maxstep', 2e-1)
    maxiter  = opt.get('maxiter', 500)
    #core_adjust = False
    maxbfgs  = opt.get('maxbfgs',3)

    #core_shells = qpp.find_core_shells(geom, .5)
    #core_list = []
    #if core_adjust:
    #    core_list = [i for i in range(n) if core_shells[i]>-1 and '_cor' in geom.atom[i] ]
    
    coord = coords_to_optimize(geom)
    coord0 = np.copy(coord)
    #initialize history
    history = []
    energy_to_optimize(coord,history,inp,data,basdefn,alpha,coord0)
    
    while maxbfgs>0:
        xopt = bfgs(energy_to_optimize, coord, fprime=gradients_to_optimize, \
                    args=(history,inp,data,basdefn,alpha,coord0), gtol=gtol, \
                    maxiter=maxiter, full_output=1, disp=1, \
                    retall=0, callback=None, maxstep = maxstep, minstep = minstep)
        if xopt.success:
            break
        coord = xopt.x
        coord0 = np.copy(coord)
        maxbfgs -= 1
    
    try:
        f = open('opt.log','a')
    except FileNotFoundError:
        f = open('opt.log','w')
    print('Optimization status: ', xopt, file=f)
    #print(xopt.message, file=f)
    f.close()
    
    

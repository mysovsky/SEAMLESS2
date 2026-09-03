import sml
from sml import np
from smlinp import write_sml_geometry
from math import sqrt

def single_point(inp,data,basdefn,ecpdefn,do_d1e = False, do_d2e = False):
    qmethods = ['hf', 'scf', 'rhf', 'uhf', 'rks', 'eom-ccsd']
    
    methods = inp['methods']
    partition = inp.get('partition','sml')
    print('Single point methods= ', methods, ' partition= ', partition)
    print('partition=',partition)
    if partition == 'pure':
        mtd = methods[0] if isinstance(methods,list) else methods 
        print('method for pure=',mtd, methods)
        if mtd== 'mm':
            results = sml.pure_MM(inp, data,do_d1e=do_d1e, do_d2e = do_d2e, mmregs = [1,2,3])
        elif mtd=='pdft':
            results= sml.periodic_dft(inp,data,basdefn,[1,2,3],charge = inp.get('charge',0), mult=inp.get('mult',1),do_d1e = do_d1e, do_d2e = do_d2e)
    elif partition in ['qmmm','add']:
        print(methods)
        if methods[0] in qmethods and methods[1] == 'mm':
            results = sml.additive_QMMM(inp, basdefn, do_d1e = do_d1e, do_d2e = do_d2e)
        else:
            print('PDFT+MM goes here')
            results = sml.additive(inp, basdefn, do_d1e = do_d1e, do_d2e = do_d2e)
    elif partition == 'sml':
        print('sml', methods)
        if methods[-1] == 'mm':
            print('sml-qm/mm')
            results =  sml.trisection_MM(inp,data, basdefn, ecpdefn, do_d1e = do_d1e, do_d2e = do_d2e)
        elif methods[2] in ['scf', 'rhf', 'uhf', 'hf']:
            results =  sml.trisection_QM(inp, basdefn, ecpdefn, do_d1e = do_d1e, do_d2e = do_d2e)
    else:
        # more complicated fragmentation patterns
        pass
    zfe=inp.get('zfe',None)
    if zfe:
        E_zfe, G_zfe = sml.zfe_add(inp,data)
        E = results[0] + E_zfe
        Gzfe_dct = {i:G_zfe[i] for i in range(len(G_zfe))}
        if not do_d1e:
            return E, results[1:]
        else:
            G = sml.gdict_add(1,results[1],1, Gzfe_dct)
            return E,G, results[-1]
    else:
        return results
        
    
# ---------------------------------------------------------------------------
    
def nactive_atoms(geom):
    n = 0
    #while geom.reg1[n]!=geom.reg1[-1]:
    while n<len(geom) and geom.reg[n][0]!=4:
        n += 1
    return n
     
# ---------------------------------------------------------------------------

def maxcoord(nparray):
    a1 = nparray.max()
    a2 = nparray.min()
    return a1 if a1>-a2 else -a2

# ---------------------------------------------------------------------------
    
def energy_to_optimize(coord,history,inp,data,basdefn,ecpdefn,alpha,coord0):
    geom = inp['geometry']
    n = nactive_atoms(geom)
    x = (1-alpha)*coord0 + alpha*coord
    k = 0
    for i in range(n):
        for j in [0,1,2]:
            geom.coord[i,j] = x[k]
            k += 1

    try:
        f = open('geom.xyz','a')
    except FileNotFoundError:
        f = open('geom.xyz','w')
    print(n,file=f)
    print('',file=f)
    for i in range(n):
        print('{:6} {:12.7f} {:12.7f} {:12.7f}'.format(geom.atom[i],geom.x[i],geom.y[i],geom.z[i]),file=f)
    f.close()
    print('before single point')
    FSCF_E, __ =  single_point(inp, data, basdefn, ecpdefn, do_d1e = False, do_d2e = False)
    iter = len(history)
    dx=np.copy(x)
    if len(history)>0:
        dx -= history[-1]['coordinates']
    else:
        dx -= dx
    maxdx = maxcoord(dx)
    history.append({'iter':iter, 'energy':FSCF_E,
                    'coordinates':x} )
    print('SML ENG: it={:4d} energy={:15.8f} max disp={:8.6f} norm disp={:7.5f}'.format(iter, FSCF_E, maxdx, np.linalg.norm(dx) ))

    try:
        f = open('opt.log','a')
    except FileNotFoundError:
        f = open('opt.log','w')
    print('SML ENG: it={:4d} energy={:15.8f} max disp={:8.6f} norm disp={:7.5f}'.format(iter, FSCF_E, maxdx, np.linalg.norm(dx) ), file = f)        
    f.close()

    f=open('geom.rst','w')
    write_sml_geometry(f,geom)
    f.close()
    
    return FSCF_E


# ---------------------------------------------------------------------------


def gradients_to_optimize(coord,history,inp,data,basdefn,ecpdefn,alpha,coord0):
    geom = inp['geometry']
    n = nactive_atoms(geom)
    x = (1-alpha)*coord0 + alpha*coord
    k=0
    for i in range(n):
        for j in [0,1,2]:
            geom.coord[i,j] = x[k]
            k+=1    
    try:
        f = open('opt.log','a')
    except FileNotFoundError:
        f = open('opt.log','w')

    #print('PZL GRD REQ', file = f)
    
    FSCF_E, FSCF_G, __ = single_point(inp, data, basdefn, ecpdefn, do_d1e = True, do_d2e = False)

    maxg = max([maxcoord(FSCF_G[a]) for a in range(n)])
    normg = sqrt(sum([FSCF_G[a].dot(FSCF_G[a]) for a in range(n)]))
    dx=np.copy(x)
    iter = len(history)
    if len(history)>0:
        dx -= history[-1]['coordinates']
    else:
        dx -= dx
    maxdx = maxcoord(dx)
    
    history.append({'iter':iter, 'energy':FSCF_E, 'gradients': FSCF_G,
                    'coordinates':x} )

    print('SML GRD: it={:4d} max grd={:8.6f}  norm grd={:7.5f}'.format(iter,  maxg, normg ))

    print('SML GRD: it={:4d} max grd={:8.6f}  norm grd={:7.5f}'.format(iter,  maxg, normg ), file = f)

    print('\nGradients:\n',file=f)
    for i in range(n):
        print('{:6} {:12.7f} {:12.7f} {:12.7f}'.format(geom.atom[i],*FSCF_G[i]),file=f)
    print('\n',file=f)

    f.close()

    G = np.zeros(3*n)
    k=0
    for i in range(n):
        for j in [0,1,2]:
            G[k] = FSCF_G[i][j]
            k+=1
    return G*alpha

def coords_to_optimize(geom):
    n = nactive_atoms(geom)
    coord = np.zeros(n*3)
    k = 0
    for i in range(n):
        #if i in core_list:
        #    continue
        for j in [0,1,2]:
            coord[k] = geom.coord[i,j]
            k += 1
    return coord



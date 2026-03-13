
import subprocess
import os
import numpy as np

def typical_cp2k_force():
    return {
        'GLOBAL': {
            'PRINT_LEVEL': 'MEDIUM',
            'PROJECT_NAME': '"cp2k"',
            'RUN_TYPE': 'ENERGY_FORCE'
        },
        'MOTION': {
            'GEO_OPT': {
                'TYPE': 'MINIMIZATION',
                'OPTIMIZER': 'LBFGS',
                'MAX_ITER': '1000',
                'MAX_DR': '1e-4',
                'MAX_FORCE': '1e-4',
                'RMS_DR': '1e-4',
                'RMS_FORCE': '1e-4',
                'STEP_START_VAL': '0'
            },
            'CONSTRAINT': {
                'FIXED_ATOMS': {
                    'COMPONENTS_TO_FIX': 'XYZ',
                    'LIST': '1'
                }
            },
            'PRINT': {
                'TRAJECTORY': {}}
        },
        'FORCE_EVAL': {
            'METHOD': 'QS',
            'STRESS_TENSOR': 'ANALYTICAL',
            'DFT': {
                'BASIS_SET_FILE_NAME': '',
                'POTENTIAL_FILE_NAME': '',
                'UKS': 'FALSE',
                'MULTIPLICITY': '1',
                'QS': {'EPS_DEFAULT': '1e-10'},
                'MGRID': {
                    'NGRIDS': '4',
                    'CUTOFF': '300',
                    'REL_CUTOFF': '60'
                },
            'SCF': {
                'SCF_GUESS': 'RESTART',
                'MAX_SCF': '50',
                'EPS_SCF': '1e-8',
                'OT': {'MINIMIZER': 'CG', 'PRECONDITIONER': 'FULL_ALL'},
                'OUTER_SCF': {'EPS_SCF': '1e-08', 'MAX_SCF': '20'}},
                'XC': {'XC_GRID': {'XC_SMOOTH_RHO': 'NN10', 'XC_DERIV': 'SPLINE2_SMOOTH'},
                       'XC_FUNCTIONAL': {
                           'PBE': {'PARAMETRIZATION': 'PBESOL'}}}
            },
            'SUBSYS': {
                'CELL': {'include':'geom.inp'},
                'KINDS': {'include':'kinds.inp'}
            },
            'PRINT': {
                'FORCES': {'value':'ON'}
            }
        }
    }

    
def stripquotes(x):
    s=''
    for a in x:
        if not a in ['"',"'"]:
            s+=a
    return s

def run_cp2k(inp):
    cp2kinp = inp['cp2k']
    mpi_run = cp2kinp.get('mpirun','mpirun')
    mpi_opt = cp2kinp.get('mpiopt',[])
    cp2krun = cp2kinp.get('cp2k','cp2k.popt')
    cp2kif = cp2kinp.get('inp','inp.inp')
    cp2kof = cp2kinp.get('out','out.out')
    comlist = []
    if mpi_run != '':
        comlist = [mpi_run]
        for x in mpi_opt:
            if x[0]=='$':
                k = x[1:]
                comlist.append(os.environ[k])
            elif x=='threads':
                threads = inp.get('threads',1)
                comlist.append(str(threads))
            else:
                comlist.append(stripquotes(x))
        comlist.append(cp2krun)
        comlist.append(cp2kif)
    with open(cp2kof, 'w') as ofile:
        print(comlist)
        subprocess.run(comlist, stdout=ofile)

def write_cp2k_geometry(geom, ofname):
        inp = []
        indt = '    '
        inp.append(indt + '&TOPOLOGY')
        inp.append(indt + '  NUMBER_OF_ATOMS ' + str(len(geom)))
        inp.append(indt + '  MULTIPLE_UNIT_CELL 1 1 1')
        inp.append(indt + '&END TOPOLOGY')
        inp.append(indt + '&CELL')
        inp.append(indt + "  A  {:13.8f} {:13.8f} {:13.8f}".format(*geom.cell[0]))
        inp.append(indt + "  B  {:13.8f} {:13.8f} {:13.8f}".format(*geom.cell[1]))
        inp.append(indt + "  C  {:13.8f} {:13.8f} {:13.8f}".format(*geom.cell[2]))
        inp.append(indt + '  MULTIPLE_UNIT_CELL 1 1 1')
        inp.append(indt + '&END CELL')
        inp.append(indt + '&COORD')
        for i in range(len(geom)):
                inp.append(indt + "  {:2s} {:12.8f} {:12.8f} {:12.8f}".format(geom.atom[i],*geom.cell.cart2frac(geom.pos(i))))
        inp.append(indt + '  SCALED T')
        inp.append(indt + '&END COORD')
        #return inp
        with open(ofname, 'w') as v:
                for line in inp:
                        print(line, file=v)

def write_cp2k_block(block, offset, f):
    for key in block:
        if key=='value': continue
        val = block[key]
        if isinstance(val,dict):
            keys = [k for k in val]
            if len(keys)==1 and keys[0].lower()=='include':
                print(' '*offset+'@include "'+val[keys[0]]+'"', file=f)
            elif 'value' in val:
                print(' '*offset +"&"+key+' '+val['value'], file=f)
                write_cp2k_block(val,offset+2,f)
                print(' '*offset +"&END "+key, file=f)
            else:
                print(' '*offset +"&"+key, file=f)
                write_cp2k_block(val,offset+2,f)
                print(' '*offset +"&END "+key, file=f)
        else:
            if isinstance(val,list):
                if val[0]=='repeat':
                    for x in val[1:]:
                        print(' '*offset + key,' ',x, file=f)
                else:
                    print(' '*offset + key,' ',*val, file=f)
            else:
                print(' '*offset + key,' ',val, file=f)
            
def write_cp2k_input(cp2kinp,f):
    if not isinstance(cp2kinp,dict):
        raise TypeError('CP2K input must be in the form of dictionary')
    write_cp2k_block(cp2kinp,0,f)

def write_cp2k_kinds(basdefn,potdefn,fname):
    with open(fname, 'w') as v:
        for at in basdefn:
            print('&KIND ',at,file=v)
            print('  BASIS_SET ORB ',basdefn[at],file=v)
            print('  POTENTIAL ',potdefn[at],file=v)
            print('&END KIND',file=v)            


def read_energy_forces(fname):
        with open(fname) as f:
                content = f.readlines()
                for i,line in enumerate(content):
                        if " ENERGY| Total FORCE_EVAL ( QS ) energy" in line: # a.u.
                                energy = float(line.split()[-1])
                        if "Atomic forces" in line: # a.u.
                                forces_start = i + 2
                        if "FORCES| Sum " in line: # a.u.
                                forces_end = i
                elements = [line.split()[2] for line in content[forces_start:forces_end]]
                forces = {(int(line.split()[1])-1):-np.array(list(map(float, line.split()[2:5]))) for line in content[forces_start:forces_end]}
                return energy,  forces, elements
            
def deep_update_existing(target_dict, update_dict):
    for key, value in update_dict.items():
        # If the key exists in target_dict and both are dictionaries, recurse
        if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
            target_dict[key] = deep_update_existing(target_dict[key], value)
            # Otherwise, add/update the value
            #        elif key in target_dict:
        else:
            target_dict[key] = value
    return target_dict

def cp2k_calculation(geom,inp,basdefn,charge = 0, mult = 1, do_d1e = False, do_d2e = False):
    write_cp2k_geometry(geom,'geom.inp')
    potdefn = inp['dftpot']
    write_cp2k_kinds(basdefn,potdefn,'kinds.inp')
    cp2kinp = typical_cp2k_force()
    if charge != 0:
        cp2kinp['FORCE_EVAL']['DFT']['CHARGE']=charge
    if mult > 1:
        cp2kinp['FORCE_EVAL']['DFT']['MULTIPLICITY']=mult
        cp2kinp['FORCE_EVAL']['DFT']['UKS']=True
    cp2kopt = inp.get('cp2kopt',{})
    if cp2kopt!={}:
        deep_update_existing(cp2kinp, cp2kopt)
    inpname = inp.get('cp2k',{}).get('inp','inp.inp')
    write_cp2k_input(cp2kinp, open(inpname,'w'))
    run_cp2k(inp)
    return read_energy_forces(inp.get('cp2k',{}).get('out','out.out'))

def collect_cp2k_lines(fname):
    res=[]
    with open(fname) as f:
        for l in f:
            if l.split()[0].lower()=='@include':
                res += collect_cp2k_lines(stripquotes(l.split()[1]))
            else:
                l1 = l.split('#')[0]
                l1 = l1.split('!')[0]
                if l1!= '':
                    res.append(l1)
    return res

def read_cp2k_block(name,lines,i):
    j=i+1
    block={}
    # special blocks - COORD, CELL
    if name.upper() in ['COORD','CELL']:
        while j<len(lines):
            l=lines[j]
            if l.split()[0].upper()=='&END' and l.split()[1].upper()==name.upper():
                break
            j+=1
        if j>=len(lines):
            print('EOF:', name, l)
            raise EOFError('Premature end of cp2k input')
    while j<len(lines):
        l=lines[j]
        print(l)
        first = l.split()[0]
        if first.upper()=='&END':
            if l.split()[1].upper()!=name.upper():
                raise SyntaxError(l+'do not match '+name +' block')
            else:
                return block,j+1
        elif first[0]=='&':
            print(first,l)
            block[first[1:]],k = read_cp2k_block(first[1:],lines,j)
            print(block)
            j=k
        else:
            llist = l.split()
            if len(llist)==2:
                val = llist[1]
            else:
                val =  [llist[k] for k in range(1,len(llist))]
            block[first]=val
            j+=1
    raise EOFError('Premature end of cp2k input')

def read_cp2k_input(fname):
   lines = collect_cp2k_lines(fname)
   cp2kinp={}
   i=0
   while i<len(lines):
       l = lines[i]
       first = l.split()[0]
       if first[0]=='&':
           cp2kinp[first[1:]],j=read_cp2k_block(first[1:],lines,i)
           i=j
   return cp2kinp
           
           
           
   
    

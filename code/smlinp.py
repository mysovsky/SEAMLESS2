from sml import qpp
from sml import psi4

key_synonims = {'runtype':   ['run','runtyp','runtype'],
                'memory':    ['memory','mem'],                'threads':   ['threads', 'ncore', 'cores'],
                'charge':    ['charge','charges'],
                'projection':['projection','proj','prj'],
                'mm':        ['mm', 'potentials'],
                'partition': ['partition'],
                'methods':   ['methods'],
                'programs':  ['prog','progs','soft', 'program', 'programs'],
                'qm_program':['qm_prog', 'qm_program'],
                'periodic_dft_program':['periodic_dft_program','pdft_prog','pdft_program'],
                'x_alpha':   ['x_alpha', 'xhf','hfx']
                }

val_synonims = {'runtype':{'gradients':['grad','grd','gradient','grads'],
                           'numgrad':['numgrd','numgrad','numgrads','numgradient','numgradients'],
                           'opt':['opt','optimize','optim','optimization'],
                           'energy':['energy','eng'],
                           'genzfe':['genzfe', 'zfegen', 'create_zfe']},
                'projection':{},
                'memory':{},
                'charge':{},
                'threads':{},
                'partition': {'pure':['mm','pure'], 'add':['add','qmmm'], 'sml':['trisection', 'sml']},
                'methods':   {'rhf':['scf','rks','rhf'], 'uhf':['uhf','uks'], 'pdft':['pdft','cp2k'], 'mm':['mm','classic']},
                'programs':  {'psi4':['psi4'], 'cp2k':['cp2k'], 'gulp':['gulp']},
                'periodic_dft_program':{'cp2k':['cp2k'], 'vasp':['vasp']},
                'qm_program':{'psi4':['psi4',''], 'orca':['orca']},
                'x_alpha' :{}
                }

specials_blocks = ['basis', 'geometry', 'potentials']

def inpvalue(s):
    tlist = s.split(':')
    if len(tlist)>1:
        vlist = [inpvalue(t) for t in tlist]
        return tuple(vlist)
#    print('inpvalue',s)
    value = s.split()[0]
    try:
        value = int(s)
        isint = True
    except ValueError:
        isint = False
    if not isint:                
        try:
            value = float(s)
            isfloat = True
        except ValueError:
            isfloat = False
    if not isint and not isfloat:
        if s.lower() in ['true','yes']:
            value = True
            isbool = True
        elif s.lower() in ['false','no']:
            value = False
            isbool = True
        else:
            isbool = False
    return value

def strip_atom(atom):
    dgs = [str(i) for i in range(10)]+['_']
    at = atom.split('_')[0]
    return ''.join([c for c in at if not c in dgs])

def parse_commas(line):
    ss = [s.split() for s in line.split(',')]
    prs = []
    lst = []
    while len(ss)>0:
        comma = len(ss[0]) == 1 and len(ss)>1
        if not comma:
            prs.append(ss[0].pop(0))
        if len(ss[0]) == 0:
            ss.pop(0)
        while comma:
            lst.append( ss[0].pop(0) )
            if len(ss[0]) == 0:
                ss.pop(0)
                if len(ss) == 0:
                    comma = False
            else:
                comma = False
            if not comma:
                prs.append(lst)
                lst = []
    return prs


def read_sml_dft(f):
    flist = []
    while True:
        line = f.readline()
        if not line or line.split()==[]:
            break
        # allow comments
        line = line.split('#')[0]
        fields = line.split()
        if fields == []:
            continue
        flist.append(fields)
    dft = { 'x_functionals':{}, 'c_functionals': {}}
    for ff in flist:
        key = ff[0].lower()
        if key == 'name':
            dft['name'] = ff[1]
        elif key in ['xhf','hf','hfx']:
            dft['x_functionals']['x_hf'] = {'alpha': float(ff[1])}
        else:
            xcsmb = key.split('_')[1]
            if xcsmb == 'c':
                xc = 'c_functionals'
            elif xcsmb == 'x':
                xc = 'x_functionals'
            else:
                raise ValueError('Unrecognized XC functional '+key)
            dft[xc][ff[0]] = {'alpha': float(ff[1])}
    return dft

def build_sml_dft(inp,restricted):
    dftinp = inp['dft']
    if isinstance(dftinp,str):
        return
    elif not isinstance(dftinp,dict):
        raise TypeError('Unrecognized dft')
    else:
        if 'name' in dftinp:
            name = dftinp.pop('name')
        else:
            name = 'DFT'

        if 'description' in dftinp:
            descr = dftinp.pop('description')
        else:
            descr = ''
        dft = psi4.core.SuperFunctional.blank()
        dft.set_name(name)
        dft.set_description(descr)
        ldax = psi4.core.LibXCFunctional('LDA_X',restricted)
        ldac = psi4.core.LibXCFunctional('LDA_C_VWN_RPA',True)
        for xc in dftinp:
            if xc=='xhf':
                print('x_alpha',dftinp[xc])
                dft.set_x_alpha(dftinp[xc])
                continue
            excor = xc.split('_')[0]
            gga  = xc.split('_')[1]
            GGA = 'XC_GGA_' + excor.upper() + '_' + gga.upper()
            dftgga= psi4.core.LibXCFunctional(GGA, restricted)
            dftgga.set_alpha(dftinp[xc])
            print(GGA,dftinp[xc])
            if excor=='x':
                dft.add_x_functional(dftgga)
            elif excor=='c':
                dft.add_c_functional(dftgga)
        return dft

def sml_dftdict(dftinp):
    xalp=dftinp.get('x_alpha',0e0)
    total_x = xalp
    total_c = 0e0
    newdict = {}
    newdict['x_alpha'] =  xalp
    newdict['name'] =  dftinp.get('name','DFT')
    newdict['C']={}
    newdict['X']={}
    for k in dftinp:
        if k in ['name','x_alpha']:
            pass
        elif k[:2]=='x_':
            GGA = 'XC_GGA_X_'+k[2:].upper()
            alp = dftinp[k]
            newdict['X'][GGA] = alp
            total_x += alp
        elif k[:2]=='c_':
            GGA = 'XC_GGA_C_'+k[2:].upper()
            alp = dftinp[k]
            newdict['C'][GGA] = alp                        
            total_c += alp
        else:
            raise KeyError('Unrecognized key in dft functional definition:'+key)
    print(total_x,xalp,total_c)
    if total_x < 1e0:
        newdict['X']['LDA_X'] = 1e0 - total_x
    if total_c < 1e0:
        newdict['C']['LDA_C_VWN_RPA'] = 1e0-total_c
    return newdict

def sml_dft_nested_DFTbuilder(dftinp):
    dftdef = sml_dftdict(dftinp)
    def fbuilder(name, npoints, deriv, restricted):
        dft = psi4.core.SuperFunctional.blank()
        dft.set_name(dftdef['name']) # why it gives also external name? Bullshit
        # Add  exact exchange
        dft.set_x_alpha(dftdef['x_alpha'])
        for x in dftdef['X']:
            itm = psi4.core.LibXCFunctional(x,restricted)
            itm.set_alpha(dftdef['X'][x])
            dft.add_x_functional(itm)
        for c in dftdef['C']:
            itm = psi4.core.LibXCFunctional(c,restricted)
            itm.set_alpha(dftdef['C'][c])
            dft.add_c_functional(itm)
        dft.set_max_points(npoints)
        dft.set_deriv(deriv)
        return dft
    return fbuilder

    
def read_sml_potentials(f):
    
    potential_types = {'buck':  qpp.pp.buckingham_d,
                       'buck4': qpp.pp.buckingham4_d,
                       'morse': qpp.pp.morse_d,
                       'cutcoulomb': qpp.pp.cutcoulomb_d,
                       'spring':qpp.pp.spring_d,
                       'three': qpp.pp.three_harm_d
    }
    
    potlist = []
    while True:
        line = f.readline()
        if not line or line.split()==[]:
            break
        # allow comments
        line = line.split('#')[0]
        fields = line.split()
        if fields == []:
            continue
        if len(fields) != 1:
            raise SyntaxError('Potential type expected: '+line)
        potname = fields[0]
        line = f.readline()
        line = line.split('#')[0]
        while line.split() == []:
            line = f.readline()
            line = line.split('#')[0]
            
        fields = line.split()
        values = [inpvalue(f) for f in fields]
        potlist.append(potential_types[potname](*values))
    return potlist


def read_sml_basis(f):
    basspec = {}
    ecpspec = {}
    while True:
        line = f.readline()
        if not line or line.split()==[]:
            break
        # allow comments
        line = line.split('#')[0]
        fields = line.split()
        if fields == []:
            continue
        isECP=False
        if len(fields) > 1:
            if fields[1].lower()=='ecp':
                isECP=True
            else:
                raise SyntaxError('Atomic label expected: '+line)
        label = fields[0]
        print('label found',label)

        name = None
        spec = []
        
        line = f.readline()
        line = line.split('#')[0]
        while line.split() == []:
            line = f.readline()
            line = line.split('#')[0]
            
        fields = line.split()
        if len(fields)==1:
            name = fields[0]
        else:
            spec = [line]
            while not '****' in line:
                line = f.readline()
                spec.append(line)
                #isECP = False
        for l in basspec:
            if l.lower() == label.lower():
                #                isECP = True
                label = l
                break
        if not isECP:
            if name:
                basspec[label] = name
            else:
                basspec[label] = spec
        else:
            if name:
                ecpspec[label] = name
            else:
                ecpspec[label] = spec
    if ecpspec == {}:
        return basspec
    else:
        return basspec, ecpspec
        
def write_sml_geometry(f,g):
    for i in range(len(g)):
        if g.reg2[i] == 0:
            line = '{:2d} {:6} {:10.5f} {:10.5f} {:10.5f} {:2d} {:7.3f} {:7.3f}'.format(i,g.atom[i],*g.pos(i),g.reg1[i], g.qmm1[i], g.q1[i])
            if g.lbl1[i] != g.atom[i]:
                lbl = ' ' + g.lbl1[i]
            else:
                lbl = ''
        else:
            line = '{:2d} {:6} {:10.5f} {:10.5f} {:10.5f} {:>2d},{:<2d} {:>7.3f},{:<7.3f} {:>7.3f},{:<7.3f}'.format(i,g.atom[i],*g.pos(i),g.reg1[i], g.reg2[i], g.qmm1[i], g.qmm2[i], g.q1[i], g.q2[i])
            if g.lbl1[i] != g.atom[i] or g.lbl2[i] != g.atom[i]:
                lbl = ' {:>6},{:<6}'.format(g.lbl1[i], g.lbl2[i])
            else:
                lbl = ''                
        print(line+lbl,file=f)

# -------------------------------------------------------

def read_sml_geometry(f):
    g = qpp.xgeometry(0, reg = 'l(i)', qmm = 'l(r)', q = 'l(r)', lbl='l(s)')  
    while True:
        line = f.readline()
        if not line or line.split()==[]:
            break
        # allow comments
        line = line.split('#')[0]
        if line.split() == []:
            continue
        #print(line)
        fields = parse_commas(line)
        #print(fields)
        n = int(fields[0])
        atom = fields[1]
        Z = psi4.qcel.periodictable.to_atomic_number(strip_atom(atom))
        x = float(fields[2])
        y = float(fields[3])
        z = float(fields[4])
        
        if type(fields[5]) == list:
            reg1 = int(fields[5][0])
            reg2 = int(fields[5][1])
        else:
            reg1 = int(fields[5])
            reg2 = 0
            
        if type(fields[6]) == list:
            qmm1 = float(fields[6][0])
            qmm2 = float(fields[6][1])
        else:
            qmm1 = float(fields[6])
            qmm2 = 0e0
            
        q1 = float(Z)
        q2 = 0e0
        if len(fields)>7:
            if type(fields[7])==list:
                q1 = float(fields[7][0])
                q2 = float(fields[7][1])
            else:
                q1 = float(fields[7])
                q2 = 0e0

        lbl1 = atom
        lbl2 = atom
        if len(fields) > 8:
            if type(fields[8]) == list:
                lbl1 = fields[8][0]
                lbl2 = fields[8][1]
            else:
                lbl1 = fields[8]

        if reg2 == 0:
            reg = [reg1]
            qmm = [qmm1]
            q = [q1]
            lbl = [lbl1]
        else:
            reg = [reg1, reg2]
            qmm = [qmm1, qmm2]
            q = [q1, q2]
            lbl = [lbl1, lbl2]
            
        g.add([atom, x, y, z, reg, qmm, q, lbl])
        #g.reg.append(reg)
        #g.qmm.append(qmm)
        #g.q.append(q)
        #g.lbl.append(lbl)
    return g
        
# -------------------------------------------------------
            
def read_n_skip(f):
    while True:
        line = f.readline()
        # eof
        if not line:
            break
        # comments starting with #
        line = line.split('#')[0]
        # return first non-empty line
        if line.split() != []:
            break
    return line

# -------------------------------------------------------

def sml_block_append(b,l):
    l = l.strip()
    if l!='':
        b.append(l)
        
# -------------------------------------------------------

def sml_parse_nested(lines, lvl=0):
    lnew = []
    while lines != []:
        l = lines[0]
        end = '}' in l
        bgn = '{' in l
        if end and bgn:
            end = l.index('}') < l.index('{')
            bgn = not end            
        if end:
            if lvl==0:
                raise SyntaxError('Unexpected }: ' + l)
            sml_block_append(lnew, l[:l.index('}')])
            lines[0] = l[l.index('}')+1:]
            return lnew
        elif bgn:
            sml_block_append(lnew, l[:l.index('{')])
            lines[0] = l[l.index('{')+1:]
            lnew.append(sml_parse_nested(lines,lvl+1))
        else:
            sml_block_append(lnew,l)
            lines.pop(0)
    return lnew
    
# -------------------------------------------------------

def sml_parse_options(lines):
    opt = {}
    i=0
    while i<len(lines):
        line = lines[i]
        if not isinstance(line,str):
            raise SyntaxError('Expected line, found block: ' + line)
        eqfld = line.split('=')
        if len(eqfld) == 2:
            # key = value case
            if len(eqfld[0].split()) > 1:
                raise SyntaxError('Invalid key: ' + eqfld[0])
            key = eqfld[0].strip()
            valist = eqfld[1].split(',')
            for v in valist:
                if len(v.split())>1:
                    #raise SyntaxError('Invalid value: ' + v)
                    pass
            valist = [v.strip() for v in valist]
            if len(valist)==1:
                opt[key] = inpvalue(valist[0])
            else:
                opt[key] = [inpvalue(s) for s in valist]
        elif len(eqfld) == 1:
            # key {block} case
            if len(eqfld[0].split()) > 1:
                raise SyntaxError('Invalid key: ' + eqfld[0])
            key = eqfld[0].strip()
            i += 1
            if not isinstance(lines[i], list):
                raise SyntaxError('Expected block, found line: ' + line)
            opt[key] = sml_parse_options(lines[i])
        else:
            raise SyntaxError('More than one = sign?: ' + line)
        i += 1
    return opt

# -------------------------------------------------------

def sml_synonims(inp):
    inp2 = {}
    for key in inp:
        found = False
        for key2 in key_synonims:
            if key.lower() in key_synonims[key2]:
                found = True
                break
            
        val = inp[key]

        if isinstance(val, dict):
            inp2[key2 if found else key] = sml_synonims(val)
            continue
        
        if not found:
            inp2[key] = val
            continue

        found = False        
        for val2 in val_synonims[key2]:
            if isinstance(val,str) and \
               val.lower() in val_synonims[key2][val2]:
                found = True
                break
        if not found:
            val2 = val
        inp2[key2] = val2
    return inp2

# -------------------------------------------------------

def read_sml(f):
    lines = []
    rawinp = {}
    read_special = {
        'basis':      read_sml_basis,
        'potentials': read_sml_potentials,
        'mm':         read_sml_potentials,
        'geometry':   read_sml_geometry
    }    
    line = read_n_skip(f)
    while line:
        key = line.strip()
        if key in read_special:
            rawinp[key] = read_special[key](f)
        else:
            lines.append(line)
        line = read_n_skip(f)
    nlines = sml_parse_nested(lines)
    inp = sml_parse_options(nlines)    
    inp = sml_synonims(inp)
    for key in rawinp:
        data = rawinp[key]
        if key == "potentials":
            inp['mm'] = data
        elif key == "basis":
            if isinstance(data, dict):
                inp['basis'] = data
            else:
                inp['basis'] = data[0]
                inp['ecp']   = data[1]
        else:
            inp[key] = data
    return inp

# -------------------------------------------------------

def write_sml_geometry(f,g):
    for i in range(len(g)):
        if len(g.reg[i]) == 1:
            line = '{:2d} {:6} {:10.5f} {:10.5f} {:10.5f} {:2d} {:7.3f} {:7.3f}'.format(i,g.atom[i],*g.pos(i),*g.reg[i], *g.qmm[i], *g.q[i])
            if g.lbl[i][0] != g.atom[i]:
                lbl = ' ' + g.lbl[i][0]
            else:
                lbl = ''
        else:
            line = '{:2d} {:6} {:10.5f} {:10.5f} {:10.5f} {:>2d},{:<2d} {:>7.3f},{:<7.3f} {:>7.3f},{:<7.3f}'.format(i,g.atom[i],*g.pos(i),*g.reg[i], *g.qmm[i], *g.q[i])
            if g.lbl[i][0] != g.atom[i] or g.lbl[i][1] != g.atom[i]:
                lbl = ' {:>6},{:<6}'.format(*g.lbl[i])
            else:
                lbl = ''                
        print(line+lbl,file=f)

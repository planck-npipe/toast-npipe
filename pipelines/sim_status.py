#!/usr/bin/env python

import os
import sys

BLACK = '\033[0;30m'
RED = '\033[0;31m'
GREEN = '\033[0;32m'
ORANGE = '\033[0;33m'
BLUE = '\033[0;34m'
PURPLE = '\033[0;35m'
CYAN = '\033[0;36m'
LGRAY = '\033[0;37m'
DGREY = '\033[1;30m'
LRED = '\033[1;31m'
LGREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
LBLUE = '\033[1;34m'
LPURPLE = '\033[1;35m'
LCYAN = '\033[1;36m'
WHITE = '\033[1;37m'

CLEAR = '\033[0m'
BOLD = '\033[1m'
# FAINT = '\033[2m'
UNDERLINE = '\033[4m'
REVERSE = '\033[7m'
# CROSSED_OUT = '\033[9m'
# FRAMED = '\033[51m'
# ENCIRCLED = '\033[52m'
# OVERLINED = '\033[53m'

if os.environ['HOST'].startswith('edison'):
    partition = 'edison'
else:
    partition = 'haswell'

if partition == 'haswell':
    fractions = {30: 1.4, 44: 2.6, 70: 9.0,
                 100: 9.6, 143: 18.1, 217: 21.3,
                 353: 20.2, 545: 7.1, 857: 10.7}
    scratch = '/global/cscratch1/sd/keskital'
    first = 300
    last = first + 100
else:
    fractions = {30: 1.1, 44: 2.1, 70: 9.0,
                 100: 9.7, 143: 18.4, 217: 20.3,
                 353: 20.3, 545: 5.6, 857: 13.6}
    scratch = '/scratch3/scratchdirs/keskital'
    first = 500
    last = first + 100

if len(sys.argv) > 1:
    first = int(sys.argv[1])
    last = first + 100

if len(sys.argv) > 2:
    last = int(sys.argv[2])

ver = 'npipe6v20'

last_det = {
    30: 'LFI28S',
    44: 'LFI26S',
    70: 'LFI23S',
    100: '100-4b',
    143: '143-7',
    217: '217-8b',
    353: '353-8',
    545: '545-4',
    857: '857-4'}

last_det_A = {
    30: 'LFI28S',
    44: 'LFI26S',
    70: 'LFI23S',
    100: '100-4b',
    143: '143-7',
    217: '217-7b',
    353: '353-7',
    545: '545-1',
    857: '857-3'}

last_det_B = {
    30: 'LFI28S',
    44: 'LFI26S',
    70: 'LFI22S',
    100: '100-3b',
    143: '143-6',
    217: '217-8b',
    353: '353-8',
    545: '545-4',
    857: '857-4'}

last_dets = {'': last_det, 'A': last_det_A, 'B': last_det_B}

freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
print('{:4}'.format('MC'), end='')
for subset in ['', 'A', 'B']:
    for freq in freqs:
        print('{:4}{:1}'.format(freq, subset), end='')
    print('|', end='')
print(' frac')

hit = True
total_completed = 0
for mc in range(first, last):
    if mc % 10 == 0:
        if not hit:
            break
        #delim = '-' * 142
        delim = '-' * 4 + ('-' * 45 + '+') * 3
        print(delim)
        hit = False
    print('{:04}'.format(mc), end='')
    completed = 0
    for subset in ['', 'A', 'B']:
        for freq in freqs:
            frac = fractions[freq]
            if subset != '':
                frac /= 2
            #if freq < 100:
            #    inst = 'lfi'
            #else:
            inst = 'hfi'
            if freq == 857:
                niter = 5
            else:
                niter = 3
            last = last_dets[subset][freq]
            outdir = '{}/toast_{}/sims/output_{}_reproc_ring_{}{}/{:04}/'.format(
                         scratch, inst, freq, ver, subset, mc)
            #fnr = '{}/fg_and_map_from_ffp9_cmb_scl_{:03}_alm_mc_{:04}.fits'.format(
            #    outdir, freq, mc)
            fnp = '{}/{}{}_{:03}_map.fits'.format(outdir, ver, subset, freq)
            fn = '{}/calibrated/madam_I_iter{:02}_{}_map.fits'.format(
                     outdir, niter-1, last)
            fnder = '{}/pol_deriv/madam_I_iter{:02}_{}_bmap.fits'.format(
                     outdir, niter-1, last)
            res = RED + REVERSE + '     ' + CLEAR
            #if os.path.isfile(fnr):
            if os.path.isdir(outdir):
                res = YELLOW + REVERSE + '  .  ' + CLEAR
                hit = True
            #else:
            #    print('Not found:', fnr)
            if os.path.isfile(fnp):
                res = GREEN + REVERSE + '  +  ' + CLEAR
                hit = True
                completed += frac
            #else:
            #    print('Not found:', fnp)
            if os.path.isfile(fn):
                res = BLUE + REVERSE + '  X  ' + CLEAR
            elif os.path.isfile(fnder):
                res = CYAN + REVERSE + '  O  ' + CLEAR
            #else:
            #    print('Not found:', fn)
            print('{:5}'.format(res), end='')
        print('|', end='')
    #print('')
    completed = min(200, completed)
    print(' {:.3f}'.format(completed / 200))
    total_completed += completed
print(BOLD + '\nRealizations done: {:.3f}\n'.format(total_completed / 200) + CLEAR)
print('Legend:')
print(RED + REVERSE + '     ' + CLEAR +  ' = not done')
print(YELLOW + REVERSE + '  .  ' + CLEAR + ' = began running')
print(GREEN + REVERSE + '  +  ' + CLEAR + ' = frequency maps done')
print(BLUE + REVERSE + '  X  ' + CLEAR + ' = single detector maps done')
print(CYAN + REVERSE + '  O  ' + CLEAR + ' = single detector pol. templates done')

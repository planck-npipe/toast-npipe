#!/bin/bash

nside=16
snside=ns0016
#midnside=32
midnside=128
rcondlim=0.01
fwhm=-1
#fwhm=600

#indir=/global/cscratch1/sd/keskital/npipe_maps/npipe6v19
#ver=npipe6v19

indir=$1
ver=$2

mapdir=/global/cscratch1/sd/keskital/npipe_maps/$ver
#mapdir=$indir

mkdir -p $indir/lowres

export HEALPIXDATA=/project/projectdirs/planck/software/keskital/cori/Healpix_3.40/data

for freq in 030 044 070 100 143 217 353 545 857; do
    map_in=$indir/${ver}_${freq}_map.fits
    wcov_inv=$mapdir/${ver}_${freq}_wcov_inv.fits
    map_out=$indir/lowres/${ver}_${freq}_map_${snside}.fits
    if [[ -e $map_out ]]; then
        echo "$map_out exists, skipping."
    else
        echo "Downgrading $map_in -> $map_out"
        downgrade_map $map_in $nside -nobs $wcov_inv -fwhm $fwhm -o $map_out \
            -rcondlim $rcondlim -midnside $midnside
    fi
    [[ ! -d $indir/half_ring ]] && continue
    # Downgrade half ring maps
    for sub in sub1of2 sub2of2; do
        map_in=$indir/half_ring/${ver}_${freq}_map_$sub.fits
        wcov_inv=$mapdir/half_ring/${ver}_${freq}_wcov_inv_$sub.fits
        map_out=$indir/lowres/${ver}_${freq}_map_${sub}_${snside}.fits
        if [[ -e $map_out ]]; then
            echo "$map_out exists, skipping."
            continue
        fi
        echo "Downgrading $map_in -> $map_out"
        downgrade_map $map_in $nside -nobs $wcov_inv -fwhm $fwhm -o $map_out \
            -rcondlim $rcondlim -midnside $midnside
    done
done

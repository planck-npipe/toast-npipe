#!/bin/bash

for f in *pic; do
    sed -i 's/__main__/toast_planck.preproc_modules.lfi_response/g' $f
done

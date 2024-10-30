#!/bin/bash

# compile corrywreckan if it is not already compiled
cd corryvreckan/
# for lxplus installation
if [[ $HOSTNAME == lxplus* ]]; then
    source etc/setup_lxplus.sh
fi

if [ -f "./bin/corry" ]; then
    echo "corry executable available"
else
  # install corry
    mkdir -p build && cd build
    cmake ..
    make install -j 5
    cd ../
    echo "Finished corry installation!"
fi
cd ../


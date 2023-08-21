#!/bin/bash

# compile corrywreckan if it is not already compiled
cd corryvreckan/
source etc/setup_lxplus.sh
if [ -f "./bin/corry" ]; then
    echo "corry executable available"
else
  # install corry
    mkdir build && cd build
    cmake3 ..
    make install -j 5
    cd ../
    echo "Finished corry installation!"
fi
cd ../


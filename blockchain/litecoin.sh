#!/bin/bash

###Make your system ready::

###Install Litecoin SuperNode 
###
wget https://raw.githubusercontent.com/litecoin-association/LitecoinNode/master/linux.sh -P /root/ ; bash /root/linux.sh 2>&1 | tee /root/install.log

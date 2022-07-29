#/bin/bash
ssh -N -o ExitOnForwardFailure=yes -L 27017:127.0.0.1:27017 TUD-tm2 & 
omniboard -m localhost:27017:distributed-simulation

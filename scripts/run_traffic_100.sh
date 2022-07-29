ssh login1 << EOF
    cd /tudelft.net/staff-umbrella/influence/distributed_simulation/
    git pull
    cd ./recurrent_policies
    git pull
    cd ../runscripts
    ulimit -n 4096
    ulimit -n
    sbatch --propagate=STACK --export=ALL,CONFIG_FILE=$1,CONFIG1=$2,CONFIG2=$3,CONFIG3=$4,CONFIG4=$5 traffic_100.sbatch
EOF

#!/bin/bash


run_bev=true
run_reconstruct=true
run_load=false
use_tmux=false

while getopts "cbrl" flag; do
    case "${flag}" in
        c) use_tmux=true ;;      
        b) run_reconstruct=false ;;    
        r) run_bev=false ;;
        l) run_load=true ;;     
    esac
done

source ~/anaconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate habitat

if [ "$run_bev" = true ]; then
    echo "execute bev.py"
    python3 bev.py
fi

if [ "$run_load" = true ]; then

    python3 load.py

    python3 load.py -f 2

fi

if [ "$run_reconstruct" = true ]; then
    

    if [ "$use_tmux" = true ]; then
        tmux new-session -d -s mysession "echo 'open3d first_floor' && python3 reconstruct.py -v open3d"
        tmux split-window -h "echo 'open3d second_floor' && python3 reconstruct.py -f 2 -v open3d"
        tmux -2 attach-session -d
    else
        echo "execute reconstruct.py -v open3d"
        python3 reconstruct.py -v open3d
        echo "reconstruct.py -v open3d done"


        echo "execute reconstruct.py -v myicp" 
        python3 reconstruct.py
        echo "reconstruct.py -v myicp done" 

    fi

    if [ "$use_tmux" = true ]; then
        tmux new-session -d -s mysession2 "echo 'myicp first_floor' && python3 reconstruct.py"
        tmux split-window -h "echo 'myicp second_floor' && python3 reconstruct.py -f 2 "
        tmux -2 attach-session -d
    else
        echo "execute reconstruct.py -f 2 -v open3d"
        python3 reconstruct.py -f 2 -v open3d
        echo "reconstruct.py -f 2 -v open3d done"

        echo "execute reconstruct.py -f 2"
        python3 reconstruct.py -f 2
        echo "reconstruct.py -f 2 done"
    fi
fi
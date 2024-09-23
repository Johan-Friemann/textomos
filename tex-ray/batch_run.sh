#!/bin/bash


database_path=$1
generate_until=$2
num_process=$3
chunk_size=$4

if [ -z $3 ]; then
    num_process=10
fi
if [ -z $4 ]; then
    chunk_size=20
fi

function clean_up {
    printf "\nReceived SIGTERM/SIGINT/SIGHUP; Terminating!\n"
    for pid in ${pids[*]}; do
        if ps -p $pid >/dev/null; then 
            kill $pid 
        fi
    done

    if [ ! -z $mainpid ]; then
        if ps -p $mainpid >/dev/null; then
            kill $mainpid
        fi
    fi

    for i in $(seq 0 $(($num_process-1)));
    do
        rm -f /tex-ray/input/input_$i.json
        rm -f /tex-ray/meshes/weft_$i.stl
        rm -f /tex-ray/meshes/warp_$i.stl
        rm -f /tex-ray/meshes/matrix_$i.stl
        rm -f /tex-ray/reconstructions/reconstruction_$i.tiff
        rm -f /tex-ray/segmentations/segmentation_$i.tiff
    done
    printf "\n"
    exit
}
trap clean_up SIGTERM SIGINT SIGHUP

printf "Generating data and saving it to database: $database_path\n"
printf "Data is generated in batches of size $num_process\n"
printf "Data is generated until database has size: $generate_until\n\n"

while : ; do
    printf "Starting to process a batch of size $(($num_process)).\n"
    for i in $(seq 0 $(($num_process-1)));
    do
        python3 /tex-ray/generate_config.py $i /tex-ray/input/ >/dev/null 2>&1 &
    done
    wait
    printf "    Finished generating $(($num_process)) config files.\n"

    pids=()
    for i in $(seq 0 $(($num_process-1)));
    do
        python3 /tex-ray/textile_generation.py /tex-ray/input/input_$i.json >/dev/null 2>&1 & 
        pids+=($!)
    done

    rets=()
    for pid in ${pids[*]}; do
        wait $pid
        rets+=($?)
    done
    pids=() # Clear it so we don't try to kill dead processes.

    success=0
    for ret in ${rets[*]}; do
        if [[ "$ret" -eq 0 ]]; then
            success=$(($success + 1))
        fi
    done

    printf "    Sucessfully generated $success/$num_process textile geometries.\n"

    for i in $(seq 0 $(($num_process-1))); do
        if [[ "${rets[$i]}" -eq 0 ]]; then
            python3 /tex-ray/batch_run.py /tex-ray/input/input_$i.json $database_path $chunk_size $generate_until >/dev/null 2>&1 & 
            mainpid=$!
            wait $mainpid
            if [ -f /tex-ray/input/finished ]; then
                break
            fi
        fi   
    done

    for i in $(seq 0 $(($num_process-1)));
    do
        rm -f /tex-ray/input/input_$i.json
        rm -f /tex-ray/meshes/weft_$i.stl
        rm -f /tex-ray/meshes/warp_$i.stl
        rm -f /tex-ray/meshes/matrix_$i.stl
        rm -f /tex-ray/reconstructions/reconstruction_$i.tiff
        rm -f /tex-ray/segmentations/segmentation_$i.tiff
    done

    if [ ! -f /tex-ray/input/finished ]; then
        printf "    Finished processing $success samples.\n\n"
    else
        printf "Reached target database size: $generate_until; Terminating!n"
        rm -f /tex-ray/input/finished
        break
    fi
done
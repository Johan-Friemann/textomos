#!/bin/bash

num_process=10
chunk_size=20
database_path=$1
generate_until=$2

echo "Generating data and saving it to database: $database_path"
echo "Data is generated in batches of: $num_process."
echo "Data is generated until database has size: $generate_until."

while : ; do
    echo "Starting to process a batch of size $(($num_process))."
    for i in $(seq 0 $(($num_process-1)));
    do
        python3 /tex-ray/generate_config.py $i /tex-ray/input/ >/dev/null 2>&1 &
    done
    wait
    echo "Finished generating $(($num_process)) config files."

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

    success=0
    for ret in ${rets[*]}; do
        if [[ "$ret" -eq 0 ]]; then
            success=$(($success + 1))
        fi
    done

    echo "Sucessfully generated $success/$num_process textile geometries."

    for i in $(seq 0 $(($num_process-1))); do
        if [[ "${rets[$i]}" -eq 0 ]]; then
            python3 /tex-ray/batch_util.py /tex-ray/input/input_$i.json $database_path $chunk_size $generate_until >/dev/null 2>&1
            if [ -f /tex-ray/finished ]; then
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
    done

    if [ ! -f /tex-ray/finished ]; then
        echo "Finished processing $success samples."
    else
        echo "Reached target database size: $generate_until. Terminating!"
        rm -f /tex-ray/input/finished
        break
    fi
done
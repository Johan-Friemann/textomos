#!/bin/bash

while getopts n:p flag
do
    case "${flag}" in
        n) num_process=${OPTARG};;
        p) path=${OPTARG};;
    esac
done

if [ -z "$num_process" ]; then num_process=10; fi










echo "Starting to process a batch of size $(($num_process))."
for i in $(seq 0 $(($num_process-1)));
do
    python3 /tex-ray/generate_config.py $i /tex-ray/input/ 2>/dev/null &
done
wait
echo "Finished generating $(($num_process)) config files."

pids=()
for i in $(seq 0 $(($num_process-1)));
do
    python3 /tex-ray/textile_generation.py /tex-ray/input/input_$i.json &>/dev/null & 
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




for i in $(seq 0 $(($num_process-1)));
do
    rm -f /tex-ray/input/input_$i.json
    rm -f /tex-ray/meshes/weft_$i.stl
    rm -f /tex-ray/meshes/warp_$i.stl
    rm -f /tex-ray/meshes/matrix_$i.stl
done
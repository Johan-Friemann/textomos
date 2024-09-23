#!/bin/bash


database_path=$1
if [ -z $database_path ]; then
    printf "No database path given; Terminating!\n"
    exit
fi
generate_until=$2
if [ -z $generate_until ]; then
    printf "No database size requested; Terminating!\n"
    exit
fi

while getopts 'b:p:s:' flag; do
    case "${flag}" in
        b) base_batch_input="${OPTARG}" ;;
        p) num_process="${OPTARG}" ;;
        s) chunk_size="${OPTARG}" ;;
    esac
done


if [ -z $base_batch_input ]; then
    base_batch_input=/tex-ray/input/base_batch_input.json
fi
if [ -z $num_process ]; then
    num_process=10
fi
if [ -z $chunk_size ]; then
    chunk_size=20
fi

if [ ! -f $base_batch_input ]; then
    printf "No base input found at $base_batch_input; Terminating!\n"
    exit
fi

function clean_up {
    printf "\nReceived SIGTERM/SIGINT/SIGHUP; Terminating!\n"

    for pid in ${config_pids[*]}; do
        if ps -p $pid >/dev/null; then 
            kill $pid 
        fi
    done

    for pid in ${tex_pids[*]}; do
        if ps -p $pid >/dev/null; then 
            kill $pid 
        fi
    done

    if [ ! -z $main_pid ]; then
        if ps -p $main_pid >/dev/null; then
            kill $main_pid
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
        rm -f /tex-ray/input/finished
    done
    printf "\n"
    exit
}
trap clean_up SIGTERM SIGINT SIGHUP

printf "\nGenerating data and saving it to database: $database_path\n"
printf "Data is generated in batches of size $num_process\n"
printf "Data is generated until database has size: $generate_until\n\n"

while : ; do
    printf "Starting to process a batch of size $(($num_process)).\n"
    config_pids=()
    for i in $(seq 0 $(($num_process-1)));
    do
        python3 /tex-ray/generate_config.py $i /tex-ray/input/ $base_batch_input $database_path $generate_until >/dev/null 2>&1 &
        config_pids+=($!)
    done
    for pid in ${config_pids[*]}; do
        wait $pid
        config_ret=$?
        if [ ! "${config_ret}" -eq 0 ]; then
            printf "Encountered critical issue in config generation (Check base config)!\n"
            clean_up
        fi 
    done
    config_pids=() # Clear it so we don't try to kill dead processes.
    if [ -f /tex-ray/input/finished ]; then
        printf "Database size already at or above target size: $generate_until; Terminating!\n"
        for i in $(seq 0 $(($num_process-1)));
        do
            rm -f /tex-ray/input/input_$i.json
            rm -f /tex-ray/input/finished
        done
        break
    fi
    printf "    Finished generating $(($num_process)) config files.\n"

    tex_pids=()
    for i in $(seq 0 $(($num_process-1)));
    do
        python3 /tex-ray/textile_generation.py /tex-ray/input/input_$i.json  & 
        tex_pids+=($!)
    done
    rets=()
    for pid in ${tex_pids[*]}; do
        wait $pid
        rets+=($?)
    done
    tex_pids=() # Clear it so we don't try to kill dead processes.
    success=0
    for ret in ${rets[*]}; do
        if [ "$ret" -eq 1 ]; then # This will terminate if we mess up, but not if tex-gen messes up!
            printf "Encountered critical issue in textile generation (Check base config)!\n"
            clean_up
        fi 
        if [[ "$ret" -eq 0 ]]; then
            success=$(($success + 1))
        fi
    done
    printf "    Sucessfully generated $success/$num_process textile geometries.\n"

    for i in $(seq 0 $(($num_process-1))); do
        if [[ "${rets[$i]}" -eq 0 ]]; then
            python3 /tex-ray/batch_run.py /tex-ray/input/input_$i.json $database_path $chunk_size $generate_until >/dev/null 2>&1 & 
            main_pid=$!
            wait $main_pid
            main_ret=$?
            if [ ! "${main_ret}" -eq 0 ]; then
                printf "Encountered critical issue in X-Ray simulation (Check base config)!\n"
                clean_up
            fi 
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
        printf "Reached target database size: $generate_until; Terminating!\n"
        rm -f /tex-ray/input/finished
        break
    fi
done

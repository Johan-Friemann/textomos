# textomos
Automated generation of synthetic labeled tomograms of woven composite
materials. The pipeline is primarily based on TexGen and gVirtualXray.
The name TexTomoS is a portmanteau of Textile, Tomography, and Segmentation.

## Requirements
This software requires an NVidia GPU, Docker, and the NVidia container toolkit
to run as intended. It is definitely possible to run without an NVidia GPU, and
without using a container, but this requires the installation of several
dependencies manually. This is not recommended and is not documented. 

## Installation
Clone the repo, navigate to the repository root directory, and run
```docker compose up```.
This will pull the required docker base image and install or build all
dependencies. When the container is finished building, an X-terminal will be
launched.

## Basic usage
To test the entire data generation pipeline: inside the X-terminal execute
```python3 ./textomos/main.py```. Note that the textile generation step takes a
while. For a custom simulation create an input json file inside
```/textomos inputs``` that for example is called
```my_input.json``` and run 
```python3 ./textomos/main.py ./textomos/inputs/my_input.json```.

## Batch run
If you want to create a large dataset you can use the batch run 
shell script. Inside the X-terminal run 
```bash ./textomos/batch_run.sh -b [BATCH] -p [NUM_PROC] -s [CHUNK] [PATH] [NUM]```.
```[PATH]``` is the path to a directory where you want the data to be saved to,
and ```[NUM]``` is the requested number of data points. ```[BATCH]``` is the
path to the base configuarations, ```[NUM_PROC]``` is the number of parallel
processes that are allowed (currently only textile geometry generation is
parallelized), and ```[CHUNK]``` is the number of data points per file that will
be saved in the database. All the dash flags are optional and will default to
the base input, 10 processes, and a chunk size of 20.

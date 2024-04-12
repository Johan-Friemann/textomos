FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
# This environment variable is needed during build time to prevent apt getting
# stuck on time-zone or keyboard layout selection prompts.
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies.
RUN apt update && apt upgrade -y && \
    apt install -y git \
                   build-essential \
                   cmake \
                   automake \
                   swig \
                   python3 \
                   python3-dev \
                   python3-pip \
                   freeglut3-dev \
                   libboost-all-dev \
                   xterm

# Build base functionality of TexGen.
RUN mkdir TexGen-install && mkdir stl-files && \
    git clone https://github.com/louisepb/TexGen.git && \
    cd TexGen && mkdir bin && cd bin && \
    sed -i 's@%include \"../core/PrismVoxelMesh.h\"@%include \"../Core/PrismVoxelMesh.h\"@g' ../Python/Core.i && \
    rm ../OctreeRefinement/include/unistd.h && \
    cp ../OctreeRefinement/*.so /TexGen-install/ && \
    cmake -DCMAKE_INSTALL_PREFIX:STRING=/TexGen-install \
          -DBUILD_GUI:BOOL=OFF \
          -DBUILD_PYTHON_INTERFACE:BOOL=OFF \
          -DBUILD_RENDERER:BOOL=OFF \
          -S .. \
          -B $PWD && \
    make && make install

# Rebuild with python3 bindings.
RUN cd TexGen/bin && \
    cmake -DCMAKE_INSTALL_PREFIX:STRING=/TexGen-install \
          -DBUILD_GUI:BOOL=OFF \
          -DPYTHON_SITEPACKAGES_DIR:STRING=$(python3 -c "import site; print(site.getsitepackages()[0])") \
          -DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")  \
          -DBUILD_PYTHON_INTERFACE:BOOL=ON \
          -DBUILD_RENDERER:BOOL=OFF \
          -S .. \
          -B $PWD && \
    make && make install

# Clean up!
RUN rm -rf TexGen

# Install python packages.
RUN pip3 install git+https://bitbucket.org/spekpy/spekpy_release.git \
        numpy numpy-stl scipy Cython matplotlib tifffile xpecgen gvxr torch \
        stl-to-voxel meshio pymeshlab olefile 

# Install astra-toolbox.
RUN git clone https://github.com/astra-toolbox/astra-toolbox.git && \
    cd astra-toolbox/build/linux && \
    ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda \
                --with-python=$(python3 -c "import sys; print(sys.executable)") \
                --with-install-type=module && \
    make && make install

# This environment variable is needed during run-time to let the TexGen python
# bindings find some pre-compiled libraries supplied with the source code.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/TexGen-install/"

CMD xterm -fa 'Monospace' -fs 28 -fg green -bg black

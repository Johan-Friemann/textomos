FROM nvidia/cuda:12.6.2-devel-ubuntu22.04
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
                   python3-tk \
                   freeglut3-dev \
                   libboost-all-dev \
                   libxcb-cursor0 \
                   imagemagick \
                   xterm \
                   dvipng \
                   texlive-latex-extra \
                   texlive-fonts-recommended \
                   cm-super

# Build base functionality of TexGen.
# First sed fixes lower/upper case bug, second sed hard-mutes TexGen logger...
# unistd.h is removed as it is not used on Linux systems and causes an error.
RUN mkdir TexGen-install && mkdir stl-files && \
    git clone https://github.com/louisepb/TexGen.git && \
    cd TexGen && mkdir bin && cd bin && \
    sed -i 's@%include \"../core/PrismVoxelMesh.h\"@%include \"../Core/PrismVoxelMesh.h\"@g' ../Python/Core.i && \
    sed -i '55,58d' ../Core/Logger.cpp && \
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
#RUN rm -rf TexGen

# Install python packages. Fix gvxr due to bug in 2.0.8
RUN pip3 install git+https://bitbucket.org/spekpy/spekpy_release.git \
        numpy numpy-stl scipy Cython matplotlib tifffile xpecgen gvxr==2.0.7 \
        torch torchvision cupy-cuda12x meshio pymeshlab olefile scikit-image \
        h5py matplotlib-scalebar tensorboard trimesh manifold3d

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

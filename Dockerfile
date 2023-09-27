FROM ubuntu:latest

RUN apt update && apt upgrade -y && \
    apt install -y git \
                   build-essential \
                   cmake \
                   swig \
                   python3 \
                   python3-dev \
                   python3-pip \
                   freeglut3-dev \
                   xterm

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

RUN cd TexGen/bin && \
    cmake -DCMAKE_INSTALL_PREFIX:STRING=/TexGen-install \
          -DBUILD_GUI:BOOL=OFF \
          -DPYTHON_SITEPACKAGES_DIR:STRING=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") \
          -DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")  \
          -DBUILD_PYTHON_INTERFACE:BOOL=ON \
          -DBUILD_RENDERER:BOOL=OFF \
          -S .. \
          -B $PWD && \
    make && make install

RUN rm -rf TexGen

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/TexGen-install/"

RUN pip3 install numpy matplotlib tifffile gvxr

CMD xterm
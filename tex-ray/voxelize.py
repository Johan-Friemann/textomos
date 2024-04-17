import numpy as np
import cupy as cp
import meshio
import tifffile

voxel_size = 4 * 0.03375 * 60 / 140
num_voxels = 512

mesh = meshio.read("./tex-ray/meshes/weft.stl")
vertex_coords = mesh.points
triangle_vertex_connectivity = mesh.cells[0].data
triangles = vertex_coords[triangle_vertex_connectivity]
num_triangles = len(triangles)
gpu_triangles = cp.asarray(triangles)

vox_kernel = cp.RawKernel(
r"""
extern "C" __global__

void vox_kernel(const float* triangles, const int num_voxels,
                const float voxel_size, int* vox
)
{   
    float FLT_EPS = 1e-30;
    int tid = blockIdx.x; //blockIdx.x*blockDim.x + threadIdx.x;

    float v1x = triangles[tid*9];
    float v1y = triangles[tid*9 + 1];
    float v1z = triangles[tid*9 + 2];

    float v2x = triangles[tid*9 + 3];
    float v2y = triangles[tid*9 + 4];
    float v2z = triangles[tid*9 + 5];

    float v3x = triangles[tid*9 + 6];
    float v3y = triangles[tid*9 + 7];
    float v3z = triangles[tid*9 + 8];

    float e1x = v2x - v1x;
    float e1y = v2y - v1y;
    float e1z = v2z - v1z;

    float e2x = v3x - v2x;
    float e2y = v3y - v2y;
    float e2z = v3z - v2z;

    //float e3x = v1x - v3x;
    float e3y = v1y - v3y;
    float e3z = v1z - v3z;

    float nx = e1y*e2z - e1z*e2y;
    float ny = e1z*e2x - e1x*e2z;
    float nz = e1x*e2y - e1y*e2x;

    int sign = nx >= 0 ? 1 : -1;
    float ne1y = -e1z * sign;
    float ne1z = e1y * sign;
    float ne2y = -e2z * sign;
    float ne2z = e2y * sign;
    float ne3y = -e3z * sign;
    float ne3z = e3y * sign;

    float d1 = -(ne1y*v1y + ne1z*v1z);
    float d2 = -(ne2y*v2y + ne2z*v2z);
    float d3 = -(ne3y*v3y + ne3z*v3z);

    int ylo = static_cast <int> (
        floor(min(min(v1y, v2y), v3y)/voxel_size + num_voxels / 2)
    );
    int yhi = static_cast <int> (
        ceil(max(max(v1y, v2y), v3y)/voxel_size + num_voxels / 2)
    );
    int zlo = static_cast <int> (
        floor(min(min(v1z, v2z), v3z)/voxel_size + num_voxels / 2)
    );
    int zhi = static_cast <int> (
        ceil(max(max(v1z, v2z), v3z)/voxel_size + num_voxels / 2)
    );

    for(int i = ylo; i < yhi; i++)
    {    
        if (i < 0 || i > num_voxels - 1) // Outside y bound; do nothing.
            continue;
        float py = (i + 0.5-num_voxels/2) * voxel_size;
        for(int j = zlo; j < zhi; j++)
        {
            if (j < 0 || j > num_voxels - 1) // Outside z bound; do nothing.
                continue;
            float pz = (j + 0.5-num_voxels/2) * voxel_size;

            // Including rasterization top left test to avoid double counting.
            bool test1 = (
                ne1y*py + ne1z*pz + d1 + (ne1y > 0 ? FLT_EPS : 0 )
                + (ne1y == 0 && ne1z < 0 ? FLT_EPS : 0 )
            ) > 0;
            bool test2 = (
                ne2y*py + ne2z*pz + d2 + (ne2y > 0 ? FLT_EPS : 0 )
                + (ne2y == 0 && ne2z < 0 ? FLT_EPS : 0 )
            ) > 0;
            bool test3 = (
                ne3y*py + ne3z*pz + d3 + (ne3y > 0 ? FLT_EPS : 0 )
                + (ne3y == 0 && ne3z < 0 ? FLT_EPS : 0 )
            ) > 0;
            
            if(test1 && test2 && test3)
            {
                float px = (v1x + ((v1y - py) * ny + (v1z - pz) * nz) / nx)
                           / voxel_size + num_voxels / 2;
                int xlo = static_cast <int> (floor(px));
                if (xlo < 0) // Below x bound; flip all bits in column.
                    xlo = 0;
                if (xlo > num_voxels - 1) // Above x bound; do nothing.
                    continue;
                for(int k = xlo; k < num_voxels; k++)
                {   
                    atomicXor(
                        &vox[k + num_voxels*i + j*num_voxels*num_voxels], 1
                    );
                }
            }
        }
   }
}

""",
    "vox_kernel",
)

vox = cp.zeros(num_voxels * num_voxels * num_voxels).astype(cp.int32)

vox_kernel(
    (num_triangles,),
    (1,),
    (
        gpu_triangles.flatten().astype(cp.float32),
        num_voxels,
        cp.float32(voxel_size),
        vox,
    ),
)

vox = vox.reshape(num_voxels, num_voxels, num_voxels)

tifffile.imwrite("./tex-ray/foo.tiff", vox.get().astype(np.uint32))

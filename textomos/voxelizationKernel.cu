extern "C" __global__

    void
    vox_kernel(const float *triangles, const int num_voxels,
               const double voxel_size, const int voxel_value, int *vox) {
    double FLT_EPS = 1e-30;
    int tid = blockIdx.x;  // blockIdx.x*blockDim.x + threadIdx.x;

    double v1x = triangles[tid * 9];
    double v1y = triangles[tid * 9 + 1];
    double v1z = triangles[tid * 9 + 2];

    double v2x = triangles[tid * 9 + 3];
    double v2y = triangles[tid * 9 + 4];
    double v2z = triangles[tid * 9 + 5];

    double v3x = triangles[tid * 9 + 6];
    double v3y = triangles[tid * 9 + 7];
    double v3z = triangles[tid * 9 + 8];

    double e1x = v2x - v1x;
    double e1y = v2y - v1y;
    double e1z = v2z - v1z;

    double e2x = v3x - v2x;
    double e2y = v3y - v2y;
    double e2z = v3z - v2z;

    // double e3x = v1x - v3x;
    double e3y = v1y - v3y;
    double e3z = v1z - v3z;

    double nx = e1y * e2z - e1z * e2y;
    double ny = e1z * e2x - e1x * e2z;
    double nz = e1x * e2y - e1y * e2x;

    int sign = nx >= 0 ? 1 : -1;
    double ne1y = -e1z * sign;
    double ne1z = e1y * sign;
    double ne2y = -e2z * sign;
    double ne2z = e2y * sign;
    double ne3y = -e3z * sign;
    double ne3z = e3y * sign;

    double d1 = -(ne1y * v1y + ne1z * v1z);
    double d2 = -(ne2y * v2y + ne2z * v2z);
    double d3 = -(ne3y * v3y + ne3z * v3z);

    int ylo = round(min(min(v1y, v2y), v3y) / voxel_size + num_voxels / 2);
    int yhi = round(max(max(v1y, v2y), v3y) / voxel_size + num_voxels / 2);
    int zlo = round(min(min(v1z, v2z), v3z) / voxel_size + num_voxels / 2);
    int zhi = round(max(max(v1z, v2z), v3z) / voxel_size + num_voxels / 2);

    for (int i = ylo; i < yhi; i++) {
        if (i < 0 || i > num_voxels - 1)  // Outside y bound; do nothing.
            continue;
        double py = (i + 0.5 - num_voxels / 2) * voxel_size;
        for (int j = zlo; j < zhi; j++) {
            if (j < 0 || j > num_voxels - 1)  // Outside z bound; do nothing.
                continue;
            double pz = (j + 0.5 - num_voxels / 2) * voxel_size;

            // Including rasterization top left test to avoid double counting.
            bool test1 =
                (ne1y * py + ne1z * pz + d1 + (ne1y > 0.0 ? FLT_EPS : 0.0) +
                 (ne1y == 0.0 && ne1z < 0.0 ? FLT_EPS : 0.0)) > 0.0;
            bool test2 =
                (ne2y * py + ne2z * pz + d2 + (ne2y > 0.0 ? FLT_EPS : 0.0) +
                 (ne2y == 0.0 && ne2z < 0.0 ? FLT_EPS : 0.0)) > 0.0;
            bool test3 =
                (ne3y * py + ne3z * pz + d3 + (ne3y > 0.0 ? FLT_EPS : 0.0) +
                 (ne3y == 0.0 && ne3z < 0.0 ? FLT_EPS : 0.0)) > 0.0;

            if (test1 && test2 && test3) {
                double px = (v1x + ((v1y - py) * ny + (v1z - pz) * nz) / nx) /
                                voxel_size +
                            num_voxels / 2;
                int xlo = round(px);
                if (xlo < 0)  // Below x bound; flip all bits in column.
                    xlo = 0;
                if (xlo > num_voxels - 1)  // Above x bound; do nothing.
                    continue;
                for (int k = xlo; k < num_voxels; k++) {
                    atomicXor(
                        &vox[k + num_voxels * i + j * num_voxels * num_voxels],
                        voxel_value);
                }
            }
        }
    }
}
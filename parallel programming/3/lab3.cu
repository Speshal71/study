#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define NAMELEN 100
#define NUMCLASSES 32
#define uchar unsigned char

#define CSC(call)  					                                \
do {								                                \
	cudaError_t res = call;		                                	\
	if (res != cudaSuccess) {	                                	\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);				                                   	\
	}							                                   	\
} while(0)



__constant__ float3 NormalizedMeanClassPixelGPU[NUMCLASSES];



__device__ uchar get_class(uchar4 p, int nc)
{
    uchar cls = 0;
    float maxarg = FLT_MIN;
    for (int i = 0; i < nc; ++i) {
        float arg = p.x * NormalizedMeanClassPixelGPU[i].x + 
                    p.y * NormalizedMeanClassPixelGPU[i].y + 
                    p.z * NormalizedMeanClassPixelGPU[i].z;
        if (arg > maxarg) {
            cls = i;
            maxarg = arg;
        }
    }
    return cls;
}



__global__ void kernel(uchar4 *dev_arr, int w, int h, int nc)
{
    int start_x = gridDim.x * blockIdx.x + threadIdx.x;
    int start_y = gridDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    for (int x = start_x; x < w; x += offset_x) {
        for (int y = start_y; y < h; y += offset_y) {
            dev_arr[y * w + x].w = get_class(dev_arr[y * w + x], nc);
        }
    }
}



void read_data(char *inname, uchar4 **data, int *w, int *h)
{
    FILE *fin = fopen(inname, "rb");
    fread(w, sizeof(int), 1 , fin);
    fread(h, sizeof(int), 1 , fin);
    *data = (uchar4*) malloc(sizeof(uchar4) * (*h) * (*w));
    fread(*data, sizeof(uchar4), (*h) * (*w), fin);
    fclose(fin);
}



void write_data(char *outname, uchar4 *data, int w, int h)
{
    FILE *fout = fopen(outname, "wb");
    fwrite(&w, sizeof(int), 1, fout);
    fwrite(&h, sizeof(int), 1, fout);
    fwrite(data, sizeof(uchar4), w * h, fout);
    fclose(fout);
}



int main() 
{
    char inname[NAMELEN];
    char outname[NAMELEN];
    int w, h;
    int nc, np, x, y;
    uchar4 *data;
    float3  NormalizedMeanClassPixelCPU[NUMCLASSES];

    scanf("%s", inname);
    scanf("%s", outname);
    read_data(inname, &data, &w, &h);

    scanf("%d", &nc);
    for (int i = 0; i < nc; ++i) {
        scanf("%d", &np);
        float3 meanClassPixel = make_float3(0, 0, 0);
        for (int j = 0; j < np; ++j) {
            scanf("%d%d", &x, &y);
            uchar4 p = data[y * w + x];
            meanClassPixel.x += p.x;
            meanClassPixel.y += p.y;
            meanClassPixel.z += p.z;
        }
        meanClassPixel.x /= np;
        meanClassPixel.y /= np;
        meanClassPixel.z /= np;
        
        float norm = sqrt(meanClassPixel.x * meanClassPixel.x + meanClassPixel.y * meanClassPixel.y + meanClassPixel.z * meanClassPixel.z);

        NormalizedMeanClassPixelCPU[i].x = meanClassPixel.x / norm;
        NormalizedMeanClassPixelCPU[i].y = meanClassPixel.y / norm;
        NormalizedMeanClassPixelCPU[i].z = meanClassPixel.z / norm;
    }

    CSC(cudaMemcpyToSymbol(NormalizedMeanClassPixelGPU, NormalizedMeanClassPixelCPU, sizeof(float3) * NUMCLASSES, 0, cudaMemcpyHostToDevice));

    uchar4 *dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(uchar4) * h * w));
    CSC(cudaMemcpy(dev_arr, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));

    kernel<<<dim3(16, 16), dim3(16, 16)>>>(dev_arr, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_arr, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));
   
    write_data(outname, data, w, h);

    CSC(cudaFree(dev_arr));
    free(data);

    return 0;
}
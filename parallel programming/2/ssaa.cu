#include <stdio.h>
#include <stdlib.h>


#define NAMELEN 100

#define CSC(call)  					                                \
do {								                                \
	cudaError_t res = call;		                                	\
	if (res != cudaSuccess) {	                                	\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);				                                   	\
	}							                                   	\
} while(0)



texture<uchar4, 2, cudaReadModeElementType> tex;


__global__ void kernel(uchar4 *dev_arr, int w, int h, int dw, int dh)
{
    int start_x = gridDim.x * blockIdx.x + threadIdx.x;
    int start_y = gridDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    int x, y;
    int xo, yo;
    int xl, yl;
    int3 part_sum;
    uchar4 p;

    for (x = start_x; x < w; x += offset_x) {
        for (y = start_y; y < h; y += offset_y) {
            part_sum = make_int3(0, 0, 0);
            xl = x * dw + dw;
            yl = y * dh + dh;
            for (xo = x * dw; xo < xl; ++xo) {
                for (yo = y * dh; yo < yl; ++yo) {
                    p = tex2D(tex, xo, yo);
                    part_sum.x += p.x;
                    part_sum.y += p.y;
                    part_sum.z += p.z;
                }
            }
            dev_arr[y * w + x] = make_uchar4(part_sum.x / (dw * dh), part_sum.y / (dw *dh), part_sum.z / (dw * dh), 0);
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



cudaArray *init_texture(uchar4 *data, int w, int h)
{
    cudaArray *tex_arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&tex_arr, &ch, w, h));
    CSC(cudaMemcpyToArray(tex_arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false; 

    CSC(cudaBindTextureToArray(tex, tex_arr, ch));

    return tex_arr;
}



int main() 
{
    char inname[NAMELEN];
    char outname[NAMELEN];
    int w, h;
    int new_w, new_h;
    uchar4 *data;

    scanf("%s", inname);
    scanf("%s", outname);
    scanf("%d", &new_w);
    scanf("%d", &new_h);

    read_data(inname, &data, &w, &h);

    cudaArray *tex_arr = init_texture(data, w, h);

    uchar4 *dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(uchar4) * new_w * new_h));
    
    data = (uchar4 *) realloc(data, sizeof(uchar4) * new_w * new_h);

    kernel<<<dim3(16, 16), dim3(16, 16)>>>(dev_arr, new_w, new_h, w / new_w, h / new_h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_arr, sizeof(uchar4) * new_w * new_h, cudaMemcpyDeviceToHost));

    write_data(outname, data, new_w, new_h);

    CSC(cudaUnbindTexture(tex));
    CSC(cudaFreeArray(tex_arr));
    CSC(cudaFree(dev_arr));
    free(data);

    return 0;
}
#include <curand_kernel.h>
#include <iostream>

#define ROUNDS 1000000000
#define ROLLS 231
#define BLOCKSIZE 1024

__global__ void roll_cuda(int *results, unsigned long long seed, int numBlocks)
{
    
    int highest = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);

    int current;

    const size_t runs = (ROUNDS / (BLOCKSIZE * numBlocks))
                      + ((threadIdx.x + blockIdx.x * numBlocks) < (ROUNDS % (BLOCKSIZE * numBlocks)));
    for (size_t i = 0; i < runs; i++)
    {
        current = __popc(curand(&state) & curand(&state));
        current += __popc(curand(&state) & curand(&state));
        current += __popc(curand(&state) & curand(&state));
        current += __popc(curand(&state) & curand(&state));
        current += __popc(curand(&state) & curand(&state));
        current += __popc(curand(&state) & curand(&state));
        current += __popc(curand(&state) & curand(&state));
        current += __popc(curand(&state) & curand(&state) << 25);
        highest = max(highest, current);
    }
    results[idx] = highest;
}

int main()
{
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    int smCount = prop.multiProcessorCount;
    
    int highest = 0;
    int *results;
    
    
    int maxActiveBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, roll_cuda, BLOCKSIZE, 0);
    int numBlocks = smCount * maxActiveBlocks;
    cudaMallocManaged(&results, numBlocks * BLOCKSIZE * sizeof(*results));
    
    float totalTime=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    roll_cuda<<<numBlocks, BLOCKSIZE>>>(results, time(NULL), numBlocks);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalTime, start, stop);

    for (size_t i = 0; i < numBlocks * BLOCKSIZE; i++)
    {
        highest = std::max(highest, results[i]);
    }

    std::cout << "My record is: " << (int)highest << ".\nIt took me " << totalTime << "ms.\n";
    return 0;
}

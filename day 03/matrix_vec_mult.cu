#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void matrixVectorMultiply(const float* matrix, const float* vector, float* result, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += matrix[row * N + col] * vector[col];
        }
        result[row] = sum;
    }
}

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printVector(const float* vector, int size, const char* name) {
    printf("%s (%dx1):\n", name, size);
    for (int i = 0; i < size; i++) {
        printf("%.2f ", vector[i]);
    }
    printf("\n\n");
}

int main() {
    const int N = 10;
    float *h_matrix, *h_vector, *h_result;

    h_matrix = (float *)malloc(N * N * sizeof(float));
    h_vector = (float *)malloc(N * sizeof(float));
    h_result = (float *)malloc(N * sizeof(float));

    if (!h_matrix || !h_vector || !h_result) {
        std::cerr << "Failed to allocate host memory!" << std::endl;
        return -1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_matrix[i * N + j] = 1.0f;
        }
        h_vector[i] = 2.0f;
        h_result[i] = 0.0f;
    }

    float *d_matrix, *d_vector, *d_result;
    
    checkCudaError(cudaMalloc(&d_matrix, N * N * sizeof(float)), "Failed to allocate device memory for matrix");
    checkCudaError(cudaMalloc(&d_vector, N * sizeof(float)), "Failed to allocate device memory for vector");
    checkCudaError(cudaMalloc(&d_result, N * sizeof(float)), "Failed to allocate device memory for result");

    checkCudaError(cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice), 
                   "Failed to copy matrix to device");
    checkCudaError(cudaMemcpy(d_vector, h_vector, N * sizeof(float), cudaMemcpyHostToDevice), 
                   "Failed to copy vector to device");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    matrixVectorMultiply<<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, N);
    
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    checkCudaError(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost), 
                   "Failed to copy result to host");

    printMatrix(h_matrix, N, N, "Matrix A");
    printVector(h_vector, N, "Vector B");
    printVector(h_result, N, "Result C = A * B");

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
    
    free(h_matrix);
    free(h_vector);
    free(h_result);

    return 0;
}

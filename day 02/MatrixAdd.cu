#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Row-wise parallelization - each thread handles one row
__global__ void MatrixAdd_RowParallel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
    if (row < N) {
        // Each thread processes one entire row
        for (int col = 0; col < N; col++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}



// Kernel 2: 2D parallelization - each thread handles one element
__global__ void MatrixAdd_ElementParallel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if thread is within matrix bounds (fixed the logic error)
    if (row < N && col < N) {
        int index = row * N + col;
        C[index] = A[index] + B[index];
    }
}

// Kernel 3: Column-wise parallelization - each thread handles one column
__global__ void MatrixAdd_ColumnParallel(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within matrix bounds
    if (col < N) {
        // Each thread processes one entire column
        for (int row = 0; row < N; row++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}



// Function to check CUDA errors
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

// Function to print a matrix in a readable format
void printMatrix(const float* matrix, int N, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    const int N = 10;  // Matrix size (N x N)
    const int matrixSize = N * N * sizeof(float);
    
    // Host matrices
    float *h_A, *h_B, *h_C;

    // Allocate host memory
    h_A = (float *)malloc(matrixSize);
    h_B = (float *)malloc(matrixSize);
    h_C = (float *)malloc(matrixSize);
    
    // Check for allocation errors
    if (!h_A || !h_B || !h_C) {
        std::cerr << "Failed to allocate host memory!" << std::endl;
        return -1;
    }

    // Initialize input matrices with sample values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 1.0f;  // Matrix A filled with 1.0
            h_B[i * N + j] = 2.0f;  // Matrix B filled with 2.0
            h_C[i * N + j] = 0.0f;  // Matrix C initialized to 0.0
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void **)&d_A, matrixSize), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void **)&d_B, matrixSize), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void **)&d_C, matrixSize), "Failed to allocate device memory for C");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice), "Failed to copy B to device");

    // Configure kernel launch parameters
    dim3 blockSize(32, 16);  // 32 threads in x, 16 threads in y
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the 2D parallelization kernel (most efficient for this case)
    MatrixAdd_ElementParallel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Wait for kernel to complete
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost), "Failed to copy result to host");
    // Print results in a clean format
    printMatrix(h_A, N, "Matrix A");
    printMatrix(h_B, N, "Matrix B");
    printMatrix(h_C, N, "Result Matrix C = A + B");

    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Matrix addition completed successfully!\n");
    return 0;
}
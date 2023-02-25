#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <CL/cl.h>
#include <vector>

#define MAX_SOURCE_SIZE (0x1000)

// initialization of the starting matrix

void initializeMatrix(int size, float* M)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == 0 || i == size-1 || j == 0 || j == size-1) {
                M[i*size+j]    = 0.5;
            } else if (i == 1 || i == size-2 || j == 1 || j == size-2) {
                M[i*size+j] = 0.2;
            } else {
                M[i*size+j] = 0.0;
            }
        }
    }
}

// function for sequential code

void sequential(float** mainMatrix, float** secondaryMatrix, int size)
{
    float* holder;
    for (int i = 2; i < size-2; i++) {
        for (int j = 2; j < size-2; j++) {

            // Main Equation
            (*secondaryMatrix)[i*size+j] =
                                    sqrt(
                                        (*mainMatrix)[(i-1) * size + (j-1)] +
                                        (*mainMatrix)[(i-1) * size + (j+1)] +
                                        (*mainMatrix)[(i+1) * size + (j-1)] +
                                        (*mainMatrix)[(i+1) * size + (j+1)])
                                        *
                                    sqrt(
                                        (*mainMatrix)[(i) * size + (j-2)] +
                                        (*mainMatrix)[(i) * size + (j+2)] +
                                        (*mainMatrix)[(i-2) * size + (j)] +
                                        (*mainMatrix)[(i+2) * size + (j)]);
            
            // Alexander Hecke's Equation
            // (*secondaryMatrix)[i*size+j] =
            //                         sqrt(
            //                             ((*mainMatrix)[(i-1) * size + (j-1)] +
            //                             (*mainMatrix)[(i-1) * size + (j+1)] +
            //                             (*mainMatrix)[(i+1) * size + (j-1)] +
            //                             (*mainMatrix)[(i+1) * size + (j+1)])
            //                             *
            //                             ((*mainMatrix)[(i) * size + (j-2)] +
            //                             (*mainMatrix)[(i) * size + (j+2)] +
            //                             (*mainMatrix)[(i-2) * size + (j)] +
            //                             (*mainMatrix)[(i+2) * size + (j)]))/4;

            
        }
    }
    holder = *mainMatrix;
    *mainMatrix = *secondaryMatrix;
    *secondaryMatrix = holder;
}


int main(int argc, char* argv[]){
    
    printf("\n\n");

    // timers
    float S_TIME;
    float P_TIME;

    //matrices used in sequential
    float* seqMat;
    float* tempSeqMat;

    //default inputs
    int iteration = 70;
    size_t SIZE = 8192;
    size_t locSize = 32;
    float e = 1e-7;

    //buffer size
    int bufferSize = SIZE * SIZE * sizeof(float);

    //interactive
    int inp = 0;

    printf("For interactive setting enter 1, for default 0: ");
    scanf("%d", &inp);
    
    if (inp == 1){
        printf("Number of iterations: ");
        scanf("%d", &iteration);
        printf("Matrix size: ");
        scanf("%d", &SIZE);
        printf("Local work size: ");
        scanf("%d", &locSize);
    }
    

    printf("--------------------------------------------------------\n");
    printf("\t  Matrix Size: %d\n\t   Iterations: %d\n", SIZE, iteration);
    

    seqMat = (float *)malloc(bufferSize);
    tempSeqMat = (float *)malloc(bufferSize);
    
    initializeMatrix(SIZE, seqMat);
    initializeMatrix(SIZE, tempSeqMat);

    clock_t S_START = clock();

    printf("--------------------------------------------------------\n");
    printf("\tSequential\n\n");
    
    for (int i = 0; i < iteration; i++) {
        sequential(&seqMat, &tempSeqMat, SIZE);
    }

    clock_t S_END = clock();

    S_TIME = (float)(S_END-S_START) / CLOCKS_PER_SEC;

    printf("   Sequential runtime: %f seconds", S_TIME);
    printf("\n");

    for(int i=50;i<54;i++){
        for(int j=50;j<54;j++){
            printf("%e\t",seqMat[i+SIZE*j]);
        }
        printf("\n");
    }
//----------------------------------------------------
    //parallel part
    
    
    //parallel matrices
    float* mainMatrix;
    float* secondaryMatrix;
    
    mainMatrix = (float *)malloc(bufferSize);
    secondaryMatrix = (float *)malloc(bufferSize);
    
    initializeMatrix(SIZE, mainMatrix);
    initializeMatrix(SIZE, secondaryMatrix);
  
    FILE *kernelFile;
    char *kernelSource;
    size_t kernelSize;
    printf("--------------------------------------------------------\n");
    printf("\tParallel using OpenCL\n\n");

    

    kernelFile = fopen("assignment-float.cl", "r");

        if (!kernelFile) {
            printf("Kernel file not found.\n");
            exit(-1);
        }
        kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
        kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
        
    fclose(kernelFile);

    // Getting platform and device information
    

    cl_platform_id platformId = NULL;
    cl_device_id deviceID = NULL;
    cl_uint errNumDevices;
    cl_uint errNumPlatforms;

    cl_int err = clGetPlatformIDs(1, &platformId, NULL);

    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, NULL);

    // Creating context.
    cl_context context = clCreateContext(0, 1, &deviceID, NULL, NULL,  &err);

    // Creating command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &err);

    // Memory buffers for each array
    cl_mem mainBuffer = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, bufferSize, mainMatrix, &err);

    cl_mem secondaryBuffer = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, bufferSize, secondaryMatrix, &err);
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &err);
    
    // const char* build_options = "-cl-finite-math-only -cl-mad-enable -cl-single-precision-constant -cl-denorms-are-zero -cl-unsafe-math-optimizations";
    // err = clBuildProgram(program, 1, &deviceID, build_options, NULL, NULL);
    err = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

    clock_t P_START = clock();

    cl_mem holder;

    size_t localSize[2] = {locSize,locSize};
    size_t globalSize[2] = {SIZE,SIZE};

    cl_kernel kernel = clCreateKernel(program, "iterVectors", &err);

    
    //iteration part
    for(int t=0;t<iteration;t++){

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mainBuffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&secondaryBuffer);
        err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
        
        //swapping buffers
        holder=mainBuffer;
        mainBuffer = secondaryBuffer;
        secondaryBuffer=holder;
    }

    printf("\t\tError: %d\n", err);
        
    err = clEnqueueReadBuffer(commandQueue, mainBuffer, CL_TRUE, 0, bufferSize, mainMatrix, 0, NULL, NULL);


    
    clock_t P_END = clock();

    P_TIME = (float)(P_END-P_START) / CLOCKS_PER_SEC;
    
    
    //parallel prints
    printf("Parallel part runtime: %f seconds\n", P_TIME);

    for(int i=50;i<54;i++){
        for(int j=50;j<54;j++){
            printf("%e\t",mainMatrix[i*SIZE+j]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------------\n");


    printf("\tSPEEDUP -> %f times\n", S_TIME/P_TIME);

// test
    int err_count = 0;
    // // float e = 2.225E-307; // smallest positive float value
    // float e = FLT_MIN;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
                if (seqMat[i*SIZE+j] - mainMatrix[i*SIZE+j] > e || mainMatrix[i*SIZE+j] - seqMat[i*SIZE+j] > e)
                {
                    err_count++;
                }
        }
    }
    printf("errorcount -> %d\n", err_count);
    printf("%d %% of cells are different with epsilon: %f", (err_count*100)/(SIZE*SIZE), e);

    err = clFlush(commandQueue);
    err = clFinish(commandQueue);
    err = clReleaseCommandQueue(commandQueue);
    err = clReleaseKernel(kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(mainBuffer);
    err = clReleaseMemObject(secondaryBuffer);
    err = clReleaseContext(context);

    printf("\n\n");

    return 0;
}

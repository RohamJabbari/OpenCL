__kernel void iterVectors(__global float *mainMatrix, __global float *secondaryMatrix) {

    int y = get_global_id(0);
    int x = get_global_id(1);

    int size = get_global_size(0);

    if(x>=2 && x<size-2 && y>=2 && y<size-2)
    {
        // Main Equation
        secondaryMatrix[x*size+y]=
                                sqrt(
                                mainMatrix[(x-1) * size + (y-1)] +
                                mainMatrix[(x-1) * size + (y+1)] +
                                mainMatrix[(x+1) * size + (y-1)] +
                                mainMatrix[(x+1) * size + (y+1)])
                                *
                                sqrt(
                                mainMatrix[(x) * size + (y-2)] +
                                mainMatrix[(x) * size + (y+2)] +
                                mainMatrix[(x-2) * size + (y)] +
                                mainMatrix[(x+2) * size + (y)]);
        
        // Alexander Hecke's Equation
        // secondaryMatrix[x*size+y] =
        //                         native_sqrt(
        //                             (mainMatrix[(x-1) * size + (y-1)] +
        //                             mainMatrix[(x-1) * size + (y+1)] +
        //                             mainMatrix[(x+1) * size + (y-1)] +
        //                             mainMatrix[(x+1) * size + (y+1)])
        //                             *
        //                             (mainMatrix[(x) * size + (y-2)] +
        //                             mainMatrix[(x) * size + (y+2)] +
        //                             mainMatrix[(x-2) * size + (y)] +
        //                             mainMatrix[(x+2) * size + (y)]))/4;
    }
}

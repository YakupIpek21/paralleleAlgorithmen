export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${CL_DEVICE+set}" != set ; then
    export CL_DEVICE=0
fi

if test "${LOCALSIZE+set}" != set ; then
    export LOCALSIZE=32
fi

gcc -O3 -Wall -Werror $OCLLIB -std=c99  countsort_bench.c -o countsort_bench_cl_D$CL_DEVICE -DCL_DEVICE=$CL_DEVICE -DLOCALSIZE=$LOCALSIZE -lm
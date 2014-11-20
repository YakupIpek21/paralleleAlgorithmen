export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${LOCALSIZE+set}" != set ; then
    export LOCALSIZE=256
fi

if test "${CL_DEVICE+set}" != set ; then
    export CL_DEVICE=0
fi

gcc -O2 -std=c99 $OCLLIB hillissteele.c dSFMT.c  -o hillissteele_cl_D$CL_DEVICE -DLOCALSIZE=$LOCALSIZE -DCL_DEVICE=$CL_DEVICE -lm
gcc -O2 -std=c99 $OCLLIB downsweep.c dSFMT.c  -o downsweep_cl_D$CL_DEVICE -DLOCALSIZE=$LOCALSIZE -DCL_DEVICE=$CL_DEVICE -lm
#gcc -O2 -std=c99 $OCLLIB prefixglobal.c dSFMT.c  -o prefixglobal_D$CL_DEVICE -DLOCALSIZE=$LOCALSIZE -DCL_DEVICE=$CL_DEVICE -lm
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpfunctions.c"

#define VALUE float


cl_ulong getProfileTotalTime(cl_event* event,int index);

void printProfileInfo(cl_ulong time, char* message);

cl_event* allocateMemoryForEvent(int numberOfEvents);

void printResult(VALUE* matrix, int amount, int elemsInRow, char* info);

int roundUp(int localWidthOrHeight, int globalWidthOrHeight);
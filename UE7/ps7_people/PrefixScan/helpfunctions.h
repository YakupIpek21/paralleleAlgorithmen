#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpfunctions.c"

#define VALUE float

//kernel execution time is computed for defined event, whichTime is 0 for start_time and 1 for end_time
cl_ulong getProfileTime(cl_event* event,int index, int whichTime);

void printProfileInfo(cl_ulong start, cl_ulong end, char* message);

cl_event* allocateMemoryForEvent(int numberOfEvents);

void printResult(VALUE* matrix, int amount, int elemsInRow, char* info);

int roundUp(int localWidthOrHeight, int globalWidthOrHeight);
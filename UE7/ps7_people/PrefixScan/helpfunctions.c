#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define VALUE float

cl_ulong getProfileTotalTime(cl_event* event,int index){
  cl_ulong time_start;
  cl_ulong time_end;
  clWaitForEvents(1 , (event+index));	//profile data should only be read when the command associated with the event is finished
  clGetEventProfilingInfo(*(event+index), CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(*(event+index), CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  
  return time_end-time_start;
}

//calc difference between start and end, print result
void printProfileInfo(cl_ulong time, char* message){
  printf("\n%s %0.2f ms\n",message,(double) ((time)/1000000));
}


cl_event* allocateMemoryForEvent(int numberOfEvents){
 cl_event* events=(cl_event*) malloc(sizeof(cl_event)*numberOfEvents);
 return events;
}

void printResult(VALUE* matrix, int amount, int elemsInRow, char* info){
  printf("%s\n", info);
  for(int i=0; i<amount; i++){
    printf("%0.0f\t", matrix[i]);
    if((i+1)%elemsInRow==0) printf("\n");
  }
  printf("\n");
}

//
int roundUp(int localWidthOrHeight, int globalWidthOrHeight){
  while(1){
      if(((globalWidthOrHeight)%localWidthOrHeight)==0){
	break;
      }else{
	globalWidthOrHeight++;
      }
  }
  return globalWidthOrHeight;
}
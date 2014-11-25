#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define VALUE float

/*------------------------------------------------Implementation--------------------------------------*/

//kernel execution time is computed for defined event, whichTime is 0 for start_time and 1 for end_time
cl_ulong getProfileTime(cl_event* event,int index, int whichTime){
  cl_ulong time_start;
  cl_ulong time_end;
  clWaitForEvents(1 , (event+index));	//profile data should only be read when the command associated with the event is finished
  clGetEventProfilingInfo(*(event+index), CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(*(event+index), CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  
  if(whichTime){
    return time_end;
  }
  else{
    return time_start;
  }
}

void printProfileInfo(cl_ulong start, cl_ulong end, char* message){
  printf("\n%s %0.2f ms\n",message,(double) ((end-start)/1000000));
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
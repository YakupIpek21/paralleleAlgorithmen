#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "time_ms.h"
#include "cl_utils.h"
#include "dSFMT.h"
#include "helpfunctions.h"

#ifndef LOCALSIZE
  #define LOCALSIZE 256
#endif

#ifndef VALUE
  #define VALUE float
#endif

#define KERNEL_FILE_NAME "./prefixglobal.cl"

int getPowerOfTwo(int number);

int main(int argc, char** argv){
	
	if(argc != 2) {
		printf("Usage: search [elements]\nExample: scan 10000\n");
		return -1;
	}
	
	unsigned long long start_time = time_ms();
	int event_amount=2;
	cl_event* events=allocateMemoryForEvent(event_amount);
	cl_ulong start, end;
	int elems = atoi(argv[1]);
	
	cl_int err;
	
	size_t localWorkGroupSize[1]={LOCALSIZE};
	size_t globalWorkGroupSize[1]={roundUp(LOCALSIZE, elems)};
	
	int howManyWorkGroups=globalWorkGroupSize[0]/LOCALSIZE;	//does not have to be power of two
	const int localBufferSize=getPowerOfTwo(howManyWorkGroups);
	
	VALUE *data = (VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result_seq=(VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result=(VALUE*)malloc(elems*sizeof(VALUE));
	
	memset(result,0,elems*sizeof(VALUE));
	memset(result_seq,0,elems*sizeof(VALUE));
	
	// initialize random number generator
	dsfmt_t rand_state;
	dsfmt_init_gen_rand(&rand_state, (uint32_t)time(NULL));

	// initialize data set (fill randomly)
	for(int j=0; j<elems; ++j) {
		data[j] = dsfmt_genrand_close1_open2(&rand_state);
	}
	
	printResult(data, elems, 4, "INPUT");
	
	
	/*Sequential Scan*/
	for(int i=1; i<elems; i++){
	    result_seq[i]=result_seq[i-1]+data[i-1];
	}
	
	printResult(result_seq, elems, 4, "Sequential Algorithm OUTPUT");
	
	
	//ocl initialization
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);
  
  
	// create memory buffer
	cl_mem mem_data=clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_USE_HOST_PTR,elems*sizeof(VALUE), data, &err);
	cl_mem mem_result=clCreateBuffer(context, CL_MEM_WRITE_ONLY, elems*sizeof(VALUE), NULL,&err);
	cl_mem mem_temporal=clCreateBuffer(context, CL_MEM_READ_WRITE, howManyWorkGroups*sizeof(VALUE), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create Buffer");
    
    
	// create kernel from source
	char tmp[1024];
	sprintf(tmp,"-DLOCALSIZE=%i  -DVALUE=%s",LOCALSIZE,EXPAND_AND_QUOTE(VALUE));
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel = clCreateKernel(program, "prefix_scan", &err);
	CLU_ERRCHECK(err,"Could not load source program");
    
	
	/*improved implementation*/
	// set arguments
	cluSetKernelArguments(kernel, 6, sizeof(cl_mem), (void *)&mem_data, sizeof(cl_mem), (void*)&mem_result,
			      sizeof(cl_mem), (void*)&mem_temporal, sizeof(int), (void*)&elems,
			      sizeof(const int), (void*)&localBufferSize, sizeof(VALUE)*localBufferSize,NULL);
	
	//execute kernel  	     
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalWorkGroupSize, localWorkGroupSize, 0, NULL, &(events[1])), "Failed to enqueue 2D kernel");		      

	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result, 0, NULL, &(events[1])),"Failed to read Min Values");
 
	
	printResult(result, elems, 4, "IMPROVED IMPLEMENTATION OUTPUT");
	
	start=getProfileTime(events,1,0);
	end=getProfileTime(events,1,1);
	printProfileInfo(start,end,"Improved Algorithm Time:");
	
	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));
	printf("Done, took %16llu ms\n", time_ms()-start_time);
    
	
	
	// finalization
	
	for(int i=0; i<event_amount; i++){
	    clReleaseEvent(events[i]);
	}
	
	err =  clFinish(command_queue);
	err |= clReleaseKernel(kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(mem_data);
	err |= clReleaseMemObject(mem_result);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
    
	free(events);
	free(result);
	free(result_seq);
	
	return EXIT_SUCCESS; 
}

int getPowerOfTwo(int number){
  int two=2;
  while(1){
    if(two==number || two>2) break;
    two*=2;
  }
  return two;
}


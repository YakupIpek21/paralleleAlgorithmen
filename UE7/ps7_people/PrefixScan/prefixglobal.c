#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "time_ms.h"
#include "cl_utils.h"
#include "helpfunctions.h"

#ifndef LOCALSIZE
  #define LOCALSIZE 256
#endif

#ifndef VALUE
  #define VALUE int
#endif

#define KERNEL_FILE_NAME "./prefixglobal.cl"

int getPowerOfTwo(int number);
void verifyResult(VALUE* want_f, VALUE* have_f, int entries, char* text);

int main(int argc, char** argv){
	
	srand(time(NULL));
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
	size_t globalWorkGroupSize[1]={getPowerOfTwo(roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)/2))};
	
	printf("LOCALSIZE: %d\n",LOCALSIZE);
	printf("Log1: %d\n",roundUp(LOCALSIZE, elems));
	printf("Log2: %d\n",roundUp(LOCALSIZE, elems)/2);
	printf("Log3: %d\n",roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)/2));
	printf("Log4: %d\n",getPowerOfTwo(roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)/2)));
	printf("GlobalSize: %d\n",getPowerOfTwo(roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)/2)));
	
	int howManyWorkGroups=globalWorkGroupSize[0]/LOCALSIZE;	//does not have to be power of two
	
	printf("Workgroups: %d\n",howManyWorkGroups);
	
	
	int sum_length=getPowerOfTwo(roundUp(LOCALSIZE,elems)/LOCALSIZE);
	
	VALUE *data = (VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result_seq=(VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result=(VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result_hillissteele=(VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *sum=(VALUE*)malloc(howManyWorkGroups*sizeof(VALUE));
	VALUE *sum_hillis=(VALUE*)malloc(sum_length*sizeof(VALUE));
	
	memset(result,0,elems*sizeof(VALUE));
	memset(result_seq,0,elems*sizeof(VALUE));
	
	
	// initialize data set (fill randomly)
	for(int j=0; j<elems; ++j) {
		data[j] =rand()%121;
	}
	
	//printResult(data, elems, 4, "INPUT");
	
	
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
 	cl_mem mem_result=clCreateBuffer(context, CL_MEM_READ_WRITE, elems*sizeof(VALUE), NULL,&err);
	cl_mem mem_sum=clCreateBuffer(context, CL_MEM_READ_WRITE, howManyWorkGroups*sizeof(VALUE), NULL, &err);
	cl_mem mem_sum_hillis=clCreateBuffer(context, CL_MEM_READ_WRITE, sum_length*sizeof(VALUE), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create Buffer");
    
    
	// create kernel from source
	char tmp[1024];
	sprintf(tmp," -DVALUE=%s",EXPAND_AND_QUOTE(VALUE));
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel_downSweep = clCreateKernel(program, "prefix_scan_downSweep", &err);
	cl_kernel kernel_hillissteele=clCreateKernel(program, "prefix_scan_hillissteele", &err);
	cl_kernel kernel_last_stage= clCreateKernel(program, "prefix_scan_last_stage", &err);
	CLU_ERRCHECK(err,"Could not load source program");
    

	
	/*-----------------------------DOWNSWEEP--------------------------------*/
	// set arguments
	int border=elems/2;
	int flag=1;
	printf("Border: %d\n",border);
	cluSetKernelArguments(kernel_downSweep, 6, sizeof(cl_mem), (void *)&mem_data, sizeof(cl_mem), (void*)&mem_result,
			      sizeof(cl_mem), (void*)&mem_sum,sizeof(VALUE)*LOCALSIZE*2, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);

	//execute kernel  	     
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_downSweep, 1, NULL, globalWorkGroupSize, localWorkGroupSize, 0, NULL, &(events[1])), "Failed to enqueue 2D kernel");		      
	
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result, 0, NULL, &(events[1])),"Failed to read Result Values");
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum, CL_TRUE, 0, howManyWorkGroups*sizeof(VALUE), sum, 0, NULL, &(events[1])),"Failed to read Sum Values");
	clFinish(command_queue);

	/*for(int i=0; i<howManyWorkGroups; i++){
	    printf("Sums[%d]:\t%f\n",i, sum[i]);
	    printf("\n");
	}*/
	
	/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/
	flag=0;
	border=howManyWorkGroups/2;
	cluSetKernelArguments(kernel_downSweep, 6, sizeof(cl_mem), (void *)&mem_sum, sizeof(cl_mem), (void*)&mem_sum,
			      sizeof(cl_mem), (void*)&mem_sum,sizeof(VALUE)*howManyWorkGroups, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);
	
	howManyWorkGroups>1 ? globalWorkGroupSize[0]=howManyWorkGroups/2:howManyWorkGroups;
	howManyWorkGroups>1 ? localWorkGroupSize[0]=howManyWorkGroups/2:howManyWorkGroups;
	//printf("GlobalSize: %d\tLocalWorkGroupSize: %d\n",globalWorkGroupSize[0], localWorkGroupSize[0]);
	//execute kernel

	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_downSweep, 1, NULL, globalWorkGroupSize, localWorkGroupSize, 0, NULL, &(events[1])), "Failed to enqueue 2D kernel");		      

	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum, CL_TRUE, 0, howManyWorkGroups*sizeof(VALUE), sum, 0, NULL, &(events[1])),"Failed to read Sum Values");
	clFinish(command_queue);
	

	/*for(int i=0; i<howManyWorkGroups; i++){
	    printf("Sums_Prefix[%d]:\t%f\n",i, sum[i]);
	    printf("\n");
	}*/
	
	/*++++++++++++++++++++++++++++++*/
	border=howManyWorkGroups;
	cluSetKernelArguments(kernel_last_stage, 3, sizeof(cl_mem), (void *)&mem_result, sizeof(cl_mem), (void*)&mem_sum, sizeof(int), (void*)&border);
	globalWorkGroupSize[0]=roundUp(LOCALSIZE,getPowerOfTwo(roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)))/2);
	localWorkGroupSize[0]=LOCALSIZE;
	//printf("GlobalSize: %d\tLocalWorkGroupSize: %d\n",globalWorkGroupSize[0], localWorkGroupSize[0]);
	printf("GLOBALSIZE: %d\tLOCALSIZE %d\n",globalWorkGroupSize[0],localWorkGroupSize[0]);
	//execute kernel  	     
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_last_stage, 1, NULL, globalWorkGroupSize, localWorkGroupSize, 0, NULL, &(events[1])), "Failed to enqueue 2D kernel");		      
	
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result, 0, NULL, &(events[1])),"Failed to read Result Values");
	
	/*---------------------------------------HILLISSTEELE----------------------------------------------------------*/
	flag=1;
	border=elems;
	cluSetKernelArguments(kernel_hillissteele, 6, sizeof(cl_mem), (void *)&mem_data, sizeof(cl_mem), (void*)&mem_result,
			      sizeof(cl_mem), (void*)&mem_sum_hillis,sizeof(VALUE)*LOCALSIZE*2, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);

	//execute kernel
	globalWorkGroupSize[0]=roundUp(LOCALSIZE,elems);
	localWorkGroupSize[0]=LOCALSIZE;
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_hillissteele, 1, NULL, globalWorkGroupSize, localWorkGroupSize, 0, NULL, &(events[1])), "Failed to enqueue 2D kernel");		      
	
	howManyWorkGroups=globalWorkGroupSize[0]/LOCALSIZE;
	
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result_hillissteele, 0, NULL, &(events[1])),"Failed to read Result Values");
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum_hillis, CL_TRUE, 0, sum_length*sizeof(VALUE), sum_hillis, 0, NULL, &(events[1])),"Failed to read Sum_1 Values");
	clFinish(command_queue);
	
	for(int i=0; i<sum_length; i++){
	    printf("Sums_Hillis[%d]:\t%f\n",i, sum_hillis[i]);
	    printf("\n");
	}

	/*++++++++++++++++++++++++++++++++++++++*/
	
	
	flag=0;
	border=getPowerOfTwo(howManyWorkGroups);
	cluSetKernelArguments(kernel_hillissteele, 6, sizeof(cl_mem), (void *)&mem_sum_hillis, sizeof(cl_mem), (void*)&mem_sum_hillis,
			      sizeof(cl_mem), (void*)&mem_sum_hillis,sizeof(VALUE)*howManyWorkGroups*2, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);

	//execute kernel
	globalWorkGroupSize[0]=getPowerOfTwo(howManyWorkGroups);
	localWorkGroupSize[0]=getPowerOfTwo(howManyWorkGroups);
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_hillissteele, 1, NULL, globalWorkGroupSize, localWorkGroupSize, 0, NULL, &(events[1])), "Failed to enqueue 2D kernel");		      
	
	howManyWorkGroups=globalWorkGroupSize[0]/LOCALSIZE;
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum_hillis, CL_TRUE, 0, sum_length*sizeof(VALUE), sum_hillis, 0, NULL, &(events[1])),"Failed to read Sum2 Values");
	clFinish(command_queue);
	
	for(int i=0; i<sum_length; i++){
	    printf("Sums_Prefix_Hillis[%d]:\t%f\n",i, sum_hillis[i]);
	    printf("\n");
	}
	/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
	
	
	cluSetKernelArguments(kernel_last_stage, 3, sizeof(cl_mem), (void *)&mem_result, sizeof(cl_mem), (void*)&mem_sum, sizeof(int), (void*)&border);
	globalWorkGroupSize[0]=roundUp(LOCALSIZE,getPowerOfTwo(roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)))/2);
	localWorkGroupSize[0]=LOCALSIZE;
	//printf("GlobalSize: %d\tLocalWorkGroupSize: %d\n",globalWorkGroupSize[0], localWorkGroupSize[0]);
	printf("GLOBALSIZE: %d\tLOCALSIZE %d\n",globalWorkGroupSize[0],localWorkGroupSize[0]);
	//execute kernel  	     
	//CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_last_stage, 1, NULL, globalWorkGroupSize, localWorkGroupSize, 0, NULL, &(events[1])), "Failed to enqueue 2D kernel");		      
	
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result_hillissteele, 0, NULL, &(events[1])),"Failed to read Result Values");
	
	printResult(result_hillissteele, elems, 4, "HILLISSTEELE OUTPUT");
	
	//printResult(result, elems, 4, "IMPROVED IMPLEMENTATION OUTPUT");
	verifyResult(result_seq,result,elems, "Verifying result of DownSweep for bigger array size");
	verifyResult(result_seq,result_hillissteele,elems, "Verifying result of HILLISSTEELE for bigger array size");
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
	err |= clReleaseKernel(kernel_downSweep);
	err |= clReleaseKernel(kernel_last_stage);
	err |= clReleaseKernel(kernel_hillissteele);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(mem_data);
	err |= clReleaseMemObject(mem_result);
	err |= clReleaseMemObject(mem_sum);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
    
	free(events);
	free(result);
	free(result_hillissteele);
	free(result_seq);
	free(sum);
	free(sum_hillis);
	
	return EXIT_SUCCESS; 
}

int getPowerOfTwo(int number){
  int two=2;
  while(1){
    if(two==number || two>number) break;
    two*=2;
  }
  return two;
}

void verifyResult(VALUE* want_f, VALUE* have_f, int entries, char* text){
  printf("\n%s\n",text);
  bool check=true;
  int want=0;
  int have=0;
  for(int i=0; i<entries; i++){
      want=(int) *(want_f+i);
      have=(int) *(have_f+i);
      if(want!=have) check=false;
  }
  if(!check){
      printf("--------------------------Wrong Result--------------------------\n");
  }else{
      printf("--------------------------Correct Result--------------------------\n");
  }
}

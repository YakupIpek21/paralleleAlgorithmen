#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "time_ms.h"
#include "cl_utils.h"
#include "helpfunctions.h"

#ifndef CL_DEVICE
  #define CL_DEVICE 0
#endif

#ifndef LOCALSIZE
  #define LOCALSIZE 256
#endif

#ifndef VALUE
  #define VALUE int
#endif

#define KERNEL_FILE_NAME "./prefixglobal.cl"

int getPowerOfTwo(int number);
void verifyResult(VALUE* want_f, VALUE* have_f, int entries, char* text);
void printSumBuffer(VALUE* sum, int entries, char* message);

int main(int argc, char** argv){
	
	srand(time(NULL));
	
	if(argc != 2) {
		printf("Usage: search [elements]\nExample: scan 10000\n");
		return -1;
	}
	
	unsigned long long start_time = time_ms();
	int event_amount=2;
	int elems = atoi(argv[1]);
	
	cl_int err;
	cl_event* events=allocateMemoryForEvent(event_amount);
	cl_ulong total_downsweep=0,total_hillissteele=0;
	size_t localWorkGroupSize_downSweep[1]={LOCALSIZE};	//must be power of two
	size_t globalWorkGroupSize_downSweep[1]={getPowerOfTwo(roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)/2))};	//calculating
	
	size_t localWorkGroupSize_hillissteele[1]={LOCALSIZE};	//must be power of two
	size_t globalWorkGroupSize_hillissteele[1]={roundUp(LOCALSIZE,elems)};	//calculating worksize
	
	
	int howManyWorkGroups=globalWorkGroupSize_downSweep[0]/LOCALSIZE;	//quotient is power of two, since dividend and divisor are power of two
	int sumBuffer_length_downSweep=howManyWorkGroups;
	int sumBuffer_length_hillis=getPowerOfTwo(roundUp(LOCALSIZE,elems)/LOCALSIZE);	

	VALUE *data = (VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result_seq=(VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result=(VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *result_hillissteele=(VALUE*)malloc(elems*sizeof(VALUE));
	VALUE *sum=(VALUE*)malloc(sumBuffer_length_downSweep*sizeof(VALUE));
	VALUE *sum_hillis=(VALUE*)malloc(sumBuffer_length_hillis*sizeof(VALUE));
	

	memset(sum_hillis,0,sumBuffer_length_hillis*sizeof(VALUE));
	memset(result_seq,0,elems*sizeof(VALUE));
	
	// initialize data set (fill randomly)
	for(int j=0; j<elems; ++j) {
		data[j] =rand()%121;
	}
	
//	printResult(data, elems, 4, "INPUT");
	
	/*Sequential Scan*/
	for(int i=1; i<elems; i++){
	    result_seq[i]=result_seq[i-1]+data[i-1];
	}
	
//	printResult(result_seq, elems, 4, "Sequential Algorithm OUTPUT");
		
	//ocl initialization
	size_t deviceInfo;
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t), &deviceInfo,NULL );
  
	
	// create memory buffer
	cl_mem mem_data=clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_USE_HOST_PTR,elems*sizeof(VALUE), data, &err);
 	cl_mem mem_data_hillis=clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_USE_HOST_PTR,elems*sizeof(VALUE), data, &err);
	cl_mem mem_result=clCreateBuffer(context, CL_MEM_READ_WRITE, elems*sizeof(VALUE), NULL,&err);
	cl_mem mem_result_tmp=clCreateBuffer(context, CL_MEM_READ_WRITE, elems*sizeof(VALUE), NULL,&err);
	cl_mem mem_sum=clCreateBuffer(context, CL_MEM_READ_WRITE, sumBuffer_length_downSweep*sizeof(VALUE), NULL, &err);
	cl_mem mem_sum_hillis=clCreateBuffer(context, CL_MEM_READ_WRITE, sumBuffer_length_hillis*sizeof(VALUE), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create Buffer");
    
	err=clEnqueueWriteBuffer(command_queue, mem_sum_hillis, CL_TRUE, 0, sumBuffer_length_hillis*sizeof(VALUE), sum_hillis, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to write values into mem_sum");
	

	// create kernel from source
	char tmp[1024];
 	sprintf(tmp,"-DVALUE=%s", EXPAND_AND_QUOTE(VALUE));
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel_downSweep = clCreateKernel(program, "prefix_scan_downSweep", &err);
	cl_kernel kernel_hillissteele=clCreateKernel(program, "prefix_scan_hillissteele", &err);
	cl_kernel kernel_last_stage= clCreateKernel(program, "prefix_scan_last_stage", &err);
	CLU_ERRCHECK(err,"Could not load source program");
    

	
	/*-------------------------------------DOWNSWEEP-----------------------------------------------*/
	// set arguments
	int border=elems/2;
	int flag=1;
	
	cluSetKernelArguments(kernel_downSweep, 6, sizeof(cl_mem), (void *)&mem_data, sizeof(cl_mem), (void*)&mem_result,
			      sizeof(cl_mem), (void*)&mem_sum,sizeof(VALUE)*LOCALSIZE*2, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);

	//execute kernel  	     
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_downSweep, 1, NULL, globalWorkGroupSize_downSweep, localWorkGroupSize_downSweep, 0, NULL, &(events[1])), "DownSweep_Failed to enqueue 2D kernel");		      
	clFinish(command_queue);
	total_downsweep+=getProfileTotalTime(events,1);
	
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result, 0, NULL, NULL),"DownSweep_Failed to read Result Values");
	
	/*
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum, CL_TRUE, 0, sumBuffer_length_downSweep*sizeof(VALUE), sum, 0, NULL, NULL),"Failed to read Sum Values");
	clFinish(command_queue);
	printSumBuffer(sum, sumBuffer_length_downSweep,"DOWNSWEEP SUM");
	*/
	err=clEnqueueCopyBuffer(command_queue, mem_result, mem_result_tmp, 0, 0, elems*sizeof(VALUE),0,NULL,NULL);
	CLU_ERRCHECK(err,"DownSweep_Failed during copying buffer");
	
	
	/*+++++++++++++++++++++++++++++++++DOWNSWEEP-ON-SUM-BUFFER+++++++++++++++++++++++++++++++++++++++*/
	flag=0;
	border=sumBuffer_length_downSweep/2;	//since sumbuffer_length is power of two no further adaption is needed
	cluSetKernelArguments(kernel_downSweep, 6, sizeof(cl_mem), (void *)&mem_sum, sizeof(cl_mem), (void*)&mem_sum,
			      sizeof(cl_mem), (void*)&mem_sum,sizeof(VALUE)*sumBuffer_length_downSweep, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);
	
	howManyWorkGroups>1 ? globalWorkGroupSize_downSweep[0]=howManyWorkGroups/2:howManyWorkGroups;	//if 1 workgroup make adaption
	howManyWorkGroups>1 ? localWorkGroupSize_downSweep[0]=howManyWorkGroups/2:howManyWorkGroups;	//if 1 workgroup make adaption
	
	
	//execute kernel
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_downSweep, 1, NULL, globalWorkGroupSize_downSweep, localWorkGroupSize_downSweep, 0, NULL,&(events[1])), "DownSweep_Failed to enqueue 2D kernel");		      
	clFinish(command_queue);
	total_downsweep+=getProfileTotalTime(events,1);
	/*
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum, CL_TRUE, 0, sumBuffer_length_downSweep*sizeof(VALUE), sum, 0, NULL, NULL),"Failed to read Sum Values");	
	printSumBuffer(sum, sumBuffer_length_downSweep,"DOWNSWEEP SUM PREFIX");
	*/
	
	/*+++++++++++++++++++++++++++++++++DOWNSWEEP-LAST-STAGE(Add Sums)++++++++++++++++++++++++++++++++++++++++*/
	border=sumBuffer_length_downSweep;
	flag=1;
	cluSetKernelArguments(kernel_last_stage, 4, sizeof(cl_mem), (void *)&mem_result_tmp, sizeof(cl_mem), (void*)&mem_sum, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);
	globalWorkGroupSize_downSweep[0]=getPowerOfTwo(roundUp(LOCALSIZE,roundUp(LOCALSIZE, elems)/2));
	localWorkGroupSize_downSweep[0]=LOCALSIZE;
	
	//printf("GLOBALSIZE: %d\tLOCALSIZE %d\n",globalWorkGroupSize[0],localWorkGroupSize[0]);
	
	//execute kernel  	     
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_last_stage, 1, NULL, globalWorkGroupSize_downSweep, localWorkGroupSize_downSweep, 0, NULL, &(events[1])), "DownSweep_Failed to enqueue 2D kernel");		      
	clFinish(command_queue);
	total_downsweep+=getProfileTotalTime(events,1);
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result_tmp, CL_TRUE, 0, elems*sizeof(VALUE), result, 0, NULL, NULL),"DownSweep_Failed to read Result Values");
	
	
	/*---------------------------------------HILLISSTEELE----------------------------------------------------------*/
	
	
	flag=1;
	border=elems;
	
	cluSetKernelArguments(kernel_hillissteele, 6, sizeof(cl_mem), (void *)&mem_data_hillis, sizeof(cl_mem), (void*)&mem_result,
			      sizeof(cl_mem), (void*)&mem_sum_hillis,sizeof(VALUE)*LOCALSIZE*2, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);

	//execute kernel	
	//printf("GlobalSize: %d\tLocalWorkGroupSize: %d\n",globalWorkGroupSize[0], localWorkGroupSize[0]);
	//printf("Amount of WorkGroups: %d\n", globalWorkGroupSize[0]/localWorkGroupSize[0]);
	
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_hillissteele, 1, NULL, globalWorkGroupSize_hillissteele, localWorkGroupSize_hillissteele, 0, NULL, &(events[0])), "Hillissteele_Failed to enqueue 2D kernel_Inputbuffer");		      
	
	clFinish(command_queue);
	total_hillissteele+=getProfileTotalTime(events,0);
	//read values back from device
	/*
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result_hillissteele, 0, NULL, NULL),"Failed to read Result Values");
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum_hillis, CL_TRUE, 0, sumBuffer_length_hillis*sizeof(VALUE), sum_hillis, 0, NULL, NULL),"Failed to read Sum_1 Values");
	printSumBuffer(sum_hillis, sumBuffer_length_hillis, "HILLISSTEELE SUM");
	printResult(result_hillissteele,elems, 4, "HILLISSTEELE Temporary OUTPUT");
	*/
	
	
	/*++++++++++++++++++++++++++++++++++++++HILLISSTEELE-ON-SUM-BUFFER+++++++++++++++++++++++++++++++++++++*/
	
	flag=0;
	border=sumBuffer_length_hillis;
	cluSetKernelArguments(kernel_hillissteele, 6, sizeof(cl_mem), (void *)&mem_sum_hillis, sizeof(cl_mem), (void*)&mem_sum_hillis,
			      sizeof(cl_mem), (void*)&mem_sum_hillis,sizeof(VALUE)*howManyWorkGroups*2, NULL, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);

	//execute kernel
	globalWorkGroupSize_hillissteele[0]=sumBuffer_length_hillis;
	localWorkGroupSize_hillissteele[0]=sumBuffer_length_hillis;
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_hillissteele, 1, NULL, globalWorkGroupSize_hillissteele, localWorkGroupSize_hillissteele, 0, NULL, &(events[0])), "Hillissteele_Failed to enqueue 2D kernel_Sumbuffer");		      
	
	clFinish(command_queue);
	total_hillissteele+=getProfileTotalTime(events,0);
	
	//read values back from device
	/*
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_sum_hillis, CL_TRUE, 0, sumBuffer_length_hillis*sizeof(VALUE), sum_hillis, 0, NULL, NULL),"Failed to read Sum2 Values");
	printSumBuffer(sum_hillis, sumBuffer_length_hillis, "HILLISSTEELE SUM PREFIX");
	*/
	
	/*+++++++++++++++++++++++++++++++++++++HILLISSTEELE-LAST-STAGE(Add Sums)++++++++++++++++++++++++++++++++++++++++*/
	
	flag=0;
	border=sumBuffer_length_hillis;
	cluSetKernelArguments(kernel_last_stage, 4, sizeof(cl_mem), (void *)&mem_result, sizeof(cl_mem), (void*)&mem_sum_hillis, sizeof(int), (void*)&border,
			      sizeof(int), (void*)&flag);
	
	globalWorkGroupSize_hillissteele[0]=roundUp(LOCALSIZE,elems);
	localWorkGroupSize_hillissteele[0]=LOCALSIZE;
	
	//printf("GLOBALSIZE: %d\tLOCALSIZE %d\n",globalWorkGroupSize[0],localWorkGroupSize[0]);
	
	//execute kernel  	     
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_last_stage, 1, NULL, globalWorkGroupSize_hillissteele, localWorkGroupSize_hillissteele, 0, NULL, &(events[0])), "Hillissteele_Failed to enqueue kernel_Last_stage");		      
	clFinish(command_queue);
	total_hillissteele+=getProfileTotalTime(events,0);
	
	//read values back from device
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, elems*sizeof(VALUE), result_hillissteele, 0, NULL, NULL),"Hillissteele_Failed to read Result Values");
	
	
	/*-------------------------FINISHED---------------------------------------------*/
	
	//printResult(result_hillissteele, elems, 4, "HILLISSTEELE OUTPUT");
	//printResult(result, elems, 4, "IMPROVED IMPLEMENTATION OUTPUT");
	
	//verify results
	verifyResult(result_seq,result,elems, "Verifying result of DownSweep for bigger array size");
	verifyResult(result_seq,result_hillissteele,elems, "Verifying result of HILLISSTEELE for bigger array size");
	
	
	printProfileInfo(total_downsweep,"Improved Algorithm Time:");
	printProfileInfo(total_hillissteele,"Hillis & Steele Time:");
	printf("\nDEVICE INFO MAX_WORK_GROUP_SIZE: %d\n", (int) deviceInfo);
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
	err |= clReleaseMemObject(mem_data_hillis);
	err |= clReleaseMemObject(mem_result);
	err |= clReleaseMemObject(mem_result_tmp);
	err |= clReleaseMemObject(mem_sum);
	err |= clReleaseMemObject(mem_sum_hillis);
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

void printSumBuffer(VALUE* sum, int entries, char* message){
  for(int i=0; i<entries; i++){
	    printf("%s\t[%d]:\t%f\n",message,i, *(sum+i));
	    printf("\n");
  }
}

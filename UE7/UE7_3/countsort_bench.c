#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "time_ms.h"
#include "people.h"
#include "cl_utils.h"

#ifndef LOCALSIZE
  #define LOCALSIZE 32
#endif

#ifndef CL_DEVICE
  #define CL_DEVICE 0
#endif

#define MAX_AGE 10

#define KERNEL_FILE_NAME "./countsort.cl"

void countSortLastStage(person_t* people,person_t* people_sorted, int* tmp_hist, int entries);
void prefixSum(int* tmp, int* hist);
void calcHistogram(person_t* people, int* hist, int entries);
void generate_list(person_t* people, int entries);
void printHist(int* hist);
void printPeople(person_t* people, int entries);
void write_ages(person_t* people, int* ages, int entries);
int roundUp(int from, int to);
int checkPowerOfTwo(int num);
void verify(int* seq, int* par,char* info);

int main(int argc, char** argv){
  
  unsigned long start_time,end_time;
  
  if(argc != 3) {
    printf("Usage: list_gen [amount][any number]\nExample: ./list_gen 5 1229");
    return EXIT_FAILURE;
  }
  if(!checkPowerOfTwo(LOCALSIZE)){
    printf("LOCALSIZE must be a power of two; Redefine LOCALSIZE and try again");
    return EXIT_FAILURE;
  }
  
  int amount_people= atoi(argv[1]);
  time_t startRand=atoi(argv[2]);
  srand(time(&startRand));
  
  person_t* people=(person_t*)malloc(amount_people*sizeof(person_t));
  person_t* people_sorted=(person_t*)malloc(amount_people*sizeof(person_t));
  int* count_sort_hist=(int*)malloc(MAX_AGE*sizeof(int));
  int* tmp_hist=(int*)malloc(MAX_AGE*sizeof(int));
  int* tmp_hist_parallel=(int*)malloc(MAX_AGE*sizeof(int));
  int* tmp_prefix_parallel=(int*)malloc(MAX_AGE*sizeof(int));
  int* tmp_hist_seq=(int*)malloc(MAX_AGE*sizeof(int));
  int* tmp_ages=(int*)malloc(amount_people*sizeof(int));
  int* zero=(int*)calloc(MAX_AGE,sizeof(int));
  

  generate_list(people, amount_people);
  
  write_ages(people,tmp_ages, amount_people);
  
 // printPeople(people,amount_people);
  
  /*------------------Countsort-Sequentail-------------*/
  
 // printf("\n----------------------------\n");
  
  start_time = time_ms();
  
  calcHistogram(people, count_sort_hist,amount_people);
  
  //printHist(count_sort_hist);
  memcpy(tmp_hist_seq,count_sort_hist,sizeof(int)*MAX_AGE);
  
  prefixSum(tmp_hist, count_sort_hist);
  
  countSortLastStage(people, people_sorted, count_sort_hist, amount_people);
  
  end_time=time_ms();
  
  //printPeople(people_sorted,amount_people); 
  
 /*-----------------Countsort-Parallel-----------------*/
 
  size_t localWorkGroupSize[1]={LOCALSIZE};
  size_t globalSize[1]={roundUp(LOCALSIZE,amount_people)};
  int workgroups=globalSize[0]/localWorkGroupSize[0];
  
  printf("LOCALSIZE: %d\n",LOCALSIZE);
  printf("Workgroups: %d\tGlobalSize: %d\tLocalWorkGroupSIze: %d\n",workgroups,(int) globalSize[0],(int)localWorkGroupSize[0]);
  
  //initialize device
  size_t max_WorkGroup;
  cl_context context;
  cl_command_queue command_queue;
  cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t), &max_WorkGroup,NULL );
  
  cl_int err;
 
  //memory buffer
  cl_mem mem_idata = clCreateBuffer(context, CL_MEM_READ_ONLY, amount_people*sizeof(int), NULL, &err);
  cl_mem mem_odata = clCreateBuffer(context, CL_MEM_READ_WRITE, workgroups*sizeof(int), NULL, &err);
  cl_mem mem_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MAX_AGE*sizeof(int), NULL, &err);
  CLU_ERRCHECK(err,"Failed to create memory buffer");
  
  //write buffer
  err = clEnqueueWriteBuffer(command_queue, mem_idata, CL_TRUE, 0, amount_people*sizeof(int), tmp_ages, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(command_queue, mem_result, CL_TRUE, 0, MAX_AGE*sizeof(int), zero, 0, NULL, NULL);
  CLU_ERRCHECK(err, "Failed to write buffer");
  
  //create kernel from source
 // char tmp[1024];
  cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, "");
  cl_kernel kernel_occurence = clCreateKernel(program,"occurence", &err);
  cl_kernel kernel_prefix = clCreateKernel(program,"prefixSum", &err);
  cl_kernel kernel_lastStep = clCreateKernel(program,"lastStep", &err);
  CLU_ERRCHECK(err, "Failed to create kernel from program");
 
  //Countsort Step 1
  int flag_one=-1;
  int flag_two=-1;
  int counter=1;
  int boundary=amount_people;
  bool finished=false;
  
  for(int i=0; i<MAX_AGE; i++){
	
	while(!finished){
	  
	  if(counter==1) {flag_one=1;  counter++;}
	  else{ flag_one=0;}
	  if(workgroups==1) {flag_two=1;}
	  else flag_two=0;
	  
	  cluSetKernelArguments(kernel_occurence, 8, sizeof(cl_mem), (void*)&mem_idata, sizeof(cl_mem), (void*)&mem_idata, 
				sizeof(cl_mem),(void*)&mem_result,sizeof(int)*LOCALSIZE,NULL,sizeof(int), (void*)&i,
				sizeof(int),(void*)&flag_one, sizeof(int),(void*)&flag_two, sizeof(int),(void*)&boundary);
	  
	  CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_occurence, 1, NULL, globalSize, localWorkGroupSize, 0, NULL, NULL), "Failed to enqueue 2D kernel");	
	  boundary=workgroups;
	
	  if(workgroups==1) finished=true;
	  else{
	      globalSize[0]=roundUp(LOCALSIZE,workgroups);
	      workgroups=globalSize[0]/localWorkGroupSize[0];
	  }
	  
	}
	flag_one=-1;
	flag_two=-1;
	counter=1;
	boundary=amount_people;
	globalSize[0]=roundUp(LOCALSIZE,amount_people);
	workgroups=globalSize[0]/localWorkGroupSize[0];
	finished=false;
	err = clEnqueueWriteBuffer(command_queue, mem_idata, CL_TRUE, 0, amount_people*sizeof(int), tmp_ages, 0, NULL, NULL);
      
  }
 
  //read from buffer
  CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, sizeof(int) * MAX_AGE, tmp_hist_parallel, 0, NULL, NULL),"Occurence-Failed to read back buffer");
  //printHist(tmp_hist);
  
  //-------------Countsort Second Step
  
  boundary=MAX_AGE%2==0 ? MAX_AGE/2: (MAX_AGE+1)/2;
  localWorkGroupSize[0]=max_WorkGroup;
  globalSize[0]= max_WorkGroup;
  int n=localWorkGroupSize[0]*2;
  cluSetKernelArguments(kernel_prefix, 5, sizeof(cl_mem), (void*)&mem_result, sizeof(cl_mem), (void*)&mem_result, 
			sizeof(int)*localWorkGroupSize[0]*2,NULL, sizeof(int),(void*)&n, sizeof(int),(void*)&boundary);
  
  CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel_prefix, 1, NULL, globalSize, localWorkGroupSize, 0, NULL, NULL), "Failed to enqueue 2D kernel");	
  CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, sizeof(int) * MAX_AGE, tmp_prefix_parallel, 0, NULL, NULL),"Occurence-Failed to read back buffer");
  
  printHist(count_sort_hist);
    printf("\n");
  printHist(tmp_prefix_parallel);

  
  
  verify(tmp_hist_seq,tmp_hist_parallel, "Histogram");
  verify(count_sort_hist,tmp_hist_parallel,"Prefix");
  printf("Sequential Time: %9lu ms\n", end_time - start_time);
  
  
  printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));
  
  //finalization
  err =  clFinish(command_queue);
  err |= clReleaseKernel(kernel_occurence);
  err |= clReleaseKernel(kernel_prefix);
  err |= clReleaseKernel(kernel_lastStep);
  err |= clReleaseProgram(program);
  err |= clReleaseMemObject(mem_idata);
  err |= clReleaseMemObject(mem_odata);
  err |= clReleaseMemObject(mem_result);
  err |= clReleaseCommandQueue(command_queue);
  err |= clReleaseContext(context);
  CLU_ERRCHECK(err, "Failed during ocl cleanup");
  
  free(people);
  free(people_sorted);
  free(count_sort_hist);
  free(tmp_hist);
  free(tmp_ages);
  free(zero);
  free(tmp_hist_parallel);
  free(tmp_hist_seq);
  free(tmp_prefix_parallel);
  return EXIT_SUCCESS; 
}

void verify(int* seq, int* par,char* info){
  bool correct=true;
  for(int i=0; i<MAX_AGE; i++){
    if((*(seq+i))!=(*(par+i))) correct=false;;
  }
  if(correct) printf("--------%s-Correct------------\n",info);
  else printf("--------%s-False------------\n",info);
}

int checkPowerOfTwo(int num){
 if((num != 1) && (num & (num - 1))) return 0;
 else return 1;
}

int roundUp(int from, int to){
  
  return to%from!=0 ? to+(from-(to%from)): to;
}

void write_ages(person_t* people, int* ages, int entries){
  for(int i=0; i<entries; i++){
      *(ages+i)=(people+i)->age;
    //  printf("AGES: %d\n",*(ages+i));
  }
}

void countSortLastStage(person_t* people,person_t* people_sorted, int* hist, int entries){
  
  //last stage
  for(int i=0; i<entries; i++){
      int index_input=(people+i)->age;
      int index_hist=*(hist+index_input);
      *(people_sorted+index_hist)=*(people+i);
      (*(hist+index_input))++;
  }
}

void generate_list(person_t* people, int entries){
  //generate people randomly
  int random=0;
  for(int i=0; i<entries;i++){
    random=rand()%MAX_AGE;
    gen_name((people+i)->name);
    (people+i)->age=random;
  }
}

void prefixSum(int* tmp, int* hist){
    //prefix sum
  tmp[0]=0;
  for(int i=1; i<MAX_AGE; i++){
      tmp[i]=hist[i-1]+tmp[i-1];
  }
  memcpy(hist, tmp, MAX_AGE * sizeof(int));
}

void calcHistogram(person_t* people, int* hist, int entries){
   //calc histogram
  int count=0;
  for(int i=0; i<MAX_AGE;i++){
      for(int j=0; j<entries; j++){
	  if((people+j)->age==i){
	      count++;
	      
	  }
      }
      (*(hist+i))=count;
      count=0;
  }
  
}

void printPeople(person_t* people, int entries){
  for(int i=0; i<entries; i++){
      printf("%d\t|\t%s\n",(people+i)->age, (people+i)->name);
  }
}

void printHist(int* hist){
 for(int i=0; i<MAX_AGE; i++){
      printf("Index[%d]:\t%d\t",i, *(hist+i));
      if((i+1)%5==0) printf("\n");
  }
}
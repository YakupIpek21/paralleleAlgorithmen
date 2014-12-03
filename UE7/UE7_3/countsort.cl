

__kernel void occurence(__global int* input, __global int* output, __global int* result, __local int* tmp, int element,
			int flag_one, int flag_two, int boundary){

  int globalID= get_global_id(0);
  int localID= get_local_id(0);
 // printf("Boundary: %d\n", boundary);
  if(globalID<boundary){
    
    if(flag_one){
      tmp[localID]= (input[globalID]==element) ? 1:0;	//when found write 1 else 0
     // printf("Element: %d, tmp[%d]: %d, input[%d]: %d\n",element,localID, tmp[localID],globalID, input[globalID]);
     // if(element==2){printf("GLOBALID: %d, ELEMENT: %d\n",globalID,element);}
      //if(input[globalID]==element){printf("GroupID: %d\tElement: %d\n",get_group_id(0),element);}
    }else{	//if reduction has to be done more than once
      tmp[localID]=output[globalID];
      //if(element==1){printf("GLOBALID: %d, ELEMENT: %d\n",globalID,element);}
    }
  }else{
    tmp[localID]=0;	//identity element for sum
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  //reduction 

  for(int offset=get_local_size(0)/2; offset>0; offset/=2){
	if(localID<offset){
	    int other= tmp[localID+offset];
	    tmp[localID]+=other;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }

  //Save results in global buffers with respect to group_id
  if(localID==0){
      if(flag_two){
	    result[element]=tmp[localID];
      }else{
	    output[get_group_id(0)]=tmp[localID];
	  //  printf("Element: %d, Output: %d\n",element,output[get_group_id(0)]);
      }
  }	

}

__kernel void prefixSum(__global int* input, __global int* output, __local int* tmp ,int n, int boundary){

  int thid = get_global_id(0);
  int offset = 1;

  if(thid<boundary){
    tmp[2*thid] = input[2*thid];
    tmp[2*thid+1] = input[2*thid+1];
  }else{
    tmp[2*thid] = 0;
    tmp[2*thid+1] = 0;
  }

  for(int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(thid < d) {
      int ai = offset*(2*thid+1) - 1;
      int bi = offset*(2*thid+2) - 1;
      tmp[bi] += tmp[ai];
    }
    offset *= 2;
  }
  if(thid == 0){ 
    tmp[n-1] = 0; // clear the last element
  } 


  for(int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(thid< d) {
      int ai = offset*(2*thid+1) - 1;
      int bi = offset*(2*thid+2) - 1;
      int t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if(thid<boundary){
      output[2*thid] = tmp[2*thid]; // write results to device memory
      output[2*thid+1] = tmp[2*thid+1];
  }
}

__kernel void lastStep(){}
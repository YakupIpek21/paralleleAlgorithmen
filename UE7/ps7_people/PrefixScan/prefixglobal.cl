

__kernel void prefix_scan_downSweep(__global VALUE* data, __global VALUE* result, __global VALUE* sumBuffer, __local VALUE* tmp, int border,int flag){

  int thid= get_global_id(0);
  int localIndex= get_local_id(0);
  int workGroupSize=get_local_size(0);
  int offset=1;
  if(thid<border){
      tmp[2*localIndex]=data[2*thid];
      tmp[2*localIndex+1]=data[2*thid+1];
  }else{
      tmp[2*localIndex]=0;
      tmp[2*localIndex+1]=0;
  }

  int n=get_local_size(0)*2;
  for(int d=n>>1; d>0; d>>=1){

      barrier(CLK_LOCAL_MEM_FENCE);
      
      if(localIndex<d){
	  int ai=offset*(2*localIndex+1)-1;
	  int bi=offset*(2*localIndex+2)-1;
	  tmp[bi]+=tmp[ai];
      }
     offset*=2;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if(flag){ //safe sums if necessary
      if(localIndex==0){ //last workitems within workgroup
	  sumBuffer[get_group_id(0)]=tmp[(workGroupSize*2-1)];
      }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if(localIndex==0) {tmp[workGroupSize*2-1]=0;}

  for(int d=1; d<n; d*=2){
      offset>>=1;
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localIndex<d){

	int ai=offset*(2*localIndex+1)-1;
	int bi=offset*(2*localIndex+2)-1;
	VALUE t= tmp[ai];
	tmp[ai]=tmp[bi];
	tmp[bi]+=t;
      }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  //save result
  result[2*thid]=tmp[2*localIndex];
  result[2*thid+1]=tmp[2*localIndex+1];
}

__kernel void prefix_scan_hillissteele(__global VALUE* data, __global VALUE* result, __global VALUE* sumBuffer, __local VALUE* tmp, int border,int flag){
  
  int thid=get_global_id(0);
  int localIndex=get_local_id(0);
  int pout=0, pin=1;
  int n=get_local_size(0);

  if(thid<border){
    tmp[localIndex]=(localIndex>0) ? data[thid-1] : 0;
  }else{
    tmp[localIndex]=(VALUE)0;
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset=1; offset<n; offset <<=1){
      pout= 1-pout;
      pin = 1-pout;
      if(localIndex>=offset){
	  tmp[pout*n+localIndex]=tmp[pin*n+localIndex]+tmp[pin*n+localIndex-offset];
      }else{
	  tmp[pout*n+localIndex]=tmp[pin*n+localIndex];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(flag){

    if(localIndex==n-1){ // since exclusive scan, last element must be added to total sum
	  sumBuffer[get_group_id(0)]=tmp[pout*n+localIndex]+data[thid];
	//  printf("GLOBALID: %d, Data_last_value: %0.0f\n",thid, data[thid]);
	//  printf("GLOBALID: %d, LOCALID: %d,SUM VALUE: %0.0f\n",thid,localIndex,tmp[localIndex]+data[thid]);
    } 
    
  }
  result[thid]=tmp[pout*n+localIndex];
}


__kernel void prefix_scan_last_stage(__global VALUE* data, __global VALUE* sums, int border, int flag){
  
  int groudId= get_group_id(0);
  int thid= get_global_id(0);
  if(flag){	//downsweep
      if(groudId<border-1){
	data[2*thid+get_local_size(0)*2]+=sums[groudId+1];
	data[2*thid+1+get_local_size(0)*2]+=sums[groudId+1];
      }
  }else{	//hillissteele
	if(groudId<border-1){
	  data[thid+get_local_size(0)]+=sums[groudId+1];
	}
  }
}

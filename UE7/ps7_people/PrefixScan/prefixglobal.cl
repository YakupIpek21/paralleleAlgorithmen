

__kernel void prefix_scan(__global VALUE* data, __global VALUE* result, __global VALUE* temporaryBuffer, int border,
			  const int localSizeTemp, __local VALUE* tmp2){

  __local VALUE tmp[LOCALSIZE];

  int thid= get_global_id(0);
  int localIndex= get_local_id(0);
  int offset=1;
  
  if(thid<border){
      tmp[2*thid]=data[2*thid];
      tmp[2*thid+1]=data[2*thid+1];
  }else{
      tmp[2*thid]=0;
      tmp[2*thid+1]=0;
  }

  int n=get_global_size(0);
  
  for(int d=n>>1; d>0; d>>=1){

      barrier(CLK_LOCAL_MEM_FENCE);
      
      if(thid<d){
	  int ai=offset*(2*thid+1)-1;
	  int bi=offset*(2*thid+2)-1;
	  tmp[bi]+=tmp[ai];
      }
     offset*=2;
  }

  if(thid==0) {tmp[n-1]=0;}


  for(int d=1; d<n; d*=2){
      offset>>=1;
      barrier(CLK_LOCAL_MEM_FENCE);
      if(thid<d){

	int ai=offset*(2*thid+1)-1;
	int bi=offset*(2*thid+2)-1;
	VALUE t= tmp[ai];
	tmp[ai]=tmp[bi];
	tmp[bi]+=t;
      }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  //next step, write last value within workgroup into temporary buffer
  offset=1;
  
  if(localIndex==(get_local_size(0)-1)){ //last workitems within workgroup
      temporaryBuffer[get_group_id(0)]=tmp[thid];
  }

  //do prefix scan on temporary buffer

  if(thid<localSizeTemp){
      tmp2[2*thid]=temporaryBuffer[2*thid];
      tmp2[2*thid+1]=temporaryBuffer[2*thid+1];
  }else{
      tmp2[2*thid]=0;
      tmp2[2*thid+1]=0;
  }

  n=localSizeTemp;
  
  for(int d=n>>1; d>0; d>>=1){

      barrier(CLK_LOCAL_MEM_FENCE);
      
      if(thid<d){
	  int ai=offset*(2*thid+1)-1;
	  int bi=offset*(2*thid+2)-1;
	  tmp2[bi]+=tmp2[ai];
      }
     offset*=2;
  }

  if(thid==0) {tmp2[n-1]=0;}


  for(int d=1; d<n; d*=2){
      offset>>=1;
      barrier(CLK_LOCAL_MEM_FENCE);
      if(thid<d){

	int ai=offset*(2*thid+1)-1;
	int bi=offset*(2*thid+2)-1;
	VALUE t= tmp2[ai];
	tmp2[ai]=tmp2[bi];
	tmp2[bi]+=t;
      }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  //last step add scanned block sum i to all values of scanned block i+1
  if(get_group_id(0)>0){
    tmp[2*thid]+=tmp2[get_group_id(0)-1];  
    tmp[2*thid+1]+=tmp2[get_group_id(0)-1];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  //save result
  result[2*thid]=tmp[2*thid];
  result[2*thid+1]=tmp[2*thid+1];

}
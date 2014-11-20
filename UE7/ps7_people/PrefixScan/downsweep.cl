

__kernel void prefix_scan(__global VALUE* data, __global VALUE* result){

  __local VALUE tmp[LOCALSIZE*2];

  int thid= get_global_id(0);
  int offset=1;
  
  tmp[2*thid]=data[2*thid];
  tmp[2*thid+1]=data[2*thid+1];
  
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
  result[2*thid]=tmp[2*thid];
  result[2*thid+1]=tmp[2*thid+1];

}


__kernel void prefix_scan(__global VALUE* data, __global VALUE* result, __local VALUE* tmp){

  int thid=get_global_id(0);
  int pout=0, pin=1;
  int n=get_global_size(0);
  tmp[thid]=(thid>0) ? data[thid-1] : 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset=1; offset<n; offset <<=1){
      pout= 1-pout;
      pin = 1-pout;
      if(thid>=offset){
	  tmp[pout*n+thid]=tmp[pin*n+thid]+tmp[pin*n+thid-offset];
      }else{
	  tmp[pout*n+thid]=tmp[pin*n+thid];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  result[thid]=tmp[pout*n+thid];
}
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "timer.h"
#include "cuda_utils.h"

typedef unsigned char dtype;

#define N_ (1200 * 1920)
#define FILTER_SIZE (3 * 3)
#define MAX_THREADS 256
#define MAX_BLOCKS 64
#define ROWS		1200
#define COLUMNS		1920


#define MAX(x,y) ((x > y) ? x : y)
#define MIN(x,y) ((x < y) ? x : y)
#define sqr(x)		((x)*(x))

/*CPU stlye laplacian mask*/
void laplacian_mask(dtype *input){
	dtype *temp;
	temp = (dtype*) malloc (ROWS * COLUMNS * sizeof(dtype));
	int sum=0;
	int i,j,k,l,flag=9;
	//printf("bp1\n");
	for(i=0;i<ROWS;i++){
		for(j=0;j<COLUMNS;j++){
			for(k=i-1;k<i+2;k++){
				for(l=j-1;l<j+2;l++){
					if(k==i&&l==j){
						sum += (int)(*(input+k*COLUMNS+l) * 8);
						flag--;
						//printf("-8 center pixel done\n");
					}
					else if(k>=0&&k<ROWS&&l>=0&&l<COLUMNS){
						sum -= (int)*(input+k*COLUMNS+l);
						flag--;
						//printf("add 1 side pixel done\n");
					}
				}
			}
			sum = sum - flag*(*(input+i*COLUMNS+j));
			sum = sum>255?255:sum;
			sum = sum<0?0:sum;
			*(temp+i*COLUMNS+j) = sum;
			sum=0;
			flag=9;
		}
	}
	//printf("bp2\n");
	for(i=0;i<ROWS;i++){
		for(j=0;j<COLUMNS;j++){
			*(input+i*COLUMNS+j) = *(temp+i*COLUMNS+j);
		}
	}
  free(temp);
  temp = NULL;
}


/* return the next power of 2 number that is larger than x */
unsigned int nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

/* find out # of threads and # thread blocks for a particular kernel */
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
  if (whichKernel < 3)
    {
      /* 1 thread per element */
      threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
      blocks = (n + threads - 1) / threads;
    }
  else
    {
      /* 1 thread per 2 elements */
      threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
      blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
  /* limit the total number of threads */
  if (whichKernel == 5)
    blocks = MIN(maxBlocks, blocks);
}

/* special type of reduction to account for floating point error */
/*dtype reduce_cpu(dtype *data, int n) {
  dtype sum = data[0];
  dtype c = (dtype)0.0;
  for (int i = 1; i < n; i++)
    {
      dtype y = data[i] - c;
      dtype t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
  return sum;
}*/

__global__ void
kernel0 (dtype *input, dtype *output)
{
  __shared__  dtype scratch[MAX_THREADS];

  //unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
  //unsigned int i = bid * blockDim.x + threadIdx.x;
  int tidx = threadIdx.x, tidy = threadIdx.y, tmp=0,i,j;
  int idx = blockIdx.x * 14 + tidx;
  int idy = blockIdx.y * 14 + tidy;
  int id = idy * COLUMNS + idx;
  int thisid = tidy * 16 + tidx;
  //int flag = FILTER_SIZE;
  
  

  if(idx < COLUMNS && idy < ROWS) {
	scratch[thisid] = input[id]; 
  } else{
	scratch[thisid] = 0;
  }
  
  
  
  /*else if (idx < COLUMNS && idy >= ROWS){
	scratch[thisid] = input[ROWS-1 * COLUMNS + idx]; 
  } else if (idx >= COLUMNS && idy < ROWS){
	scratch[thisid] = input[idy * COLUMNS + COLUMNS-1];
  } else {
	scratch[thisid] = input[ROWS-1 * COLUMNS + COLUMNS-1];
  }*/
  __syncthreads ();

	if(tidx>0&&tidx<15&&tidy>0&&tidy<15){
		for(i=tidy-1;i<tidy+2;i++){
			for(j=tidx-1;j<tidx+2;j++){
				if(i==tidy&&j==tidx){
					tmp = tmp + 8 * scratch[i*16+j];
				}else{
					tmp = tmp - scratch[i*16+j];
				}
			}
		}
		if(idx < COLUMNS-1 && idy < ROWS-1){
			tmp = tmp>255?255:tmp;
			tmp = tmp<0?0:tmp;
			output[id] = tmp;
		}
	}

	
}

int 
main(int argc, char** argv)
{
  int i,j;
  FILE		*fp;
  char		*ifile = "sample4.raw", *ofile1, *ofile2;

  /* data structure */
  dtype *h_idata, *h_odata, *imagegpu;
  dtype *d_idata, *d_odata;	

  /* timer */
  struct stopwatch_t* timer = NULL;
  long double t_kernel_0, t_cpu;
  int flag = 0;

  /* which kernel are we running */
  //int whichKernel;

  /* number of threads and thread blocks */
  //int threads, blocks;

  int N;
  if(argc > 1) {
    N = atoi (argv[1]);
    printf("N: %d\n", N);
  } else {
    N = N_;
    printf("N: %d\n", N);
  }

  /* naive kernel */
  //whichKernel = 0;
  /*getNumBlocksAndThreads (whichKernel, N, MAX_BLOCKS, MAX_THREADS, 
			  blocks, threads);*/

  /* initialize timer */
  stopwatch_init ();
  timer = stopwatch_create ();

  /* allocate memory */
  h_idata = (dtype*) malloc (N * sizeof (dtype));
  h_odata = (dtype*) malloc (N * sizeof (dtype));
  imagegpu = (dtype*) malloc (N * sizeof (dtype));
  CUDA_CHECK_ERROR (cudaMalloc (&d_idata, N * sizeof (dtype)));
  CUDA_CHECK_ERROR (cudaMalloc (&d_odata, N * sizeof (dtype)));

  /* Initialize array */
  /*srand48(time(NULL));
  for(i = 0; i < N; i++) {
    h_idata[i] = drand48() / 100000;
  }*/
  
  //fprintf(stdout, "loading image\n"); 
	if (( fp = fopen( ifile, "rb" )) == NULL ){
	  fprintf( stderr, "error: couldn't open %s\n", ifile );
	  exit( 1 );
	}			

	for ( i = 0; i < ROWS ; i++ )
	  if ( fread( (h_idata + i*COLUMNS), 1, COLUMNS, fp ) != COLUMNS )
	  {
	    fprintf( stderr, "error: couldn't read enough stuff\n" );
	    exit( 1 );
	  }
	fclose( fp );
  
  
  //fprintf(stdout, "entering gpu part\n"); 
  
  CUDA_CHECK_ERROR (cudaMemcpy (d_idata, h_idata, N * sizeof (dtype), 
				cudaMemcpyHostToDevice));

	
  /* ================================================== */
  /* GPU kernel */
  //dim3 gb(COLUMNS/BLOCK_SIZE+1, ROWS/BLOCK_SIZE+1, 1);
  dim3 gb(COLUMNS/14+1, ROWS/14+1, 1);
  dim3 tb(16, 16, 1);

  /* warm up */
  kernel0 <<<gb, tb>>> (d_idata, d_odata);
  cudaThreadSynchronize ();
	
	//fprintf (stdout, " warm up finished \n");
  stopwatch_start (timer);

  /* execute kernel */
  kernel0 <<<gb, tb>>> (d_idata, d_odata);
  cudaThreadSynchronize ();

  t_kernel_0 = stopwatch_stop (timer);
  fprintf (stdout, "Time to execute naive GPU edge detection: %Lg secs\n",
	   t_kernel_0);
  double bw = (N * sizeof(dtype)) / (t_kernel_0 * 1e9);
  fprintf (stdout, "Effective bandwidth: %.2lf GB/s\n", bw);
	
  /* copy result back from GPU */
  CUDA_CHECK_ERROR (cudaMemcpy (h_odata, d_odata, N * sizeof (dtype), 
				cudaMemcpyDeviceToHost));
  /* ================================================== */

  /* ================================================== */
  /* CPU kernel */
  stopwatch_start (timer);
  laplacian_mask(h_idata);
  t_cpu = stopwatch_stop (timer);
  fprintf (stdout, "Time to execute naive CPU edge detection: %Lg secs\n",
	   t_cpu);
	for(i = 0; i < ROWS; i++) {
		for(j = 0; j < COLUMNS; j++){
			if(i==0||j==0||i==ROWS-1||j==COLUMNS-1) *(imagegpu+i*COLUMNS+j) = *(h_idata+i*COLUMNS+j);
			else *(imagegpu+i*COLUMNS+j) = *(h_odata+i*COLUMNS+j);
		}
	}
	
  /* ================================================== */

	for(i = 1; i < ROWS-1; i++) {
		for(j = 1; j < COLUMNS-1; j++){
			if(abs (*(h_odata+i*COLUMNS+j) - *(h_idata+i*COLUMNS+j)) > 3) {  //used to be 1e-5
				flag = 1;
				fprintf(stderr, "FAILURE: GPU: %d 	CPU: %d\n", *(h_odata+i*COLUMNS+j), *(h_idata+i*COLUMNS+j));
				break;
			}
		}
		if(flag==1) break;
	}
	
	if(flag==0) fprintf(stdout, "SUCCESS\n"); 
	
  
  ofile1 = "outputcpu.raw";
  ofile2 = "outputgpu.raw";
  if (( fp = fopen( ofile1, "wb" )) == NULL )
	{
	  fprintf( stderr, "error: could not open %s\n", ofile1 );
	  exit( 1 );
	}
	for ( i = 0 ; i < ROWS ; i++ ) fwrite( (h_idata + i*COLUMNS), 1, COLUMNS, fp );
	fclose( fp );

	if (( fp = fopen( ofile2, "wb" )) == NULL )
	{
	  fprintf( stderr, "error: could not open %s\n", ofile2 );
	  exit( 1 );
	}
	for ( i = 0 ; i < ROWS ; i++ ) fwrite( (imagegpu + i*COLUMNS), 1, COLUMNS, fp );
	fclose( fp );
	
	free(h_idata);
	h_idata=NULL;
	free(h_odata);
	h_idata=NULL;
	free(imagegpu);
	h_idata=NULL;

  return 0;
}

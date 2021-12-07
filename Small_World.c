#Setting up the runtime with necessary libraries
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update
!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-9.2
#
#Installing an extension to interface a python runtime (Colab) with the nvcc compiler
!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git
#Load the extension
%load_ext nvcc_plugin
%%time
	%%cu
	#include <stdio.h>
	#include <stdlib.h>
  #include <stdbool.h>
	#include <math.h>
	#include <curand_kernel.h>
  #include <errno.h>
  #include <time.h>
	#define N 300
	#define M 1
  #define k 150
  #define beta 0.1
	#define S 100000


	
	
	__device__ int rast[N][10000]={0};
	__device__ int index1[N]={0};
	
	FILE* infile;
	FILE* outfile;
	FILE* outfile2;
	
	typedef struct queue{
	    //structure to hold last M entries
	    float arr[M];
	    int front;
	} queue;
	
	__device__
	queue initialize(){
	    queue q;
	    for(int i=0;i<M;i++)
	        q.arr[i] = -1;
	    q.front =0;
	    return q;
	}
	
	__device__
	void enqueue(queue* q,float a){
	    q->arr[q->front]=a;
	    q->front=(q->front+1)%M;
	}
	
	
	__device__
	int gcd(int a, int b)
	{
	    if(a==0)
	        return b;
	    return gcd(b%a,a);
	}
	
	__device__
	void rotate(queue* q)
	{
	    //Order the spike times in ascending order
	    int shift = 1;
	    int i=0;
	    while((i+1) < M && q->arr[i+1] > q->arr[i])
	    {
	        i++;
	        shift++;
	    }
	    if(shift==M)
	      return;
	    int blocks = gcd(shift,M);
	    for(int b=0;b<blocks;b++)
	    {
	        float temp = q->arr[b];
	        float next = temp;
	        int prev = b-shift + M;
	        while(prev!=b)
	        {
	            temp = q->arr[prev];
	            q->arr[prev] = next;
	            next = temp;
	            prev = prev-shift;
	            if(prev<0)
	                prev += M;
	        }
	        q->arr[prev] = next;
	    }
	}
	
	
	__global__ void run_simulations(int graphs, int inits, int* d_C_arr, float* d_times, float* d_times1, float* d_indx, float* d_sz,float a, float Vth, float g, int spikes, int* d_sync)
	{
	    //printf("%d",index1[3]);
		  int idx1=0;
		  int idx2=0;
		  int idx3=0;
			int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	    if(idx >= graphs*inits)
	        return;
	    curandState state;
	    curand_init(idx,0,0,&state);   //initialize every thread with a random seed
	    float arr[N];
	    for(int i=0;i<N;i++){
	        arr[i] = curand_uniform(&state)*Vth;

					//if(idx==7)
					//printf("%f,",arr[i]);
					
	    }
	    queue times[N];
		  
	    for(int i=0;i<N;i++)
	        times[i] = initialize();
	    int firing[N];
	    int size;
	    int fired[N];
	    int infiring[N];
	    float time = 0;
	    //printf("%f\n",time);
	    for(int s=0;s<spikes;s++){
	        size = 0;
	        float xm = -1;
	        for(int i=0;i<N;i++){
	            fired[i] = 0;
	            infiring[i] = 0;
	        }
	        for(int i=0;i<N;i++){
	            if(arr[i] > xm){
	                size = 1;
	                firing[0] = i;
	                xm = arr[i];
	            }
	            else if(arr[i] == xm){
	                firing[size] = i;
	                size++;
	            }
						//printf("%d,\n",s);
	        }
					
	        float tau = log((a - xm)/(a - Vth));
					//printf("%f,\n",time);
	        time += tau;
					//printf("%f,%f,\n",time,tau);
					//printf("%f,%d,\n",time,size);
					//printf("%d,\n",size);
					
	        for(int i=0;i<N;i++){
	            arr[i] = arr[i]*exp(-tau) + a*(1-exp(-tau));
							//if(idx==7)
			         //printf("new elemtn=%f",arr[i]);
	        }
	        for(int i=0;i<size;i++){
	            infiring[firing[i]] = 1;
	        }
	        for(int p=0;p<size;p++){
	            int ptr = firing[p];
	            fired[ptr] = 1;
	            arr[ptr] = 0;
	            for(int i=0;i<N;i++){
	                if(i==ptr || fired[i] || infiring[i])
	                    continue;
	                arr[i] += g*d_C_arr[(idx/inits)*N*N+i*N+ptr];
	                if(arr[i] >= Vth){
	                    if(infiring[i])
	                        continue;
	                    arr[i] = 0;
	                    fired[i] = 1;                    
	                }
	            }
	        }
					
	        for(int i=0;i<N;i++)
	        {
							//if(idx==7 && fired[i]&& i==1)
	            // printf("%f,\n",time);
							//if(idx==7 && fired[i]&& i==2)
	             //printf("%f,\n",time);
							//if(idx==7 && fired[i]&& i==7)
	             //printf("%f,\n",time);
							if(fired[i]) {             
	                enqueue(&times[i],time);
									 rast[i][index1[i]]=time;
									 //printf("%f\n",rast[i][index1[i]]);
									 index1[i]++;
									 //printf("%d\n",index1[i]);
									//printf("%f,\n",times[i].front);
									//printf("%f,%d,\n",time,i);
									//printf("%d\n",idx1);
									if(idx1<S){
											d_times1[idx1]=time;
											idx1=idx1+1;
									}
									if(idx2<S){
											d_indx[idx2]=(float)i;
											idx2++;
											
									}
									if(idx3<S){
											d_sz[idx3]=(float)size;
										
											idx3++;
											
									}
									//printf("%f,%d,%d,\n",time,i,size);
									
									
									
									
									
							}
	        }
	        float all = 0;
	        for(int i=0; i<N; i++){
	            all += arr[i];
	        }
	
	        if(all == 0)    //synchronization check
	        {
	            d_sync[idx]=1;
	            break;
	        }
	    }
	    for(int i=0;i<N;i++)
	    {
	        //printf("%f,\n",times[i].arr[0]);
					rotate(&times[i]);
					float top=times[i].front;
					
	    }
	    int offset = M*N*idx;
	    for(int i=0;i<N;i++)
	    {
	        for(int j=0;j<M;j++)
	        {
	            d_times[offset+M*i+j] = times[i].arr[j];
	        }
	    }
	}

	
	void dfs(int** arr,int* visited,int len,int start){
	    //Depth first search to check if graphs are connected
	    visited[start] = 1;
	    for(int i=0;i<len;i++){
	        if(visited[i] || i==start)
	            continue;
	        if(arr[start][i])
	            dfs(arr,visited,len,i);
	    }
	}
	
	int is_connected(int** arr,int len){
	    //Set up depth first search and decide if a graph is connected
	    int* visited = (int*) calloc(len,sizeof(int));
	    dfs(arr,visited,len,0);
	    int sum = 0;
	    for(int i=0;i<len;i++)
	        sum += visited[i];
	    return (sum==len)?1:0;
	}
	
	void binary_dump(int* arr, int len, FILE* outfile)
	{
	    //Dump binary data to file
	    unsigned char* b = (unsigned char*) malloc((len+7)/8*sizeof(char));
	    int i;
	    fprintf(outfile,"%d\n",len);
	    for(i=0;(i+8)<=len;i+=8)
	    {
	        b[i/8] = (char)((arr[i]<<7)+(arr[i+1]<<6)+(arr[i+2]<<5)+(arr[i+3]<<4)+(arr[i+4]<<3)+(arr[i+5]<<2)+(arr[i+6]<<1)+(arr[i+7]<<0));
	    }
	    b[i/8] = 0;
	    for(int j=0;j<len/8;j++)
	    {
	        fprintf(outfile,"%c",b[j]);
	    }
	
	    for(;i<len;i++)
	    {
	        fprintf(outfile,"%d",arr[i]);
	    }
	    fprintf(outfile, "\n");

	    free(b);
	}
	
	void float_dump(float* arr, int len, FILE* outfile)
	{
	    //Dump floating point data to a file
	    //fprintf(outfile, "\n%d\n", len);
		  //printf("%s\n",outfile);
	    for (int i = 0; i < len; ++i)
	    {
	        fprintf(outfile, "%f, ", arr[i]);
					 //printf("%f,\n ", arr[i]);
	    }
	}
	
	void initialize_files(int graphs, int inits, int minremove, int maxremove, int steps, int spikes)
	{
	    //Initialize files with simulation metadata
	    FILE* adjacency = fopen("adjacency.txt","w");
	    FILE* sync = fopen("sync.txt","w");
	    FILE* times = fopen("times.txt","w");
		  FILE* times1 = fopen("times1.txt","w");
		  FILE* indx = fopen("indx.txt","w");
		  FILE* sz = fopen("size.txt","w");
	    FILE* filelist[6] = {adjacency,sync,times,times1,indx,sz};
	    for(int i=0;i<3;i++)
	    {
	        fprintf(filelist[i],"Type: %d\n",i);
	        fprintf(filelist[i],"N: %d\n",N);
	        fprintf(filelist[i],"M: %d\n",M);
	        fprintf(filelist[i],"g: %d\n",graphs);
	        fprintf(filelist[i],"i: %d\n",inits);
	        fprintf(filelist[i],"mns: %d\n",minremove);
	        fprintf(filelist[i],"mxs: %d\n",maxremove);
	        fprintf(filelist[i],"stp: %d\n",steps);
	        fprintf(filelist[i],"spikes: %d\n",spikes);
	    }
	    fclose(adjacency);
	    fclose(sync);
	    fclose(times);
		  fclose(times1);
		  fclose(indx);
		  fclose(sz);
	}
	
	void dump(int graphs, int inits, int* C_arr, int* synchronized, float* times, float* times1, float* indx,float* sz)
	{
	    //Dump all simulation data by calling corresponding functions
	    FILE* adjacency = fopen("adjacency.txt","a");
	    FILE* sync = fopen("sync.txt","a");
	    FILE* timesfile = fopen("times.txt","a");
		  FILE* timesfile1 = fopen("times1.txt","a");
		  FILE* indxfile = fopen("indx.txt","a");
		  FILE* sizefile= fopen("size.txt","a");
	    binary_dump(C_arr,N*N*graphs,adjacency);
	    binary_dump(synchronized,graphs*inits,sync);
	    float_dump(times, N*M*graphs*inits, timesfile);
		  float_dump(times1,S, timesfile1);
		  float_dump(indx, S, indxfile);
		  float_dump(sz, S, sizefile);
	    fclose(adjacency);
	    fclose(sync);
	    fclose(timesfile);
		  fclose(timesfile1);
		  fclose(indxfile);
		  fclose(sizefile);
	}

  void init_global(int** C){
      for(int i=0;i<N;i++){
          for(int j=0;j<N;j++)
              C[i][j] = 1;
      }
  }
  void init_global1(int** C){
      for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
          if(j==i){
              C[i][j]=0;
              continue;
          }
          int val=abs(i-j)%(N-1-(k/2));
          if(val>0 && val<=(k/2)){
              C[i][j]=1;
          }
          else{
              C[i][j]=0;
          }
      }   
    }
    time_t t;
    srand((unsigned) time(&t));
    for(int i=0;i<N;i++){
        for(int j=i+1;j<=i+k/2;j++){
            int v=j%N;
            int num=rand()%100;
            float p=(float)(num+1)/100;
						
            if(p<=beta){
                int flag=0;
                while(flag != 1){
                    int new1=rand()%N;
                    if(i==new1){
                        continue;
                    }
                    else if(C[i][new1]==1){
                        continue;
                    }
                    else{
                        C[i][v]=0;
                        C[v][i]=0;
                        C[i][new1]=1;
                        C[new1][i]=1;
                        flag=1;
                    }
                }
            }
        }
    }
  }

  int minDistance(int dist[],bool sptSet[]){
    int min=INT_MAX,min_index;
    for(int v=0;v<N;v++){
        if(sptSet[v]==false && dist[v] <=min){
            min=dist[v],min_index=v;
        }
    }
    return min_index;
  }
  int dijkstra(int** C,int src){
    int dist[N];
    bool sptSet[N];

    for(int i=0;i<N;i++){
        dist[i]=INT_MAX,sptSet[i] = false;
    }
    dist[src]=0;
    for(int count=0;count<N-1;count++){
        int u=minDistance(dist,sptSet);
        sptSet[u]=true;
        for(int v=0;v<N;v++){
            if( !sptSet[v] && C[u][v] && dist[u] != INT_MAX && dist[u]+C[u][v]<dist[v])
                dist[v]=dist[u]+C[u][v];
        }
    }
    int sum=0;
    for(int i=0;i<N;i++){
        if(dist[i]==INT_MAX){
            dist[i]=0;
        }
        sum=sum+dist[i];
    }
    return sum;
  }
	
	int main(void){
					int u=14;
					int minremove =0, maxremove =10000;
					int stepsize = 100;
					int graphs = 1;
					int inits = 1;
					int spikes = 100000;
			
					float g = 0.06/N;
					float a = 1;
					float Vth = 0.8;
			
					initialize_files(graphs, inits, minremove, maxremove, stepsize, spikes);
			
					cudaEvent_t start,stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
			
					srand(time(0));
					int** C = (int**) malloc(N*sizeof(int*));  //Array to store each array that's generated
					for(int i=0;i<N;i++)
							C[i] = (int*) malloc(N*sizeof(int));
					
					int* C_arr;       //Hold all graphs in flattened
					C_arr = (int*) malloc(N*N*graphs*sizeof(int));
					int* d_C_arr;     //Initialized to be used in kernel (GPU) code 
					cudaMalloc(&d_C_arr,N*N*graphs*sizeof(int));
					int* d_sync;      //For the kernel code to indicate whether synchronization took place 
					cudaMalloc(&d_sync,graphs*inits*sizeof(int));
					float* times;     //Holds all the spike times
					times = (float*) malloc(M*N*graphs*inits*sizeof(float));
					float* d_times;   //To be used in kernel code
					cudaMalloc(&d_times,M*N*graphs*inits*sizeof(float));
					cudaMemcpy(d_times,times,M*N*graphs*inits*sizeof(float),cudaMemcpyHostToDevice);
					float* times1;     //Holds all the spike times
					times1 = (float*) malloc(spikes*sizeof(float));
					float* d_times1;   //To be used in kernel code
					cudaMalloc(&d_times1,spikes*sizeof(float));
					cudaMemcpy(d_times1,times1,spikes*sizeof(float),cudaMemcpyHostToDevice);
					float* indx;     //Holds all the spike times
					indx = (float*) malloc(spikes*sizeof(float));
					float* d_indx;   //To be used in kernel code
					cudaMalloc(&d_indx,spikes*sizeof(float));
					cudaMemcpy(d_indx,indx,spikes*sizeof(float),cudaMemcpyHostToDevice);
					float* sz;     //Holds all the spike times
					sz = (float*) malloc(spikes*sizeof(float));
					float* d_sz;   //To be used in kernel code
					cudaMalloc(&d_sz,spikes*sizeof(float));
					cudaMemcpy(d_sz,indx,spikes*sizeof(float),cudaMemcpyHostToDevice);
					
					for(int synremove=minremove;synremove<=maxremove;synremove+=stepsize){
							
							
							cudaEventRecord(start); //Start timer
							int restarts = 0;  
							float av=0;     //Track number of restarts since last connected graph
							for(int gr=0;gr<graphs;gr++){
									init_global1(C);

									for(int syn=0;syn<synremove;){
											int n1 = rand()%N;
											int n2 = rand()%N;
											if(C[n1][n2] == 1 && n1 != n2){
													C[n1][n2] = 0;
													C[n2][n1] = 0;
													syn += 1;
													//printf("%d,\n",syn);
											}
									}
									if(!is_connected(C,N)){
											gr--;
											restarts++;
											if(restarts >= 10000){
													printf("No connected graphs for %d synapses being removed.\n",synremove);
													exit(0);
											}
											continue;
									}
									float av_path=0;
									int path;
									for(int i=0;i<N;i++){
										path=dijkstra(C,i);
										av_path=av_path+path;
									}
									av_path=av_path/(N*(N-1));
									av=av+av_path;
									restarts = 0;
									int offset = gr*N*N;

									for(int i=0;i<N;i++){
											for(int j=0;j<N;j++){
													C_arr[offset+N*i+j] = C[i][j];
											}
									}
							}
							av=av/graphs;
							cudaMemcpy(d_C_arr,C_arr,graphs*N*N*sizeof(int),cudaMemcpyHostToDevice);
							int* sync = (int*) malloc(graphs*inits*sizeof(int));
							for(int i=0;i<graphs*inits;i++)
									sync[i]=0;
							cudaMemcpy(d_sync,sync,graphs*inits*sizeof(int),cudaMemcpyHostToDevice);
			
							int grid_size, block_size; //block_size is the number of threads, grid_size is the number of blocks to deploy
							cudaOccupancyMaxPotentialBlockSize(&grid_size,&block_size,run_simulations);
							printf("Grid size: %d, block size: %d\n",grid_size,block_size);
							printf("Average Path Length: %f\n",av);
							printf("Serial portion of program completed. Starting the simulation.\n");
							run_simulations<<<(graphs*inits+block_size-1)/block_size,block_size>>>(graphs,inits,d_C_arr,d_times,d_times1,d_indx,d_sz,a,Vth,g,spikes,d_sync);
							cudaDeviceSynchronize();
							cudaEventRecord(stop);
			
							cudaMemcpy(sync,d_sync,graphs*inits*sizeof(int),cudaMemcpyDeviceToHost);
							cudaMemcpy(times,d_times,M*N*graphs*inits*sizeof(float),cudaMemcpyDeviceToHost);
							cudaMemcpy(times1,d_times1,spikes*sizeof(float),cudaMemcpyDeviceToHost);
							cudaMemcpy(indx,d_indx,spikes*sizeof(float),cudaMemcpyDeviceToHost);
							cudaMemcpy(sz,d_sz,spikes*sizeof(float),cudaMemcpyDeviceToHost);
							dump(graphs,inits,C_arr,sync,times,times1,indx,sz);
							for(int bin=0;bin<M*N*graphs*inits;bin++){
									//printf("%f,",times[bin]);
							}
							int tot = 0;
							for(int i=0;i<graphs*inits;i++)
									tot+=(sync[i]==1);
							printf("Synchronization probability is: %f for %d synapses removed\n",(float)tot/(graphs*inits),synremove);
							float milliseconds = 0.0f;
							cudaEventElapsedTime(&milliseconds,start,stop);
							printf("Time taken: %f milliseconds\n",milliseconds);
							free(sync);
							for(int i=0;i<N;i++){
							 for(int j=0;j<10000;j++){
									if(rast[i][j]==0){
											break;
									}
									else{
											printf("%f,",rast[i][j]);
									}
							}
					}
					}
					free(C_arr);
					free(times);
					free(times1);
					free(indx);
					free(sz);
					cudaFree(&d_sync);
					cudaFree(&d_C_arr);
					cudaFree(&d_times);
					cudaFree(&d_times1);
					cudaFree(&d_indx);
					cudaFree(&d_sz);
					
					
					
		
	    
	    return 0;
	}


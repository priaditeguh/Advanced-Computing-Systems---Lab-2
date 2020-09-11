#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <math.h>
// #include <tgmath.h> 

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

/* Utility function/macro, used to do error checking.
   Use this function/macro like this:
   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
   And to check the result of a kernel invocation:
   checkCudaCall(cudaGetLastError());
*/
#define checkCudaCall(result) {                                     \
    if (result != cudaSuccess){                                     \
        cerr << "cuda error: " << cudaGetErrorString(result);       \
        cerr << " in " << __FILE__ << " at line "<< __LINE__<<endl; \
        exit(1);                                                    \
    }                                                               \
}

__global__ void testKernel(unsigned char *grayImage_cuda,unsigned char *tempImage_cuda, 
	const int width, const int height) 
{

}
/*
	GPU function to change rgb picture to grayscale
*/
__global__ void rgb2grayCudaKernel(unsigned char *_input, unsigned char *_output,
	const int width, const int height) 
{
	// get location
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// check if location is still inside picture
	if (x < width && y < height) {
		float grayPix = 0.0f;
		int index = (y * width) + x;
		int size = width * height;
		float r = static_cast< float >(_input[index]);
		float g = static_cast< float >(_input[size + index]);
		float b = static_cast< float >(_input[(2 * size) + index]);

		// count appropriate gray pixel value
		grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);
		// set gray value to correct pixel
		_output[index] = static_cast< unsigned char >(grayPix);
	}

	__syncthreads();
}

/*
	CPU function to change rgb picture to grayscale
*/
void rgb2grayCuda (unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) 
{
	// NSTimer initTime = NSTimer("initTime", false, false);
	// initTime.start();
	// cudaSetDevice(1);
	// initTime.stop();
	// cout << "initialization : \t\t" << initTime.getElapsed() << " seconds." << endl;
	
	NSTimer rgbTime = NSTimer("rgbTime", false, false);
	rgbTime.start();
	
	// default block size
	int warpSize = 32;

	unsigned char *_input, *_output;

	// allocate memory to device
	checkCudaCall(cudaMalloc((void**) &_input, sizeof(unsigned char)*width*height*3)); 
	checkCudaCall(cudaMalloc((void**) &_output, sizeof(unsigned char)*width*height)); 
	// make sure no memory is left laying around
	checkCudaCall(cudaMemset(_output, 0, sizeof(unsigned char)*width*height));
	// copy data to memory in device
	checkCudaCall(cudaMemcpy(_input, inputImage, sizeof(unsigned char) * width * height*3, cudaMemcpyHostToDevice));
	
	// set block and grid size
	const dim3 dimBlock(warpSize, warpSize);
  	const dim3 dimGrid( ceil(width/(float)warpSize), ceil(height/(float)warpSize), 1);
  	
  	// start timer
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	kernelTime.start();
	// run kernel function
	rgb2grayCudaKernel<<<dimGrid, dimBlock>>>(_input, _output, width, height);
	// 	stop timer
	kernelTime.stop();
  	
	// synchronize
	checkCudaCall(cudaThreadSynchronize());
	
	// transfer from device to host
	checkCudaCall(cudaMemcpy(grayImage, _output, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost));
	
	// free device memory
	cudaFree(_input);
	cudaFree(_output);

	rgbTime.stop();
	cout << "rgb2grayCuda (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "rgb2grayCuda (total): \t\t" << rgbTime.getElapsed() << " seconds." << endl;
}

void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height ) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			float grayPix = 0.0f; 
			float r = static_cast< float >(inputImage[(y * width) + x]);
			float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

			grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

			grayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
		}
	}
	// /Kernel
	kernelTime.stop();
	
	cout << fixed << setprecision(6);
	cout << "rgb2gray (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

__global__ void histogram1DCudaKernel_first(unsigned char *grayImage_cuda,unsigned int *tempImage_cuda)
{
  // declare shared memory to store every 256 pixel
  __shared__ int temp[256];

  int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
  
  // Scan every 256 pixel and store it on shared memory 
  temp[threadIdx.x] = grayImage_cuda[pixel_index];
  
  // wait untill all thread finish working
  __syncthreads();
  
  // This loop count the 256 histogram value from the stored 256 pixel
  for(int i=0;i<256;i++)
  {
    if(temp[i] == threadIdx.x)
      tempImage_cuda[pixel_index]++;
  }
}

__global__ void histogram1DCudaKernel_second(unsigned int *tempImage_cuda, unsigned int *histogram_cuda, int n)
{ 
  // The second kernel counts the total histogram value from height*width/256 elements for each grayImage value
  for(int i=0;i<n;i++)
  {
    histogram_cuda[threadIdx.x] += tempImage_cuda[i*blockDim.x+ threadIdx.x];
  }
}

void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height, 
				 unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
				 const unsigned int BAR_WIDTH
         ) 
{
	NSTimer totalTime = NSTimer("contrastTime", false, false);
	totalTime.start();
	unsigned int max = 0;
  
	memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));
  
	unsigned char *grayImage_cuda;
	unsigned int *histogram_cuda;
	unsigned int *tempImage_cuda;

	int n = ceil(height*width/256);

	dim3 grid_1(n);
	dim3 block_1(256);

	dim3 grid_2(1);
	dim3 block_2(256);

	// Kernel
	// Alloc space for GPU device 
	checkCudaCall(cudaMalloc(&histogram_cuda, HISTOGRAM_SIZE*sizeof(unsigned int)));
	checkCudaCall(cudaMalloc(&grayImage_cuda, height*width*sizeof(unsigned char)));
	cudaMalloc(&tempImage_cuda, height*width*sizeof(unsigned int));

	// Initialize the variable with zero
	checkCudaCall(cudaMemset(histogram_cuda, 0, HISTOGRAM_SIZE*sizeof(unsigned int)));
	checkCudaCall(cudaMemset(tempImage_cuda, 0, height*width*sizeof(unsigned int)));

	// Copy grayImage to GPU device
	checkCudaCall(cudaMemcpy(grayImage_cuda, grayImage, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice));

	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	kernelTime.start();

	// Launch histogram1DCudaKernel kernel on GPU kernel
	histogram1DCudaKernel_first<<<grid_1,block_1>>>(grayImage_cuda,tempImage_cuda);
	histogram1DCudaKernel_second<<<grid_2,block_2>>>(tempImage_cuda,histogram_cuda,n);

	kernelTime.stop();

	// Copy result back to CPU host
	checkCudaCall(cudaMemcpy(histogram, histogram_cuda, HISTOGRAM_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// Clean memory at GPU devices
	checkCudaCall(cudaFree(histogram_cuda));
	checkCudaCall(cudaFree(grayImage_cuda));
  	
 	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) 
	{
		if ( histogram[i] > max ) 
		{
			max = histogram[i];
		}
	}

	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) 
	{
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( unsigned int y = value; y < HISTOGRAM_SIZE; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}
  
	totalTime.stop();

	cout << fixed << setprecision(6);
	cout << "histogram (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "histogram (total): \t\t" << totalTime.getElapsed() << " seconds." << endl;
}

/*
	Kernel function to improve contrast
*/
__global__ void contrast1DCudaKernel(unsigned char *_input, unsigned char *_output, const int width, const int height, 
				int min, int max) 
{
	// get correct pixel location
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float diff = max - min;

	// check if x and y is still inside picture
	if (x < width && y < height) {
		unsigned char pixel = _input[(y * width) + x];
		// improve contrast
		if ( pixel < min ) 
		{
			pixel = 0;
		}
		else if ( pixel > max ) 
		{
			pixel = 255;
		}
		else 
		{
			pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
		}
		// set pixel with new value		
		_output[(y * width) + x] = pixel;
	}
}

/*
	GPU function to improve contrast
*/
void contrast1DCuda(unsigned char *grayImage, const int width, const int height, 
				unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
				const unsigned int CONTRAST_THRESHOLD) 
{
	NSTimer contrastTime = NSTimer("contrastTime", false, false);
	contrastTime.start();
	
	// default block size
	int warpSize = 32;
	unsigned char *_input, *_output;

	unsigned int i = 0;
	
	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i--;
	}
	unsigned int max = i;
	
	// allocate memory for gpu variable
	checkCudaCall(cudaMalloc((void**) &_input, sizeof(unsigned char)*width*height)); 
	checkCudaCall(cudaMalloc((void**) &_output, sizeof(unsigned char)*width*height)); 
	// make sure no memory is left laying around
	// checkCudaCall(cudaMemset(_output, 0, sizeof(unsigned char)*width*height));
	// copy data to memory in device
	checkCudaCall(cudaMemcpy(_input, grayImage, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice));
	
	// set block dimension and grid dimension
	const dim3 dimBlock(warpSize, warpSize, 1);
  	const dim3 dimGrid( ceil(width/(float)warpSize), ceil(height/(float)warpSize), 1);

  	// start timer
	NSTimer kernelTime = NSTimer("RGB Gray Cuda kernelTime", false, false);
	kernelTime.start();
	// run the kernel function
	contrast1DCudaKernel<<<dimGrid, dimBlock>>>(_input, _output, width, height, min, max);
	// stop timer
	kernelTime.stop();
  	
	// // synchronize
	checkCudaCall(cudaThreadSynchronize());
	// transfer from device to host
	checkCudaCall(cudaMemcpy(grayImage, _output, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost));

	// free device memory
	cudaFree(_input);
	cudaFree(_output);

	contrastTime.stop();
	cout << fixed << setprecision(6);
	cout << "contrast1DCuda (kernel): \t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "contrast1DCuda (total): \t" << contrastTime.getElapsed() << " seconds." << endl;
}


void contrast1D(unsigned char *grayImage, const int width, const int height, 
				unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
				const unsigned int CONTRAST_THRESHOLD) 
{
	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for (int x = 0; x < width; x++ ) 
		{
			unsigned char pixel = grayImage[(y * width) + x];

			if ( pixel < min ) 
			{
				pixel = 0;
			}
			else if ( pixel > max ) 
			{
				pixel = 255;
			}
			else 
			{
				pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
			}
			
			grayImage[(y * width) + x] = pixel;
		}
	}
	// /Kernel
	kernelTime.stop();
	
	cout << fixed << setprecision(6);
	cout << "contrast1D (cpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
}

__global__ void triangularSmoothKernel(unsigned char *grayImage, unsigned char *smoothImage, float *filter) 
{
  // declare shared memory for 5*5 window and filter weight
  __shared__ float window[25];
  __shared__ float filterWeight[25];

  // calculate the index for the shared memory variable and 
  // index for window
  int cache_index = blockDim.x * threadIdx.y + threadIdx.x;
  int index_x = blockIdx.x + threadIdx.x - 2;
  int index_y = blockIdx.y + threadIdx.y - 2;
  
  if ( (index_x < 0) || (index_x > gridDim.x) || (index_y < 0) || (index_y > gridDim.y) )
  {
    // In this branch, the index is out of image range 
    filterWeight[cache_index] = 0.0f;
    window[cache_index] = 0.0f;
  } else {
    // each window element is multiplied with filter weight 
    int filter_index = gridDim.x * index_y + index_x;
    filterWeight[cache_index] = filter[cache_index]; 
    window[cache_index] = (static_cast< float > (grayImage[filter_index])) * filter[cache_index];
  } 
  
  // wait untill all thread finish working
  __syncthreads();
  
  if(cache_index%25 == 0)
  { 
    // declare image output index
    int block_index = gridDim.x * blockIdx.y + blockIdx.x;
    
    float smoothPix = 0.0f;
    float filterSum = 0.0f;
    
    // sum all the multiplication results
    for(int i=0;i<25;i++)
    {
      smoothPix += window[i];
      filterSum += filterWeight[i];
    }
    
    // store division results to output
    smoothImage[block_index] = static_cast< unsigned char >(smoothPix/filterSum);
  }
}

void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,
					  const float *filter) 
{
	NSTimer totalTime = NSTimer("rgbTime", false, false);
	totalTime.start();
	
	unsigned char *grayImage_cuda;
	unsigned char *smoothImage_cuda;
	float *filter_cuda;

	dim3 grid(width,height);
	dim3 block(5,5);

	// Kernel

	// Alloc space for GPU device 
	checkCudaCall(cudaMalloc(&grayImage_cuda, height*width*sizeof(unsigned char)));
	checkCudaCall(cudaMalloc(&smoothImage_cuda, height*width*sizeof(unsigned char)));
	checkCudaCall(cudaMalloc(&filter_cuda,25*sizeof(float)));

	checkCudaCall(cudaMemset(smoothImage_cuda, 0, height*width*sizeof(unsigned char)));

	// Copy grayImage to GPU device
	checkCudaCall(cudaMemcpy(grayImage_cuda, grayImage, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaCall(cudaMemcpy(filter_cuda, filter, 25*sizeof(float), cudaMemcpyHostToDevice));

	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	kernelTime.start();
	// Launch triangularSmoothKernel kernel on GPU kernel
	triangularSmoothKernel<<<grid,block>>>(grayImage_cuda,smoothImage_cuda,filter_cuda);
	kernelTime.stop();

	// Copy result back to CPU host
	checkCudaCall(cudaMemcpy(smoothImage, smoothImage_cuda, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Clean memory at GPU devices
	checkCudaCall(cudaFree(smoothImage_cuda));
	checkCudaCall(cudaFree(grayImage_cuda));

	totalTime.stop();

	cout << fixed << setprecision(6);
	cout << "triangularSmooth (kernel): \t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (total): \t" << totalTime.getElapsed() << " seconds." << endl;
}

void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,
					  const float *filter) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			unsigned int filterItem = 0;
			float filterSum = 0.0f;
			float smoothPix = 0.0f;

			for ( int fy = y - 2; fy < y + 3; fy++ ) 
			{
				for ( int fx = x - 2; fx < x + 3; fx++ ) 
				{
					if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ) 
					{
						filterItem++;
						continue;
					}

					smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
					filterSum += filter[filterItem];
					filterItem++;
				}
			}

			smoothPix /= filterSum;
			smoothImage[(y * width) + x] = static_cast< unsigned char >(smoothPix);
		}
	}
	// /Kernel
	kernelTime.stop();
	
	cout << fixed << setprecision(6);
	cout << "triangularSmooth (cpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
}



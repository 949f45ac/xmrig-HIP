#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>

#ifdef __NVCC__
#include <cuda_runtime.h>
#endif

#ifdef __HIPCC__
__constant__
#else
const
#endif
uint64_t keccakf_rndc[24] ={
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

#include <algorithm>
#include "cryptonight.h"
#include "cuda_extra.h"
#include "cuda_aes.hpp"
#include "cuda_keccak.hpp"
#include "cuda_blake.hpp"
#include "cuda_groestl.hpp"
#include "cuda_jh.hpp"
#include "cuda_skein.hpp"
#include "cuda_device.hpp"
#include "common/xmrig.h"

__constant__ uint8_t d_sub_byte[16][16] ={
	{0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76 },
	{0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0 },
	{0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15 },
	{0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75 },
	{0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84 },
	{0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf },
	{0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8 },
	{0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2 },
	{0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73 },
	{0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb },
	{0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79 },
	{0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08 },
	{0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a },
	{0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e },
	{0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf },
	{0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 }
};

__constant__ uint32_t aes_gf[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

__device__ __forceinline__ void cryptonight_aes_set_key( uint32_t * __restrict__ key, const uint32_t * __restrict__ data )
{
	int i, j;
	uint8_t temp[4];

	MEMSET4( key, 0, 40 );
	MEMCPY4( key, data, 8 );

#pragma unroll
	for ( i = 8; i < 40; i++ )
	{
		*(uint32_t *) temp = key[i - 1];
		if ( i % 8 == 0 )
		{
			*(uint32_t *) temp = ROTR32( *(uint32_t *) temp, 8 );
			for ( j = 0; j < 4; j++ )
				temp[j] = d_sub_byte[( temp[j] >> 4 ) & 0x0f][temp[j] & 0x0f];
			*(uint32_t *) temp ^= aes_gf[i / 8 - 1];
		}
		else
		{
			if ( i % 8 == 4 )
			{
#pragma unroll
				for ( j = 0; j < 4; j++ )
					temp[j] = d_sub_byte[( temp[j] >> 4 ) & 0x0f][temp[j] & 0x0f];
			}
		}

		key[i] = key[( i - 8 )] ^ *(uint32_t *) temp;
	}
}

__device__ __forceinline__ void mix_and_propagate( uint32_t* state )
{
    uint32_t tmp0[4];
    for(size_t x = 0; x < 4; ++x)
        tmp0[x] = (state)[x];

    // set destination [0,6]
    for(size_t t = 0; t < 7; ++t)
        for(size_t x = 0; x < 4; ++x)
            (state + 4 * t)[x] = (state + 4 * t)[x] ^ (state + 4 * (t + 1))[x];

    // set destination 7
    for(size_t x = 0; x < 4; ++x)
        (state + 4 * 7)[x] = (state + 4 * 7)[x] ^ tmp0[x];
}

template<bool CN_HEAVY>
__global__ void cryptonight_extra_gpu_prepare( int threads, uint32_t * __restrict__ d_input, uint32_t len, uint32_t startNonce,
											   uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_state_p1,
											   uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b,
											   uint32_t * __restrict__ d_ctx_key1, uint32_t * __restrict__ d_ctx_key2 )
{
	int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x );
	__shared__ uint32_t sharedMemory[1024];

	if (CN_HEAVY) {
        cn_aes_gpu_init( sharedMemory );
        __syncthreads( );
    }

	if ( thread >= threads )
		return;

	uint32_t ctx_state[50];
	uint32_t ctx_a[4];
	uint32_t ctx_b[4];
	uint32_t ctx_key1[40];
	uint32_t ctx_key2[40];
	uint32_t input[21];

	memcpy( input, d_input, len );
	//*((uint32_t *)(((char *)input) + 39)) = startNonce + thread;
	uint32_t nonce = startNonce + thread;
	for ( int i = 0; i < sizeof (uint32_t ); ++i )
		( ( (char *) input ) + 39 )[i] = ( (char*) ( &nonce ) )[i]; //take care of pointer alignment

	cn_keccak( (uint8_t *) input, len, (uint8_t *) ctx_state );
	cryptonight_aes_set_key( ctx_key1, ctx_state );
	cryptonight_aes_set_key( ctx_key2, ctx_state + 8 );
	XOR_BLOCKS_DST( ctx_state, ctx_state + 8, ctx_a );
	XOR_BLOCKS_DST( ctx_state + 4, ctx_state + 12, ctx_b );
        __threadfence_block();


	memcpy( d_ctx_state + thread * 50, ctx_state, 50 * 4 );
	memcpy( d_ctx_state_p1 + thread * 50, ctx_state, 50 * 4);
	memcpy( d_ctx_a + thread * 4, ctx_a, 4 * 4 );



	memcpy( d_ctx_b + thread * 12, ctx_b, 4 * 4 );

	if (!CN_HEAVY) {
	// bx1
		XOR_BLOCKS_DST( ctx_state + 16, ctx_state + 20, ctx_b );
		memcpy( d_ctx_b + thread * 12 + 4, ctx_b, 4 * 4 );
		// division_result
		memcpy( d_ctx_b + thread * 12 + 2 * 4, ctx_state + 24, 4 * 2 );
		// sqrt_result
		memcpy( d_ctx_b + thread * 12 + 2 * 4 + 2, ctx_state + 26, 4 * 2 );
	}
	//// endif


	memcpy( d_ctx_key1 + thread * 40, ctx_key1, 40 * 4 );
	memcpy( d_ctx_key2 + thread * 40, ctx_key2, 40 * 4 );

	if (CN_HEAVY) {
        for (int i = 0; i < 16; i++) {
            for (size_t t = 4; t < 12; ++t) {
                cn_aes_pseudo_round_mut(sharedMemory, ctx_state + 4u * t, ctx_key1);
            }
            // scipt first 4 * 128bit blocks = 4 * 4 uint32_t values
            mix_and_propagate(ctx_state + 4 * 4);
        }
        // double buffer to move manipulated state into phase1
        memcpy(d_ctx_state_p1 + thread * 50, ctx_state, 50 * 4);
	}
}

template<bool CN_HEAVY>
__global__ void cryptonight_extra_gpu_final( int threads, uint64_t target,
											 uint32_t* __restrict__ d_res_count, uint32_t * __restrict__ d_res_nonce,
											 uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2 )
{
	const uint32_t thread = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

	__shared__ uint32_t sharedMemory[1024];

    if (CN_HEAVY)
    {
        cn_aes_gpu_init( sharedMemory );
        __syncthreads( );
	}

	if ( thread >= threads )
		return;

	int i;
	uint32_t * __restrict__ ctx_state = d_ctx_state + thread * 50;
	uint64_t hash[4];
	uint64_t state[25];

	uint32_t* state32 = reinterpret_cast<uint32_t*>(state);

//#pragma unroll
	for ( i = 0; i < 50; i++ )
		state32[i] = ctx_state[i];

	if (CN_HEAVY) {
        uint32_t key[40];

        // load keys
        MEMCPY8( key, d_ctx_key2 + thread * 40, 20 );

        for(int i=0; i < 16; i++)
        {
            for(size_t t = 4; t < 12; ++t)
            {
                cn_aes_pseudo_round_mut( sharedMemory, state32 + 4u * t, key );
            }
            // scipt first 4 * 128bit blocks = 4 * 4 uint32_t values
            mix_and_propagate(state32 + 4 * 4);
        }
	}

	cn_keccakf2( state );

	__syncthreads();
	switch ( ( (uint8_t *) state )[0] & 0x03 )
	{
	case 0:
		cn_blake( (const uint8_t *) state, 200, (uint8_t *) hash );
		break;
	case 1:
		cn_groestl( (const BitSequence *) state, 200, (BitSequence *) hash );
		break;
	case 2:
		cn_jh( (const BitSequence *) state, 200, (BitSequence *) hash );
		break;
	case 3:
		cn_skein( (const BitSequence *) state, 200, (BitSequence *) hash );
		break;
	default:
		break;
	}

	// Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
	// and expect an accurate result for target > 32-bit without implementing carries
	__syncthreads();

	if ( hash[3] < target )
	{
		uint32_t idx = atomicInc( d_res_count, 0xFFFFFFFF );

		if(idx < 10)
			d_res_nonce[idx] = thread;
	}
	__syncthreads();
}

extern "C" void cryptonight_extra_cpu_set_data( nvid_ctx* ctx, const void *data, uint32_t len )
{
	ctx->inputlen = len;
	hipMemcpyAsync( ctx->d_input, data, len, hipMemcpyHostToDevice, *ctx->stream );
	exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );
}

extern "C" int cryptonight_extra_cpu_init(nvid_ctx* ctx, xmrig::Algo algo)
{
	hipError_t err;
	err = hipSetDevice(ctx->device_id);
	if(err != hipSuccess)
	{
		printf("GPU %d: %s", ctx->device_id, hipGetErrorString(err));
		return 0;
	}

	//hipDeviceReset();
#ifdef __HCC__
	hipSetDeviceFlags(hipDeviceScheduleSpin);
	hipDeviceSetCacheConfig(hipFuncCachePreferL1);
#else
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

	ctx->stream = new hipStream_t;
	hipStreamCreate(ctx->stream);

	size_t wsize = ctx->device_blocks * ctx->device_threads;
	hipMalloc(&ctx->d_ctx_state, 50 * sizeof(uint32_t) * wsize);
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);

	uint f;
	if (algo == xmrig::CRYPTONIGHT_HEAVY) {
		hipMalloc(&ctx->d_ctx_state_p1, 50 * sizeof(uint32_t) * wsize);
		exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);
		f = 2;
	} else {
		ctx->d_ctx_state_p1 = ctx->d_ctx_state;
		f = 1;
	}
	hipMalloc(&ctx->d_long_state, (size_t)MEMORY * wsize * f);
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);

	hipMalloc(&ctx->d_ctx_key1, 40 * sizeof(uint32_t) * wsize);
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);
	hipMalloc(&ctx->d_ctx_key2, 40 * sizeof(uint32_t) * wsize);
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);
	hipMalloc(&ctx->d_ctx_text, 32 * sizeof(uint32_t) * wsize);
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);
	hipMalloc(&ctx->d_ctx_a, 4 * sizeof(uint32_t) * wsize);
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);
	hipMalloc(&ctx->d_ctx_b, 3 * 4 * sizeof(uint32_t) * wsize);
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);
	hipMalloc(&ctx->d_input, 21 * sizeof (uint32_t ) );
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__);
	hipMalloc(&ctx->d_result_count, sizeof (uint32_t ) );
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );
	hipMalloc(&ctx->d_result_nonce, 10 * sizeof (uint32_t ) );
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );
	return 1;
}

void set_grid_block(nvid_ctx* ctx, dim3 * grid, dim3 * block, int maxthreads) {
	uint32_t blocks = ctx->device_blocks;
	uint32_t threads = ctx->device_threads;
	int mpcount = ctx->device_mpcount;

#ifdef __HIP_PLATFORM_HCC__
        // TODO calculate only once
	int nexthalved = blocks / 2;
	int minblocks = mpcount;
	while (minblocks % 4 == 0) {
		minblocks /= 2;
	}

	while (threads < maxthreads && nexthalved >= minblocks && nexthalved % minblocks == 0) {
		blocks /= 2;
		threads *= 2;

		nexthalved = blocks / 2;
	}
#else
        // TODO optimize
	uint32_t wsize = blocks * threads;

	threads = 128;
	blocks = ( wsize + threads - 1 ) / threads;
#endif
        //printf("T x B = %d x %d\n", threads, blocks);
	*grid = dim3( blocks );
	*block = dim3( threads );
}

extern "C" void cryptonight_extra_cpu_prepare(nvid_ctx* ctx, uint32_t startNonce, bool heavy)
{
	uint32_t wsize = ctx->device_blocks * ctx->device_threads;

/*	dim3 grid( ( wsize + threadsperblock - 1 ) / threadsperblock );
	dim3 block( threadsperblock );*/
	dim3 grid, block;
	set_grid_block(ctx, &grid, &block, 128);

	if (heavy) {
		hipLaunchKernelGGL(cryptonight_extra_gpu_prepare<true>, dim3(grid), dim3(block), 0, *ctx->stream, wsize, ctx->d_input, ctx->inputlen, startNonce,
						   ctx->d_ctx_state, ctx->d_ctx_state_p1, ctx->d_ctx_a, ctx->d_ctx_b, ctx->d_ctx_key1, ctx->d_ctx_key2);
	} else {
		hipLaunchKernelGGL(cryptonight_extra_gpu_prepare<false>, dim3(grid), dim3(block), 0, *ctx->stream, wsize, ctx->d_input, ctx->inputlen, startNonce,
						   ctx->d_ctx_state, ctx->d_ctx_state_p1, ctx->d_ctx_a, ctx->d_ctx_b, ctx->d_ctx_key1, ctx->d_ctx_key2);
	}
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );
}

extern "C" void cryptonight_extra_cpu_final(nvid_ctx* ctx, uint32_t startNonce, uint64_t target, uint32_t* rescount, uint32_t *resnonce, bool heavy)
{
	uint32_t wsize = ctx->device_blocks * ctx->device_threads;
	dim3 grid, block;
	set_grid_block(ctx, &grid, &block, 256);

	hipMemsetAsync( ctx->d_result_nonce, 0xFF, 10 * sizeof (uint32_t ), *ctx->stream );
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );
	hipMemsetAsync( ctx->d_result_count, 0, sizeof (uint32_t ), *ctx->stream );
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );

	if (heavy) {
		hipLaunchKernelGGL(cryptonight_extra_gpu_final<true>, dim3(grid), dim3(block), 0, *ctx->stream, wsize, target, ctx->d_result_count, ctx->d_result_nonce, ctx->d_ctx_state, ctx->d_ctx_key2);
	} else {
		hipLaunchKernelGGL(cryptonight_extra_gpu_final<false>, dim3(grid), dim3(block), 0, *ctx->stream, wsize, target, ctx->d_result_count, ctx->d_result_nonce, ctx->d_ctx_state, ctx->d_ctx_key2);
	}

	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );

	hipMemcpyAsync( rescount, ctx->d_result_count, sizeof (uint32_t ), hipMemcpyDeviceToHost, *ctx->stream );
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );
	hipMemcpyAsync( resnonce, ctx->d_result_nonce, 10 * sizeof (uint32_t ), hipMemcpyDeviceToHost, *ctx->stream );
	exit_if_cudaerror(ctx->device_id, __FILE__, __LINE__ );
#if DEBUG
	printf ("Run for startnonce %d with target %016lX over.\n", startNonce, target);
#endif
	for(int i=0; i < *rescount; i++) {
#if DEBUG
		printf ("Found raw resnonce %d.\n", resnonce[i]);
#endif
		resnonce[i] += startNonce;
	}
}

extern "C" int cuda_get_devicecount()
{
	hipError_t err;
	int deviceCount = 0;
	err = hipGetDeviceCount(&deviceCount);
	if(err != hipSuccess)
	{
		if(err == hipErrorNoDevice)
			printf("No CUDA devices?\n");
		else
			printf("Unable to query number of CUDA devices!\n");
		return 0;
	}

	return deviceCount;
}

#ifndef CUDART_VERSION
#define CUDART_VERSION 0
#endif

extern "C" int cuda_get_deviceinfo(nvid_ctx* ctx, xmrig::Algo algo)
{
	hipError_t err;
	int version;

	err = hipDriverGetVersion(&version);
	if(err != hipSuccess)
	{
		printf("Unable to query CUDA driver version! Is an nVidia driver installed?\n");
		return 0;
	}

	if(version < 0) //CUDART_VERSION)
	{
		printf("Driver does not support CUDA %d.%d API! Update your nVidia driver!\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
		return 0;
	}

	int GPU_N = cuda_get_devicecount();
	if(GPU_N == 0)
	{
		return 0;
	}

	if(ctx->device_id >= GPU_N)
	{
		printf("Invalid device ID!\n");
		return 0;
	}

	hipDeviceProp_t props;
	err = hipGetDeviceProperties(&props, ctx->device_id);
	if(err != hipSuccess)
	{
		printf("\nGPU %d: %s\n%s line %d\n", ctx->device_id, hipGetErrorString(err), __FILE__, __LINE__);
		return 0;
	}

	ctx->device_name = strdup(props.name);
	ctx->device_mpcount = props.multiProcessorCount;
#if 0 // def __HCC__
        // Treat HCC as Cuda 4.0 feature-wise, for now
	ctx->device_arch[0] = 4;
	ctx->device_arch[1] = 0;
#else
	ctx->device_arch[0] = props.major;
	ctx->device_arch[1] = props.minor;
#endif

	int shift, threads_to_set;
	if (props.multiProcessorCount < 20) {
		// Small Polaris
		threads_to_set = 64;
		shift = SMALL_POLARIS_SHIFT;
		ctx->autolower = 0;
	} else if (props.multiProcessorCount < 40) {
		// Big Polaris
		threads_to_set = 8;
		shift = LARGE_POLARIS_SHIFT;
		ctx->autolower = 1;
	} else {
		// Vega
		threads_to_set = 32;
		shift = VEGA_SHIFT;
		ctx->autolower = 0;
	}

	if(ctx->device_threads == -1)
	{
		ctx->device_threads = threads_to_set;
		printf("INFO: Set %s threads to: %d.\n", ctx->device_name, ctx->device_threads);
	}

	printf("INFO: Selected shift: %d.\n", shift);
	int d = 1 << shift;

	if(ctx->device_blocks == -1)
	{
		int blocks_for_sector = d / ctx->device_threads;
		size_t freeMemory = props.totalGlobalMem;

		size_t memory_97 = (freeMemory * size_t(97)) / 100;
		size_t memory_85 = (freeMemory * size_t(85)) / 100;

		printf("INFO: %s free memory: %lu.\n", ctx->device_name, freeMemory);
		printf("INFO: Blocks for sector: %d.\n", blocks_for_sector);

		ctx->device_blocks = 0;
		while(true)
		{
			int next_size = ctx->device_blocks + blocks_for_sector;
			size_t nextmem = size_t(next_size) * size_t(ctx->device_threads) * size_t(2u * 1024u * 1024u);
			if( nextmem > memory_97 ) {
				break;
			} else if(nextmem > memory_85 && (ctx->device_blocks % ctx->device_mpcount == 0) && (ctx->device_threads > 8)) {
				break;
			} else {
				ctx->device_blocks = next_size;
			}
		}
	}

	int t = ctx->device_threads * ctx->device_blocks;
	int rest = t % d;
	if (rest > 0) {
		uint other_shift = d >> MIXED_SHIFT_DOWNDRAFT;
		if (shift == VEGA_SHIFT) {
			if (rest % other_shift == 0) {
				printf("INFO: Total number of threads %d (threads*blocks) is not divisible by %d. Will divide the remainder by %d.\n",
					   t, d,  other_shift);
				ctx->mixed_shift = true;
			} else {
				printf("INFO: Total number of threads %d (threads*blocks) is not divisible by %d. Please at least make sure the remainder is divisible by %d.\n",
					   t, d,  other_shift);
				return 0;
			}
		} else {
			int suggested = ctx->device_blocks - (rest / ctx->device_threads);
			printf("INFO: Total number of threads %d (threads*blocks) is not divisible by %d. Auto-lowering blocks to %d.\n",
				   t, d, suggested);
			ctx->device_blocks = suggested;
		}
	}

	return 1;
}

int cryptonight_gpu_init(nvid_ctx *ctx, xmrig::Algo algo)
{
    return cryptonight_extra_cpu_init(ctx, algo); //  CRYPTONIGHT_MEMORY);
}

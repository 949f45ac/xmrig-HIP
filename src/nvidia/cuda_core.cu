#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>
#ifdef __HCC__
#include <hip/hcc_detail/device_functions.h>
#else
#include <vector_functions.h>
#include <cuda_runtime.h>
#endif

#include <sys/time.h>

#ifdef _WIN32
#include <windows.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
    if (waitTime > 0)
    {
        if (waitTime > 100)
        {
            // use a waitable timer for larger intervals > 0.1ms

            HANDLE timer;
            LARGE_INTEGER ft;

            ft.QuadPart = -(10*waitTime); // Convert to 100 nanosecond interval, negative value indicates relative time

            timer = CreateWaitableTimer(NULL, TRUE, NULL);
            SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
            WaitForSingleObject(timer, INFINITE);
            CloseHandle(timer);
        }
        else
        {
            // use a polling loop for short intervals <= 100ms

            LARGE_INTEGER perfCnt, start, now;
            __int64 elapsed;

            QueryPerformanceFrequency(&perfCnt);
            QueryPerformanceCounter(&start);
            do {
		SwitchToThread();
                QueryPerformanceCounter((LARGE_INTEGER*) &now);
                elapsed = (__int64)((now.QuadPart - start.QuadPart) / (float)perfCnt.QuadPart * 1000 * 1000);
            } while ( elapsed < waitTime );
        }
    }
}
#else
#include <unistd.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
	usleep(waitTime);
}
#endif

#include "cryptonight.h"
#include "cuda_extra.h"
#include "cuda_aes.hpp"
#include "cuda_device.hpp"
#include "phase2.cu"

/* sm_2X is limited to 2GB due to the small TLB
 * therefore we never use 64bit indices
 */
#if 1 // defined(XMR_STAK_LARGEGRID) // && (__CUDA_ARCH__ >= 300)
typedef uint64_t IndexType;
#else
typedef int IndexType;
#endif

__device__ __forceinline__ uint64_t cuda_mul128( uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi )
{
	*product_hi = __umul64hi( multiplier, multiplicand );
	return (multiplier * multiplicand );
}

template< typename T >
__device__ __forceinline__ void storeGlobal128AsyncGlc( T* adr, T const & val ) {
#ifdef __HCC__
	uint32_t * const val32 = (uint32_t*) &val;
	asm volatile (EMIT_STORE("%4, v[19:22]") " glc"
				  :
				  : "{v19}" (val32[0]), "{v20}" (val32[1]), "{v21}" (val32[2]), "{v22}" (val32[3]), "r" ( adr ));
#else
	*adr = val;
#endif
}


#undef HEAVY

// Number of threads per block to use in phase 1 and 3
#ifdef __HIP_PLATFORM_HCC__
#define P13T 256
#else
#define P13T 128
#endif
#define ENABLE_LAUNCH_BOUNDS 0

#if ENABLE_LAUNCH_BOUNDS
__launch_bounds__( P13T )
#endif


template<bool HEAVY, bool MIXED_SHIFT, int SEC_SHIFT>
__global__ void cryptonight_core_gpu_phase1( int threads, uint64_t * __restrict__ long_state_64, uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	const uint64_t thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ) >> 3;
	const int subRaw = ( hipThreadIdx_x & 7 );
	const int sub = ( hipThreadIdx_x & 7 ) << 2;

	INIT_SHIFT(3)

	uint4 * const long_state = reinterpret_cast<uint4*>(long_state_64) + BASE_OFF(thread, threads);

	const int batchsize = (0x80000 >> (HEAVY ? 1 : 2));
	const int start = 0;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40];
	uint4 text;

	MEMCPY8( key, ctx_key1 + thread * 40, 20 );

	// first round
	//text = ctx_state[thread * 50 + sub + 16];
	// text = *reinterpret_cast<uint4*>(ctx_state + thread * 50 + sub + 16);
	MEMCPY8( &text, ctx_state + thread * 50 + sub + 16, 2 );

	__syncthreads( );
	// #pragma unroll
	for ( int i = start; i < end; i += 8 )
	{
		cn_aes_pseudo_round_mut( sharedMemory, (uint32_t*) &text, key );

		// int offset = ((thread << 19) + (sub + i) ) / 4;
		int offset = SCRATCH_INDEX(subRaw + i);
		storeGlobal128AsyncGlc(long_state + offset, text);
	}
}


#if ENABLE_LAUNCH_BOUNDS
__launch_bounds__( P13T )
#endif
template<bool HEAVY, bool MIXED_SHIFT, int SEC_SHIFT>
__global__ void cryptonight_core_gpu_phase3( int threads, const uint64_t * __restrict__ long_state_64, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ) >> 3;
	int subRaw = ( hipThreadIdx_x & 7 );
	int sub = ( hipThreadIdx_x & 7 ) << 2;

	INIT_SHIFT(3)

	const uint4 * __restrict__ long_state = reinterpret_cast<const uint4*>(long_state_64) + BASE_OFF(thread, threads);

	const int batchsize = (0x80000 >> (HEAVY ? 1 : 2));
	const int start = 0;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40];
	uint4 text;

	// memc_<10>( key, d_ctx_key2 + thread * 40 );
	MEMCPY8( key, d_ctx_key2 + thread * 40, 20 );
	// text = *reinterpret_cast<uint4*>(d_ctx_state + thread * 50 + sub + 16);
	MEMCPY8( &text, d_ctx_state + thread * 50 + sub + 16, 2 );

	__syncthreads( );
	#pragma unroll 8
	for ( int i = start; i < end; i += 8 )
	{
		uint4 l = long_state[SCRATCH_INDEX(subRaw+i)];
		// text ^= l;
		text.x ^= l.x;
		text.y ^= l.y;
		text.z ^= l.z;
		text.w ^= l.w;

		cn_aes_pseudo_round_mut( sharedMemory, reinterpret_cast<uint32_t*>(&text), key );

		if (HEAVY) {
			text.x ^= __shfl( text.x, (subRaw+1)&7, 8 );
			text.y ^= __shfl( text.y, (subRaw+1)&7, 8 );
			text.z ^= __shfl( text.z, (subRaw+1)&7, 8 );
			text.w ^= __shfl( text.w, (subRaw+1)&7, 8 );
		}
	}

	//memcpy(d_ctx_state + thread * 50 + sub + 16, &text, sizeof(uint4));
	// memc_<1>(d_ctx_state + thread * 50 + sub + 16, &text);
	MEMCPY8(d_ctx_state + thread * 50 + sub + 16, &text, 2);
	__syncthreads( );
}




#define HEAVY (VARIANT == xmrig::VARIANT_XHV)

// extern "C"
template<xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
void cryptonight_core_cpu_hash(nvid_ctx* ctx, uint32_t nonce)
{
	dim3 grid, block;
	bool lowered = false; // VARIANT != xmrig::VARIANT_2 && !HEAVY && !ctx->is_vega && ctx->device_mpcount > 20 && ctx->device_threads > 4;
	if (lowered) {
		grid = dim3( ctx->device_blocks << 1 );
		block = dim3( ctx->device_threads >> 1);
	} else if (false) { // round blocks up
		int blocks = ctx->device_blocks ;
		if (blocks % ctx->device_mpcount > 0) {
			int m = blocks / ctx->device_mpcount;
			blocks = (m+1) * ctx->device_mpcount;
			printf("Rounded blocks up to %d\n", blocks);
		}

		grid = dim3( blocks );
		block = dim3( ctx->device_threads );
	} else {
		grid = dim3( ctx->device_blocks );
		block = dim3( ctx->device_threads );
	}

#if DEBUG
	timespec timespecc;
	clock_gettime(CLOCK_REALTIME, &timespecc);

	printf("Nonce %d P2 Schedule start at %ld \n", nonce, timespecc.tv_nsec);
#endif

	dim3 block2( ctx->device_threads * 2);
	dim3 block8( ctx->device_threads << 3 );
	dim3 block16( ctx->device_threads << 4 );

	dim3 p1_3_grid((ctx->device_blocks * ctx->device_threads * 8) / P13T);
	dim3 p1_3_block( P13T );

	int partcount = 1 << ctx->device_bfactor;

	/* bfactor for phase 1 and 3
	 *
	 * phase 1 and 3 consume less time than phase 2, therefore we begin with the
	 * kernel splitting if the user defined a `bfactor >= 5`
	 */
	int bfactorOneThree = ctx->device_bfactor - 4;
	if( bfactorOneThree < 0 )
		bfactorOneThree = 0;

	int partcountOneThree = 1 << bfactorOneThree;

#ifdef __HIP_PLATFORM_NVCC__
	cryptonight_core_gpu_phase1<HEAVY, MIXED_SHIFT, SEC_SHIFT><<<dim3(p1_3_grid), dim3(p1_3_block), 0, ctx->stream>>>(ctx->device_blocks*ctx->device_threads, ctx->d_long_state, ctx->d_ctx_state_p1, ctx->d_ctx_key1);
#else
	hipLaunchKernelGGL(cryptonight_core_gpu_phase1<HEAVY, MIXED_SHIFT, SEC_SHIFT>, dim3(p1_3_grid), dim3(p1_3_block), 0, ctx->stream, ctx->device_blocks*ctx->device_threads, ctx->d_long_state, ctx->d_ctx_state_p1, ctx->d_ctx_key1);
#endif
	exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );
#if DEBUG
	printf("Starting run for nonce %d\n", nonce);
#endif

	if (VARIANT == xmrig::VARIANT_2) {
#ifdef __HIP_PLATFORM_NVCC__
		cryptonight_core_gpu_phase2_monero_v8<MIXED_SHIFT, SEC_SHIFT><<<dim3(grid), dim3(block), 0, ctx->stream>>>(
			ctx->device_blocks*ctx->device_threads,
			ctx->d_long_state,
			ctx->d_ctx_a,
			ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#else
		hipLaunchKernelGGL(cryptonight_core_gpu_phase2_monero_v8<MIXED_SHIFT, SEC_SHIFT>,
						   dim3(grid), dim3(block), 0, ctx->stream, ctx->device_blocks*ctx->device_threads,
						   ctx->d_long_state,
						   ctx->d_ctx_a,
						   ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#endif
	} else if (HEAVY) {
#ifdef __HIP_PLATFORM_NVCC__
		cryptonight_core_gpu_phase2_heavy<VARIANT, MIXED_SHIFT, SEC_SHIFT><<<dim3(grid), dim3(block), 0, ctx->stream>>>(
			ctx->device_blocks*ctx->device_threads,
			ctx->d_long_state,
			ctx->d_ctx_a,
			ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#else
		hipLaunchKernelGGL(cryptonight_core_gpu_phase2_heavy<VARIANT, MIXED_SHIFT, SEC_SHIFT>,
						   dim3(grid), dim3(block), 0, ctx->stream, ctx->device_blocks*ctx->device_threads,
						   ctx->d_long_state,
						   ctx->d_ctx_a,
						   ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#endif
	} else if (lowered) {
#ifdef __HIP_PLATFORM_NVCC__
		cryptonight_core_gpu_phase2<VARIANT, false, SEC_SHIFT-1><<<dim3(grid), dim3(block), 0, ctx->stream>>>(
			ctx->device_blocks*ctx->device_threads,
			ctx->d_long_state,
			ctx->d_ctx_a,
			ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#else
		hipLaunchKernelGGL(cryptonight_core_gpu_phase2<VARIANT, false, SEC_SHIFT-1>,
						   dim3(grid), dim3(block), 0, ctx->stream, ctx->device_blocks*ctx->device_threads,
						   ctx->d_long_state,
						   ctx->d_ctx_a,
						   ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#endif
	} else {
#ifdef __HIP_PLATFORM_NVCC__
		cryptonight_core_gpu_phase2<VARIANT, MIXED_SHIFT, SEC_SHIFT><<<dim3(grid), dim3(block), 0, ctx->stream>>>(
			ctx->device_blocks*ctx->device_threads,
			ctx->d_long_state,
			ctx->d_ctx_a,
			ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#else
		hipLaunchKernelGGL(cryptonight_core_gpu_phase2<VARIANT, MIXED_SHIFT, SEC_SHIFT>,
						   dim3(grid), dim3(block), 0, ctx->stream, ctx->device_blocks*ctx->device_threads,
						   ctx->d_long_state,
						   ctx->d_ctx_a,
						   ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
#endif
	}

	exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );

	for ( int i = 0; i < (HEAVY + 1); i++ )
	{
#ifdef __HIP_PLATFORM_NVCC__
		cryptonight_core_gpu_phase3<HEAVY, MIXED_SHIFT, SEC_SHIFT><<<dim3(p1_3_grid), dim3(p1_3_block), 0, ctx->stream>>>(
			ctx->device_blocks*ctx->device_threads,
			ctx->d_long_state,
			ctx->d_ctx_state,
			ctx->d_ctx_key2);
#else
		hipLaunchKernelGGL(cryptonight_core_gpu_phase3<HEAVY, MIXED_SHIFT, SEC_SHIFT>, dim3(p1_3_grid), dim3(p1_3_block), 0, ctx->stream,
						   ctx->device_blocks*ctx->device_threads,
						   ctx->d_long_state,
						   ctx->d_ctx_state,
						   ctx->d_ctx_key2);
#endif
		exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );
	}
}

template<bool MIXED_SHIFT, int SEC_SHIFT>
void cryptonight_gpu_hash_shifted(nvid_ctx *ctx, xmrig::Algo algo, xmrig::Variant variant, uint32_t startNonce) {

    using namespace xmrig;

    if (algo == CRYPTONIGHT || algo == CRYPTONIGHT_HEAVY) {
        switch (variant) {
        case VARIANT_2:
            cryptonight_core_cpu_hash<VARIANT_2, MIXED_SHIFT, SEC_SHIFT>(ctx, startNonce);
            break;

        // case VARIANT_1:
        //     cryptonight_core_cpu_hash<VARIANT_1, MIXED_SHIFT, SEC_SHIFT>(ctx, startNonce);
        //     break;

        // case VARIANT_XTL:
        //     cryptonight_core_cpu_hash<VARIANT_XTL, MIXED_SHIFT, SEC_SHIFT>(ctx, startNonce);
        //     break;

        // case VARIANT_MSR:
        //     cryptonight_core_cpu_hash<VARIANT_MSR, MIXED_SHIFT, SEC_SHIFT>(ctx, startNonce);
        //     break;

        // case VARIANT_XHV:
        //     cryptonight_core_cpu_hash<VARIANT_XHV, MIXED_SHIFT, SEC_SHIFT>(ctx, startNonce);
        //     break;

        default:
            printf("Only CN1, XTL, MSR, XHV supported for now, but you requested: %d\n.", variant);
			exit(1);
        }
    }
    else {
		printf("Only CN1, XTL, MSR, XHV supported for now.");
		exit(1);
		return;
    }
}

#define COMPILE_FOR_VEGA (__HIP_ARCH_GFX900__ || __HIP_ARCH_GFX906__)
#define ONLY_VEGA (COMPILE_FOR_VEGA && !(__HIP_ARCH_GFX803__ || __HIP_ARCH_GFX802__ || __HIP_ARCH_GFX801__ || __HIP_ARCH_GFX701__))

extern "C" void cryptonight_gpu_hash(nvid_ctx *ctx, xmrig::Algo algo, xmrig::Variant variant, uint32_t startNonce)
{
#if COMPILE_FOR_VEGA
	if (ctx->is_vega) {
		if (ctx->mixed_shift) {
			cryptonight_gpu_hash_shifted<true, VEGA_SHIFT>(ctx, algo, variant, startNonce);
		} else {
			cryptonight_gpu_hash_shifted<false, VEGA_SHIFT>(ctx, algo, variant, startNonce);
		}
		return;
	}
#endif

#if !ONLY_VEGA
	if (ctx->device_mpcount > 22) {
		cryptonight_gpu_hash_shifted<false, LARGE_POLARIS_SHIFT>(ctx, algo, variant, startNonce);
		return;
	}

	// else
	{
		cryptonight_gpu_hash_shifted<false, SMALL_POLARIS_SHIFT>(ctx, algo, variant, startNonce);
		return;
	}
#endif

	printf("No P2 matched!\n");
	exit(1);
}

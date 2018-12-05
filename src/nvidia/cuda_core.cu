#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>
#ifdef __HCC__
#include <hip/hcc_detail/device_functions.h>
#else
#include <vector_functions.h>
#endif

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
#include "fast_int_math_v2.hpp"
#include "cuda_device.hpp"

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

// ASM required to wedge operations on variables loaded from memory into the correct spot.
#ifdef __HCC__
#define RETIRE(x)  { \
        uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
        asm volatile("v_mov_b32 %0, %1" : "=v" (w[0]), "=v" (w[1]) : "v" (w[0]), "v" (w[1]) : "memory"); }

#define FENCE(x) { \
	uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
	asm volatile("v_mov_b32 %0, %2\n\tv_mov_b32 %1, %3" : "=v" (w[0]), "=v" (w[1]) : "v" (w[0]), "v" (w[1]) : "memory"); }

#define WAIT_FOR(x, n) asm volatile("s_waitcnt vmcnt(" #n ")\n\t": : "v" (x) : "memory"); FENCE(x)
#define PRIO(n) asm volatile ("s_setprio 0x" #n ::: "memory");
#else
#define WAIT_FOR(x, n) FENCE(x)
#define RETIRE(x) ;
#define PRIO(n) ;
#define FENCE(w) { \
	asm volatile("mov.u64 %0, %1;\n\t" : "=l" (w) : "l" (w) : "memory"); }
//	uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
//	asm volatile("mov.u32 %0, %2;\n\tmov.u32 %1, %3;" : "=r" (w[0]), "=r" (w[1]) : "r" (w[0]), "r" (w[1]) : "memory"); }
#endif


#if !__HIP_ARCH_GFX803__
#define EMIT_LOAD(args) "global_load_dwordx2 " args ", off"
#define EMIT_STORE(args) "global_store_dwordx4 " args ", off"
#else
#define EMIT_LOAD(args) "flat_load_dwordx2 " args
#define EMIT_STORE(args) "flat_store_dwordx4 " args
#endif

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

template<int N>
__device__ __forceinline__ void memc_(void * __restrict__ dst, void * __restrict__ src) {
	ulonglong2 * lDst = reinterpret_cast<ulonglong2*>(dst);
	ulonglong2 * lSrc = reinterpret_cast<ulonglong2*>(src);
	#pragma unroll
	for (int i = 0; i < N; i++) {
		lDst[i] = lSrc[i];
	}
}

#define INIT_SHIFT()													\
	const uint over = SEC_SHIFT == 8 && thread >= ((threads >> SEC_SHIFT) << SEC_SHIFT); \
	const uint concrete_shift = SEC_SHIFT - over;						\
	const uint sec_size0 = 1 << SEC_SHIFT;								\
	const uint sec_size1 = 1 << concrete_shift;							\

#define CHU_SHIFT 4
#define CHU (1 << CHU_SHIFT)
__device__ __forceinline__ uint32_t scratch_index(uint64_t offset, int shift) {
	uint32_t chunk_blob_id = offset / CHU;
	int local_chunk = offset % CHU;

	return local_chunk + (chunk_blob_id << (shift+CHU_SHIFT));
}

#define SCRATCH_INDEX(offset) scratch_index(offset, concrete_shift)

#define BASE_OFF(thread, threads) (thread / sec_size0) * sec_size0 * (MEMORY/16) + (thread % sec_size1) * CHU;

// Number of threads per block to use in phase 1 and 3
#define P13T 256
#define ENABLE_LAUNCH_BOUNDS 0

template<int SEC_SHIFT>
#if ENABLE_LAUNCH_BOUNDS
__launch_bounds__( P13T )
#endif
__global__ void cryptonight_core_gpu_phase1( int threads, uint64_t * __restrict__ long_state_64, uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	const uint64_t thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ) >> 3;
	const int subRaw = ( hipThreadIdx_x & 7 );
	const int sub = ( hipThreadIdx_x & 7 ) << 2;

	INIT_SHIFT()

	uint4 * const long_state = reinterpret_cast<uint4*>(long_state_64) + BASE_OFF(thread, threads);

	const int batchsize = (0x80000 >> 2);
	const int start = 0;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40];
	uint4 text;

	memcpy( key, ctx_key1 + thread * 40, 160 );

	// first round
	//text = ctx_state[thread * 50 + sub + 16];
	text = *reinterpret_cast<uint4*>(ctx_state + thread * 50 + sub + 16);

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

#ifdef __HCC__
// HIP builtin is broken so we need to resort to manual impl.
#define UMUL64HI(m1, m2) _gpu_mul_hi_u64(m1, m2)
#else
#define UMUL64HI(m1, m2) __umul64hi(m1, m2)
#endif

__device__ __forceinline__ ulong
_gpu_mul_hi_u64(ulong x, ulong y)
{
    ulong x0 = x & 0xffffffffUL;
    ulong x1 = x >> 32;
    ulong y0 = y & 0xffffffffUL;
    ulong y1 = y >> 32;
    ulong z0 = x0*y0;
    ulong t = x1*y0 + (z0 >> 32);
    ulong z1 = t & 0xffffffffUL;
    ulong z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

#if __HIP_ARCH_GFX900__
#define MFLAGS " slc"
#else
#define MFLAGS ""
#endif

#ifdef __HCC__
#define ASYNC_LOAD(dst0, dst1, adr)	{								\
		asm volatile(												\
			EMIT_LOAD("%0, %2") MFLAGS " \n\t"						\
			EMIT_LOAD("%1, %3") MFLAGS " \n\t"						\
			: "=v"(dst0), "=v" (dst1)								\
			: "r" (adr), "r"(adr+1) ); }

#define AL128(dst, adr) asm volatile("flat_load_dwordx4 %0, %1" : "=v"(dst) : "r"(adr));
#else
#define ASYNC_LOAD(dst0, dst1, adr)	{								\
		asm volatile( "prefetch.global.L1 [%0];" : : "l"(adr) );	\
		asm volatile( "prefetch.global.L1 [%0+8];" : : "l"(adr) );}
#endif

// Only available for HCC.
#define ASYNC_STORE(adr, src) {											\
		uint32_t * const s32 = reinterpret_cast<uint32_t*>(&src);		\
		asm volatile(EMIT_STORE("%0, v[20:23]")	MFLAGS					\
					 :													\
					 : "r" (adr),										\
					   "{v20}" (s32[0]), "{v21}" (s32[1]), "{v22}" (s32[2]), "{v23}" (s32[3])); }


#define LOAD_CHUNK(dst, offset, n) dst = long_state[offset^n];
#define STORE_CHUNK(offset, src, n) long_state[offset^n] = src;

// Only available for HCC.
#define ASYNC_STORE_CHUNK(offset, src, n) {								\
		uint32_t * const s32 = reinterpret_cast<uint32_t*>(&src);		\
		int xoff = offset^n;											\
		asm volatile(EMIT_STORE("%0, v[20:23]")	MFLAGS					\
					 :													\
					 : "r" (long_state + xoff),							\
					   "{v20}" (s32[0]), "{v21}" (s32[1]), "{v22}" (s32[2]), "{v23}" (s32[3])); }

template<xmrig::Variant VARIANT, int SEC_SHIFT>
#ifdef __HCC__
__launch_bounds__( 16 )
#else
//__launch_bounds__( 64 )
#endif
__global__ void cryptonight_core_gpu_phase2( int threads, uint64_t * __restrict__ d_long_state_64, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b, uint32_t * __restrict__ d_ctx_state, uint32_t startNonce, uint32_t * __restrict__ d_input )
{
	__shared__ uint32_t sharedMemWritable[1024];

	cn_aes_gpu_init( sharedMemWritable );

	__syncthreads( );

	const uint32_t * const __restrict__ sharedMemory = (const uint32_t*) sharedMemWritable;


	const int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x );

	if ( thread >= threads )
		return;
	INIT_SHIFT()
	int i;
    uint32_t j0, j1;
	bool same_adr;
	const int batchsize = ITER >> ( VARIANT == xmrig::VARIANT_MSR ? 3 : 2 );
	const int start = 0;
	const int end = start + batchsize;
	// ulonglong2 * __restrict__ long_state = reinterpret_cast<ulonglong2*>(d_long_state_64 + ((((IndexType) thread) << 19) / 2));
	ulonglong2 * __restrict__ long_state = reinterpret_cast<ulonglong2*>(d_long_state_64) + BASE_OFF(thread, threads);
	ulonglong2 * ctx_a = reinterpret_cast<ulonglong2*>(d_ctx_a + thread * (16/sizeof(uint32_t)));
	uint32_t * ctx_b = d_ctx_b + thread * 12;

	uint32_t * state = d_ctx_state + thread * 50;

	uint32_t tweak1_2[2];
	tweak1_2[0] = (d_input[8] >> 24) | (d_input[9] << 8);
	tweak1_2[0] ^= state[48];
	tweak1_2[1] = startNonce + thread;
	tweak1_2[1] ^= state[49];

	ulonglong2 d[2];

	// Do not do memcpy here: it somehow causes the main loop to buffer registers t_t
	ulonglong2 a = *reinterpret_cast<ulonglong2*>(ctx_a);
	d[1] = *reinterpret_cast<ulonglong2*>(ctx_b);

	j0 = SCRATCH_INDEX(( a.x & 0x1FFFF0 ) >> 4);

	ulonglong2 x64 = long_state[j0];

	__syncthreads();
	#pragma unroll 8
	for ( i = start; i < end; ++i )
	{
		#pragma unroll 2
		for ( int x = 0; x < 2; ++x )
		{
			uint32_t * const x32 = (uint32_t*) &x64;
			uint32_t * const a32 = (uint32_t*) &a;

			uint32_t * const d32 = (uint32_t*) (d+x);

			d32[0] = a32[0] ^ (t_fn0(x32[0] & 0xff) ^ t_fn1((x32[1] >> 8) & 0xff) ^ t_fn2((x32[2] >> 16) & 0xff) ^ t_fn3((x32[3] >> 24)));
			j1 = SCRATCH_INDEX((d32[0] & 0x1FFFF0 ) >> 4);

			uint64_t * adr = reinterpret_cast<uint64_t*>(long_state + j1);
//			uint64_t ldst0, ldst1;
			ulonglong2 ldst_f;

			ASYNC_LOAD(ldst_f.x, ldst_f.y, adr);
			PRIO(1)

			d32[1] = a32[1]  ^ (t_fn0(x32[1] & 0xff) ^ t_fn1((x32[2] >> 8) & 0xff) ^ t_fn2((x32[3] >> 16) & 0xff) ^ t_fn3((x32[0] >> 24)));
			d32[2] = a32[2]  ^ (t_fn0(x32[2] & 0xff) ^ t_fn1((x32[3] >> 8) & 0xff) ^ t_fn2((x32[0] >> 16) & 0xff) ^ t_fn3((x32[1] >> 24)));
			d32[3] = a32[3]  ^ (t_fn0(x32[3] & 0xff) ^ t_fn1((x32[0] >> 8) & 0xff) ^ t_fn2((x32[1] >> 16) & 0xff) ^ t_fn3((x32[2] >> 24)));

			ulonglong2 d_xored = d[0];
			d_xored.x ^= d[1].x;
			d_xored.y ^= d[1].y;

			uint64_t fork_7 = d_xored.y;

			uint8_t index;
			if(VARIANT == xmrig::VARIANT_XTL) {
				index = ((fork_7 >> 27) & 12) | ((fork_7 >> 23) & 2);
			} else {
				index = ((fork_7 >> 26) & 12) | ((fork_7 >> 23) & 2);
			}

			const uint16_t table = 0x7531;
			fork_7 ^= ((table >> index) & 0x3) << 28;

			d_xored.y = fork_7;
			ulonglong2 * adr2 = long_state + j0;

#ifdef __HCC__
			ASYNC_STORE(adr2, d_xored);
#else
			// This load is automatically vectorized by nvcc.
			ldst_f.x = *adr;
			ldst_f.y = *(adr+1);

			// Manually setting .wt here can be sliightly faster than doing a simple store.
			asm volatile( "st.global.wt.v2.u64 [%0], {%1, %2};" : : "l"( adr2 ), "l"( d_xored.x ), "l"(d_xored.y) : "memory" );
#endif

			same_adr = j1 == j0;
			uint64_t t1_64 = d[x].x;


			WAIT_FOR(ldst_f.x, 1);
			PRIO(3)
			FENCE(ldst_f.y)
			ulonglong2 y2;
			y2.x = same_adr ? d_xored.x : ldst_f.x;
			y2.y = same_adr ? d_xored.y : ldst_f.y;

			a.x += UMUL64HI(t1_64, y2.x);

			ulonglong2 a_stor;
			a_stor.x = a.x;

			a.x ^= y2.x;
			j0 = SCRATCH_INDEX((a.x & 0x1FFFF0 ) >> 4);

			adr = reinterpret_cast<uint64_t*>(long_state + j0);
			ulonglong2 ldst;

			ASYNC_LOAD(ldst.x, ldst.y, adr);
			PRIO(0)

			FENCE(t1_64)
			a.y += (t1_64 * y2.x);

			a_stor.y = a.y;

			uint32_t *  a_stor32 = (uint32_t*) &a_stor;
			a_stor32[2] ^= tweak1_2[0];
			a_stor32[3] ^= tweak1_2[1];

#ifdef __HCC__
			ASYNC_STORE(long_state+j1, a_stor);
#else
			long_state[j1] = a_stor;
#endif
			same_adr = j0 == j1;

#ifndef __HCC__
			ldst = long_state[j0];
#endif
			a.y ^= y2.y;

			WAIT_FOR(ldst.x, 1)
			FENCE(ldst.y)
			PRIO(3)
			x64.x = same_adr ? a_stor.x : ldst.x;
			x64.y = same_adr ? a_stor.y : ldst.y;
			RETIRE(ldst.y)
		}
	}

}


#if ENABLE_LAUNCH_BOUNDS
__launch_bounds__( P13T )
#endif
template<int SEC_SHIFT>
__global__ void cryptonight_core_gpu_phase3( int threads, const uint64_t * __restrict__ long_state_64, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ) >> 3;
	int subRaw = ( hipThreadIdx_x & 7 );
	int sub = ( hipThreadIdx_x & 7 ) << 2;

	INIT_SHIFT()

	const uint4 * __restrict__ long_state = reinterpret_cast<const uint4*>(long_state_64) + BASE_OFF(thread, threads);

	const int batchsize = (0x80000 >> 2);
	const int start = 0;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40];
	uint4 text;
	//memcpy( key, d_ctx_key2 + thread * 40, 160 );
	memc_<10>( key, d_ctx_key2 + thread * 40 );
	text = *reinterpret_cast<uint4*>(d_ctx_state + thread * 50 + sub + 16);

	__syncthreads( );
	#pragma unroll 8
	for ( int i = start; i < end; i += 8 )
	{
		uint4 l = long_state[SCRATCH_INDEX(subRaw+i)];
		text ^= l;

		// text.x ^= l.x;
		// text.y ^= l.y;
		// text.z ^= l.z;
		// text.w ^= l.w;

		cn_aes_pseudo_round_mut( sharedMemory, (uint32_t*) &text, key );
	}

	//memcpy(d_ctx_state + thread * 50 + sub + 16, &text, sizeof(uint4));
	memc_<1>(d_ctx_state + thread * 50 + sub + 16, &text);
	__syncthreads( );
}

__device__ __forceinline__ ulonglong2
v_add(ulonglong2 a, ulonglong2 b)
{

#ifdef __HCC__
	return a+b;
#else
	a.x += b.x;
	a.y += b.y;
	return a;
#endif
}

__device__ __forceinline__ ulonglong2
v_xor(ulonglong2 a, ulonglong2 b)
{
#ifdef __HCC__
	return a^b;
#else
	a.x ^= b.x;
	a.y ^= b.y;
	return a;
#endif
}

template<int SEC_SHIFT>
__launch_bounds__( 32, 3 )
__global__ void cryptonight_core_gpu_phase2_monero_v8( int threads, uint64_t * __restrict__ d_long_state_64, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b, uint32_t * __restrict__ d_ctx_state, uint32_t startNonce, uint32_t * __restrict__ d_input )
{
	__shared__ uint32_t sharedMemWritable[1024];
	__shared__ uint32_t RCP[256];

	cn_aes_gpu_init( sharedMemWritable );

	for(int i = hipThreadIdx_x; i < 256; i += hipBlockDim_x)
		RCP[i] = RCP_C[i];

	__syncthreads( );

	const uint32_t * const __restrict__ sharedMemory = (const uint32_t*) sharedMemWritable;


	const int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x );

	if ( thread >= threads )
		return;

	INIT_SHIFT()

	int i;
    uint32_t j0, j1;

	const int batchsize = ITER >> ( 2 );
	const int start = 0;
	const int end = (start + batchsize) * 2;
	// ulonglong2 * __restrict__ long_state = reinterpret_cast<ulonglong2*>(d_long_state_64 + ((((IndexType) thread) << 19) / 2));

	ulonglong2 * __restrict__ long_state = reinterpret_cast<ulonglong2*>(d_long_state_64) + BASE_OFF(thread, threads);

	ulonglong2 * ctx_a = reinterpret_cast<ulonglong2*>(d_ctx_a + thread * (16/sizeof(uint32_t)));
	uint32_t * ctx_b = d_ctx_b + thread * 12;

	ulonglong2 a = *reinterpret_cast<ulonglong2*>(ctx_a);

	j0 = SCRATCH_INDEX(( a.x & 0x1FFFF0 ) >> 4);

//	__local uint RCP[256];
	uint64_t division_result;
	uint32_t sqrt_result;


	ulonglong2 d = reinterpret_cast<ulonglong2*>(ctx_b)[0];
	ulonglong2 d_old = reinterpret_cast<ulonglong2*>(ctx_b)[1];
	division_result = *reinterpret_cast<uint64_t*>(ctx_b+8);
	sqrt_result = *(ctx_b+10);

	// bx0 = ((u64*)(d_ctx_b + thread * 12))[sub];
	// 	bx1 = ((u64*)(d_ctx_b + thread * 12 + 4))[sub];

	// 	division_result = ((uint64_t*)(d_ctx_b + thread * 12 + 4 * 2))[0];
	// 	sqrt_result = (d_ctx_b + thread * 12 + 4 * 2 + 2)[0];

	__syncthreads();
	#pragma unroll 2
	for ( i = start; i < end; ++i )
	{
		uint4 x32 = reinterpret_cast<uint4*>(long_state)[j0];
		ulonglong2 chunk1, chunk2, chunk3;

		if (SEC_SHIFT != 6) {
			LOAD_CHUNK(chunk1, j0, 1);
			LOAD_CHUNK(chunk2, j0, 2);
			LOAD_CHUNK(chunk3, j0, 3);
		}

		if (SEC_SHIFT < 8) PRIO(2)

		uint4 c;
		uint32_t * a32 = reinterpret_cast<uint32_t*>(&a);

		// cn_aes_single_round(sharedMemory, (uint32_t*) &x32, (uint32_t*) &c, a32);
		c.x = a32[0] ^ (t_fn0(x32.x & 0xff) ^ t_fn1((x32.y >> 8) & 0xff) ^ t_fn2((x32.z >> 16) & 0xff) ^ t_fn3((x32.w >> 24)));
		j1 = SCRATCH_INDEX((c.x & 0x1FFFF0 ) >> 4);

		c.y = a32[1]  ^ (t_fn0(x32.y & 0xff) ^ t_fn1((x32.z >> 8) & 0xff) ^ t_fn2((x32.w >> 16) & 0xff) ^ t_fn3((x32.x >> 24)));
		c.z = a32[2]  ^ (t_fn0(x32.z & 0xff) ^ t_fn1((x32.w >> 8) & 0xff) ^ t_fn2((x32.x >> 16) & 0xff) ^ t_fn3((x32.y >> 24)));
		c.w = a32[3]  ^ (t_fn0(x32.w & 0xff) ^ t_fn1((x32.x >> 8) & 0xff) ^ t_fn2((x32.y >> 16) & 0xff) ^ t_fn3((x32.z >> 24)));

		if (SEC_SHIFT == 6) {
			LOAD_CHUNK(chunk1, j0, 1);
			LOAD_CHUNK(chunk2, j0, 2);
			LOAD_CHUNK(chunk3, j0, 3);
		}

		STORE_CHUNK(j0, v_add(chunk3, d_old), 1);
		STORE_CHUNK(j0, v_add(chunk1, d), 2);
		STORE_CHUNK(j0, v_add(chunk2, a), 3);
		{
			ulonglong2 d_xored = v_xor(d, *reinterpret_cast<ulonglong2*>(&c));
			long_state[j0] = d_xored;
		}

		ulonglong2 y2 = long_state[j1];
		uint64_t t1_64 = c.x | (((uint64_t) c.y) << 32);

		// ==== LOAD 2 : chunks ====
		LOAD_CHUNK(chunk1, j1, 1);
		LOAD_CHUNK(chunk2, j1, 2);
		LOAD_CHUNK(chunk3, j1, 3);

		// 			// Use division and square root results from the _previous_ iteration to hide the latency
		// tmp.s0 ^= division_result.s0;
		// tmp.s1 ^= division_result.s1 ^ sqrt_result;
		// {
		// 	uint32_t * y32 = (uint32_t*) &y2;
		// 	uint32_t * dr32 = reinterpret_cast<uint32_t*>(&division_result);
		// 	y32[0] ^= dr32[0];
		// 	y32[1] ^= dr32[1] ^ sqrt_result;
		// }

		PRIO(3)

		// // Most and least significant bits in the divisor are set to 1
		// // to make sure we don't divide by a small or even number,
		// // so there are no shortcuts for such cases
		const uint din = ( (c.x) + (sqrt_result << 1)) | 0x80000001UL;
		// Quotient may be as large as (2^64 - 1)/(2^31 + 1) = 8589934588 = 2^33 - 4
		// We drop the highest bit to fit both quotient and remainder in 32 bits
		uint64_t n_division_result = fast_div_v2(RCP, reinterpret_cast<ulonglong2*>(&c)->y, din);
		// Use division_result as an input for the square root to prevent parallel implementation in hardware
		uint32_t n_sqrt_result = fast_sqrt_v2(t1_64 + n_division_result); // 0x00AABBCCFF555 c.x + division_result

		y2.x ^= division_result ^ (((uint64_t) sqrt_result) << 32);

		division_result = n_division_result;
		sqrt_result = n_sqrt_result;


		uint64_t hi = UMUL64HI(t1_64, y2.x);
		uint64_t lo = t1_64 * y2.x;


		ulonglong2 result_mul;// = ulonglong2(hi, lo);
		result_mul.x = hi;
		result_mul.y = lo;

		// 	ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1)) ^ result_mul;
		chunk1 = v_xor(chunk1, result_mul);
		// 	ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
		// 	result_mul ^= chunk2;
		result_mul = v_xor(result_mul, chunk2);

		STORE_CHUNK(j1, v_add(chunk3, d_old), 1);
		STORE_CHUNK(j1, v_add(chunk1, d), 2);
		STORE_CHUNK(j1, v_add(chunk2, a), 3);

		// 	SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + ((ulong2 *)(b_x + 1))[0]);
		// 	SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + ((ulong2 *)b_x)[0]);
		// 	SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
		// }

	    a = v_add(a, result_mul);

		// a.y += result_mul.y; // lo
		// a.x += result_mul.x; // hi
		long_state[j1] = a;
		PRIO(0)

		a = v_xor(a, y2);
		j0 = SCRATCH_INDEX(( a.x & 0x1FFFF0 ) >> 4);

		d_old = d;
		d = *reinterpret_cast<ulonglong2*>(&c);
	}
}

// extern "C"
template<xmrig::Variant VARIANT, int SEC_SHIFT>
void cryptonight_core_cpu_hash(nvid_ctx* ctx, uint32_t nonce)
{
	dim3 grid, block;
	if (VARIANT != xmrig::VARIANT_2 && ctx->autolower && ctx->device_threads > 4) {
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

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		hipLaunchKernelGGL(cryptonight_core_gpu_phase1<SEC_SHIFT>, dim3(p1_3_grid), dim3(p1_3_block), 0, 0, ctx->device_blocks*ctx->device_threads,
			// ctx->device_shift, i,
			ctx->d_long_state, ctx->d_ctx_state, ctx->d_ctx_key1);
		exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}
	if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );

	for ( int i = 0; i < partcount; i++ )
	{
#if DEBUG
		printf("Starting run for nonce %d\n", nonce);
#endif

		if (VARIANT == xmrig::VARIANT_2) {
			hipLaunchKernelGGL(cryptonight_core_gpu_phase2_monero_v8<SEC_SHIFT>,
							   dim3(grid), dim3(block), 0, 0, ctx->device_blocks*ctx->device_threads,
							   // ctx->device_shift,
							   // i,
							   ctx->d_long_state,
							   ctx->d_ctx_a,
							   ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);

		} else {
			hipLaunchKernelGGL(cryptonight_core_gpu_phase2<VARIANT, SEC_SHIFT>,
							   dim3(grid), dim3(block), 0, 0, ctx->device_blocks*ctx->device_threads,
							   // ctx->device_shift,
							   // i,
							   ctx->d_long_state,
							   ctx->d_ctx_a,
							   ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
		}

		exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		hipLaunchKernelGGL(cryptonight_core_gpu_phase3<SEC_SHIFT>, dim3(p1_3_grid), dim3(p1_3_block), 0, 0, ctx->device_blocks*ctx->device_threads,
			// ctx->device_shift, i,
			ctx->d_long_state,
			ctx->d_ctx_state, ctx->d_ctx_key2);
		exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );
	}
}

template<int SEC_SHIFT>
void cryptonight_gpu_hash_shifted(nvid_ctx *ctx, xmrig::Algo algo, xmrig::Variant variant, uint32_t startNonce) {

    using namespace xmrig;

    if (algo == CRYPTONIGHT) {
        switch (variant) {
        case VARIANT_1:
            cryptonight_core_cpu_hash<VARIANT_1, SEC_SHIFT>(ctx, startNonce);
            break;

		case VARIANT_2:
            cryptonight_core_cpu_hash<VARIANT_2, SEC_SHIFT>(ctx, startNonce);
            break;

        case VARIANT_XTL:
            cryptonight_core_cpu_hash<VARIANT_XTL, SEC_SHIFT>(ctx, startNonce);
            break;

        case VARIANT_MSR:
            cryptonight_core_cpu_hash<VARIANT_MSR, SEC_SHIFT>(ctx, startNonce);
            break;

        default:
            break;
        }
    }
    else {
		printf("Only CN1, XTL, MSR supported for now.");
		return;
    }
}

#define ONLY_VEGA !(__HIP_ARCH_GFX803__ || __HIP_ARCH_GFX802__ || __HIP_ARCH_GFX801__ || __HIP_ARCH_GFX701__)

extern "C" void cryptonight_gpu_hash(nvid_ctx *ctx, xmrig::Algo algo, xmrig::Variant variant, uint32_t startNonce)
{
#if __HIP_ARCH_GFX900__ || __HIP_ARCH_GFX906__
	if (ctx->device_mpcount > 50) {
		cryptonight_gpu_hash_shifted<VEGA_SHIFT>(ctx, algo, variant, startNonce);
		return;
	}
#endif

#if __HIP_ARCH_GFX803__ || __HIP_ARCH_GFX701__
	if (ctx->device_mpcount > 22) {
		cryptonight_gpu_hash_shifted<LARGE_POLARIS_SHIFT>(ctx, algo, variant, startNonce);
		return;
	}

	// else
	{
		cryptonight_gpu_hash_shifted<SMALL_POLARIS_SHIFT>(ctx, algo, variant, startNonce);
		return;
	}
#endif
}

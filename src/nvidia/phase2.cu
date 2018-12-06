#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>

#ifdef __HCC__
#include <hip/hcc_detail/device_functions.h>
#else
#include <vector_functions.h>
#endif

#include "cryptonight.h"
#include "cuda_extra.h"
#include "cuda_aes.hpp"
#include "fast_int_math_v2.hpp"


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


template<xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
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

#define IS_V1 0
#define CN_HEAVY 1

template<xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
#ifdef __HCC__
__launch_bounds__( 16 )
#else
//__launch_bounds__( 64 )
#endif
__global__ void cryptonight_core_gpu_phase2_heavy( int threads, uint64_t * __restrict__ d_long_state_64, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b, uint32_t * __restrict__ d_ctx_state, uint32_t startNonce, uint32_t * __restrict__ d_input )
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
	const int batchsize = ITER >> ( 3 );
	const int start = 0;
	const int end = start + batchsize;
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

	ulonglong2 a = *reinterpret_cast<ulonglong2*>(ctx_a);
	d[1] = *reinterpret_cast<ulonglong2*>(ctx_b);

	j0 = SCRATCH_INDEX(( a.x & 0x3FFFF0 ) >> 4);

	ulonglong2 x64 = long_state[j0];

	__syncthreads();
	#pragma unroll 2
	for ( i = start; i < end; ++i )
	{
		#pragma unroll 2
		for ( int x = 0; x < 2; ++x )
		{
			uint32_t * const x32 = (uint32_t*) &x64;
			uint32_t * const a32 = (uint32_t*) &a;

			uint32_t * const d32 = (uint32_t*) (d+x);


			// WAIT_FOR(x64.x, 0);
			// FENCE(x64.y);
			d32[0] = a32[0] ^ (t_fn0(x32[0] & 0xff) ^ t_fn1((x32[1] >> 8) & 0xff) ^ t_fn2((x32[2] >> 16) & 0xff) ^ t_fn3((x32[3] >> 24)));
			j1 = SCRATCH_INDEX((d32[0] & 0x3FFFF0 ) >> 4);

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

			if (0) {
				uint64_t fork_7 = d_xored.y;

				uint8_t index = ((fork_7 >> 26) & 12) | ((fork_7 >> 23) & 2);

				const uint16_t table = 0x7531;
				fork_7 ^= ((table >> index) & 0x3) << 28;

				d_xored.y = fork_7;
			}
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
			j0 = SCRATCH_INDEX((a.x & 0x3FFFF0 ) >> 4);

			adr = reinterpret_cast<uint64_t*>(long_state + j0);


			int64_t n_;
			int32_t d_;
			asm (
				EMIT_LOAD("%0, %2") MFLAGS " \n\t"
				"flat_load_dword %1, %3" MFLAGS " \n\t"
				: "=v"(n_), "=v" (d_)
				: "r" (adr), "r"(adr+1)
				: "memory");
			PRIO(0)

			FENCE(t1_64)
			a.y += (t1_64 * y2.x);

			a_stor.y = a.y;

			uint32_t *  a_stor32 = (uint32_t*) &a_stor;
			if (0) {
				a_stor32[2] ^= tweak1_2[0];
				a_stor32[3] ^= tweak1_2[1];
			}

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

			WAIT_FOR(n_, 1);
			FENCE32(d_);
			PRIO(3);

			int64_t n = same_adr ? (int64_t) a_stor.x : n_;
			int32_t d = same_adr ? (int32_t) a_stor.y : d_;

			int64_t q = fast_div_heavy(n, d | 0x5);

			// *((__global long*)(Scratchpad + (IDX((idx0) >> 4)))) = n ^ q;
			// *reinterpret_cast<int64_t*>(long_state + j0) = n ^ q;
			int j2 = SCRATCH_INDEX((((~d) ^ q) & 0x3FFFF0) >> 4);

			x64.y = long_state[j2].y;
			// uint64_t ldst;

			// ASYNC_LOAD(ldst, x64.y, long_state+j2);

			int64_t nxq = n^q;
			asm ("flat_store_dwordx2 %0, %1"	MFLAGS
				 :
				 : "r" (long_state+j0), "v" (nxq) : "memory");

			x64.x = long_state[j2].x;

			// WAIT_FOR(ldst, 1);
			// FENCE(x64.y);
			// x64.x = j2 == j0 ? nxq : ldst;
			// RETIRE(ldst);
			j0 = j2;
		}
	}

}

#undef CN_HEAVY

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

template<bool MIXED_SHIFT, int SEC_SHIFT>
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

#define VARIANT (xmrig::VARIANT_2)

	ulonglong2 * __restrict__ long_state = reinterpret_cast<ulonglong2*>(d_long_state_64) + BASE_OFF(thread, threads);

#undef VARIANT

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

		PRIO(3)

		// // Most and least significant bits in the divisor are set to 1
		// // to make sure we don't divide by a small or even number,
		// // so there are no shortcuts for such cases
		const uint din = ( (c.x) + (sqrt_result << 1)) | 0x80000001UL;
		// Quotient may be as large as (2^64 - 1)/(2^31 + 1) = 8589934588 = 2^33 - 4
		// We drop the highest bit to fit both quotient and remainder in 32 bits
		uint64_t n_division_result = fast_div_v2(RCP, reinterpret_cast<ulonglong2*>(&c)->y, din);
		// Use division_result as an input for the square root to prevent parallel implementation in hardware
		uint32_t n_sqrt_result = fast_sqrt_v2(t1_64 + n_division_result);

		y2.x ^= division_result ^ (((uint64_t) sqrt_result) << 32);

		division_result = n_division_result;
		sqrt_result = n_sqrt_result;


		uint64_t hi = UMUL64HI(t1_64, y2.x);
		uint64_t lo = t1_64 * y2.x;


		ulonglong2 result_mul = ulonglong2(hi, lo);

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

	    a = v_add(a, result_mul);

		long_state[j1] = a;
		PRIO(0)

		a = v_xor(a, y2);
		j0 = SCRATCH_INDEX(( a.x & 0x1FFFF0 ) >> 4);

		d_old = d;
		d = *reinterpret_cast<ulonglong2*>(&c);
	}
}

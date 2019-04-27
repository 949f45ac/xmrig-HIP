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

#include "crypto/CryptoNight_constants.h"

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
			: "r" (adr), "r"(adr+1) : "memory"); }

#define AL128(dst, adr) asm volatile(EMIT_WIDE_LOAD("%0, %1") MFLAGS : "=v"(dst) : "r"(adr) : "memory");


// Only available for HCC.
#define ASYNC_STORE(adr, src) {											\
		uint32_t * const s32 = reinterpret_cast<uint32_t*>(&src);		\
		asm volatile(EMIT_STORE("%0, v[20:23]")	MFLAGS					\
					 :													\
					 : "r" (adr),										\
					   "{v20}" (s32[0]), "{v21}" (s32[1]), "{v22}" (s32[2]), "{v23}" (s32[3])); }


#define AS128(adr, src) { \
		asm volatile(EMIT_STORE("%0, %1") MFLAGS						\
					 :													\
					 : "r" (adr),										\
					   "r" (*reinterpret_cast<__uint128_t*>(&src))		\
					 : "memory"); }

#define ASYNC_STORE_CHUNK(offset, src, n) {								\
		uint32_t * const s32 = reinterpret_cast<uint32_t*>(&src);		\
		int xoff = offset^n;											\
		asm volatile(EMIT_STORE("%0, v[20:23]")	MFLAGS					\
					 :													\
					 : "r" (long_state + xoff),							\
					   "{v20}" (s32[0]), "{v21}" (s32[1]), "{v22}" (s32[2]), "{v23}" (s32[3])); }
#else
#define ASYNC_LOAD(dst0, dst1, adr)	{								\
		asm volatile( "prefetch.global.L1 [%0];" : : "l"(adr) );	\
		asm volatile( "prefetch.global.L1 [%0+8];" : : "l"(adr) );}
#endif

#define LOAD_CHUNK(dst, offset, n) dst = long_state[offset^n];
#define STORE_CHUNK(offset, src, n) long_state[offset^n] = src;

template<xmrig::Algo ALGO, xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
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

	INIT_SHIFT(0)
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

	const uint64_t tweak = tweak1_2[0] | (((uint64_t) tweak1_2[1]) << 32);

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
			uint4 x32 = *reinterpret_cast<uint4*>(&x64);
			uint4 c = make_uint4(0, 0, 0, 0);

			c.x = ((uint32_t) a.x) ^ (t_fn0(x32.x & 0xff) ^ t_fn1((x32.y >> 8) & 0xff) ^ t_fn2((x32.z >> 16) & 0xff) ^ t_fn3((x32.w >> 24)));
			j1 = SCRATCH_INDEX((c.x & 0x1FFFF0) >> 4);
			uint64_t * adr = reinterpret_cast<uint64_t*>(long_state + j1);
//			uint64_t ldst0, ldst1;
			ulonglong2 ldst_f;

			ASYNC_LOAD(ldst_f.x, ldst_f.y, adr);
			PRIO(1);

			c.y = (a.x >> 32) ^ (t_fn0(x32.y & 0xff) ^ t_fn1((x32.z >> 8) & 0xff) ^ t_fn2((x32.w >> 16) & 0xff) ^ t_fn3((x32.x >> 24)));
			c.z = ((uint32_t) a.y) ^ (t_fn0(x32.z & 0xff) ^ t_fn1((x32.w >> 8) & 0xff) ^ t_fn2((x32.x >> 16) & 0xff) ^ t_fn3((x32.y >> 24)));
			c.w = (a.y >> 32) ^ (t_fn0(x32.w & 0xff) ^ t_fn1((x32.x >> 8) & 0xff) ^ t_fn2((x32.y >> 16) & 0xff) ^ t_fn3((x32.z >> 24)));

			d[x] = *reinterpret_cast<ulonglong2*>(&c);

			ulonglong2 d_xored = d[0];
			d_xored.x ^= d[1].x;
			d_xored.y ^= d[1].y;

			uint64_t fork_7 = d_xored.y;

			uint8_t index;
			if (VARIANT == xmrig::VARIANT_XTL) {
				index = ((fork_7 >> 27) & 12) | ((fork_7 >> 23) & 2);
			} else {
				index = ((fork_7 >> 26) & 12) | ((fork_7 >> 23) & 2);
			}

			const uint16_t table = 0x7531;
			fork_7 ^= ((table >> index) & 0x3) << 28;

			d_xored.y = fork_7;
			ulonglong2 * adr2 = long_state + j0;

#ifdef __HCC__
			AS128(adr2, d_xored);
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
			j0 = SCRATCH_INDEX((a.x & 0x1FFFF0) >> 4);

			adr = reinterpret_cast<uint64_t*>(long_state + j0);
			ulonglong2 ldst;

			ASYNC_LOAD(ldst.x, ldst.y, adr);
			PRIO(0)

			FENCE(t1_64)
			a.y += (t1_64 * y2.x);

			a_stor.y = a.y ^ tweak;

#ifdef __HCC__
			AS128(long_state+j1, a_stor);
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

#define ALGO (xmrig::CRYPTONIGHT_HEAVY)
template<xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
#ifdef __HCC__
__launch_bounds__( 64 )
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

	INIT_SHIFT(0)

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

	const uint64_t tweak = tweak1_2[0] | (((uint64_t) tweak1_2[1]) << 32);

	ulonglong2 d[2];

	ulonglong2 a = *reinterpret_cast<ulonglong2*>(ctx_a);
	d[1] = *reinterpret_cast<ulonglong2*>(ctx_b);

	j0 = SCRATCH_INDEX(( a.x & 0x3FFFF0 ) >> 4);

	__syncthreads();
	#pragma unroll 2
	for ( i = start; i < end; ++i )
	{
		#pragma unroll 2
		for ( int x = 0; x < 2; ++x )
		{
			ulonglong2 x64 = long_state[j0];
			uint4 x32 = make_uint4(x64.x, x64.x >> 32, x64.y, x64.y >> 32);
			if (VARIANT == xmrig::VARIANT_TUBE) { x32 = ~x32; }
			uint4 c = make_uint4(0, 0, 0, 0);

			c.x = ((uint32_t) a.x) ^ (t_fn0(x32.x & 0xff) ^ t_fn1((x32.y >> 8) & 0xff) ^ t_fn2((x32.z >> 16) & 0xff) ^ t_fn3((x32.w >> 24)));
			j1 = SCRATCH_INDEX((c.x & 0x3FFFF0) >> 4);
//			uint64_t * adr = reinterpret_cast<uint64_t*>(long_state + j1);
//			uint64_t ldst0, ldst1;
			ulonglong2 y2 = long_state[j1];

			// ASYNC_LOAD(ldst_f.x, ldst_f.y, adr);
			PRIO(1);

			if (VARIANT == xmrig::VARIANT_TUBE) { x32.x ^= c.x; }

			c.y = (a.x >> 32) ^ (t_fn0(x32.y & 0xff) ^ t_fn1((x32.z >> 8) & 0xff) ^ t_fn2((x32.w >> 16) & 0xff) ^ t_fn3((x32.x >> 24)));
			if (VARIANT == xmrig::VARIANT_TUBE) { x32.y ^= c.y; }
			c.z = ((uint32_t) a.y) ^ (t_fn0(x32.z & 0xff) ^ t_fn1((x32.w >> 8) & 0xff) ^ t_fn2((x32.x >> 16) & 0xff) ^ t_fn3((x32.y >> 24)));
			if (VARIANT == xmrig::VARIANT_TUBE) { x32.z ^= c.z; }
			c.w = (a.y >> 32) ^ (t_fn0(x32.w & 0xff) ^ t_fn1((x32.x >> 8) & 0xff) ^ t_fn2((x32.y >> 16) & 0xff) ^ t_fn3((x32.z >> 24)));

			d[x] = *reinterpret_cast<ulonglong2*>(&c);

			ulonglong2 d_xored = d[0];
			d_xored.x ^= d[1].x;
			d_xored.y ^= d[1].y;

			if (VARIANT == xmrig::VARIANT_TUBE || VARIANT == xmrig::VARIANT_1) {
				uint64_t fork_7 = d_xored.y;

				uint8_t index = ((fork_7 >> 26) & 12) | ((fork_7 >> 23) & 2);

				const uint16_t table = 0x7531;
				fork_7 ^= ((table >> index) & 0x3) << 28;

				d_xored.y = fork_7;
			}

			// AS128(adr2, d_xored);
			long_state[j0] = d_xored;


			same_adr = j1 == j0;
			uint64_t t1_64 = d[x].x;


			// WAIT_FOR(ldst_f.x, 1);
			PRIO(3);
			// FENCE(ldst_f.y)
			// ulonglong2 y2;
			if (same_adr) {
				y2 = d_xored;
			}

			a.x += UMUL64HI(t1_64, y2.x);

			ulonglong2 a_stor;
			a_stor.x = a.x;

			a.x ^= y2.x;
			j0 = SCRATCH_INDEX((a.x & 0x3FFFF0) >> 4);

			int64_t * adr = reinterpret_cast<int64_t*>(long_state + j0);

			int64_t n = *adr;
			int32_t d = *reinterpret_cast<int32_t*>(adr+1);


			FENCE(t1_64)
			a.y += (t1_64 * y2.x);

			a_stor.y = a.y;

			if (VARIANT == xmrig::VARIANT_TUBE) {
				a_stor.y ^= tweak ^ a_stor.x;

			}

			if (VARIANT == xmrig::VARIANT_1) {
				a_stor.y ^= tweak;
			}


			long_state[j1] = a_stor;

			same_adr = j0 == j1;

			a.y ^= y2.y;

			if (same_adr) {
				n = (int64_t) a_stor.x;
				d = (int32_t) a_stor.y;
			}
			int64_t q = fast_div_heavy(n, d | 0x5);

			if (VARIANT == xmrig::VARIANT_XHV) {
				asm ("V_NOT_B32 %0, %1" : "=v"(d) : "v"(d));
			}

			uint64_t nnn = d ^ q;
			j0 = SCRATCH_INDEX((nnn & 0x3FFFF0) >> 4);


			int64_t nxq = n^q;

			*adr = nxq;
		}
	}
}
#undef ALGO

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


template<xmrig::Algo ALGO, xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
__launch_bounds__( 32, 3 )
__global__ void cryptonight_core_gpu_phase2_monero_v8( int threads, uint64_t * __restrict__ d_long_state_64, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b, uint32_t * __restrict__ d_ctx_state, uint32_t startNonce, uint32_t * __restrict__ d_input )
{
	__shared__ uint32_t sharedMemWritable[1024];

	cn_aes_gpu_init( sharedMemWritable );

	__syncthreads( );

	const uint32_t * const __restrict__ sharedMemory = (const uint32_t*) sharedMemWritable;

	const int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x );

	if ( thread >= threads )
		return;

	INIT_SHIFT(0);

	ulonglong2 * __restrict__ long_state = reinterpret_cast<ulonglong2*>(d_long_state_64) + BASE_OFF(thread, threads);

	ulonglong2 * ctx_a = reinterpret_cast<ulonglong2*>(d_ctx_a + thread * (16/sizeof(uint32_t)));
	uint32_t * ctx_b = d_ctx_b + thread * 12;

	ulonglong2 a = *reinterpret_cast<ulonglong2*>(ctx_a);
	ulonglong2 d = reinterpret_cast<ulonglong2*>(ctx_b)[0];
	ulonglong2 d_old = reinterpret_cast<ulonglong2*>(ctx_b)[1];
	uint64_t division_result = reinterpret_cast<uint64_t*>(ctx_b)[4];
	uint32_t sqrt_result = ctx_b[10];

	const uint32_t mask = xmrig::cn_select_mask<ALGO>();

	ulonglong2 foo = make_ulonglong2(55, 77);
	foo -= d;
	foo -= d_old;
	foo.x += division_result;
	foo.x += sqrt_result;

	const uint32_t end = ( xmrig::cn_select_iter<ALGO, VARIANT>() );
	constexpr const bool reverse = VARIANT == xmrig::VARIANT_RWZ;

	uint32_t j0 = SCRATCH_INDEX(( a.x & mask ) >> 4);
	ulonglong2 chunk1, chunk2, chunk3;

	uint4 x32 = reinterpret_cast<uint4*>(long_state)[j0];

	if (reverse) {
		LOAD_CHUNK(chunk1, j0, 3);
		LOAD_CHUNK(chunk2, j0, 2);
		LOAD_CHUNK(chunk3, j0, 1);
	} else {
		LOAD_CHUNK(chunk1, j0, 1);
		LOAD_CHUNK(chunk2, j0, 2);
		LOAD_CHUNK(chunk3, j0, 3);
	}

	__syncthreads();
	// #pragma unroll 2
	for ( int i = 0; i < end; ++i )
	{
		uint4 c = make_uint4(0, 0, 0, 0);

		c.x = ((uint32_t) a.x) ^ (t_fn0(x32.x & 0xff) ^ t_fn1((x32.y >> 8) & 0xff) ^ t_fn2((x32.z >> 16) & 0xff) ^ t_fn3((x32.w >> 24)));
		uint32_t j1 = SCRATCH_INDEX(( c.x & mask ) >> 4);
		ulonglong2 y2 = long_state[j1];

		ulonglong2 chunk1_l, chunk2_l, chunk3_l;
		LOAD_CHUNK(chunk3_l, j1, 3);
		LOAD_CHUNK(chunk1_l, j1, 1);
		LOAD_CHUNK(chunk2_l, j1, 2);
		FENCE32(c.x);

		c.y = (a.x >> 32) ^ (t_fn0(x32.y & 0xff) ^ t_fn1((x32.z >> 8) & 0xff) ^ t_fn2((x32.w >> 16) & 0xff) ^ t_fn3((x32.x >> 24)));
		c.z = ((uint32_t) a.y) ^ (t_fn0(x32.z & 0xff) ^ t_fn1((x32.w >> 8) & 0xff) ^ t_fn2((x32.x >> 16) & 0xff) ^ t_fn3((x32.y >> 24)));
		c.w = (a.y >> 32) ^ (t_fn0(x32.w & 0xff) ^ t_fn1((x32.x >> 8) & 0xff) ^ t_fn2((x32.y >> 16) & 0xff) ^ t_fn3((x32.z >> 24)));


		ulonglong2 chunk1_stored = v_add(chunk3, d_old);
		ulonglong2 chunk2_stored = v_add(chunk1, d);
		ulonglong2 chunk3_stored = v_add(chunk2, a);

		STORE_CHUNK(j0, chunk1_stored, 1);
		STORE_CHUNK(j0, chunk2_stored, 2);
		STORE_CHUNK(j0, chunk3_stored, 3);

		ulonglong2 d_xored = v_xor(d, *reinterpret_cast<ulonglong2*>(&c));
		long_state[j0] = d_xored;

		uint pattern = j0 ^ j1;
		if (pattern < 4) {
			switch (pattern) {
			case 1: // Pairwise swap
				y2 = chunk1_stored;
				chunk1_l = d_xored;

				chunk2_l = chunk3_stored;
				chunk3_l= chunk2_stored;
				break;

			case 2: // Reverse + swap
				y2 = chunk2_stored;
				chunk1_l = chunk3_stored;

				chunk2_l = d_xored;
				chunk3_l = chunk1_stored;
				break;

			case 3: // Reverse
				y2 = chunk3_stored;
				chunk1_l = chunk2_stored;

				chunk2_l = chunk1_stored;
				chunk3_l = d_xored;
				break;

			case 0: // Id
			default:
				y2 = d_xored;
				chunk1_l = chunk1_stored;

				chunk2_l = chunk2_stored;
				chunk3_l = chunk3_stored;
			}
		}

		FENCE32(c.x);
		uint64_t t1_64 = c.x | (((uint64_t) c.y) << 32);

		const uint din = (c.x + (sqrt_result << 1)) | 0x80000001UL;
		uint64_t n_division_result = fast_div_v2(reinterpret_cast<ulonglong2*>(&c)->y, din);
		uint32_t n_sqrt_result = fast_sqrt_v2(t1_64 + n_division_result);
		FENCE32(n_sqrt_result);

#if 0 // if ONLY_VEGA -- Only faster with unroll=4 and we cannot #if the unroll
		uint4 dl = make_uint4(d_old.x, d_old.x >> 32, d_old.y, d_old.y >> 32);
		asm volatile(
			"v_add_co_u32_e32  %0, vcc, %8, %4 \n\t"
			"v_addc_co_u32_e32 %1, vcc, %9, %5, vcc \n\t"
			"v_add_co_u32_e32  %2, vcc, %10, %6 \n\t"
			"v_addc_co_u32_e32 %3, vcc, %11, %7, vcc \n\t"
			: "=v" (dl.x), "=v" (dl.y), "=v" (dl.z), "=v" (dl.w)
			: "v" ((uint32_t)chunk3.x), "v" ((uint32_t)(chunk3.x >> 32)), "v" ((uint32_t)chunk3.y), "v" ((uint32_t)(chunk3.y>>32)),
			  "v" (dl.x), "v" (dl.y), "v" (dl.z), "v" (dl.w) : "vcc", "memory");

		reinterpret_cast<uint4*>(long_state)[j1^1] = dl;
#else

#endif

		y2.x ^= division_result ^ (((uint64_t) sqrt_result) << 32);

		division_result = n_division_result;
		sqrt_result = n_sqrt_result;

		uint64_t hi = UMUL64HI(t1_64, y2.x);
		uint64_t lo = t1_64 * y2.x;

		ulonglong2 result_mul = make_ulonglong2(hi, lo);
		ulonglong2 rm_init = result_mul;

		result_mul = v_xor(result_mul, chunk2_l);


/////
		ulonglong2 a_later = a;
		ulonglong2 a_stor = v_add(a, result_mul);
		a = v_xor(a_stor, y2);

		j0 = SCRATCH_INDEX(( a.x & mask ) >> 4);
		x32 = reinterpret_cast<uint4*>(long_state)[j0];
		if (reverse) {
			LOAD_CHUNK(chunk1, j0, 3);
			LOAD_CHUNK(chunk2, j0, 2);
			LOAD_CHUNK(chunk3, j0, 1);
		} else {
			LOAD_CHUNK(chunk1, j0, 1);
			LOAD_CHUNK(chunk2, j0, 2);
			LOAD_CHUNK(chunk3, j0, 3);
		}
//////

		if (reverse) {
		} else {

		}
		chunk1_l = v_xor(chunk1_l, rm_init);

		if (reverse) {
			chunk1_stored = v_add(chunk1_l, d_old);
			STORE_CHUNK(j1, chunk1_stored, 1);
			chunk2_stored = v_add(chunk3_l, d);
			STORE_CHUNK(j1, chunk2_stored, 2);
		} else {
			chunk1_stored = v_add(chunk3_l, d_old);
			STORE_CHUNK(j1, chunk1_stored, 1);
			chunk2_stored = v_add(chunk1_l, d);
			STORE_CHUNK(j1, chunk2_stored, 2);
		}

		chunk3_stored = v_add(chunk2_l, a_later);
		STORE_CHUNK(j1, chunk3_stored, 3);

		long_state[j1] = a_stor;
		// PRIO(0); ////// normal spot below~


		d_old = d;
		d = *reinterpret_cast<ulonglong2*>(&c);


		if (SEC_SHIFT < 8) { PRIO(2); }

		pattern = j0 ^ j1;
		if (pattern < 4) {
			switch (pattern) {
			case 1: // Pairwise swap
				x32 = *reinterpret_cast<uint4*>(&chunk1_stored);
				chunk1 = a_stor;

				chunk2 = chunk3_stored;
				chunk3= chunk2_stored;
				break;

			case 2: // Reverse + swap
				x32 = *reinterpret_cast<uint4*>(&chunk2_stored);
				chunk1 = chunk3_stored;

				chunk2 = a_stor;
				chunk3 = chunk1_stored;
				break;

			case 3: // Reverse
				x32 = *reinterpret_cast<uint4*>(&chunk3_stored);
				chunk1 = chunk2_stored;

				chunk2 = chunk1_stored;
				chunk3 = a_stor;
				break;

			case 0: // Id
			default:
				x32 = *reinterpret_cast<uint4*>(&a_stor);
				chunk1 = chunk1_stored;

				chunk2 = chunk2_stored;
				chunk3 = chunk3_stored;
			}
		}
	}

	foo -=  d_old;
	*ctx_a = foo;
}

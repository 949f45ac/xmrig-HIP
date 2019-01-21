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

#define HEAVY 0

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

#undef HEAVY
#define HEAVY 1

template<xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
#ifdef __HCC__
__launch_bounds__( 32 )
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

	ulonglong2 x64 = long_state[j0];

	__syncthreads();
	#pragma unroll 2
	for ( i = start; i < end; ++i )
	{
		#pragma unroll 2
		for ( int x = 0; x < 2; ++x )
		{
			uint4 x32 = make_uint4(x64.x, x64.x >> 32, x64.y, x64.y >> 32);
			if (VARIANT == xmrig::VARIANT_TUBE) { x32 = ~x32; }
			uint4 c = make_uint4(0, 0, 0, 0);

			c.x = ((uint32_t) a.x) ^ (t_fn0(x32.x & 0xff) ^ t_fn1((x32.y >> 8) & 0xff) ^ t_fn2((x32.z >> 16) & 0xff) ^ t_fn3((x32.w >> 24)));
			j1 = SCRATCH_INDEX((c.x & 0x3FFFF0) >> 4);
			uint64_t * adr = reinterpret_cast<uint64_t*>(long_state + j1);
//			uint64_t ldst0, ldst1;
			ulonglong2 ldst_f;

			ASYNC_LOAD(ldst_f.x, ldst_f.y, adr);
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
			j0 = SCRATCH_INDEX((a.x & 0x3FFFF0) >> 4);

			adr = reinterpret_cast<uint64_t*>(long_state + j0);

			int64_t n_;
			int32_t d_;
#ifdef __HCC__
			asm (
				EMIT_LOAD("%0, %2") MFLAGS " \n\t"
				"flat_load_dword %1, %3" MFLAGS " \n\t"
				: "=v"(n_), "=v" (d_)
				: "r" (adr), "r"(adr+1)
				: "memory");
#endif
			PRIO(0)

			FENCE(t1_64)
			a.y += (t1_64 * y2.x);

			a_stor.y = a.y;

			if (VARIANT == xmrig::VARIANT_TUBE) {
				a_stor.y ^= tweak ^ a_stor.x;

			}

			if (VARIANT == xmrig::VARIANT_1) {
				a_stor.y ^= tweak;
			}

#ifdef __HCC__
			AS128(long_state+j1, a_stor);
#else
			long_state[j1] = a_stor;
#endif
			same_adr = j0 == j1;

#ifndef __HCC__
			n_ = adr[0];
			d_ = adr[1];
#endif
			a.y ^= y2.y;

			WAIT_FOR(n_, 1);
			FENCE32(d_);
			PRIO(3);

			int64_t n = same_adr ? (int64_t) a_stor.x : n_;
			int32_t d = same_adr ? (int32_t) a_stor.y : d_;

			int64_t q = fast_div_heavy(n, d | 0x5);

			uint64_t nnn = (VARIANT == xmrig::VARIANT_XHV ? (~d) : d) ^ q;
			j0 = SCRATCH_INDEX((nnn & 0x3FFFF0) >> 4);

			x64.y = long_state[j0].y;

			int64_t nxq = n^q;
#ifdef __HCC__
			asm ("flat_store_dwordx2 %0, %1"	//MFLAGS
				 :
				 : "r" (adr), "v" (nxq) : "memory");
#else
			*adr = nxq;
#endif

			x64.x = long_state[j0].x;

			PRIO(0);
		}
	}

}

#undef HEAVY
#define HEAVY 0

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


template<xmrig::Variant VARIANT, bool MIXED_SHIFT, int SEC_SHIFT>
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

	ulonglong2 foo = make_ulonglong2(55, 77);
	foo -= d;
	foo -= d_old;
	foo.x += division_result;
	foo.x += sqrt_result;

	const uint32_t end = ( ITER >> (VARIANT == xmrig::VARIANT_HALF ? 2 : 1) );

	__syncthreads();
	#pragma unroll 2
	for ( int i = 0; i < end; ++i )
	{
		uint32_t j0 = SCRATCH_INDEX(( a.x & 0x1FFFF0 ) >> 4);
		ulonglong2 chunk1, chunk2, chunk3;
		uint4 x32 = reinterpret_cast<uint4*>(long_state)[j0];
		LOAD_CHUNK(chunk1, j0, 1);
		LOAD_CHUNK(chunk2, j0, 2);
		LOAD_CHUNK(chunk3, j0, 3);

		if (SEC_SHIFT < 8) { PRIO(2); }

		uint4 a4 = make_uint4(a.x, a.x >> 32, a.y, a.y >> 32);
		uint4 c = cn_aes_single_round(sharedMemory, x32, a4);

		uint32_t j1 = SCRATCH_INDEX((c.x & 0x1FFFF0 ) >> 4);

		STORE_CHUNK(j0, v_add(chunk3, d_old), 1);
		STORE_CHUNK(j0, v_add(chunk1, d), 2);
		STORE_CHUNK(j0, v_add(chunk2, a), 3);
		{
			ulonglong2 d_xored = v_xor(d, *reinterpret_cast<ulonglong2*>(&c));
			long_state[j0] = d_xored;
		}

		// ==== LOAD 2 : chunks ====
		LOAD_CHUNK(chunk3, j1, 3);
		ulonglong2 y2 = long_state[j1];
		LOAD_CHUNK(chunk1, j1, 1);
		LOAD_CHUNK(chunk2, j1, 2);

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
		STORE_CHUNK(j1, v_add(chunk3, d_old), 1);
		FENCE32(sqrt_result);
#endif

		y2.x ^= division_result ^ (((uint64_t) sqrt_result) << 32);

		division_result = n_division_result;
		sqrt_result = n_sqrt_result;

		uint64_t hi = UMUL64HI(t1_64, y2.x);
		uint64_t lo = t1_64 * y2.x;

		ulonglong2 result_mul = make_ulonglong2(hi, lo);

		chunk1 = v_xor(chunk1, result_mul);
		result_mul = v_xor(result_mul, chunk2);

		STORE_CHUNK(j1, v_add(chunk1, d), 2);
		STORE_CHUNK(j1, v_add(chunk2, a), 3);

		a = v_add(a, result_mul);

		long_state[j1] = a;
		PRIO(0);

		a = v_xor(a, y2);

		d_old = d;
		d = *reinterpret_cast<ulonglong2*>(&c);
	}

	foo -=  d_old;
	*ctx_a = foo;
}

#undef HEAVY

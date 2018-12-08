// #include <hip/hip_runtime.h>
#pragma once

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 520
/* avoid red underlining */

struct uint3
{
	unsigned int x, y, z;
};

struct uint3  threadIdx;
struct uint3  blockIdx;
struct uint3  blockDim;
#define __funnelshift_r(a,b,c) 1
#define __syncthreads()
#define asm(x)
#define __shfl(a,b,c) 1
#endif

#define MEMORY         (1 << 21) // 2 MiB / 2097152 B
#define MEMORY_LITE    (1 << 20) // 1048576
#define ITER           (1 << 20) // 1048576
#define AES_BLOCK_SIZE  16
#define AES_KEY_SIZE    32
#define INIT_SIZE_BLK   8
#define INIT_SIZE_BYTE (INIT_SIZE_BLK * AES_BLOCK_SIZE) // 128 B

#define C32(x)    ((uint32_t)(x ## U))
#define T32(x) ((x) & C32(0xFFFFFFFF))

#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t cuda_ROTL64(const uint64_t value, const int offset)
{
	uint2 result;
	if(offset >= 32)
	{
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} 
	else 
	{
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#define ROTL64(x, n) (cuda_ROTL64(x, n))
#else
#define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif

#if __CUDA_ARCH__ < 350
#define ROTL32(x, n) T32(((x) << (n)) | ((x) >> (32 - (n))))
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
#define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
#define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif

#define MEMSET8(dst,what,cnt) { \
	int i_memset8; \
	uint64_t *out_memset8 = (uint64_t *)(dst); \
	for( i_memset8 = 0; i_memset8 < cnt; i_memset8++ ) \
		out_memset8[i_memset8] = (what); }

#define MEMSET4(dst,what,cnt) { \
	int i_memset4; \
	uint32_t *out_memset4 = (uint32_t *)(dst); \
	for( i_memset4 = 0; i_memset4 < cnt; i_memset4++ ) \
		out_memset4[i_memset4] = (what); }

#define MEMCPY8(dst,src,cnt) { \
	int i_memcpy8; \
	uint64_t *in_memcpy8 = (uint64_t *)(src); \
	uint64_t *out_memcpy8 = (uint64_t *)(dst); \
	for( i_memcpy8 = 0; i_memcpy8 < cnt; i_memcpy8++ ) \
		out_memcpy8[i_memcpy8] = in_memcpy8[i_memcpy8]; }

#define MEMCPY4(dst,src,cnt) { \
	int i_memcpy4; \
	uint32_t *in_memcpy4 = (uint32_t *)(src); \
	uint32_t *out_memcpy4 = (uint32_t *)(dst); \
	for( i_memcpy4 = 0; i_memcpy4 < cnt; i_memcpy4++ ) \
		out_memcpy4[i_memcpy4] = in_memcpy4[i_memcpy4]; }

#define XOR_BLOCKS(a,b) { \
	((uint64_t *)a)[0] ^= ((uint64_t *)b)[0]; \
	((uint64_t *)a)[1] ^= ((uint64_t *)b)[1]; }

#define XOR_BLOCKS_DST(x,y,z) { \
	((uint64_t *)z)[0] = ((uint64_t *)(x))[0] ^ ((uint64_t *)(y))[0]; \
	((uint64_t *)z)[1] = ((uint64_t *)(x))[1] ^ ((uint64_t *)(y))[1]; }

#define MUL_SUM_XOR_DST(a,c,dst) { \
	const uint64_t dst0 = ((uint64_t *)dst)[0]; \
	uint64_t hi, lo = cuda_mul128(((uint64_t *)a)[0], dst0, &hi) + ((uint64_t *)c)[1]; \
	hi += ((uint64_t *)c)[0]; \
	((uint64_t *)c)[0] = dst0 ^ hi; \
	((uint64_t *)dst)[0] = hi; \
	((uint64_t *)c)[1] = atomicExch(((unsigned long long int *)dst) + 1, (unsigned long long int)lo) ^ lo; \
	}

#define E2I(x) ((size_t)(((*((uint64_t*)(x)) >> 4) & 0x1ffff)))


// ASM required to wedge operations on variables loaded from memory into the correct spot.
#ifdef __HCC__
#define RETIRE(x)  { \
        uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
        asm volatile("v_mov_b32 %0, %1" : "=v" (w[0]), "=v" (w[1]) : "v" (w[0]), "v" (w[1]) : "memory"); }

#define FENCE(x) { \
	uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
	asm volatile("v_mov_b32 %0, %2\n\tv_mov_b32 %1, %3" : "=v" (w[0]), "=v" (w[1]) : "v" (w[0]), "v" (w[1]) : "memory"); }

#define FENCE32(x) asm volatile("v_mov_b32 %0, %1\n\t" : "=v" (x) : "v" (x) : "memory");

#define WAIT_FOR(x, n) asm volatile("s_waitcnt vmcnt(" #n ")\n\t": : "v" (x) : "memory"); FENCE(x)
#define PRIO(n) asm volatile ("s_setprio 0x" #n ::: "memory");
#else
#define WAIT_FOR(x, n) FENCE(x)
#define RETIRE(x) ;
#define PRIO(n) ;
#define FENCE(w) { \
	asm volatile("mov.u64 %0, %1;\n\t" : "=l" (w) : "l" (w) : "memory"); }

#define FENCE32(w) { \
	asm volatile("mov.u32 %0, %1;\n\t" : "=l" (w) : "l" (w) : "memory"); }
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
	const uint nshift_limit = (threads >> SEC_SHIFT) << MIXED_SHIFT_DOWNDRAFT; \
	const bool over = MIXED_SHIFT && hipBlockIdx_x >= nshift_limit;		\
	const uint concrete_shift = SEC_SHIFT - (over ? MIXED_SHIFT_DOWNDRAFT : 0); \
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

#define HEAVY (VARIANT == xmrig::VARIANT_XHV)
#define ALGOMEM (HEAVY ? (MEMORY/8) : (MEMORY/16))

#define BASE_OFF(thread, threads) (thread / sec_size0) * sec_size0 * ALGOMEM + over * ((thread-1792) / sec_size1) * sec_size1 * ALGOMEM + (thread % sec_size1) * CHU;

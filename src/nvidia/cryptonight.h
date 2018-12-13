
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__ 1
#endif

#include <hip/hip_runtime.h>
#pragma once

#include <stdint.h>
#include "../common/xmrig.h"

#define VEGA_SHIFT (8)
#define LARGE_POLARIS_SHIFT (3)
#define SMALL_POLARIS_SHIFT (6)

#define MIXED_SHIFT_DOWNDRAFT (3)

typedef struct {
	int device_id;
	const char *device_name;
	int device_arch[2];
	int device_mpcount;
	int device_blocks;
	int device_threads;
	bool autolower;
	bool mixed_shift;
	int device_bfactor;
	int device_bsleep;
	int device_clockRate;
	int device_memoryClockRate;
    uint32_t device_pciBusID;
    uint32_t device_pciDeviceID;
    uint32_t device_pciDomainID;
    uint32_t syncMode;

	size_t overall_wsize_on_card = 0;
	size_t w_off = 0;

	hipStream_t stream;

	uint32_t *d_input;
	uint32_t inputlen;
	uint32_t *d_result_count;
	uint32_t *d_result_nonce;
	uint64_t *d_long_state;
	uint32_t *d_ctx_state;
	uint32_t *d_ctx_state_p1;
	uint32_t *d_ctx_a;
	uint32_t *d_ctx_b;
	uint32_t *d_ctx_key1;
	uint32_t *d_ctx_key2;
	uint32_t *d_ctx_text;
} nvid_ctx;

extern "C" {

/** get device count
 *
 * @param deviceCount[out] cuda device count
 * @return error code: 0 == error is occurred, 1 == no error
 */
int cuda_get_devicecount();
int cuda_get_deviceinfo(nvid_ctx *ctx, xmrig::Algo algo);
int cryptonight_gpu_init(nvid_ctx *ctx, xmrig::Algo algo);
/* int cryptonight_extra_cpu_init(nvid_ctx *ctx); */
int cryptonight_extra_cpu_set_gpu(nvid_ctx* ctx);
void cryptonight_extra_cpu_set_data(nvid_ctx* ctx, const void *data, uint32_t len);
void cryptonight_extra_cpu_prepare(nvid_ctx* ctx, uint32_t startNonce, bool heavy);
//void cryptonight_core_cpu_hash(nvid_ctx* ctx, uint32_t startNonce);
void cryptonight_gpu_hash(nvid_ctx *ctx, xmrig::Algo algo, xmrig::Variant variant, uint32_t startNonce);
void cryptonight_extra_cpu_final(nvid_ctx* ctx, uint32_t startNonce, uint64_t target, uint32_t* rescount, uint32_t *resnonce, bool heavy);
}

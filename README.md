# XMRig HIP

**Why use this?** Can be a bit faster than OpenCL and CUDA miners, YMMV.

HIP CryptoNight miner based on XMRig. For more info on HIP and the approach of this miner check out the original `xmr-stak-hip` project: https://github.com/949f45ac/xmr-stak-hip

This is a linux-only miner. Theoretically it does support AMD as well as nvidia cards, but nvidia cards are currently untested again.

Supported algos:
- CN/1 aka Monero7
- MSR
- XTL
- automatic algo switching

## Setup for High Vega Hashrate on Linux

Expert mode: Instead of Ubuntu take whatever distro you like, the stuff below is just to give you an idea.

### 4.18+ kernel and ROCm
- Install Ubuntu 18.10 beta, it comes with a 4.18 kernel
- Install ROCm without dkms:
Follow this guide but stop when it wants you to install `rocm-dkms`:
https://github.com/RadeonOpenCompute/ROCm/#ubuntu-support---installing-from-a-debian-repository

- Then instead do this:
`sudo apt install rocm-opencl rocm-clang-ocl rocminfo rocm-smi rocm-utils hip_hcc`

- Don't forget to add kdf rule: `echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules`

- Reboot

### 4.15 or older kernel and amdgpu-pro and ROCm
- Install Ubuntu 18.04 or 16.04
- Install ROCm without dkms:
Follow this guide but stop when it wants you to install `rocm-dkms`:
https://github.com/RadeonOpenCompute/ROCm/#ubuntu-support---installing-from-a-debian-repository

- Then instead do this:
`sudo apt install rocm-opencl rocm-clang-ocl rocminfo rocm-smi rocm-utils hip_hcc`
- Install amdgpu-pro with `--opencl=pal --headless` options, make sure its dkms module gets installed for your kernel
- Reboot

## Building the miner
- Clone this repo, `mkdir build`
- `cd build`
- `cmake .. -DCUDA_COMPILER=/opt/rocm/bin/hipcc -DHIP_PLATFORM=hcc -DHIP_ROOT_DIR=/opt/rocm/hip`
- `make -j4`

### How do I choose threads and blocks?
There’s a guide in the xmr-stak-hip repo
tl;dr
- Vega 8 GB: t = 8, b = 448
- Vega 16 GB: t = 16, b = 512
- RX 470/570+: t = 4, b = 480 (when 4 GB) or 960
- RX 460 etc. junk cards use t=8 again and b=448 or 224 (2 GB)

**Do not use bsleep / bfactor I broke them.**

Example config:

```json
{
    "algo": "cryptonight",
    "background": false,
    "colors": true,
    "donate-level": 1,
    "log-file": null,
    "print-time": 20,
    "retries": 5,
    "retry-pause": 5,
    "syslog": false,
    "threads": [
        {    // Vega 56 / 64
            "index": 0,
            "threads": 8,
            "blocks": 448,
            "bfactor": 0,
            "bsleep": 0,
            "sync_mode": 3
        },
        {    // RX 570
            "index": 1,
            "threads": 4,
            "blocks": 480,
            "bfactor": 0,
            "bsleep": 0,
            "sync_mode": 3
	}
    ],
    "pools": [
        {
            "url": "gulf.moneroocean.stream:10032",
            "user": "45FbpewbfJf6wp7gkwAqtwNc7wqnpEeJdUH2QRgeLPhZ1Chhi2qs4sNQKJX4Ek2jm946zmyBYnH6SFVCdL5aMjqRHodYYsF",
            "pass": "x",
            "keepalive": true,
            "nicehash": false
        }
    ]
}

```

## Automatic donations still go to original XMRig authors
Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via command line option `--donate-level`.

## If you want to donate to me (949f45ac) who did HIP port + optimization

* XMR: `45FbpewbfJf6wp7gkwAqtwNc7wqnpEeJdUH2QRgeLPhZ1Chhi2qs4sNQKJX4Ek2jm946zmyBYnH6SFVCdL5aMjqRHodYYsF`
* BTC: `181TVrHPjeVZuKdqEsz8n9maqFLJAzTLc`

## How do I overclock?

No idea, look at this: https://github.com/RadeonOpenCompute/ROCm/issues/463

## Command line options (unchanged)
```
  -a, --algo=ALGO           cryptonight (default) or cryptonight-lite
  -o, --url=URL             URL of mining server
  -O, --userpass=U:P        username:password pair for mining server
  -u, --user=USERNAME       username for mining server
  -p, --pass=PASSWORD       password for mining server
  -k, --keepalive           send keepalived for prevent timeout (need pool support)
  -r, --retries=N           number of times to retry before switch to backup server (default: 5)
  -R, --retry-pause=N       time to pause between retries (default: 5)
      --cuda-devices=N      List of CUDA devices to use.
      --cuda-launch=TxB     List of launch config for the CryptoNight kernel
      --cuda-max-threads=N  limit maximum count of GPU threads in automatic mode
      --cuda-bfactor=[0-12] run CryptoNight core kernel in smaller pieces
      --cuda-bsleep=N       insert a delay of N microseconds between kernel launches
      --cuda-affinity=N     affine GPU threads to a CPU
      --no-color            disable colored output
      --donate-level=N      donate level, default 5% (5 minutes in 100 minutes)
      --user-agent          set custom user-agent string for pool
  -B, --background          run the miner in the background
  -c, --config=FILE         load a JSON-format configuration file
  -l, --log-file=FILE       log all output to a file
  -S, --syslog              use system log for output messages
      --nicehash            enable nicehash support
      --print-time=N        print hashrate report every N seconds
      --api-port=N          port for the miner API
      --api-access-token=T  access token for API
      --api-worker-id=ID    custom worker-id for API
  -h, --help                display this help and exit
  -V, --version             output version information and exit
```

**Please double your threads and halven your blocks if you are migrating an existing HIP miner config from CN7.**
**For best performance, please use amdgpu-pro dkms driver.**

# XMRig HIP

A Linux CryptoNight GPU miner built on the HIP framework.

Features:
- Fast and stable mining with Vega cards, CN/2 1600+ H/s, CN/1 1700+ H/s
- Large Polaris cards (_70, _80) mine very fast, up to 10% faster than on OpenCL (both Cn7 and Cn8)
- Small Polaris cards (_50, _60) can also be mined on; Cn8 almost as fast as Cn7
- Multi algo support for Cn7/Cn8, meaning you can mine on MoneroOcean
- Configuration is automatically adjusted for optimal speed when auto-switching between Cn7 and Cn8
- Mine on a full open source stack, aside from parts of the amdgpu-pro driver

Caveat emptor
- Vega Cn7 performance in HIP is often a bit worse than OpenCL Cn7 performance
- Polaris cards need to run in a true PCIe 3.0 x8 or x16 slot – no risers!
- Nvidia cards are theoretically supported, but that needs more testing

## Setup for High Vega Hashrate on Linux

Currently the miner gets best hashrate when run in concert with
amdgpu-pro.

Hence you’re best served using a 16.04 or 18.04 Ubuntu with stock kernel.

- Install Ubuntu 18.04 or 16.04
- Install ROCm without dkms:
Follow this guide but stop when it wants you to install `rocm-dkms`:
https://github.com/RadeonOpenCompute/ROCm/#ubuntu-support---installing-from-a-debian-repository

- Then instead do this: `sudo apt install rocminfo rocm-smi rocm-utils hip_hcc`
- Add your user to video group `sudo usermod -a -G video $LOGNAME`
- Install amdgpu-pro with `--opencl=pal --headless` options, make sure its dkms module gets installed for your kernel
- Reboot

## Building the miner
- Install some build deps `sudo apt install cmake libuv1-dev libssl-dev`
- Clone this repo, `mkdir build`
- `cd build`
- `cmake .. -DCUDA_COMPILER=/opt/rocm/bin/hipcc -DHIP_PLATFORM=hcc -DHIP_ROOT_DIR=/opt/rocm/hip -DWITH_HTTPD=OFF`
- `make -j4`
- If it says "file format not recognized" in the end (while linking) just `make` once more
- Now copy `src/config.json` to your directory and configure GPUs like explained below.

### How do I choose threads and blocks?
tl;dr

Find card numbers (to specify as `"index": ` in the json) by running `/opt/rocm/bin/rocm-smi`
Use the following threads/blocks depending on card:

- Vega 8 GB: Threads = 16, Blocks = 224 (4 * 56) or 192 (3 * 64)
- Vega 16 GB: Threads = 16, Blocks = 512
- Polaris _70 or _80: Threads = 8, Blocks: Try 216, 224, 240, 252, or double that if the card has 8 GB memory.
- Polaris _50 or _60: Threads = 16, Blocks: Try 56, 64 for 2 GB, or double that if the card has 4 GB memory.

Technical note: When mining a Cn7 algorithm (msr or xtl), the miner will automatically use half the
threads and double the blocks.

More detailed explanation:

You want 16 threads on most cards, and a number of blocks that is
an integer multiple of CU count, and `T x B x 2 < (Memory in MB)`.

E.g.:
Vega 56 has 56 CU.

56 x 4 = 224

224 x 16 x 2 = 7168

7168 < 8000


In some cases taking another 0.5 * CU blocks (like blocks=6.5*CU overall) will increase speed, in
most cases it will not.

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
            "threads": 16,
            "blocks": 224,
            "bfactor": 0,
            "bsleep": 0,
            "sync_mode": 3
        },
        {    // RX 570
            "index": 1,
            "threads": 8,
            "blocks": 240,
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

## If you want to donate to me (949f45ac) who did HIP port + optimization
* XMR: `45FbpewbfJf6wp7gkwAqtwNc7wqnpEeJdUH2QRgeLPhZ1Chhi2qs4sNQKJX4Ek2jm946zmyBYnH6SFVCdL5aMjqRHodYYsF`
* BTC: `181TVrHPjeVZuKdqEsz8n9maqFLJAzTLc`

## Automatic donations still go to original XMRig authors
Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via command line option `--donate-level`.

## How do I overclock?
Use soft pp tables.
Check out this guide, for example: https://github.com/xmrminer01102018/VegaToolsNConfigs

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

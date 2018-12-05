**For best performance, please use amdgpu-pro dkms driver.**

**Crank up your sclk (core clock) for fast hashrates! Use threads=32 on Vega to save power!**

# XMRig HIP

A Linux CryptoNight GPU miner built on the HIP framework.

Features:
- Fast and stable mining with Vega cards! 1800+ CN/2, 2000+ CN/1+xtl+msr
- Large Polaris cards (_70, _80) mine very fast, up to 10% faster than on OpenCL (both Cn7 and Cn8)
- Small Polaris cards (_50, _60) can also be mined on; Cn8 faster than Cn7!
- Multi algo support for Cn7/Cn8, meaning you can mine on MoneroOcean

Caveat emptor
- Polaris cards need to run in a true PCIe 3.0 x8 or x16 slot – no risers!
- Nvidia cards are theoretically supported, but that needs more testing

## Setup for High Vega Hashrate on Linux

*Currently the miner is fast ONLY when run with amdgpu-pro dkms driver!*

Hence you’re best served using a 16.04 or 18.04 Ubuntu with stock kernel.

- Install Ubuntu 18.04 or 16.04
- Install ROCm without dkms:
Follow this guide but *stop* when it wants you to install `rocm-dkms`:
https://github.com/RadeonOpenCompute/ROCm/#ubuntu-support---installing-from-a-debian-repository

- Then instead do this: `sudo apt install rocminfo rocm-smi rocm-utils hip_hcc`
- Add your user to video group `sudo usermod -a -G video $LOGNAME`
- Install [amdgpu-pro](https://www.amd.com/en/support/kb/release-notes/rn-prorad-lin-18-30) with `--opencl=pal --headless` options, make sure its dkms module gets installed for your kernel
- Reboot

## Building the miner
```bash
# Install some build deps
sudo apt install cmake libuv1-dev libssl-dev
mkdir build
cd build
cmake .. -DCUDA_COMPILER=/opt/rocm/bin/hipcc -DHIP_PLATFORM=hcc -DHIP_ROOT_DIR=/opt/rocm/hip -DWITH_HTTPD=OFF

# First invocation of linker tends to fail
make -j4 && make -j4

# Copy config template to folder and add pool credentials
cp ../src/config.json .

# Start miner
./xmrig-hip

```

### How do I choose threads and blocks?
Miner has autoconfig to do this for you, but here’s some info still:

Find card numbers (to specify as `"index": ` in the json) by running `/opt/rocm/bin/rocm-smi`

Use the following threads/blocks depending on card.

- Vega 56: Threads = 32, Blocks = 112
- Vega 64: T=32 B=124. If you want maximal hashrate: T=16, B=248
- Vega FE: T=32 B=128, or even more blocks.
- Polaris _70 or _80: Threads = 8, Blocks: Try 248, 252 for 4 GB, double that for 8 GB.
- Polaris _50 or _60: Threads = 64, Blocks: Number of Compute Units.


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
        {    // Vega 56
            "index": 0,
            "threads": 32,
            "blocks": 112,
            "bfactor": 0,
            "bsleep": 0,
            "sync_mode": 3
        },
        {    // RX 570
            "index": 1,
            "threads": 8,
            "blocks": 248,
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

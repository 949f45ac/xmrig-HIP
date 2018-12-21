**You must use amdgpu-pro dkms driver for good performance.**

**Crank up your sclk (core clock) for fast hashrates! Power consumption stays fine with Threads=32 on Vega!**

# XMRig HIP

A Linux CryptoNight GPU miner built on the HIP framework.

Features:
- Vega 64: 2000 H/s and up on CN/2. 2100+ on CN/1 variants
- Vega 56: 1820+ H/s on CN/2 at least. 2000+ H/s on CN/1 variants.
- Large Polaris cards (_70, _80) mine very fast, up to 10% faster than on OpenCL.
- Small Polaris cards (_50, _60) are also quite fast, losing no speed on CN/2 compared to CN/1.
- Automatic algo switching (for mining on MoneroOcean)

Caveat emptor
- Polaris cards need to run in a true PCIe 3.0 x8 or x16 slot – no
  risers!
- Dual thread setups on anything but Vega 64 are quite unstable
  as of yet. Donate job or algo switching will screw Vega 56 hashrates
  until miner is restarted.
- Nvidia cards are theoretically supported, but that needs more
  testing

## Setup for High Vega Hashrate on Linux

*Currently the miner is fast ONLY when run with amdgpu-pro dkms driver!*

Hence you’re best served using a 16.04 or 18.04 Ubuntu with stock kernel.

- Install Ubuntu 18.04 or 16.04
- Install ROCm without dkms: *lol ROCm 2.0 is broken, see if you can install 1.9*
```bash
# Make sure you are generally up to date
sudo apt update
sudo apt dist-upgrade
sudo apt install libnuma-dev
sudo reboot

# Add AMD ROCm repo
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install HIP
sudo apt update
sudo apt install rocminfo rocm-smi rocm-utils hip_hcc

# Add your user to group "video"
sudo usermod -a -G video $LOGNAME

# Reboot
sudo reboot
```
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
Miner has autoconfig to do this for you, but here’s how to reach even higher hashrate by using multiple workloads per GPU:

Find card numbers (to specify as `"index": ` in the json) by running `/opt/rocm/bin/rocm-smi`

Use the following threads/blocks depending on card.

- Vega FE: threads=32, blocks=64 + threads=32, blocks=64
- Vega 64: threads=32, blocks=64 + threads=32, blocks=60
- Vega 56: threads=32, blocks=56 + threads=32, blocks=56
- Polaris 8GB: threads=8, blocks=128 + threads=8, blocks=128
- Polaris 4GB: threads=8, blocks=124 + threads=8, blocks=124
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
        // Vega 64 dual threads
        {
            "index": 0,
            "threads": 32,
            "blocks": 64,
            "bfactor": 0,
            "bsleep": 0,
            "sync_mode": 3
        },
        {
            "index": 0,
            "threads": 32,
            "blocks": 60,
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

# Optimizations
- Support two "threads" per GPU, but allocate all the large chunks of GPU memory as single blocks. (Second thread simply adds an offset to the pointers, based on first threads’ work item count.) If both threads allocate their memory blocks at the same time, there is a chance that this happens "naturally"  (calls are interleaved exactly so as to yield the same layout), but doing it programatically gives 100% guarantee it happens.

- On RX Vega, it seems that the tasks of the second thread should be scheduled with a small delay. On Polaris, though, it seems they should rather be scheduled at the exact same time. I have some crude logic for achieving these in xmrig-HIP, but I figure the new interleaving logic in xmr-stak would do a better job.

- In scratchpad striding, use chunk size of exactly 16 x 128 bit. This also means that to get the other 3 memory locations, one can always XOR the base address *after* computing the stride-modified actual address.

- On Vega, stride scratchpads in groups of 256 threads, as far as possible. (Meaning scratchpads stride through 512 MB of memory.) OpenCL miner currently always uses groups of size equal to work size – that seems to be ideal only on Polaris cards.

- It can still pay off to use more memory, so that the overall work item count is not divisible by 256. The remainder then is best divided into groups of 32.

- Using a non-hardcoded (variable) striding group size comes with a performance hit – since we run two threads, we only have to do it on one. E.g. 2048 work items on first thread, 1920 on second thread means the first can run with hardcoded 256.

## If you want to donate to me (949f45ac) who did HIP port + optimization
* XMR: `45FbpewbfJf6wp7gkwAqtwNc7wqnpEeJdUH2QRgeLPhZ1Chhi2qs4sNQKJX4Ek2jm946zmyBYnH6SFVCdL5aMjqRHodYYsF`
* BTC: `181TVrHPjeVZuKdqEsz8n9maqFLJAzTLc`

## Automatic donations still go to original XMRig authors
Default donation 5% (5 minutes in 100 minutes) can be reduced to 0% via command line option `--donate-level`.

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

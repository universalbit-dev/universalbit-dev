##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)
BlockChain |  | # Stack
---|---|---
 [NetNode](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain/bitcoin) |  | 
 [Buy-Sell](https://github.com/universalbit-dev/gekko-m4-globular-cluster/blob/master/README.md) |  | 
 [Mining](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain) |  | 

<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/images/blockchain.png" width="5%"></img>

---

## Actively participate in blockchain mining
### It does not require large investments and is useful in a blockchain environment. 

Simplemining Os :
Offers an advanced dashboard for CPU/GPU/ASIC management at a negligible monthly cost. 
[SimpleMining.net](https://simplemining.net/)
Make mining more accessible,advantages and disadvantages are being evaluated.

Hardware:
* GPU (GigaByte WindForce R9290) -- no supported in kfd  -- [ROCR EXPERIMENTS](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/what-is-rocr-runtime.html)
* ATX 620W Cooler Master 80Plus (green connector pci-e)
* ASRock Q1900M MOBO BIOS P2.00 12/20/2018
* CPU Intel(R) Celeron(R) CPU J1900 @ 1.99GHz
* RAM 4 GB DDR3
* KingSton DataTraveler 64GB (USB 3.0)

Software:
* [SimpleMining](https://simplemining.net/) Operative System ,[Pricing](https://simplemining.net/pricing)
* Writing Image to KingSton DataTraveler 64GB and Resize Partition using the largest available space (57GB)
* Update && Upgrade Ubuntu Repository
* Opencl Headers && Mesa Driver
* Vulkan-SDK 
* AMD Driver 22.40

#### [ -- PAUSE -- your Rig from SimpleMining DashBoard ]

### Ubuntu Archive (FocalFossa) Repository

```bash
sudo nano /etc/apt/sources.list
```

```bash
deb http://archive.ubuntu.com/ubuntu focal main restricted
deb http://archive.ubuntu.com/ubuntu focal-updates main restricted
deb http://archive.ubuntu.com/ubuntu focal universe
deb http://archive.ubuntu.com/ubuntu focal-updates universe
deb http://archive.ubuntu.com/ubuntu focal multiverse
deb http://archive.ubuntu.com/ubuntu focal-updates multiverse
deb http://archive.ubuntu.com/ubuntu focal-backports main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu focal-security main restricted
deb http://archive.ubuntu.com/ubuntu focal-security universe
deb http://archive.ubuntu.com/ubuntu focal-security multiverse
```

* save edited sources.list file (CTRL+SAVE)

### Update and Upgrade Operative System (from archive.ubuntu.com)

```bash
sudo apt update && sudo apt upgrade
```

### [OpenCL Headers](https://cn.khronos.org/opencl/),mesa,vulkan and microcode packages:

```bash
sudo apt-get -y install ocl-icd-opencl-dev opencl-headers mesa-common-dev mesa-opencl-icd mesa-utils-extra clinfo libvulkan1 mesa-vulkan-drivers vulkan-utils amd64-microcode intel-microcode iucode-tool thermald gdebi-core
```

### [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)

```bash
sudo wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
&& sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.2.189-focal.list https://packages.lunarg.com/vulkan/1.2.189/lunarg-vulkan-1.2.189-focal.list
&& sudo apt update
&& sudo apt install vulkan-sdk
```

### [Development and Tools packages](https://github.com/universalbit-dev/AMDVLK?tab=readme-ov-file)
```bash
sudo apt-get install libssl-dev libx11-dev libxcb1-dev x11proto-dri2-dev libxcb-dri3-dev libxcb-dri2-0-dev libxcb-present-dev libxshmfence-dev libxrandr-dev libwayland-dev
```

### Install AMD Driver 22.40  (Ubuntu 20.04) [Release Notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-22-40-6.html)
```bash
sudo wget https://repo.radeon.com/amdgpu-install/22.40.6/ubuntu/focal/amdgpu-install_5.4.50406-1_all.deb
sudo dpkg --add-architecture i386
amdgpu-install --opencl=rocr,legacy --vulkan=amdvlk,pro
```

* After this operation, > 2.8GB of additional disk space will be used. 


---
##### [ -- EXPERIMENTAL Installation of AMD Driver and DownGrade OR UpGrade Kernel version -- ]
##### [ -- WRONG Combination of Kernel version and AMD Driver could makes the operating system unusable -- ]
### Ubuntu Tools :
* Upgrade or Downgrade Kernel
### [ubuntu-mainline-kernel](https://github.com/pimlie/ubuntu-mainline-kernel.sh/blob/617171ebea0a506d57659f43bc07fb591e3c4a56/ubuntu-mainline-kernel.sh#L4) 
bash script help to change kernel version (easy way)

```bash
sudo wget https://raw.githubusercontent.com/pimlie/ubuntu-mainline-kernel.sh/master/ubuntu-mainline-kernel.sh && chmod +x ubuntu-mainline-kernel.sh && sudo mv ubuntu-mainline-kernel.sh /usr/local/bin/
```

* Find Kernel Version

```bash
sudo ubuntu-mainline-kernel.sh -r
```

* Install Kernel Version v6.2.0 (default kernel 6.1.57-sm4) 

```bash
#sudo ubuntu-mainline-kernel.sh -i <<kernel version to install>>
sudo ubuntu-mainline-kernel.sh -i v6.2.0
```

```bash
Downloading amd64/linux-headers-6.2.0-060200-generic_6.2.0-060200.202302191831_amd64.deb: 100%
Downloading amd64/linux-headers-6.2.0-060200_6.2.0-060200.202302191831_all.deb: 100%
Downloading amd64/linux-image-unsigned-6.2.0-060200-generic_6.2.0-060200.202302191831_amd64.deb: 100%
Downloading amd64/linux-modules-6.2.0-060200-generic_6.2.0-060200.202302191831_amd64.deb: 100%
```

### Edit Grub Bootloader Display MenuEntry 
```bash
sudo grub-mkconfig | grep -iE "menuentry 'Ubuntu, with Linux" | awk '{print i++ " : "$1, $2, $3, $4, $5, $6, $7}'
```
### Resources:
##### [CPU Mining](https://github.com/universalbit-dev/CityGenerator/blob/master/workers/workers.md)
##### [ESP32 Mining](https://github.com/BitMaker-hub/NerdMiner_v2)
##### [Release Note AMD Driver 22.40](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-22-40-6.html)
##### [AMDGPU Mining Ubuntu Server](https://github.com/michalslonina/Ubuntu-Server-LTS-20.04-GUIDE-AMDGPU-Mining)
##### [Bypass PCIE 3.0 atomics limitation](https://www.reddit.com/r/gpumining/comments/ptmyjd/ubuntu_20043_amdgpu_2130_opencl_rocr_rocm/)
##### [How Bitcoin Mining Really Works](https://www.freecodecamp.org/news/how-bitcoin-mining-really-works-38563ec38c87/)
##### [Web3](https://web3.freecodecamp.org/web3)
##### [MultiArchitecture](https://wiki.debian.org/Multiarch/HOWTO)

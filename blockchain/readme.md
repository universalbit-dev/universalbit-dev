# Blockchain Mining Guide

Welcome to the Blockchain Mining Guide! This document provides detailed instructions on setting up and participating in blockchain mining, including hardware and software requirements, configuration steps, and additional resources.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Support and Links](#support-and-links)
3. [Stack Overview](#stack-overview)
4. [Mining Setup](#mining-setup)
   - [Hardware Requirements](#hardware-requirements)
   - [Software Requirements](#software-requirements)
5. [Step-by-Step Guides](#step-by-step-guides)
   - [Configuring Repositories](#configuring-repositories)
   - [System Update and Upgrade](#system-update-and-upgrade)
   - [Installing Dependencies](#installing-dependencies)
6. [Warnings and Experimental Sections](#warnings-and-experimental-sections)
7. [Resources](#resources)

---

## Introduction

Blockchain mining allows you to actively participate in the blockchain environment without requiring large investments. This guide walks you through setting up a mining rig, from hardware to software, and provides useful links for further learning.

---

## Support and Links

- **[Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)**
- **[Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)**

---

## Stack Overview

| Component | Description | Links |
|-----------|-------------|-------|
| [NetNode](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain/bitcoin) | Blockchain node implementation | [Details](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain/bitcoin) |
| [Buy-Sell](https://github.com/universalbit-dev/gekko-m4-globular-cluster/blob/master/README.md) | Marketplace integration | [Details](https://github.com/universalbit-dev/gekko-m4-globular-cluster/blob/master/README.md) |
| [CPU Mining](https://github.com/universalbit-dev/CityGenerator/blob/master/workers/workers.md) | CPU-based mining setup | [Details](https://github.com/universalbit-dev/CityGenerator/blob/master/workers/workers.md) |
| [ESP32 MicroMiner](https://github.com/universalbit-dev/esptool) | MicroMiner implementation using ESP32 | [Details](https://github.com/universalbit-dev/esptool) |

---

## Mining Setup

### Hardware Requirements
- **GPU**: GigaByte WindForce R9290 (not supported in kfd) [ROCR Experiments](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/what-is-rocr-runtime.html)
- **Power Supply**: ATX 620W Cooler Master 80Plus
- **Motherboard**: ASRock Q1900M (BIOS P2.00 12/20/2018)
- **CPU**: Intel(R) Celeron(R) J1900 @ 1.99GHz
- **RAM**: 4GB DDR3
- **Storage**: Kingston DataTraveler 64GB (USB 3.0)

### Software Requirements
- **OS**: [SimpleMining](https://simplemining.net/) ([Pricing](https://simplemining.net/pricing))
- **Drivers**: AMD Driver 22.40
- **Tools**: OpenCL headers, Mesa driver, Vulkan-SDK

---

## Step-by-Step Guides

### Configuring Repositories
Edit the `sources.list` file to include necessary repositories:
```bash
sudo nano /etc/apt/sources.list
```
Add the following lines:
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
Save the file with `CTRL+O` and exit with `CTRL+X`.

### System Update and Upgrade
Run the following command to update and upgrade the operating system:
```bash
sudo apt update && sudo apt upgrade
```

### Installing Dependencies
Install OpenCL headers, Mesa drivers, Vulkan SDK, and other necessary tools:
```bash
sudo apt-get -y install ocl-icd-opencl-dev opencl-headers mesa-common-dev mesa-opencl-icd mesa-utils-extra clinfo libvulkan1 mesa-vulkan-drivers vulkan-utils amd64-microcode intel-microcode iucode-tool
```

Install the Vulkan SDK:
```bash
sudo wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.2.189-focal.list https://packages.lunarg.com/vulkan/1.2.189/lunarg-vulkan-1.2.189-focal.list
sudo apt update
sudo apt install vulkan-sdk
```

Install AMD Driver 22.40:
```bash
sudo wget https://repo.radeon.com/amdgpu-install/22.40.6/ubuntu/focal/amdgpu-install_5.4.50406-1_all.deb
sudo dpkg --add-architecture i386
amdgpu-install --opencl=rocr,legacy --vulkan=amdvlk,pro
```

---

## Warnings and Experimental Sections

### Experimental AMD Driver Installation
**Warning**: A wrong combination of kernel version and AMD Driver can make the operating system unusable. Proceed with caution.

Use the following tool to upgrade or downgrade the kernel:
```bash
sudo wget https://raw.githubusercontent.com/pimlie/ubuntu-mainline-kernel.sh/master/ubuntu-mainline-kernel.sh
chmod +x ubuntu-mainline-kernel.sh
sudo mv ubuntu-mainline-kernel.sh /usr/local/bin/
```

Find the available kernel versions:
```bash
sudo ubuntu-mainline-kernel.sh -r
```

---

## Resources

- [How Bitcoin Mining Really Works](https://www.freecodecamp.org/news/how-bitcoin-mining-really-works-38563ec38c87/)
- [Web3 Overview](https://web3.freecodecamp.org/web3)
- [Release Notes for AMD Driver 22.40](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-22-40-6.html)
- [Bypass PCIe 3.0 Atomics Limitation](https://www.reddit.com/r/gpumining/comments/ptmyjd/ubuntu_20043_amdgpu_2130_opencl_rocr_rocm/)
- [Ubuntu Archive](https://releases.ubuntu.com/)
- [MultiArchitecture How-To](https://wiki.debian.org/Multiarch/HOWTO)

---

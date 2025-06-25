# Blockchain | Bitcoin Node Setup Guide

![Bitcoin](https://img.shields.io/badge/bitcoin-2F3134?style=for-the-badge&logo=bitcoin&logoColor=white)
![Litecoin](https://img.shields.io/badge/Litecoin-A6A9AA?style=for-the-badge&logo=Litecoin&logoColor=white)

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Bitcoin Node Configuration](#bitcoin-node-configuration)
5. [Using PM2 to Manage Processes](#using-pm2-to-manage-processes)
6. [Additional Resources](#additional-resources)
7. [Support and Contributions](#support-and-contributions)

---

## Introduction

This guide explains how to set up a Bitcoin Full Node using minimal disk space (pruned node) and manage it with the [PM2 Process Manager](https://pm2.io/docs/runtime/guide/process-management/). It is part of the UniversalBit ecosystem and includes steps for setting up the node, configuring it, and running it efficiently.

![Blockchain Stack](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/images/blockchain.png)

---

## Prerequisites

Before proceeding, ensure you have the following:
- A Linux-based operating system.
- Basic knowledge of command-line tools.
- Installed software:
  - `Node.js` (Version 20 recommended, install using [nvm](https://github.com/nvm-sh/nvm)).
  - `PM2` global process manager.
  - Required dependencies for Bitcoin Core.

---

## Installation Steps

### Step 1: Clone the Repository
```bash
cd /home/$USER
git clone https://github.com/universalbit-dev/universalbit-dev.git
```

### Step 2: Install the Latest Bitcoin Full Node
**[Download and install the Bitcoin Full Node] using the following command:
```bash
curl https://bitnodes.io/install-full-node.sh | sh
```

---

## Bitcoin Node Configuration

### Minimal Disk Space Configuration (Pruned Node)
1. Copy the Bitcoin configuration file:
    ```bash
    cp bitcoin.conf /home/$USER/.bitcoin/
    ```
2. Copy the `bitcoind` binary to the project directory:
    ```bash
    cp /home/$USER/bitcoin-core/bin/bitcoind /home/$USER/universalbit-dev/blockchain/bitcoin/
    ```
   ### Resources:
   **[Bitcoin-Core](https://bitcoin.org/en/download) || [Litecoin-Core](https://litecoin.org/)**

---

### Install Node.js Dependencies
Navigate to the project directory and install the required Node.js dependencies:
```bash
cd /home/$USER/universalbit-dev/blockchain/bitcoin
npm i && npm audit fix && npm i pm2 -g
```

---

## Using PM2 to Manage Processes

### Start Bitcoin Node with PM2
Start the Bitcoin Node using the PM2 process manager:
```bash
pm2 start btc_pruned.js
```

![Bitcoin Node Process](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/btc-pruned-node.png "Bitcoin Node Process")

### PM2 Startup Script
Generate startup scripts using the [PM2 Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/) to ensure the process list is preserved across machine restarts.

![PM2 Startup Script](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/pm2_btc_startup_script.gif)

---

## Additional Resources

Here are some helpful links and resources to enhance your setup and understanding of Bitcoin mining:

- [CPU Mining](https://github.com/universalbit-dev/CityGenerator/blob/master/workers/workers.md)
- [ESP32 Mining](https://github.com/universalbit-dev/esptool)
- [AMD Driver Release Notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-22-40-6.html)
- [AMDGPU Mining on Ubuntu Server](https://github.com/michalslonina/Ubuntu-Server-LTS-20.04-GUIDE-AMDGPU-Mining)
- [Bypass PCIe 3.0 Atomics Limitation](https://www.reddit.com/r/gpumining/comments/ptmyjd/ubuntu_20043_amdgpu_2130_opencl_rocr_rocm/)
- [How Bitcoin Mining Really Works](https://www.freecodecamp.org/news/how-bitcoin-mining-really-works-38563ec38c87/)
- [Gekko-M4-Globular-Cluster](https://github.com/universalbit-dev/gekko-m4-globular-cluster)
- [Web3 Development](https://web3.freecodecamp.org/web3)
- [MultiArchitecture Guide](https://wiki.debian.org/Multiarch/HOWTO)
- [Programming Blockchain](https://programmingblockchain.gitbook.io/programmingblockchain)

---

## Support and Contributions

- Support the UniversalBit Project: [Support Page](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)
- For questions or contributions, feel free to open an issue or submit a pull request.

---

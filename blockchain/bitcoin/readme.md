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

![Blockchain Stack](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/images/btc_pruned_node.png)

---
## Prerequisites

Before proceeding, ensure you have the following:
- A Linux-based operating system.
- Basic knowledge of command-line tools.
- Installed software:
  - `Node.js` (Version 24 recommended, install using [nvm](https://github.com/nvm-sh/nvm)):
    ```bash
    # Install nvm if not already installed
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    # Load nvm (restart shell or run the following)
    export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    # Install Node.js 24
    nvm install 24
    nvm use 24
    nvm alias default 24
    ```

---

## Installation Steps

### Step 1: Clone the Repository
```bash
cd /home/$USER
git clone https://github.com/universalbit-dev/universalbit-dev.git
```

### Step 2: Install the Latest Bitcoin Full Node
**[Download and install the Bitcoin Full Node]**

```bash
curl https://bitnodes.io/install-full-node.sh | sh
```

---


## Bitcoin Node Configuration

### Minimal Disk Space Configuration (Pruned Node)

Copy the Bitcoin configuration file:

**Note: Summary of bitcoin.conf Configuration**
This `bitcoin.conf` file configures a lightweight Bitcoin node for testnet operation with limited resource usage:

- `prune=550`: Enables blockchain pruning, keeping disk usage low by only storing recent blocks (here, up to ~550 MB).
- `maxconnections=8`: Limits the maximum number of peer connections to 8, reducing network resource requirements.
- `listen=0`: Disables listening for incoming connections, making this a non-public, outbound-only node.
- `maxuploadtarget=144`: Restricts data uploaded to peers each day to 144 MB (helpful for bandwidth limitation).
- `checkblocks=1`/`checklevel=0`: Minimal blockchain verification for faster, lightweight startup and operation.
- `txindex=0`: Disables full transaction indexing, reducing storage and indexing needs.
- `testnet=1`: Runs the node on the Bitcoin testnet, not on the main Bitcoin blockchain—useful for development and testing purposes.

This setup is ideal for a resource-constrained environment, development, or testing without maintaining a full blockchain or publicly accessible node.

```bash
    cp bitcoin.conf /home/$USER/.bitcoin/
```

Copy the `bitcoind` binary to the project directory:

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

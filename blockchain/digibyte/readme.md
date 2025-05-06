# Blockchain | Digibyte Node Setup Guide

![Digibyte](https://img.shields.io/badge/digibyte-0055FF?style=for-the-badge&logo=digibyte&logoColor=white)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Digibyte Node Configuration](#digibyte-node-configuration)
5. [Using PM2 to Manage Processes](#using-pm2-to-manage-processes)
6. [Additional Resources](#additional-resources)
7. [Support and Contributions](#support-and-contributions)

---

## Introduction

This guide explains how to set up a Digibyte Full Node using minimal disk space (pruned node) and manage it with the [PM2 Process Manager](https://pm2.io/docs/runtime/guide/process-management/).

---

## Prerequisites

Before proceeding, ensure you have the following:
- A Linux-based operating system.
- Basic knowledge of command-line tools.
- Installed software:
  - `Node.js` (Version 20 recommended, install using [nvm](https://github.com/nvm-sh/nvm)).
  - `PM2` global process manager.
  - Required dependencies for Digibyte Core.

---

## Installation Steps

### Step 1: Clone the Repository
```bash
cd /home/$USER
git clone https://github.com/universalbit-dev/universalbit-dev.git
```

### Step 2: Latest Digibyte Core [digibyte-8.22.2]
Download and install [Digibyte-Core](https://github.com/DigiByte-Core/digibyte/releases)
```bash
wget https://github.com/DigiByte-Core/digibyte/releases/download/v8.22.2/digibyte-8.22.2-x86_64-linux-gnu.tar.gz
tar -xvzf digibyte-8.22.2-x86_64-linux-gnu.tar.gz
cd digibyte-8.22.2/bin/
sudo chmod a+x ./digibyted
./digibyted
#**This commands sync full Digibyte Node**
```

---

### Minimal Disk Space Configuration (Pruned Node)
## Create Digibyte Node Configuration [File](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/digibyte/digibyte.conf)

```bash
nano /home/$USER/.digibyte/digibyte.conf
```
### Edit digibyte.conf
```bash
prune=550
maxconnections=8
listen=0
maxuploadtarget=144
checkblocks=1
checklevel=0
txindex=0
testnet=1
```

```bash
cd /home/$USER/Downloads/digibyte-8.22.2/bin/
./digibyted
```
**This commands sync Pruned Digibyte Node**

---


### Install Node.js Dependencies 
Navigate to the project directory and install the required Node.js dependencies:
```bash
cd /home/$USER/universalbit-dev/blockchain/digibyte
npm install
npm audit fix
npm install pm2 -g
npm install pm2 --save
```

---

## Using PM2 to Manage Processes

### Start Digibyte Node with PM2
Start the Digibyte Node using the PM2 process manager:
```bash
pm2 start dgb_pruned.js
```

### PM2 Startup Script
Generate startup scripts using the [PM2 Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/) to ensure the process list is preserved across machine restarts.

---

## Additional Resources

Here are some helpful links and resources to enhance your setup and understanding of Digibyte mining:

- [How Digibyte Mining Works](https://www.digibytewiki.com/)
- [DigiByte Core GitHub Repository](https://github.com/digibyte/digibyte)
- [Digibyte Node Setup Guide](https://www.digibyte.org/)
- [MultiArchitecture Guide](https://wiki.debian.org/Multiarch/HOWTO)

---

## 📢 Support the UniversalBit Project
Help us grow and continue innovating!  
- [Support the UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)  
- [Learn about Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)  
- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/)

---

### Notes:
- Replace all instances of `bitcoin` with `digibyte` in the filenames, configuration paths, and binary references.
- Ensure Digibyte-specific commands, binaries, and configuration details are correctly updated.

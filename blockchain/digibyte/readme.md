# Blockchain | Digibyte Node Setup Guide

![Digibyte](https://img.shields.io/badge/digibyte-0055FF?style=for-the-badge&logo=digibyte&logoColor=white)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Digibyte Node Configuration](#minimal-disk-space-configuration-pruned-node)
5. [Using PM2 to Manage Processes](#using-pm2-to-manage-processes)
6. [Additional Resources](#additional-resources)
7. [Support and Contributions](#-support-the-universalbit-project)

---

## Introduction

This guide provides step-by-step instructions to set up a **Digibyte Full Node** using minimal disk space (pruned node) and manage it with the [PM2 Process Manager](https://pm2.io/docs/runtime/guide/process-management/).

---

## Prerequisites

Before proceeding, ensure you have the following:

- **Operating System**: A Linux-based operating system.
- **Basic Knowledge**: Familiarity with command-line tools.
- **Installed Software**:
  - `Node.js` (Version 20 recommended). Install using [nvm](https://github.com/nvm-sh/nvm).
  - `PM2` global process manager.
  - Required dependencies for Digibyte Core.

---

## Installation Steps

### Step 1: Clone the Repository

Run the following command in your terminal to clone the repository:

```bash
cd /home/$USER
git clone https://github.com/universalbit-dev/universalbit-dev.git
```

### Step 2: Install Latest Digibyte Core [v8.22.2]

Download and install the latest version of Digibyte Core:

```bash
wget https://github.com/DigiByte-Core/digibyte/releases/download/v8.22.2/digibyte-8.22.2-x86_64-linux-gnu.tar.gz
tar -xvzf digibyte-8.22.2-x86_64-linux-gnu.tar.gz
cd digibyte-8.22.2/bin/
sudo chmod a+x ./digibyted
./digibyted
```

> **Note**: The above commands sync the full Digibyte Node.

---

### Minimal Disk Space Configuration (Pruned Node)

#### Step 1: Create a Digibyte Node Configuration File

```bash
nano /home/$USER/.digibyte/digibyte.conf
```

#### Step 2: Edit `digibyte.conf`

Update the configuration file with the following settings:

```plaintext
prune=550
maxconnections=8
listen=0
maxuploadtarget=144
checkblocks=1
checklevel=0
txindex=0
testnet=1
```

#### Step 3: Start Dgb Pruned Node

Navigate to binary directory and start node:

```bash
cd /home/$USER/Downloads/digibyte-8.22.2/bin/
./digibyted
```

> **Note**: The above commands sync a Pruned Digibyte Node.

---

### Install Node.js Dependencies

Navigate to the project directory and install the necessary Node.js dependencies:

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
```bash
pm2 start dgb_pruned.js
```

### PM2 Startup Script

Generate startup scripts using the [PM2 Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/) to ensure the process list is preserved across machine restarts.

---

## Additional Resources

Here are some helpful resources to enhance your setup and understanding of Digibyte:

- [Digibyte Node Setup Guide](https://www.digibyte.org/)
- [Contribution Guidelines](https://github.com/DigiByte-Core/digibyte/wiki/Contribution-Guidelines)

---

## 📢 Support the UniversalBit Project

Help us grow and continue innovating!  
- [Support the UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)  
- [Learn about Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)  
- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/)

---

### Notes:

- Explore the [CityGenerator Project](https://github.com/universalbit-dev/CityGenerator) for a related initiative and additional resources.
- Use the latest version of **Digibyte-Core** and download it from the [Official GitHub releases page](https://github.com/DigiByte-Core/digibyte/releases).

---

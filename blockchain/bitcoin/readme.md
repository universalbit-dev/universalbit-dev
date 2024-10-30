##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)

BlockChain |  | # Stack
---|---|---
 [NetNode](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain/bitcoin) | <img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/images/bitnodes.png" width="auto"></img> | 
 [Buy-Sell](https://github.com/universalbit-dev/gekko-m4-globular-cluster/blob/master/README.md) |  | 
 [Mining](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain) |  | 

<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/images/blockchain.png" width="5%"></img>


#### Pruned Node simple explanation
[means: |storage - minimum disk space required for running a full node|](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)

* Clone project:
```bash
cd /home/$USER
git clone https://github.com/universalbit-dev/universalbit-dev.git
```
---
* Latest Bitcoin Full Node 
```bash
curl https://bitnodes.io/install-full-node.sh | sh
```

#### bitcoin node configuration file (minimal disk space)
```bash
cp bitcoin.conf /home/$USER/.bitcoin/
cp /home/$USER/bitcoin-core/bin/bitcoind /home/$USER/universalbit-dev/blockchain/bitcoin/
```

#### [Nodejs Engine 20](https://github.com/nvm-sh/nvm)
```bash
cd /home/$USER/universalbit-dev/blockchain/bitcoin
npm i && npm audit fix
npm i pm2 -g
npm i pm2 --save
```
* Start BITCOIN-NODE with [pm2 process manager](https://pm2.io/docs/runtime/guide/process-management/) 
```bash
pm2 start btc_pruned.js
```
![BTC](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/btc-pruned-node.png "btc")
##### Note: 
* [PM2 Module Not Found](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain/bitcoin/gif/readme.md)
* [Gif Tutorial](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/btc-net-node.gif)
---

#### [PM2 Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)
can generate startup scripts and configure them in order to keep your process list intact across expected or unexpected machine restarts.
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/pm2_btc_startup_script.gif" width="auto"></img>

### Resources:
##### [CPU Mining](https://github.com/universalbit-dev/CityGenerator/blob/master/workers/workers.md)
##### [ESP32 Mining](https://github.com/universalbit-dev/esptool)
##### [Release Note AMD Driver 22.40](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-22-40-6.html)
##### [AMDGPU Mining Ubuntu Server](https://github.com/michalslonina/Ubuntu-Server-LTS-20.04-GUIDE-AMDGPU-Mining)
##### [Bypass PCIE 3.0 atomics limitation](https://www.reddit.com/r/gpumining/comments/ptmyjd/ubuntu_20043_amdgpu_2130_opencl_rocr_rocm/)
##### [Gekko-M4-Globular-Cluster](https://github.com/universalbit-dev/gekko-m4-globular-cluster/blob/master/README.md)
##### [How Bitcoin Mining Really Works](https://www.freecodecamp.org/news/how-bitcoin-mining-really-works-38563ec38c87/)
##### [Web3](https://web3.freecodecamp.org/web3)
##### [MultiArchitecture](https://wiki.debian.org/Multiarch/HOWTO)
##### [Programming Blockchain](https://programmingblockchain.gitbook.io/programmingblockchain)

##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)

#### Pruned Node simple explanation
[means: |storage - minimum disk space required for running a full node|](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)

* Clone project:
```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
cd universalbit-dev/blockchain/

```
---
* Latest Bitcoin Full Node 
```bash
curl https://bitnodes.io/install-full-node.sh | sh
```
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/btc-net-node.gif" width="auto"></img>

* Edit [bitcoin configuration file](https://bitcoincoredocs.com/bitcoin-conf.html)
```bash
nano /home/your_username/.bitcoin/bitcoin.conf
```
#### bitcoin node configuration file (minimal disk space)
```bash
prune=550
maxconnections=8
listen=0
maxuploadtarget=144
checkblocks=1
checklevel=0
txindex=0
```

---

* [npm installation](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
```bash
npm i && npm i audit fix
```

* start BITCOIN-NODE with [pm2 process manager](https://pm2.io/docs/runtime/guide/process-management/) 
```bash
pm2 start btc_pruned.js
```
![BTC](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/btc-pruned-node.png "btc")

---

#### [PM2 Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/pm2_btc_startup_script.gif" width="auto"></img>

### Resources:
##### [CPU Mining](https://github.com/universalbit-dev/CityGenerator/blob/master/workers/workers.md)
##### [ESP32 Mining](https://github.com/BitMaker-hub/NerdMiner_v2)
##### [Release Note AMD Driver 22.40](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-22-40-6.html)
##### [AMDGPU Mining Ubuntu Server](https://github.com/michalslonina/Ubuntu-Server-LTS-20.04-GUIDE-AMDGPU-Mining)
##### [Bypass PCIE 3.0 atomics limitation](https://www.reddit.com/r/gpumining/comments/ptmyjd/ubuntu_20043_amdgpu_2130_opencl_rocr_rocm/)
##### [How Bitcoin Mining Really Works](https://www.freecodecamp.org/news/how-bitcoin-mining-really-works-38563ec38c87/)
##### [Web3](https://web3.freecodecamp.org/web3)
##### [MultiArchitecture](https://wiki.debian.org/Multiarch/HOWTO)
##### [Programming Blockchain](https://programmingblockchain.gitbook.io/programmingblockchain)

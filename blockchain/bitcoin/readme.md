##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)   ##### [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)   ##### [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html)

* ### [Bitcoin](https://en.wikipedia.org/wiki/Bitcoin)
* ### [Getting Started](https://bitcoin.org/en/getting-started)

#### Pruned Node simple explanation
[means: |storage - minimum disk space required for running a full node|](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)

* clone project:
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

* edit [bitcoin configuration file](https://bitcoincoredocs.com/bitcoin-conf.html)
```bash
nano /home/your_username/.bitcoin/bitcoin.conf
```
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
npm i
#to fix vulnerabilities:
npm audit fix
```
* start BITCOIN-NODE with [pm2 process manager](https://pm2.io/docs/runtime/guide/process-management/) 
```bash
pm2 start btc_pruned.js
```
![BTC](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/btc-pruned-node.png "btc")

---

### [PM2 Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/pm2_btc_startup_script.gif" width="auto"></img>


* [Programming Blockchain](https://programmingblockchain.gitbook.io/programmingblockchain)

#### Pruned Node simple explanation
[means: |storage - minimum disk space required for running a full node|](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)

[ProgrammingBlockchain](https://programmingblockchain.gitbook.io/programmingblockchain)
---

* edit bitcoin configuration file
```bash
mkdir /home/your_username/.bitcoin
nano /home/your_username/.bitcoin/bitcoin.conf
```
* add [this](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)
```bash
prune=550
maxconnections=8
listen=0
maxuploadtarget=144
checkblocks=1
checklevel=0
txindex=0
```
* save configuration file as : bitcoin.conf

---

#### [BitCoinCore](https://bitcoincore.org/)

#### [BitNodes](https://bitnodes.io)

---
* Latest Bitcoin Full Node 
```bash
curl https://bitnodes.io/install-full-node.sh | sh
```

* Run Node
```bash
npm i 
pm2 start btc_pruned.js
```
![BTC](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/btc-pruned-node.png "btc")

[Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)


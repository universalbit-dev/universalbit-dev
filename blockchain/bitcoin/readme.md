##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)

* ### [Bitcoin](https://en.wikipedia.org/wiki/Bitcoin)
* ### [Getting Started](https://bitcoin.org/en/getting-started)

#### Pruned Node simple explanation
[means: |storage - minimum disk space required for running a full node|](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)

[ProgrammingBlockchain](https://programmingblockchain.gitbook.io/programmingblockchain)
---
* Latest Bitcoin Full Node 
```bash
curl https://bitnodes.io/install-full-node.sh | sh
```
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/btc-net-node.gif" width="auto"></img>




* edit bitcoin configuration file
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
create package.json file
```bash
nano /home/your_username/bitcoin-core/package.json
```
```json
{
  "name": "bitcoin_pruned",
  "version": "1.0.0",
  "description": "",
  "main": "btc_pruned.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "author": "UniversalBit-Dev",
  "license": "MIT",
  "dependencies": {"pm2": "^5.3.1"}
}
```


create btc_pruned.js file
```bash
nano /home/your_username/bitcoin-core/btc_pruned.js
```
```js
var pm2 = require('pm2');
pm2.connect(function(err) {
  if (err) {
    console.error(err)
    process.exit(2)
}

pm2.start({
  script    : './bitcoind',
  name      : '|BITCOIN-NODE|'
},

function(err, apps) {
  if (err) {
    console.error(err)
    return pm2.disconnect()
}

pm2.list((err, list) => {
  console.log(err, list)
})
})
})
```


---


* Run Node
```bash
npm i 
pm2 start btc_pruned.js
```
![BTC](https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/btc-pruned-node.png "btc")

---

### [PM2 Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/blockchain/bitcoin/gif/pm2_btc_startup_script.gif" width="auto"></img>

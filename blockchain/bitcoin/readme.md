
#### [BitNodes](https://bitnodes.io)
---
#### Latest Bitcoin Node 
```
curl https://bitnodes.io/install-full-node.sh | sh
```
[ProgrammingBlockchain](https://programmingblockchain.gitbook.io/programmingblockchain)

#### [Configure Bitcoin Pruned Node]()

```
nano bitcoin-core/.bitcoin/bitcoin.conf
```
#### [Pruned Node](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)

```
prune=550
maxconnections=8
listen=0
maxuploadtarget=144
checkblocks=1
checklevel=0
txindex=0
```
* [Bitcoin Core integration](https://github.com/bitcoin/bitcoin)


#### Usage: Start/Stop Bitcoin Node
* bitcoin-core/bin/start.sh 
* bitcoin-core/bin/stop.sh

#### Pm2 Process Manager [Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)
```
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




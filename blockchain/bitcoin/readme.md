### Pruned Node simple explanation
means: |storage - minimum disk space required for running a full node|
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

#### Pm2 Process Manager 
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
[Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)
To automatically generate and configuration a startup script just type the command (without sudo) 
```
pm2 startup
```
Then copy/paste the displayed command onto the terminal
```
* ### EXAMPLE:
sudo su -c "env PATH=$PATH:/home/unitech/.nvm/versions/node/v14.3/bin pm2 startup <distribution> -u <user> --hp <home-path>
```
and save app list
```
pm2 save
```


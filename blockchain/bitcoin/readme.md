### Pruned Node simple explanation
means: |storage - minimum disk space required for running a full node|
#### [BitNodes](https://bitnodes.io)
[ProgrammingBlockchain](https://programmingblockchain.gitbook.io/programmingblockchain)
---

* Create and Edit bitcoin configuration file
```bash
mkdir /home/your_username/.bitcoin
nano /home/your_username/.bitcoin/bitcoin.conf
```
#### [Pruned Node](https://programmingblockchain.gitbook.io/programmingblockchain/wallet/pruned-node)
```bash
prune=550
maxconnections=8
listen=0
maxuploadtarget=144
checkblocks=1
checklevel=0
txindex=0
```
* [Bitcoin Core integration](https://github.com/bitcoin/bitcoin)

  
* Latest Bitcoin Node 
```bash
curl https://bitnodes.io/install-full-node.sh | sh
```

* ### [Nodejs Setup](https://github.com/nvm-sh/nvm)

* Run Bitcoin-core Pruned Node
```bash
cp btc_pruned.js package.json home/your_username/bitcoin-core/
cd home/your_username/bitcoin-core/
npm i 
pm2 start btc_pruned.js
```

[Startup Script Generator](https://pm2.keymetrics.io/docs/usage/startup/)
To automatically generate and configuration a startup script just type the command (without sudo) 
```bash
pm2 startup
```
Then copy/paste the displayed command onto the terminal
```bash
* ### EXAMPLE:
sudo su -c "env PATH=$PATH:/home/unitech/.nvm/versions/node/v14.3/bin pm2 startup <distribution> -u <user> --hp <home-path>
```
and save app list
```bash
pm2 save
```



### Pruned Node simple explanation
means: |storage - minimum disk space required for running a full node|


#### * [Litecoin Release](https://github.com/litecoin-project/litecoin/releases) 
#### * [Litecoin Project](https://www.litecoin.net/)

* Create and Edit litecoin configuration file
```
mkdir /home/your_username/.litecoin/
nano /home/your_username/.litecoin/litecoin.conf
```
```
prune=550
maxconnections=8
listen=0
maxuploadtarget=144
checkblocks=1
checklevel=0
txindex=0
```

* Litecoin Node
```
wget https://github.com/litecoin-project/litecoin/releases/download/v0.21.2.2/litecoin-0.21.2.2-x86_64-linux-gnu.tar.gz
tar xvzf litecoin-0.21.2.2-x86_64-linux-gnu.tar.gz
cp ltc_pruned.js package.json litecoin-0.21.2.2/
```

* Install [Pm2 Process Manager](https://pm2.io/docs/runtime/guide/process-management/)
```
npm i
```
* Run Litecoin Pruned Node
```
cd litecoin-0.21.2.2
pm2 start ltc_pruned.js
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
  
   

[Repository:](https://github.com/litecoin-project/litecoin/releases)
[Litecoin Core integration](https://github.com/litecoin-project/litecoin)




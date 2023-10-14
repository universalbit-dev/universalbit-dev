
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
cd ~/litecoin-0.21.2.2/bin
```
[Repository:](https://github.com/litecoin-project/litecoin/releases)
[Litecoin Core integration](https://github.com/litecoin-project/litecoin)




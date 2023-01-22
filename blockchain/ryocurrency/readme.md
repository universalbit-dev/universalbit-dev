
[RyoCurrency](https://ryo-currency.com/cn-gpu/) Project Description

NetNode:
[GitHub Repository](https://github.com/ryo-currency)

---
[RyoWallet](https://github.com/ryo-currency/ryo-wallet)


Donation Address: ![RYO Currency!](https://github.com/ryo-currency/ryo-wallet/blob/master/src-electron/icons/icon_64x64.png?raw=true)

RYoNsBiFU6iYi8rqkmyE9c4SftzYzWPCGA3XvcXbGuBYcqDQJWe8wp8NEwNicFyzZgKTSjCjnpuXTitwn6VdBcFZEFXLcV5Z7BjMWkkPzSzdRp

[Release](https://github.com/ryo-currency/ryo-currency/releases)



Debian / Ubuntu one liner for all dependencies  sudo apt update && sudo apt install build-essential cmake pkg-config libboost-all-dev libssl-dev libzmq3-dev libunbound-dev libsodium-dev libunwind8-dev liblzma-dev libreadline6-dev libldns-dev libexpat1-dev doxygen graphviz libpgm-dev
Cloning the repository
This repo does not use submodules, so simply clone this repo:

```
git clone https://github.com/ryo-currency/ryo-currency.git
```
If you already have a repo cloned, initialize and update:

```
cd ryo-currency
```

Build instructions
Ryo uses the CMake build system and a top-level Makefile that invokes cmake commands as needed.
On Linux and OS X
Install the dependencies
Change to the root of the source code directory, change to the most recent release branch, and build:

```
cd ryo-currency
git checkout tags/0.5.0.0
make
```

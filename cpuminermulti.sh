###
sudo apt-get install build-essential libcurl4-openssl-dev git automake libtool libjansson* libncurses5-dev libssl-dev
git clone --recursive https://github.com/tpruvot/cpuminer-multi.git
git checkout linux
./autogen.sh
./configure CFLAGS="-march=native" --with-crypto --with-curl
make
./cpuminer -help
# to run cpuminer and load settings from example cpuminer-conf.json.lyra2re configuration file issue this command:
./cpuminer -c cpuminer-conf.json.lyra2re

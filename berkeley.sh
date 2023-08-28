###https://en.wikipedia.org/wiki/Berkeley_DB
###Update System and Install Dependencies
###sudo ./berkeley.sh

apt-get update && sudo apt-get upgrade
apt-get install build-essential  libssl-dev libgmp-dev

###Download and Compile SRC BerkeleyDB-4.8
wget http://download.oracle.com/berkeley-db/db-4.8.30.NC.tar.gz
tar -xzvf db-4.8.30.NC.tar.gz
#make: *** [Makefile:2018: cxx_db.o] Error 1   one line solution
sed -i 's/__atomic_compare_exchange/__atomic_compare_exchange_db/g' db-4.8.30.NC/dbinc/atomic.h
cd db-4.8.30.NC/build_unix
../dist/configure --enable-cxx --disable-shared --with-pic --prefix=$BDB_PREFIX
make
make install

###Linking Correct Directory of BerkeleyDB-4.8
export BDB_INCLUDE_PATH="/usr/local/BerkeleyDB.4.8/include"
export BDB_LIB_PATH="/usr/local/BerkeleyDB.4.8/lib"
ln -s /usr/local/BerkeleyDB.4.8/lib/libdb-4.8.so /usr/lib/libdb-4.8.so

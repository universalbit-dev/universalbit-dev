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

#COPILOT EXPLAIN
#This shell script, `berkeley.sh`, is used to update the system, install dependencies, download, compile, and install Berkeley DB 4.8. Here's a step-by-step explanation:
#
#1. **Comments**: 
#   - The script starts with comments that provide a link to the Wikipedia page for Berkeley DB and instructions on how to run the script.
#
#2. **Update System and Install Dependencies**:
#   - `apt-get update && sudo apt-get upgrade`: Updates the package list and upgrades all installed packages.
#   - `apt-get install build-essential libssl-dev libgmp-dev`: Installs essential build tools and libraries required for compiling Berkeley DB.
#
#3. **Download and Compile Berkeley DB 4.8**:
#   - `wget http://download.oracle.com/berkeley-db/db-4.8.30.NC.tar.gz`: Downloads the Berkeley DB 4.8 source code archive.
#   - `tar -xzvf db-4.8.30.NC.tar.gz`: Extracts the downloaded archive.
#   - `sed -i 's/__atomic_compare_exchange/__atomic_compare_exchange_db/g' db-4.8.30.NC/dbinc/atomic.h`: Replaces a problematic function name in the source code to avoid compilation errors.
#   - `cd db-4.8.30.NC/build_unix`: Changes the directory to the build directory.
#   - `../dist/configure --enable-cxx --disable-shared --with-pic --prefix=$BDB_PREFIX`: Configures the build with C++ support, static linking, and position-independent code. The installation prefix is set to the `BDB_PREFIX` environment variable.
#   - `make`: Compiles the source code.
#   - `make install`: Installs the compiled binaries.
#
#4. **Linking Correct Directory of Berkeley DB 4.8**:
#   - `export BDB_INCLUDE_PATH="/usr/local/BerkeleyDB.4.8/include"`: Sets the environment variable for the include path.
#   - `export BDB_LIB_PATH="/usr/local/BerkeleyDB.4.8/lib"`: Sets the environment variable for the library path.
#   - `ln -s /usr/local/BerkeleyDB.4.8/lib/libdb-4.8.so /usr/lib/libdb-4.8.so`: Creates a symbolic link for the Berkeley DB shared library in the `/usr/lib` directory.
#
#This script automates the process of setting up Berkeley DB 4.8 on a system.


##The GNU Multiple Precision Arithmetic Library
cd
rm -rf gmp-5.1.2.tar.bz2 gmp-5.1.2
wget http://mirrors.kernel.org/gnu/gmp/gmp-5.1.2.tar.bz2
tar xjvf gmp-5.1.2.tar.bz2
cd gmp-5.1.2
./configure --enable-cxx
make
sudo make install

#COPILOT EXPLAIN
#This shell script, `gmp.sh`, automates the process of downloading, extracting, configuring, compiling, and installing the GNU Multiple Precision Arithmetic Library (GMP). Here's a step-by-step explanation:
#
#1. **Comments**:
#   - The script starts with a comment indicating that it is for the GNU Multiple Precision Arithmetic Library.
#
#2. **Change to Home Directory**:
#   - `cd`: Changes the current directory to the user's home directory.
#
#3. **Remove Existing Files and Directories**:
#   - `rm -rf gmp-5.1.2.tar.bz2 gmp-5.1.2`: Removes any existing `gmp-5.1.2.tar.bz2` file and `gmp-5.1.2` directory to ensure a clean installation.
#
#4. **Download GMP Source Code**:
#   - `wget http://mirrors.kernel.org/gnu/gmp/gmp-5.1.2.tar.bz2`: Downloads the GMP version 5.1.2 source code archive from the specified URL.
#
#5. **Extract the Archive**:
#   - `tar xjvf gmp-5.1.2.tar.bz2`: Extracts the downloaded `.tar.bz2` archive.
#
#6. **Change to Extracted Directory**:
#   - `cd gmp-5.1.2`: Changes the current directory to the extracted `gmp-5.1.2` directory.
#
#7. **Configure the Build**:
#   - `./configure --enable-cxx`: Configures the build with C++ support.
#
#8. **Compile the Source Code**:
#   - `make`: Compiles the source code.
#
#9. **Install the Library**:
#   - `sudo make install`: Installs the compiled library using superuser privileges.
#
#This script automates the entire process of setting up GMP on a system.

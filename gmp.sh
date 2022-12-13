##The GNU Multiple Precision Arithmetic Library
cd
rm -rf gmp-5.1.2.tar.bz2 gmp-5.1.2
wget http://mirrors.kernel.org/gnu/gmp/gmp-5.1.2.tar.bz2
tar xjvf gmp-5.1.2.tar.bz2
cd gmp-5.1.2
./configure --enable-cxx
make
sudo make install

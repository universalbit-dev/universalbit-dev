#!/bin/bash
###UniversalBit once again

###url:https://letsencrypt.org.
###Documentation:https://letsencrypt.org/docs/

sudo apt-get install git

#Debian11
sudo apt -y install certbot python3-certbot-apache

git clone https://github.com/letsencrypt/letsencrypt
cd letsencrypt
./letsencrypt-auto --help ###Suite Command List

###Usage:
./letsencrypt-auto --apache -d universalbit.it 
###./letsencrypt-auto certonly --standalone --email admin@youremail.com -d yourdomain.com -d www.yourdomain.com -d otherdomain.net

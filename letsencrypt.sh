#!/bin/bash
###UniversalBit once again

###Letsencrypt and Certbot on WebServer
###Src:https://letsencrypt.org.
###Documentation:https://letsencrypt.org/docs/

sudo apt-get install git
###CertBot
echo 'deb http://ftp.debian.org/debian jessie-backports main' | sudo tee /etc/apt/sources.list.d/backports.list
sudo apt-get install python-certbot-apache -t jessie-backports
#Debian11
#sudo apt -y install certbot python3-certbot-apache

git clone https://github.com/letsencrypt/letsencrypt
cd letsencrypt
./letsencrypt-auto --help ###Suite Command List

###Usage:
###./letsencrypt-auto --apache -d yourdomain.com -d www.yourdomain.com -d otherdomain.net
###./letsencrypt-auto certonly --standalone --email admin@youremail.com -d yourdomain.com -d www.yourdomain.com -d otherdomain.net

### UniversalBit once again 
...enable HTTPS on your website.

* [LetsEncrypt.org](https://letsencrypt.org)
* [Documentation](:https://letsencrypt.org/docs/)
---
### Debian11
```
apt install certbot python3-certbot-apache -y
git clone https://github.com/letsencrypt/letsencrypt
cd letsencrypt
./letsencrypt-auto --help ###Suite Command List
```
#### usage:
```
./letsencrypt-auto --apache -d yourdomain.com
```
#### certonly:
```
./letsencrypt-auto certonly --standalone --email infouniversalbits@gmail.com -d universalbit.it 
```
---
### Debian12
[LetsEncrypt Debian Wiki](https://wiki.debian.org/LetsEncrypt)
```
```
---

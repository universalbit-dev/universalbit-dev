#### [NTP](https://wiki.debian.org/NTP)
##### The Debian package will install a default set of time servers which 
should be good for most typical client installations.  However you may 
customize this for your network location.

---

#### [Read More](https://timetoolsltd.com/information/public-ntp-server/)

Installation
```
sudo apt-get install ntp
```
example: [conf](https://github.com/universalbit-dev/universalbit-dev/blob/main/ntp/ntp.conf)



Edit NTP configuration file:

```
nano /etc/ntp.conf
```


```
server 0.europe.pool.ntp.org
server 1.europe.pool.ntp.org
server 2.europe.pool.ntp.org
server 3.europe.pool.ntp.org
```  


Customization completed:
```
dpkg-reconfigure ntp
```








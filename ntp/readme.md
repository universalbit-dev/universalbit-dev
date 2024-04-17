##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)
##### [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html)


* #### [Network Time Protocol](https://en.wikipedia.org/wiki/Network_Time_Protocol)

---
#### [NTP Debian Wiki](https://wiki.debian.org/NTP)
##### The Debian package will install a default set of time servers

should be good for most typical client installations.  
However you may customize this for your network location.

---

#### [Europe — europe.pool.ntp.org](https://www.ntppool.org/zone/europe)

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
server pool.ntp.org
```  


Customization completed:
```
dpkg-reconfigure ntp
```








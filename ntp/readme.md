##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)

---

* #### [Network Time Protocol](https://en.wikipedia.org/wiki/Network_Time_Protocol)

---
#### [NTP Debian Wiki](https://wiki.debian.org/NTP)
##### The Debian package will install a default set of time servers

should be good for most typical client installations.  
However you may customize this for your network location.

---

#### [pool.ntp.org](https://www.ntppool.org)

Installation
```
sudo apt-get install ntp
```
example: [conf](https://github.com/universalbit-dev/universalbit-dev/blob/main/ntp/ntp.conf)



##### Edit NTP configuration file:

```bash
nano /etc/ntp.conf
```
```bash
server pool.ntp.org
```
```bash
dpkg-reconfigure ntp
```








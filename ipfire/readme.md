#### [IPFIRE](https://www.ipfire.org/) Customizable Router:
much more than configuration

---
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ipfire/images/IPFIRE_HPT610.jpg" width="50%"></img>


Overview:
* [ZoneConfiguration](#ZoneConfiguration)



## [Docs](https://www.ipfire.org/docs)
---
### WebGui
* [Web Interface](https://www.ipfire.org/docs/configuration)
### System
* [Backup](https://www.ipfire.org/docs/configuration/system/backup)
### Network:
* [Zone Configuration](https://www.ipfire.org/docs/configuration/network/zoneconf)
* [Domain Name System](https://www.ipfire.org/docs/configuration/network/dns-server)
* [Location Block](https://www.ipfire.org/docs/configuration/firewall/geoip-block)
* [Web Proxy](https://www.ipfire.org/docs/configuration/network/proxy)
* [URL Filter](https://www.ipfire.org/docs/configuration/network/proxy/url-filter)
* [DHCP Server](https://www.ipfire.org/docs/configuration/network/dhcp)
* [Captive Portal](https://www.ipfire.org/docs/configuration/network/captive)
* [DNS forwarding](https://www.ipfire.org/docs/configuration/network/dnsforward)
### Firewall
* [Intrusion Prevention System](https://www.ipfire.org/docs/configuration/firewall/ips)
* [Logs](https://www.ipfire.org/docs/configuration/logs/firewall)
### Add-ons
* [Install Addon](https://www.ipfire.org/docs/search?q=install+addon)


## ZoneConfiguration 
<strong>Configuring 3 NICs</strong> assignment

<strong>Red Zone</strong>  [INTERNET]
* wlan1 (Native)  

<strong>Green Zone Switch Area</strong> [LAN] 
* Bridged Together wlan2 and eth0 


<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ipfire/images/bridge_green_area.png" width="auto"></img>
[docs](https://www.ipfire.org/docs/configuration/network/zoneconf/bridge3nic2green)

---
<strong>Configuring 4 NICs</strong> assignment:


<strong>Red Zone</strong>  [INTERNET]
* wlan1 (Native)
  
<strong>Green Zone Switch Area</strong> [LAN]
* Bridge old reused  Mini PCI Express 54Mbps WiFi card,Ralink USB Wifi Adapter 300mbps and  Ethernet 40 Gbps 2000 MHz Cat 8

* Bridged wlan0 wlan2 and eth0 (Native)

<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ipfire/images/bridge_green_area_4.png" width="auto"></img>
[Network Adapter Hardware Compatibility List](https://www.ipfire.org/docs/hardware/networking)


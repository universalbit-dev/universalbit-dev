##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)   ##### [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)   ##### [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html)
---

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
* [Domain Name System](https://www.ipfire.org/docs/configuration/network/dns-server) -- [DNS Public Servers](https://www.ipfire.org/docs/dns/public-servers) 
* [Location Block](https://www.ipfire.org/docs/configuration/firewall/geoip-block)
* [Web Proxy](https://www.ipfire.org/docs/configuration/network/proxy)
* [URL Filter](https://www.ipfire.org/docs/configuration/network/proxy/url-filter)
* [DHCP Server](https://www.ipfire.org/docs/configuration/network/dhcp)
* [Captive Portal](https://www.ipfire.org/docs/configuration/network/captive)
* [VPN](https://www.ipfire.org/docs/configuration/services/openvpn) -- [OpenVPN Configuration](https://www.ipfire.org/docs/configuration/services/openvpn/config)
* [DNS forwarding](https://www.ipfire.org/docs/configuration/network/dnsforward)

### Status
[Network Status](#NetworkStatus)
* <strong>External</strong> Network Traffic (red)
* <strong>Internal</strong> Network Traffic (green)
  

### Firewall
* [Intrusion Prevention System](https://www.ipfire.org/docs/configuration/firewall/ips)
* [Logs](https://www.ipfire.org/docs/configuration/logs/firewall)
### Add-ons
* [Install Add-ons](https://www.ipfire.org/docs/search?q=install+addon)


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
* Ralink USB Wifi Adapter 300mbps
* interface:wlan1 (Native)
  
<strong>Green Zone Switch Area</strong> [LAN]
* Bridged old reused  Mini PCI Express 54Mbps WiFi card,Ralink USB Wifi Adapter 300mbps and  Ethernet 40 Gbps 2000 MHz Cat 8
* interfaces:wlan0 wlan2 and eth0 (Native)

<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ipfire/images/bridge_green_area_4.png" width="auto"></img>

---
---

<strong>Configuring 5 NICs</strong> assignment:

<strong>Red Zone</strong>  [INTERNET]
* Ralink USB Wifi Adapter 300mbps
* interface:wlan3 (Native)
  
<strong>Green Zone Switch Area</strong> [LAN]
* Bridged Ralink USB Wifi Adapters 300mbps and  Ethernet 40 Gbps 2000 MHz Cat 8
* interfaces:wlan1 wlan2 wlan4 and eth0 (Native)

<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ipfire/images/bridge_green_area_5.png" width="auto"></img>

[Network Adapter Hardware Compatibility List](https://www.ipfire.org/docs/hardware/networking)

## NetworkStatus

### External (Red)
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ipfire/images/status_external_traffic.png" width="auto"></img>

### Internal (Green)
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ipfire/images/status_internal_traffic.png"></img>


[status reports on various parts of the system](https://www.ipfire.org/docs/configuration/status)





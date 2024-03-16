* ## [GlobalPing-Cli](https://github.com/jsdelivr/globalping-cli)

* ##### [Content Delivery Network](https://en.wikipedia.org/wiki/Content_delivery_network)
---
* ##### [jsDelivr Network Map](https://www.jsdelivr.com/network#map)
```bash
curl -s https://packagecloud.io/install/repositories/jsdelivr/globalping/script.deb.sh | sudo bash
apt install globalping
```

[NodeJs v20.11.1 -- Npm 10.2.4]
* GlobalPing with [Pm2](https://pm2.keymetrics.io/) advanced process manager

#### Global
```bash
npm i
pm2 start cdn.js
```
#### United States
```bash
pm2 start cdn_united_states.js
```
#### Europe
```bash
pm2 start cdn_europe.js
```
cdn(global) cdn_europe cnd_united_states
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/pm2_start.png" width="80%"></img> 

#### Pm2 status
```bash
pm2 status
```
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/pm2_status.png" width="80%"></img> 

#### Pm2 startup
```bash
pm2 startup
```
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/systemctl_enable.png" width="80%"></img>  
```bash
systemctl enable pm2-root
pm2 save
```

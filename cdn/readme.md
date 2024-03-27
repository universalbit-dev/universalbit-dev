
#### [Rescaling Global Manufacturing](https://github.com/universalbit-dev/CityGenerator/blob/master/Fab-City_Whitepaper.pdf) 
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/rescaling_fab_city_whitepaper.png" width="40%"></img>
#### Planet - Region - Country - City
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/gif/cdn_ecosystem.gif" width="auto"></img>





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
#### Italy
```bash
pm2 start cdn_italy.js
```
#### Italy Palermo
```bash
pm2 start cdn_italy_palermo.js
```

#### Pm2 status
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/gif/cdn_and_blockchain_net_node.gif" width="auto"></img>  
* [blockchain net node](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain)


#### * consider run cdn.js cdn_europe.js in cluster mode
```bash
pm2 start cdn.js -i 1
pm2 start cdn_europe.js -i 1
```

#### Pm2 status cluster mode
```bash
pm2 status (mode:cluster)
```

```bash
┌────┬──────────────────────┬─────────────┬─────────┬─────────┬
│ id │ name                 │ namespace   │ version │ mode    │
├────┼──────────────────────┼─────────────┼─────────┼───────── 
│ 5  │ btc_pruned           │ default     │ 1.0.0   │ fork    │  
│ 7  │ cdn                  │ default     │ 1.0.0   │ cluster │  
│ 8  │ cdn_europe           │ default     │ 1.0.0   │ cluster │  
│ 9  │ cdn_italy            │ default     │ 1.0.0   │ fork    │ 
│ 10 │ cdn_italy_palermo    │ default     │ 1.0.0   │ fork    │
│ 11 │ cdn_united_states    │ default     │ 1.0.0   │ fork    │
│ 6  │ |BITCOIN-NODE|       │ default     │ 1.0.0   │ fork    │
└────┴──────────────────────┴─────────────┴─────────┴─────────┴
```


#### Pm2 startup 
```bash
pm2 startup
```
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/systemctl_enable.png" width="80%"></img>  
```bash
systemctl enable pm2-root
pm2 save
```
* Routing : Root endpoint is always https://cdn.jsdelivr.net
#### LocalNet Routing
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/router_cdn_dns_jsdelivr.png" width="80%"></img>  

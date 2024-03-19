#### [Rescaling Global Manufacturing](https://github.com/universalbit-dev/CityGenerator/blob/master/Fab-City_Whitepaper.pdf) 
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/rescaling_fab_city_whitepaper.png" width="40%"></img>


#### Planet - Country - Region - City

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
```bash
pm2 status
```

```bash
┌────┬────────────────────┬──────────┬──────┬───────────┬──────────┬──────────┐
│ id │ name               │ mode     │ ↺    │ status    │ cpu      │ memory   │
├────┼────────────────────┼──────────┼──────┼───────────┼──────────┼──────────┤
│ 0  │ btc_pruned         │ fork     │ 0    │ online    │ 0%       │ 16.9mb   │
│ 4  │ cdn                │ fork     │ 12   │ online    │ 0%       │ 51.2mb   │
│ 5  │ cdn_europe         │ fork     │ 9    │ online    │ 125%     │ 50.0mb   │
│ 7  │ cdn_italy          │ fork     │ 1    │ online    │ 0%       │ 52.4mb   │
│ 8  │ cdn_italy_palermo  │ fork     │ 0    │ online    │ 0%       │ 13.4mb   │
│ 6  │ cdn_united_states  │ fork     │ 8    │ online    │ 0%       │ 51.2mb   │
│ 1  │ |BITCOIN-NODE|     │ fork     │ 1    │ online    │ 50%      │ 539.4mb  │
└────┴────────────────────┴──────────┴──────┴───────────┴──────────┴──────────┘
```
##### [blockchain net node](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain)


#### Pm2 startup 
```bash
pm2 startup
```
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/systemctl_enable.png" width="80%"></img>  
```bash
systemctl enable pm2-root
pm2 save
```

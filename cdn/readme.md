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
├────┼──────────────────────┼─────────────┼─────────┼─────────┼──────────┼────────┼──────┼───────────┼──────────┼──────────┼──────────┼──────────┤
│ 6  │ cdn                  │ default     │ 1.0.0   │ fork    │ 44379    │ 1s     │ 344  │ online    │ 0%       │ 52.2mb   │ uni… 
│ 7  │ cdn_europe           │ default     │ 1.0.0   │ fork    │ 44399    │ 0s     │ 328  │ online    │ 0%       │ 51.0mb   │ uni… 
│ 4  │ cdn_italy            │ default     │ 1.0.0   │ fork    │ 44418    │ 0s     │ 554  │ online    │ 0%       │ 49.6mb   │ uni… 
│ 5  │ cdn_italy_palermo    │ default     │ 1.0.0   │ fork    │ 44361    │ 1s     │ 349  │ online    │ 28.1%    │ 51.9mb   │ uni… 
│ 8  │ cdn_united_states    │ default     │ 1.0.0   │ fork    │ 44328    │ 2s     │ 317  │ online    │ 1.1%     │ 52.3mb   │ uni… 
└────┴──────────────────────┴─────────────┴─────────┴─────────┴──────────┴────────┴──────┴───────────┴──────────┴──────────┴──────────┴──────────┘
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

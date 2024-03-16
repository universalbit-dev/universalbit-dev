* ### [jsDelivr Network Map](https://www.jsdelivr.com/network#map)
* ### [Content Delivery Network](https://en.wikipedia.org/wiki/Content_delivery_network)


  * ### [GlobalPing-Cli](https://github.com/jsdelivr/globalping-cli)
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

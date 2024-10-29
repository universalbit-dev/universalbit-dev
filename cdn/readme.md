##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)

#### [Rescaling Global Manufacturing](https://github.com/universalbit-dev/CityGenerator/blob/master/Fab-City_Whitepaper.pdf) 

* clone project:
```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
cd universalbit-dev/cdn/
```

### To join the [Globalping probe network](https://globalping.io) all you have to do is run our container.
```
docker run -d --log-driver local --network host --restart=always --name globalping-probe globalping/globalping-probe
```

### To join the UniversalBit network what to do:
* [snap](https://snapcraft.io/docs/installing-snap-on-ubuntu) install [GlobalPing]
```bash
snap install globalping
```

#### Required npm packages to start the ecosystem
```bash
npm i && npm audit fix
npm i pm2 -g
pm2 start ecosystem.config.js
```
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/cdn_universalbit.png" width="auto"></img>

#### heavy disk memory usage (optimization of system performance)
```bash
pm2 start autoclean.js --exp-backoff-restart-delay=10000
```
---

* [Universalbit Content Delivery Network](https://universalbitcdn.it)   
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/gif/content_delivery_network_live.gif" width="auto"></img>

* [Machine learning (ML) powered anomaly detection](https://learn.netdata.cloud/docs/machine-learning-and-anomaly-detection/machine-learning-ml-powered-anomaly-detection)
##### Anomalies may exist without adequate "lenses" or "filters" to see them and may become visible only when the tools exist to define them
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/netdata_android_device.jpg" width="40%"></img>

#### Resources:
* ##### [Github Globalping Repository](https://github.com/jsdelivr/globalping)
* ##### [Pm2 startup script](https://pm2.keymetrics.io/docs/usage/startup/)
* ##### [Globalping releases](https://github.com/jsdelivr/globalping-cli/releases)
* ##### [GlobalPing-Cli](https://github.com/jsdelivr/globalping-cli)
* ##### [Fab-City_Whitepaper](https://github.com/universalbit-dev/CityGenerator/blob/master/Fab-City_Whitepaper.pdf)
* ##### [High Availability Clusters](https://github.com/universalbit-dev/HArmadillium)
* ##### [Content Delivery Network](https://universalbitcdn.it/spaces/)
* ##### [Conceptual framework](https://en.wikipedia.org/wiki/Conceptual_framework)
* ##### [Blockchain Infrastructure](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain)


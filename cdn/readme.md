# UniversalBit Content Delivery Network (CDN) Documentation

This guide provides instructions for setting up the UniversalBit CDN and joining networks like GlobalPing. It also includes tips for performance optimization and links to additional resources.

---

## Table of Contents
1. [Support and References](#support-and-references)
2. [Overview](#overview)
3. [Project Setup](#project-setup)
4. [Joining GlobalPing Probe Network](#joining-globalping-probe-network)
5. [Performance Optimization](#performance-optimization)
6. [Resources](#resources)

---

## Support and References

- [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)
- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)
- [Bash References](https://help.ubuntu.com/community/Bash)

---

## Overview

The UniversalBit CDN is a distributed system designed to optimize content delivery. It integrates with Docker, Snap, and npm-based ecosystems to support features like machine learning-powered anomaly detection and high availability clusters.

![Router CDN DNS jsDelivr](https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/router_cdn_dns_jsdelivr.png)

For more on its conceptual and practical applications, read the [Fab-City Whitepaper](https://github.com/universalbit-dev/CityGenerator/blob/master/Fab-City_Whitepaper.pdf).

---

## Project Setup

### Clone the Repository
Set up the project by cloning the repository:
```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
cd universalbit-dev/cdn/
```

---

## Joining GlobalPing Probe Network

To join the [GlobalPing Probe Network](https://globalping.io), run the provided container. First, ensure you have Docker installed:

### Install Docker
Follow the instructions in the [Docker Engine Installation Guide](https://docs.docker.com/engine/install/ubuntu/).

### Run the GlobalPing Container
```bash
docker run -d --log-driver local --network host --restart=always --name globalping-probe globalping/globalping-probe
```

For more information about using Docker with firewalls, refer to [How to Use UFW Firewall with Docker Containers](https://blog.jarrousse.org/2023/03/18/how-to-use-ufw-firewall-with-docker-containers/).

---

## Performance Optimization

### Install Required npm Packages
Install the necessary npm packages and start the ecosystem:
```bash
npm i && npm audit fix
npm i pm2 -g
pm2 start ecosystem.config.js
```

### Heavy Disk Memory Usage Optimization
Run the following command to optimize system performance:
```bash
pm2 start autoclean.js --exp-backoff-restart-delay=10000
```

![Content Delivery Network in Action](https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/gif/content_delivery_network_live.gif)

---

## Resources

Explore additional resources to enhance your understanding and implementation:

- [UniversalBit Content Delivery Network](https://www.universalbitcdn.it)
- [Machine Learning (ML) Powered Anomaly Detection](https://learn.netdata.cloud/docs/machine-learning-and-anomaly-detection/machine-learning-ml-powered-anomaly-detection)
- [GitHub GlobalPing Repository](https://github.com/jsdelivr/globalping)
- [PM2 Startup Script](https://pm2.keymetrics.io/docs/usage/startup/)
- [GlobalPing Releases](https://github.com/jsdelivr/globalping-cli/releases)
- [GlobalPing CLI](https://github.com/jsdelivr/globalping-cli)
- [Fab-City Whitepaper](https://github.com/universalbit-dev/CityGenerator/blob/master/Fab-City_Whitepaper.pdf)
- [High Availability Clusters](https://github.com/universalbit-dev/HArmadillium)
- [Conceptual Framework](https://en.wikipedia.org/wiki/Conceptual_framework)
- [Blockchain Infrastructure](https://github.com/universalbit-dev/universalbit-dev/tree/main/blockchain)

---

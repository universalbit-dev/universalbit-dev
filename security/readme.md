# Security Projects Overview

Welcome to the **Security Projects** section of UniversalBit. This document provides an overview and relevant resources.

---

# 🛡️ Reducing Net Frustration

## 📖 Overview

Asymmetric routing occurs when packets follow different paths for outgoing and incoming traffic. 
This setup centralizes the flow through a stateful firewall (**IPFire**) and a dedicated DNS sinkhole (**Pi-hole**) to ensure path consistency, lower latency, and higher security.

### 🕸️ Physical Network Topology (RJ45)

### Port Mapping Summary
### LAN
1. **ZTE ONT (LAN01)** ➔ **Router (WAN Port)** 
This establishes the external Internet connection.
2. **Router (LAN02)** ➔ **Switch (LAN02)** 
This bridges the router's internal network to the Gigabit Switch.
3. **Switch (LAN03)** ➔ **IPFire (ThinClient #01)** 
Dedicated hardware for DMZ and Intrusion Detection.
4. **Switch (LAN04)** ➔ **Pi-hole (ThinClient #02)** 
Dedicated hardware for DNS/URL filtering.
5. **Switch (LAN01 / LAN05)** 
Available for additional hardwired devices or Access Points.

| Device | Port | Connection Target |
| --- | --- | --- |
| **ZTE ONT** | LAN01 | Router WAN |
| **Router** | LAN02 | Switch LAN02 |
| **Switch** | LAN03 | IPFire (ThinClient #01) |
| **Switch** | LAN04 | Pi-hole (ThinClient #02) |
| **Switch** | LAN01/05 | Available / AP / Extenders |

### ROUTER (Gateway)

* **IPv4 DHCP:** Enabled (Auto).
* **IPv6 DHCPv6:** Enabled (Manual) to reduce overhead/latency.
* **DNS 1:** `2a01:4f8:1c0c:8274::1` (LibreDNS)
* **DNS 2:** `2001:4860:4860::8888` (Google)
* **DNS 3:** `fe80::cc19:6f42:328c:eec8` (Local)
--> MTU 1492
--> Refresh 86400s
--> Prefix Delegate Type: Disabled

### Wifi
* **Centralized SSID:** Unified name for 2.4G/5G for seamless handoff.

| Component | Role | Configuration |
|---|---|---|
| Router Wifi | Router | Default Setup |
| Extender | Extender | Default Setup |
| Router Wifi Access Point | Access Point | Default Setup |

---

### DMZ
* **ORANGE (DMZ):** REJECT all incoming traffic by default

| Zone | Host | Firewall (DMZ) |
|---|---|---|
| ORANGE (DMZ) | HOST | REJECT all incoming traffic |
---

## 🛠️ Infrastructure Components

* **ThinClient #01:** [IPFire Linux Distro](https://www.ipfire.org/docs/installation) (Hardened firewall, DoT recursor, IDS)
  - Zones: RED, GREEN, ORANGE, BLUE
  - IDS: enabled on all zones
  - DNS protocol: TLS (recursor mode)
  - Forwarders: **IPFire TLS DNS** list
  
* **ThinClient #02:** [Pi-hole Adblocker](https://docs.pi-hole.net/main/basic-install/) (DNS/URL Filtering)
  - Upstream: IPFire TLS DNS (local)
  - Gravity: Steven Black lists (or equivalent curated lists)
  - Layer 3 inspection: Enabled (use with care / ensure performance)
  - Admin UI: restrict to management VLAN / IP

### IPFire Customization >>(The Shield)<<

* **Zone ORANGE (DMZ):** REJECT all incoming traffic by default.
* **IDS (Intrusion Detection):** Enabled on RED, GREEN, ORANGE, and BLUE zones.
* **DNS over TLS (DoT):** (Recursor Mode)
* **DNS Forwarding** (IPFire TLS DNS list)

**Firewall — DMZ Rule**
| Setting | Value |
|---|---|
| Zone | ORANGE |
| Rule for incoming traffic | REJECT All Incoming Traffic |
---

**IDS (Intrusion Detection)**
| Setting | Value |
|---|---|
| Status | Enabled |
| Applicable zones | RED, GREEN, ORANGE, BLUE |
| Ruleset | Updated Community Rules |
---

**DNS Configuration (DoT)**
| Setting | Value |
|---|---|
| Use ISP-assigned DNS servers | Disabled |
| Protocol for DNS queries | TLS |
| Enable Safe Search | Disabled |
| Include YouTube in Safe Search | Disabled |
| QNAME Minimisation | Standard |
---

**DNS Forwarder** IPFire TLS DNS list
| Zone | Nameserver | Remark |
|---|---:|---|
| noads.libredns.gr | 116.202.176.26 | LibreDNS |
| one.one.one.one | 1.1.1.1 | Cloudflare |
| dot.ffmuc.net | 5.1.66.255 | Freifunk München e.V. |
| dns.sb | 185.222.222.222 | dns.sb |
| open.dns0.eu | 193.110.81.254 | DNS0.EU Open |
| anycast.uncensoreddns.org | 91.239.100.100 | UncensoredDNS |
| dot1.applied-privacy.net | 146.255.56.98 | Foundation for Applied Privacy |
| dns.cmrg.net | 199.58.83.33 | CMRG DNS |
| dns.digitale-gesellschaft.ch | 185.95.218.42 | Digitale Gesellschaft Schweiz |
| dns3.digitalcourage.de | 5.9.164.112 | Digitalcourage e.V. |
| recursor01.dns.lightningwirelabs.com | 81.3.27.54 | Lightning Wire Labs |
| unicast.uncensoreddns.org | 89.233.43.71 | UncensoredDNS |
| unfiltered.joindns4.eu | 86.54.11.100 | DNS4EU |
| public.ns.nwps.fi | 95.217.11.63 | NWPS.fi |
| kids.ns.nwps.fi | 135.181.103.31 | NWPS.fi |
| ns0.fdn.fr | 80.67.169.12 | French Data Network (FDN) |
| ns1.fdn.fr | 80.67.169.40 | French Data Network (FDN) |
| dns.neutopia.org | 89.234.186.112 | Neutopia |
| dns.linuxpatch.com | 45.80.1.6 | LinuxPatch.com |
| kaitain.restena.lu | 158.64.1.29 | Restena Foundation |
| nl.resolv.flokinet.net | 185.246.188.51 | FlokiNET |
| getdnsapi.net | 185.49.141.37 | GetDNS |
| ro.resolv.flokinet.net | 185.247.225.17 | FlokiNET |
| dns.njal.la | 95.215.19.53 | Njalla |
---

### Pi-hole Configuration >>(The Filter)<<

* **Upstream DNS:** Configured to use the **IPFire TLS DNS** list.
* **Gravity:** Steven Black URL List enabled.
* **Layer 3:** Enabled for deep packet inspection/filtering.
---

### 🛡️ Why This Setup Works

* **Isolation:** By putting the ONT strictly into the Router WAN, you ensure the Router's NAT/Firewall handles the initial handshake.
* **Security Stack:** The Switch allows IPFire and Pi-hole to sit on the same high-speed gigabit backplane, reducing the latency for DNS queries and packet inspection.
* **Scalability:** Since you have LAN01 and LAN05 available on the Switch, you can add more APs for the "Strong Signal" Wifi strategy without rearranging the core security logic.
* **Access Points** are hardwired via RJ45 to the Gigabit Switch to maintain maximum throughput.
* **Extenders** Centralized SSID:Unified name for 2.4G/5G for seamless handoff.

> **Note:** This configuration prioritizes **Micro-networking** (low latency for multimedia) and **Macro-networking** (high security via TLS forwarding).

 

## 📢 Support the UniversalBit Project
Help us grow and continue innovating!  
- [Support the UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)  
- [Learn about Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)  
- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/)

---

## Featured Projects

### 1. [IPFire](https://github.com/universalbit-dev/universalbit-dev/tree/main/ipfire)
- **Website**: [IPFire Official Website](https://www.ipfire.org/)
- **Description**:
  - IPFire is a hardened open-source Linux distribution designed to function as a router and firewall.
  - It features a web-based management console for ease of configuration and supports the addition of server services via add-ons.
  - Originally a fork of IPCop, IPFire has been rebuilt using Linux From Scratch since version 2.
- **Key Features**:
  - Robust firewall and router capabilities.
  - Flexible add-on support for extending functionality.
  - Community-driven project with a strong security focus.
- **Learn More**:
  - [IPFire on Wikipedia](https://en.wikipedia.org/wiki/IPFire)

---

### 2. [SELKS](https://github.com/StamusNetworks/SELKS/blob/master/README.rst)
- **Website**: [SELKS Official Website](https://www.stamus-networks.com/selks)
- **Description**:
  - SELKS is a Debian-based platform for Intrusion Detection/Prevention Systems (IDS/IPS) and Network Security Monitoring (NSM).
  - Released under GPLv3 by Stamus Networks, it integrates tools like Suricata, Elasticsearch, Logstash, Kibana, and the Stamus Scirius Community Edition.
  - SELKS can be deployed via Docker Compose on Linux or Windows or as ISO images for bare-metal or air-gapped environments.
- **Key Features**:
  - Comprehensive IDS/IPS and NSM capabilities.
  - Pre-configured with powerful tools for data analysis and threat detection.
  - Flexible deployment options (Docker, bare-metal, air-gapped environments).
- **Learn More**:
  - [SELKS on GitHub](https://github.com/StamusNetworks/SELKS)

### 3. [Pi-hole](https://github.com/pi-hole/pi-hole)
- **Website**: [Pi-hole Official Website](https://pi-hole.net/)
- **Description**:
  - Pi-hole is an open‑source network-wide ad, tracker and telemetry blocker that works by acting as a DNS sinkhole for known unwanted domains.
  - It intercepts DNS queries from clients on your network and blocks requests for domains on configurable blocklists, reducing ads, trackers and some forms of malware at the DNS level.
  - Lightweight and versatile — commonly run on Raspberry Pi devices, but equally deployable on Debian/Ubuntu, VMs, containers (Docker), or cloud instances.
  - Includes the FTL (Faster Than Light) engine for fast DNS resolution, statistics and real-time query logging.
- **Key Features**:
  - Network-wide ad and tracker blocking via DNS sinkholing.
  - Web-based admin interface with dashboards, query logs and client/group management.
  - Customizable blocklists (Gravity), whitelists, blacklists, and regex filtering.
  - Optional built-in DHCP server or integration with existing DHCP services.
  - Supports upstream DNS providers and can be paired with DoH/DoT/Unbound/stubby for encrypted/resilient upstream resolution.
  - Low resource usage; easy Docker deployment and scripting/API access for automation.
  - Query analytics, per-client stats, and per-group policies.
- **Learn More**:
  - [Pi-hole GitHub repository](https://github.com/pi-hole/pi-hole)
  - [Pi-hole Documentation](https://docs.pi-hole.net/)
  - [Pi-hole on Wikipedia](https://en.wikipedia.org/wiki/Pi-hole)
---

## Additional Information
This file serves as a gateway to the UniversalBit. For further details on installation, configuration, or contributions, explore the respective project repositories and official documentation.

---

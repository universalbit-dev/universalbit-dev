# 🔐 Security Projects Overview

Welcome to the **Security Projects** section of UniversalBit.  
This document describes the layered home-lab security architecture built around **IPFire**, **Pi-hole**, and **SELKS** — forming a privacy-first, threat-aware micro-network.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Physical Network Topology](#️-physical-network-topology-rj45)
3. [Router / Gateway Configuration](#router-gateway)
4. [DHCP Allocation Table](#dhcp-server-ip-allocation-table)
5. [Wi-Fi Configuration](#wifi)
6. [DMZ Policy](#dmz)
7. [Infrastructure Components](#️-infrastructure-components)
   - [IPFire — The Shield](#ipfire-customization-the-shield)
   - [Pi-hole — The Filter](#pi-hole-configuration-the-filter)
8. [DNS Forwarder List](#dns-forwarder--ipfire-tls-dns-list)
9. [Why This Setup Works](#️-why-this-setup-works)
10. [Hardening Checklist](#-hardening-checklist)
11. [Featured Projects](#featured-projects)
12. [Support](#-support-the-universalbit-project)

---

## 📖 Overview

**Asymmetric routing** occurs when packets follow different paths for outgoing and incoming traffic.  
This architecture centralises all traffic through a layered security stack:

```
Internet
   │
[ZTE ONT]
   │  (WAN)
[Router / NAT / DHCP]
   │  (LAN)
[Gigabit Switch]
   ├──[IPFire ThinClient #01]  ← Stateful Firewall · IDS · DoT Recursor
   ├──[Pi-hole ThinClient #02] ← DNS Sinkhole · Ad/Tracker Filter
   └──[APs / Extenders / Devices]
```

**Traffic flow — DNS query lifecycle:**

1. Client issues a DNS query → routed to **Pi-hole**.
2. Pi-hole checks its Gravity blocklist; if not blocked → forwards to **IPFire DoT recursor**.
3. IPFire resolves over **TLS** via a curated upstream forwarder list.
4. All packets pass through IPFire's **stateful firewall** and **IDS** before reaching the internet.

---

## 🕸️ Physical Network Topology (RJ45)

### Port Mapping Summary

| # | Device | Port | Connection Target | Purpose |
|---|--------|------|-------------------|---------|
| 1 | **ZTE ONT** | LAN01 | Router WAN | External internet uplink |
| 2 | **Router** | LAN02 | Switch LAN02 | Bridges internal LAN to switch |
| 3 | **Switch** | LAN03 | IPFire (ThinClient #01) | Firewall / DMZ / IDS |
| 4 | **Switch** | LAN04 | Pi-hole (ThinClient #02) | DNS / URL filtering |
| 5 | **Switch** | LAN01/05 | APs / Extenders / Spare | Expansion / wireless backhaul |

---
## Router (Gateway)

| Setting | Value | Notes |
|---------|-------|-------|
| **IPv4 DHCP** | Enabled (Auto) | Default home router behavior |
| **IPv6 DHCPv6** | Enabled (Auto) | Use router defaults unless ISP requires manual tuning |
| **DNS 1** | Auto (ISP/Router Default) | Primary resolver |
| **DNS 2** | Auto (ISP/Router Default) | Secondary resolver |
| **DNS 3** | Not set (default) | Optional; omit link-local/custom resolver in public docs |
| **MTU** | Auto (Default) | Typical default is often PPPoE 1492 or Ethernet 1500 |
| **Refresh** | Default | Keep vendor default lease/renew timers |
| **Prefix Delegate Type** | Auto / Default | Depends on ISP IPv6 delegation model |
---
## DHCP Server IP Allocation Table

**Network:** `192.168.1.0/24` · **Subnet Mask:** `255.255.255.0`

> **Default (recommended):** Enable **only Router DHCP (Auto)** as the active DHCP service for the subnet.

| Server | Start IP | End IP | Purpose | Status |
|--------|:--------:|:------:|---------|--------|
| Generic Router | 192.168.1.2 | 192.168.1.253 | All LAN clients (default pool) | ✅ Enabled (Auto) |
| IPFire | 192.168.1.121 | 192.168.1.152 | Security / Firewall hosts | ⚙️ Optional (disabled unless explicitly used) |
| Pi-hole | 192.168.1.153 | 192.168.1.253 | Ad-blocking clients | ⚙️ Optional (disabled unless explicitly used) |

> **Notes:**
> - `192.168.1.1` is reserved as the gateway/router IP.
> - Keep DHCP pools non-overlapping if optional segmented scopes are enabled.
> - Use static DHCP reservations for infrastructure nodes (IPFire, Pi-hole, APs).
> - Do **not** run multiple DHCP servers on the same subnet unless strictly coordinated.
---



## Wifi

**Centralized SSID:** single unified name for 2.4 GHz / 5 GHz enables seamless client handoff.

| Component | Role | Recommended Configuration |
|-----------|------|--------------------------|
| Router Wi-Fi | Primary AP | Default setup; disable WPS |
| Extender | Range extender | Same SSID/passphrase as router |
| Access Point (RJ45 backhaul) | Secondary AP | Same SSID; set to AP mode |

> **Tip:** Use WPA3-Personal where supported. Enable client isolation on guest SSIDs.

---

## DMZ

The **ORANGE** zone in IPFire acts as a DMZ (De-Militarised Zone).

| Zone | Policy | Rationale |
|------|--------|-----------|
| ORANGE (DMZ) | **REJECT** all unsolicited inbound traffic | Only stateful reply traffic is permitted |
| RED (WAN) | Block all inbound by default | Managed via IPFire firewall rules |
| GREEN (LAN) | Allow outbound; inspect inbound | Trusted internal zone |
| BLUE (Wi-Fi) | Isolated from GREEN | Limits lateral movement from wireless clients |

> **Stateful firewall rule:** Only return traffic for connections **initiated** from allowed internal sources is permitted through the DMZ.

---

## 🛠️ Infrastructure Components

### ThinClient #01 — IPFire

[📥 Installation Guide](https://www.ipfire.org/docs/installation)

| Property | Value |
|----------|-------|
| Role | Hardened firewall, DoT recursor, IDS |
| Zones | RED · GREEN · ORANGE · BLUE |
| IDS | Enabled on **all** zones |
| DNS Protocol | TLS (Recursor Mode) |
| DNS Forwarders | IPFire TLS DNS list (see below) |

### ThinClient #02 — Pi-hole

[📥 Installation Guide](https://docs.pi-hole.net/main/basic-install/)

| Property | Value |
|----------|-------|
| Role | DNS sinkhole, ad/tracker/URL filter |
| Upstream DNS | IPFire DoT recursor (local) |
| Blocklist | Steven Black lists (curated) |
| Layer 3 inspection | ❌ Not applicable — DNS filtering only |
| Admin UI | Restrict to management VLAN / specific IP |

---

## IPFire Customization — The Shield

### Firewall — DMZ Rule

| Setting | Value |
|---------|-------|
| Zone | ORANGE |
| Incoming traffic rule | REJECT all unsolicited inbound traffic |

---

### IDS — Intrusion Detection System (Suricata)

| Setting | Value |
|---------|-------|
| Status | ✅ Enabled |
| Applicable zones | RED · GREEN · ORANGE · BLUE |
| Ruleset | Updated Community Rules |

> **Tip:** Regularly update IDS rulesets via the IPFire web UI (`IPS → Update`) to catch emerging threats.

---

### DNS Configuration — DoT (Recursor Mode)

| Setting | Value |
|---------|-------|
| Use ISP-assigned DNS | ❌ Disabled |
| Protocol for DNS queries | TLS |
| Safe Search | ✅ Enabled |
| YouTube in Safe Search | ✅ Enabled |
| QNAME Minimisation | Standard |

---

![Domain Name System Diagram](https://raw.githubusercontent.com/universalbit-dev/universalbit-dev/main/docs/assets/images/Domain_Name_System.png)

---

## DNS Forwarder — IPFire TLS DNS List

All upstream resolvers use **DNS-over-TLS (DoT)** on port 853.  
Privacy-focused, censorship-resistant providers are preferred.

| Hostname | IP Address | Provider | Notes |
|----------|:----------:|----------|-------|
| noads.libredns.gr | 116.202.176.26 | LibreDNS | Ad-free resolver |
| one.one.one.one | 1.1.1.1 | Cloudflare | High performance |
| dot.ffmuc.net | 5.1.66.255 | Freifunk München e.V. | Community / non-profit |
| dns.sb | 185.222.222.222 | dns.sb | Privacy-focused |
| open.dns0.eu | 193.110.81.254 | DNS0.EU Open | EU-based |
| anycast.uncensoreddns.org | 91.239.100.100 | UncensoredDNS | Uncensored / anycast |
| dot1.applied-privacy.net | 146.255.56.98 | Applied Privacy Foundation | No-log policy |
| dns.cmrg.net | 199.58.83.33 | CMRG DNS | Research-grade |
| dns.digitale-gesellschaft.ch | 185.95.218.42 | Digitale Gesellschaft CH | Swiss non-profit |
| dns3.digitalcourage.de | 5.9.164.112 | Digitalcourage e.V. | German civil rights org |
| recursor01.dns.lightningwirelabs.com | 81.3.27.54 | Lightning Wire Labs | |
| unicast.uncensoreddns.org | 89.233.43.71 | UncensoredDNS | Unicast endpoint |
| unfiltered.joindns4.eu | 86.54.11.100 | DNS4EU | EU initiative |
| public.ns.nwps.fi | 95.217.11.63 | NWPS.fi | Finnish provider |
| kids.ns.nwps.fi | 135.181.103.31 | NWPS.fi | Family-safe filter |
| ns0.fdn.fr | 80.67.169.12 | French Data Network (FDN) | Non-profit |
| ns1.fdn.fr | 80.67.169.40 | French Data Network (FDN) | Non-profit |
| dns.neutopia.org | 89.234.186.112 | Neutopia | |
| dns.linuxpatch.com | 45.80.1.6 | LinuxPatch.com | |
| kaitain.restena.lu | 158.64.1.29 | Restena Foundation | Luxembourg academic |
| nl.resolv.flokinet.net | 185.246.188.51 | FlokiNET | Iceland / NL privacy host |
| getdnsapi.net | 185.49.141.37 | GetDNS | Reference implementation |
| ro.resolv.flokinet.net | 185.247.225.17 | FlokiNET | Romania endpoint |
| dns.njal.la | 95.215.19.53 | Njalla | Privacy-first registrar |

> **Tip:** Use at least 3–5 diverse providers to avoid single-provider dependency. Prefer non-profit / European providers for GDPR-aligned privacy guarantees.

---

## Pi-hole Configuration — The Filter

| Setting | Value |
|---------|-------|
| Upstream DNS | IPFire DoT recursor (local, `192.168.1.121`) |
| Gravity blocklist | Steven Black URL list (ads + malware + social) |
| Layer 3 inspection | ❌ DNS filtering only — no packet inspection |
| DHCP role | Optional; defer to router or IPFire if preferred |

> **Tip:** Add supplemental blocklists such as [oisd.nl](https://oisd.nl) or [hagezi/dns-blocklists](https://github.com/hagezi/dns-blocklists) in Pi-hole's Gravity to broaden coverage without impacting performance.

---

## 🛡️ Why This Setup Works

| Benefit | Detail |
|---------|--------|
| **Isolation** | ONT → Router WAN keeps NAT/firewall as the first packet filter, before traffic reaches internal hosts |
| **Performance** | Gigabit switch backplane keeps IPFire ↔ Pi-hole DNS latency under 1 ms |
| **Scalability** | LAN01/LAN05 switch ports support additional APs without touching core security topology |
| **Throughput** | APs hardwired via RJ45 maintain maximum Wi-Fi backhaul speed |
| **Seamless roaming** | Unified SSID across router, extenders, and APs enables seamless 2.4 G / 5 G handoff |
| **Encrypted DNS** | All upstream resolution uses DNS-over-TLS — no plaintext DNS leaving the network |
| **Threat detection** | Suricata IDS covers all four IPFire zones, logging and alerting on known signatures |

> **Design philosophy:** This configuration balances **Micro-networking** (low latency for multimedia / real-time traffic) with **Macro-networking** (high security via TLS forwarding, IDS, and DNS sinkholing).

---

## 🔒 Hardening Checklist

- [ ] Disable WPS on all Wi-Fi access points
- [ ] Enable WPA3-Personal where hardware supports it
- [ ] Restrict Pi-hole admin UI to a management IP / VLAN
- [ ] Set static DHCP leases for IPFire and Pi-hole (prevents IP drift)
- [ ] Schedule automatic IDS ruleset updates in IPFire (`IPS → Update`)
- [ ] Review Pi-hole query logs weekly for anomalous domains
- [ ] Keep IPFire and Pi-hole on the latest stable releases
- [ ] Disable IPv6 on zones where not needed to reduce attack surface
- [ ] Enable IPFire location-based blocking for high-risk ASNs (optional)
- [ ] Enable client isolation on guest / IoT SSIDs

---

## Featured Projects

### 1. [IPFire](https://github.com/universalbit-dev/universalbit-dev/tree/main/ipfire)

- **Website:** [ipfire.org](https://www.ipfire.org/)
- **Description:** IPFire is a hardened open-source Linux distribution functioning as a router and firewall. Built using Linux From Scratch (since v2), it offers a web-based management console and an extensive add-on ecosystem.
- **Key Features:**
  - Stateful firewall with zone-based policy (RED / GREEN / ORANGE / BLUE)
  - Integrated Suricata IDS/IPS across all zones
  - DNS-over-TLS recursor with QNAME minimisation
  - Web proxy, traffic shaping, VPN (IPsec / OpenVPN / WireGuard)
  - Add-on packages: Zeek, ClamAV, Asterisk, and more
- **Learn More:** [IPFire on Wikipedia](https://en.wikipedia.org/wiki/IPFire)

---

### 2. [SELKS](https://github.com/StamusNetworks/SELKS/blob/master/README.rst)

- **Website:** [stamus-networks.com/selks](https://www.stamus-networks.com/selks)
- **Description:** SELKS is a Debian-based IDS/IPS and Network Security Monitoring (NSM) platform released under GPLv3 by Stamus Networks. It integrates Suricata, Elasticsearch, Logstash, Kibana, and Scirius CE into a single deployable unit.
- **Key Features:**
  - Full IDS/IPS + NSM pipeline out of the box
  - Kibana dashboards for real-time threat visualisation
  - Docker Compose deployment (Linux / Windows) or bare-metal ISO
  - Air-gapped environment support
- **Learn More:** [SELKS on GitHub](https://github.com/StamusNetworks/SELKS)

---

### 3. [Pi-hole](https://github.com/pi-hole/pi-hole)

- **Website:** [pi-hole.net](https://pi-hole.net/)
- **Description:** Pi-hole is a network-wide DNS sinkhole that blocks ads, trackers, and telemetry for every device on the network — no client-side software required. It includes the FTL (Faster Than Light) engine for fast DNS resolution, statistics, and real-time query logging.
- **Key Features:**
  - Network-wide ad and tracker blocking via DNS sinkholing
  - Web admin UI with dashboards, query logs, and per-client/group policies
  - Gravity blocklist engine with regex, whitelist, and blacklist support
  - Optional built-in DHCP server
  - Pairs with DoT / DoH / Unbound / stubby for encrypted upstream resolution
  - Low resource usage; Docker-friendly; REST API for automation
- **Learn More:**
  - [Pi-hole Documentation](https://docs.pi-hole.net/)
  - [Pi-hole on Wikipedia](https://en.wikipedia.org/wiki/Pi-hole)

---

## 📢 Support the UniversalBit Project

Help us grow and continue innovating!

- [💖 Support the UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)
- [📖 Learn about Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)
- [📚 Bash Reference Manual](https://www.gnu.org/software/bash/manual/)

---

## Additional Information

This file serves as the entry point for UniversalBit's security stack documentation.  
For installation, configuration, or contribution details, explore the respective project repositories and their official documentation linked above.

---

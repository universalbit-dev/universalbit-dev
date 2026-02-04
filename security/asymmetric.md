# Stop IP spoofing on Ubuntu: asymmetric routing & rp_filter

When routing between a source and destination is asymmetric, a stateful firewall that only sees one direction of a connection may drop the return packets because it does not recognize the session. The ASCII diagram below illustrates the problem:

```
[ SOURCE ]
|
| (Path A) ------> [ FIREWALL 1 ] ------> [ DESTINATION ]
|
| <------ [ FIREWALL 2 ] <------ (Path B) -----+
|                                              |
|                                              [ PACKET DROPPED ]
|                                              (Firewall 2 doesn't recognize the session!)
```

Problem summary
- Stateful firewalls track sessions by observing packets in both directions. If the reply packet comes back to the destination via a different firewall (asymmetric routing), that firewall may not have the outgoing state and will therefore drop the return packet.
- Attackers can exploit asymmetric paths to hide spoofed source addresses, causing confusing drops and possible denial-of-service and excessive resource usage by user-space firewalls/connection tracking.

Kernel-level (gold standard) fix: rp_filter
- Rather than relying solely on user-space firewall state, let the kernel drop obviously spoofed packets early using reverse-path filtering (`rp_filter`).
- `rp_filter` checks whether the incoming packet's source IP is reachable via the interface the packet arrived on. If not, the kernel treats it as a spoof (a "martian").
- This is lightweight (kernel-level) and prevents many spoofing attempts before they reach higher protocol layers and consume RAM/CPU.

Quick 3-step setup (simple and robust)
1. Persist configuration (two common approaches):
   - Traditional: `sudo nano /etc/sysctl.conf` and add the lines below.
   - Preferred on modern systems: create `/etc/sysctl.d/99-rpfilter.conf` with the same lines to avoid overwriting distribution defaults.

Add these lines:
```
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.log_martians = 1
```

2. Apply immediately:
```
sudo sysctl --system    # reloads /etc/sysctl.conf and /etc/sysctl.d/*.conf
# or for a one-off change:
sudo sysctl -w net.ipv4.conf.all.rp_filter=1
sudo sysctl -w net.ipv4.conf.all.log_martians=1
```

3. Verify:
```
sysctl net.ipv4.conf.all.rp_filter
sysctl net.ipv4.conf.default.rp_filter
sysctl net.ipv4.conf.all.log_martians
```

Strict vs Loose modes
- `rp_filter = 0` : disabled.
- `rp_filter = 1` : strict mode (recommended for single‑homed hosts or symmetric routing). The kernel requires that the route for the source IP points out the same interface the packet arrived on.
- `rp_filter = 2` : loose mode (useful for multi‑homed/dual‑WAN setups). Loose mode only checks that the source is reachable via any interface — it is more permissive and will allow asymmetric routing that has a valid route back to the source.

Practical advice for multi‑homed setups
- If you run dual‑WAN or policy-based routing, strict mode (`1`) may break valid asymmetric flows. Use `rp_filter = 2` on hosts/interfaces expected to see asymmetric traffic and keep `1` on others.
- Set per‑interface values when necessary:
```
sudo sysctl -w net.ipv4.conf.eth0.rp_filter=1
sudo sysctl -w net.ipv4.conf.eth1.rp_filter=2
```
- Persist per‑interface settings in your sysctl config file using the same keys (for example, in `/etc/sysctl.d/99-rpfilter.conf`).

Logging martians (useful for troubleshooting)
- Enable `net.ipv4.conf.all.log_martians = 1` to have the kernel log packets it considers spoofed (martians). The logs help identify misconfigured routing, accidental asymmetry, or active spoofing.
- View logs:
```
# systemd (journal):
sudo journalctl -k -f | grep -i martian

# or traditional syslog:
sudo grep -i martian /var/log/syslog
```
- Example martian line (format varies by kernel/distribution):
```
kernel: martian source 203.0.113.5 from 198.51.100.7 on dev eth0
```

Integration with stateful firewalls
- `rp_filter` is complementary to connection tracking. The kernel-level filter reduces spoofed traffic before it reaches conntrack tables and user-space firewalls (iptables/nftables), saving memory and reducing false drops.
- Keep your firewall rules allowing established/related traffic to avoid blocking legitimate replies. Example iptables rule:
```
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
```

Testing and rollback
- Apply changes during a maintenance window if possible: misconfigurations in multi‑homed environments can cause service disruption.
- Quick rollback:
```
sudo sysctl -w net.ipv4.conf.all.rp_filter=0
sudo sysctl -w net.ipv4.conf.all.log_martians=0
```

IPv6 note
- `rp_filter` applies to IPv4 only. IPv6 routing and validation use different controls (e.g., RA/ND handling, firewall rules) and should be considered separately.

Summary / Recommendation
- For most single‑homed servers and simple networks: enable strict reverse‑path filtering (`rp_filter=1`) and enable martian logging.
- For multi‑homed or policy‑routed hosts: prefer per‑interface `rp_filter` settings or `rp_filter=2` where needed, and combine with careful monitoring of martian logs.
- Prefer placing this check in the kernel — it’s inexpensive and removes spoofed packets before they reach user‑space firewalls.

Further reading and notes
- Use `/etc/sysctl.d/99-rpfilter.conf` on modern systems to persist settings cleanly.
- If you want a version focused more on martian logging for technical audiences, I can produce a second document with example logs, log rotation guidance, and automated alerting snippets.
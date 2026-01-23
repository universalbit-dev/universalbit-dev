[![Support UniversalBit Project](https://img.shields.io/badge/Support%20UniversalBit-Project-blue?style=flat-square&logo=github)](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)
[![Disambiguation](https://img.shields.io/badge/Disambiguation-Wikipedia-yellow?style=flat-square&logo=wikipedia)](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)
[![Bash Reference Manual](https://img.shields.io/badge/Bash-Reference%20Manual-brightgreen?style=flat-square&logo=gnu-bash)](https://www.gnu.org/software/bash/manual/html_node/index.html)
[![Join Mastodon](https://img.shields.io/badge/Join-Mastodon-6364ff?logo=mastodon&logoColor=white&style=flat-square)](https://mastodon.social/invite/wTHp2hSD)

---

# letsencrypt.md — Enable HTTPS via Let's Encrypt

The **letsencrypt.md** file provides instructions for enabling HTTPS on your website using Let's Encrypt.

## Key Points

1. **References**
   - [LetsEncrypt.org](https://letsencrypt.org)
   - [Documentation](https://letsencrypt.org/docs/)

2. **Setup for Debian 11**
   ```bash
   apt install certbot python3-certbot-apache -y
   git clone https://github.com/letsencrypt/letsencrypt
   cd letsencrypt
   ./letsencrypt-auto --help   # Suite Command List
   ```
   **Usage Examples**
   ```bash
   ./letsencrypt-auto --apache -d yourdomain.com
   ```
   **Certonly Example**
   ```bash
   ./letsencrypt-auto certonly --standalone --email infouniversalbits@gmail.com -d universalbit.it 
   ```

3. **Setup for Debian 12**
   - [LetsEncrypt Debian 12 Setup](https://www.server-world.info/en/note?os=Debian_12&p=ssl&f=2)

---

## 🎉 Latest News: Let's Encrypt Domain & IP Validation

Let's Encrypt has made updates regarding how certificates are granted, particularly affecting domain and IP validation methods.  
As of January 2026:
- **Domain validation**: You must prove ownership/control of the domain via HTTP/HTTPS challenge, DNS challenge (TXT record), or use the Certbot with supported options.
- **IP certificate support**: Let’s Encrypt _does not_ issue certificates directly to IP addresses unless proper reverse DNS and proof is provided. Make sure your domain resolves correctly and you pass validation steps.
- **Firewall/Network Controls**: Ensure Let's Encrypt servers can reach your public server via the validation method you select.

More info and current updates on validation requirements are available here:
- [Let’s Encrypt Validation Methods](https://letsencrypt.org/docs/challenge-types/)
- [Let’s Encrypt Community Announcements](https://community.letsencrypt.org/c/announcements/6)
  
---

### Debian 11 Quick Setup
```bash
apt install certbot python3-certbot-apache -y
git clone https://github.com/letsencrypt/letsencrypt
cd letsencrypt
./letsencrypt-auto --help   # List commands
```

**Obtain and Install (Apache):**
```bash
./letsencrypt-auto --apache -d yourdomain.com
```

**Obtain Only (Standalone mode):**
```bash
./letsencrypt-auto certonly --standalone --email -d 
```

---

### Debian 12 Setup
See [LetsEncrypt Debian Guide](https://www.server-world.info/en/note?os=Debian_12&p=ssl&f=2)

---

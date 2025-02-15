##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)

copilot explain

The `letsencrypt.md` file provides instructions for enabling HTTPS on your website using Let's Encrypt. Here are the key points:

1. **References**:
   - Links to [LetsEncrypt.org](https://letsencrypt.org) and [Documentation](https://letsencrypt.org/docs/).

2. **Setup for Debian 11**:
   - Install Certbot and its Apache plugin:
     ```bash
     apt install certbot python3-certbot-apache -y
     ```
   - Clone the Let's Encrypt repository and navigate to it:
     ```bash
     git clone https://github.com/letsencrypt/letsencrypt
     cd letsencrypt
     ```
   - Access the help command for `letsencrypt-auto`:
     ```bash
     ./letsencrypt-auto --help
     ```

3. **Usage Examples**:
   - Obtain and install a certificate for Apache:
     ```bash
     ./letsencrypt-auto --apache -d yourdomain.com
     ```
   - Obtain a certificate only, using standalone mode:
     ```bash
     ./letsencrypt-auto certonly --standalone --email infouniversalbits@gmail.com -d universalbit.it
     ```

4. **Setup for Debian 12**:
   - Link to Let's Encrypt Debian 12 setup [guide](https://www.server-world.info/en/note?os=Debian_12&p=ssl&f=2).
---


### UniversalBit once again 
...enable HTTPS on your website.

* [LetsEncrypt.org](https://letsencrypt.org)
* [Documentation](https://letsencrypt.org/docs/)
---
### Debian11
```
apt install certbot python3-certbot-apache -y
git clone https://github.com/letsencrypt/letsencrypt
cd letsencrypt
./letsencrypt-auto --help ###Suite Command List
```
#### usage:
```
./letsencrypt-auto --apache -d yourdomain.com
```
#### certonly:
```
./letsencrypt-auto certonly --standalone --email infouniversalbits@gmail.com -d universalbit.it 
```
---
### Debian12
[LetsEncrypt Debian](https://www.server-world.info/en/note?os=Debian_12&p=ssl&f=2)


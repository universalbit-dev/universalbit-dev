##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)   ##### [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)   ##### [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html)
---

## [UniversalBitCDN](https://universalbitcdn.it/)
### [content-delivery-network](https://en.wikipedia.org/wiki/Content_delivery_network) basic configuration:
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/cdn/images/netdata_android_device.jpg" width="40%"></img>

### [NetData](https://en.wikipedia.org/wiki/Netdata):

* Like all open source programs, [jsdelivr.com](https://en.wikipedia.org/wiki/JSDelivr) Network.

##### Technically, it's fine

* [Ubuntu 22.04 • 2.29 GHz • 1 Cores • x86_64 • 957.41 MiB RAM • 35.00 GiB HD]

##### There's more open source software in a smartphone than in linux


* WebServer [Nginx](https://en.wikipedia.org/wiki/Nginx)
```bash
sudo apt install nginx
```

* [CertBot](https://en.wikipedia.org/wiki/Let's_Encrypt#Software_implementation) (LetsEncrypt) HTTPS
##### [LetsEncrypt](https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-22-04)
```bash
sudo snap install core; sudo snap refresh core
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
sudo ufw allow 'Nginx Full'
sudo ufw delete allow 'Nginx HTTP'
```
<gif>

##### [Nginx as Reverse Proxy](https://www.digitalocean.com/community/tutorials/how-to-configure-nginx-as-a-reverse-proxy-on-ubuntu-22-04)

<gif>

```
 sudo nano /etc/nginx/sites-enabled/default
```
##### default
```
server {
	listen 80 default_server;
	listen [::]:80 default_server;
	root /var/www/html;
	index index.html index.htm index.nginx-debian.html;
	server_name _;

	location / {
        try_files $uri $uri/ =404;
	}
}

server {

	root /var/www/html;
	index index.html index.htm index.nginx-debian.html;
        server_name www.universalbitcdn.it; # managed by Certbot
	location / {
	try_files $uri $uri/ =404;
	}

    listen [::]:443 ssl; # managed by Certbot
    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/universalbitcdn.it/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/universalbitcdn.it/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

}

server {
    if ($host = www.universalbitcdn.it) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


	listen 80 ;
	listen [::]:80 ;
    server_name www.universalbitcdn.it;
    return 404; # managed by Certbot

}
```
##### universalbitcdn
```
sudo nano /etc/nginx/sites-enabled/universalbitcdn
```

```
server {
    server_name universalbitcdn.it;   
    location / {
        proxy_pass http://localhost:19999;
        include proxy_params;
    }
    listen [::]:443 ssl ipv6only=on; # managed by Certbot
    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/universalbitcdn.it/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/universalbitcdn.it/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

}

server {
    if ($host = universalbitcdn.it) {
        return 301 https://$host$request_uri;
    } # managed by Certbot
    listen 80;
    listen [::]:80;
    server_name universalbitcdn.it;
    return 404; # managed by Certbot
}

```

##### Reconfigure [TimeZone](https://en.wikipedia.org/wiki/Time_zone) (Europe/Italy)
```bash
dpkg-reconfigure tzdata
```
##### [Swap space](https://en.wikipedia.org/wiki/Memory_paging#Unix_and_Unix-like_systems)
<gif>
  
```bash
sudo fallocate -l 6G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

##### [Ufw](https://en.wikipedia.org/wiki/Uncomplicated_Firewall)

##### Firewall Rules:

```bash
ufw allow 443
ufw allow 853
ufw allow 953
ufw allow 53
```

<gif>

##### [Fail2Ban](https://en.wikipedia.org/wiki/Fail2ban)

```bash
apt install fail2ban
```

<gif>

##### [Pihole](https://en.wikipedia.org/wiki/Pi-hole) Low Level URL filter

  -- Server DNS Enabled -- (PiHole) 

```bash
curl -sSL https://install.pi-hole.net | bash
```

<gif>

##### note pihole setup:
Configuration DNS : 4.2.2.2 (Low Level)
Admin Web Interface: Disabled
Logging Level : No Log


* [Nodejs](https://en.wikipedia.org/wiki/Node.js) Javascript Engine:

<gif>

##### [NVM](https://github.com/nvm-sh/nvm) node version manager :
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

##### [NetData](https://github.com/netdata/netdata)
```bash
wget -O /tmp/netdata-kickstart.sh https://get.netdata.cloud/kickstart.sh && sh /tmp/netdata-kickstart.sh
```

##### Content Delivery Network [GlobalPing](https://www.jsdelivr.com/globalping) jsdelivr.com
from universalbit-dev repository:

<gif>

```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
```

-- GLOBAL - USA - EUROPE - ITALY - PALERMO --
```bash
cd universalbit-dev/cdn
npm i
npm i pm2 -g
pm2 start ecosystem.config.js
```

##### [HaCluster](https://en.wikipedia.org/wiki/High-availability_cluster) Environment && [HAproxy](https://en.wikipedia.org/wiki/HAProxy) Environment

```bash
apt install haproxy heartbeat
```




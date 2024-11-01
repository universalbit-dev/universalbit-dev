##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)

---

##### [NetData](https://github.com/netdata/netdata)
```bash
wget -O /tmp/netdata-kickstart.sh https://get.netdata.cloud/kickstart.sh && sh /tmp/netdata-kickstart.sh
```

* WebServer [Nginx](https://nginx.org/en/docs/)
```bash
sudo apt install nginx
```

##### [LetsEncrypt](https://letsencrypt.org/)
* [CertBot](https://en.wikipedia.org/wiki/Let's_Encrypt#Software_implementation) (LetsEncrypt) HTTPS

```bash
sudo snap install core; sudo snap refresh core
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
```
#### [UFW](https://help.ubuntu.com/community/UFW)
```bash
sudo ufw allow 'Nginx Full'
sudo ufw delete allow 'Nginx HTTP'
```


##### [Nginx as Reverse Proxy](https://www.digitalocean.com/community/tutorials/how-to-configure-nginx-as-a-reverse-proxy-on-ubuntu-22-04)
```
 sudo nano /etc/nginx/sites-enabled/default
```

canonical: www.universalbitcdn.it
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

domain: universalbitcdn.it
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

##### [Fail2Ban](https://github.com/fail2ban/fail2ban)
```bash
apt install fail2ban
```

##### Reconfigure [TimeZone](https://en.wikipedia.org/wiki/Time_zone) (Europe/Italy)
```bash
dpkg-reconfigure tzdata
```

##### [UniversalBitCdn.it](https://universalbitcdn.it/)

---

#### Wikipedia Resources:
* [NetData](https://en.wikipedia.org/wiki/Netdata)
* [Nginx](https://en.wikipedia.org/wiki/Nginx)
* [UFW](https://en.wikipedia.org/wiki/Uncomplicated_Firewall)
* [NetWork Time Protocol](https://en.wikipedia.org/wiki/Network_Time_Protocol)



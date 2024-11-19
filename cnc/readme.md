##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/) -- [CNC Router Machines](https://github.com/universalbit-dev/cnc-router-machines)


### Internet of normal things in life


##### CNCjs is a full-featured web-based interface for CNC controllers running Grbl, Marlin, Smoothieware, or TinyG.
---

* Linux app image cncjs v1.10.3
```bash
wget https://github.com/cncjs/cncjs/releases/download/v1.10.3/cncjs-app-1.10.3-linux-x86_64.AppImage
```

* ## [CncJS Desktop App](https://github.com/cncjs/cncjs/releases/download/v1.10.3/cncjs-app-1.10.3-linux-x86_64.AppImage)
  ##### [releases](https://github.com/cncjs/cncjs/releases)


# Upload [GRBL](https://github.com/grbl/grbl) to Arduino Nano Shield V3.
Grbl is a no-compromise, high performance, low cost alternative to parallel-port-based motion control for CNC milling. It will run on a vanilla Arduino (Duemillanove/Uno) as long as it sports an Atmega 328.

[Ubuntu 24.04 LTS]
Required packages: 
```bash
sudo apt install arduino zip unzip wget
```
GRBL Release : [v1.1](https://github.com/gnea/grbl/archive/refs/tags/v1.1h.20190825.zip) 
---


* Release:[GRBL1.1](https://github.com/gnea/grbl/releases)

```bash
#Download Release GRBL 1.1 (.zip) 
cd ~/Downloads
wget https://github.com/gnea/grbl/archive/refs/tags/v1.1h.20190825.zip
unzip grbl-1.1h.20190825.zip
#create .zip archive of grbl folder 
cd grbl-1.1h.20190825/ && zip -r grbl.zip grbl
#rename folder grbl-1.1h.20190825 to GRBL
cp -R ~/Downloads/grbl-1.1h.20190825 GRBL
#copy GRBL to ...~/Arduino/libraries directory
cp -R GRBL /home/universalbit/Arduino/libraries/
```

run [Arduino-IDE](https://www.arduino.cc/en/software) 1.8.19
<strong>Upload</strong> grbl.zip created previously

* Arduino-IDE == > Sketch ==> Include Library ===> Add .ZIP Library 

---
### Ready to connect Arduino Nano Shield V3 to [CNCJS](https://github.com/universalbit-dev/cnc-router-machines) for uploading gcode and start cnc milling simulation.

### Resources:
* [CNC MACHINES](https://github.com/universalbit-dev/cnc-router-machines)
* [CNC wiki](https://en.wikipedia.org/wiki/CNC_router)
* [Universal G-Code Sender (UGS)](https://universalgcodesender.com/)


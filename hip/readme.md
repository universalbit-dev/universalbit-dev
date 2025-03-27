[Introduction to the HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)


[Enhance HArmadillium Project -- Experimental --](https://github.com/universalbit-dev/HArmadillium?tab=readme-ov-file)
---
Ubuntu 24.04 LTS noble
* kernel version: 6.8.0-35-lowlatency
---
* Update and Upgrade Ubuntu Os
```bash
sudo apt-get update && sudo apt-get upgrade
```

* [AMD Repo](https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/)

```bash
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.3.60304-1_all.deb
sudo chmod a+x amdgpu-install_6.3.60304-1_all.deb 
```
[gdebi](https://wiki.ubuntu-it.org/AmministrazioneSistema/InstallareProgrammi/Gdebi)
```bash
sudo apt install gdebi
sudo gdebi ./amdgpu-install_6.3.60304-1_all.deb
```


```bash
sudo amdgpu-install --usecase=hip
```

* [Blender HIP](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html)
* [AMD Radeon Software Instructions](https://amdgpu-install.readthedocs.io/en/latest/)

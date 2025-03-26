[Introduction to the HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)
 
[ThinClient HP-T630 ==> Experimental Driver Installation]

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
sudo apt-get install ./amdgpu-install_6.3.60304-1_all.deb
```

```bash
sudo amdgpu-install --usecase=hip
```
* [HIP offers several benefits over OpenCL](https://gpuopen.com/wp-con:ent/uploads/2016/01/7637_HIP_Datasheet_V1_7_PrintReady_US_WE.pdf)
* [Blender HIP](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html)
* [AMD Radeon Software Instructions](https://amdgpu-install.readthedocs.io/en/latest/)

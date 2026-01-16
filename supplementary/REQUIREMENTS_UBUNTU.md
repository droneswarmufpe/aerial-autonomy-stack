# Pre-installation Steps for AAS on Ubuntu

> These instructions are tested using Ubuntu 22.04.5 LTS

## Install Ubuntu with NVIDIA Driver

- Get/install an OS from a startup disk based on Ubuntu 22 or newer (e.g. `ubuntu-22.04.5-desktop-amd64.iso`)
  - Choose "Normal installation", "Download updates while installing Ubuntu", no "Install third-party software"
- Update the OS
  - Run "Software Updater" and restart
  - "Update All" in "Ubuntu Software" (including `killall snap-store && sudo snap refresh snap-store`)
  - Update and restart for "Device Firmware" as necessary
- In "Software & Updates", select `nvidia-driver-580 (proprietary, tested)`

```sh
sudo apt update && sudo apt upgrade

nvidia-smi                          # Should report "Driver Version: 580.65.06, CUDA Version: 13.0"

# Set NVIDIA Performance Mode
sudo prime-select nvidia            # Alternatively, use `nvidia-settings` -> "PRIME profiles" -> "NVIDIA (Performance Mode)"
# Reboot and check in Ubuntu's "Settings" -> "About" -> "Graphics" is your NVIDIA card
# `nvidia-smi` will show applications like Firefox and VSCode as Type G processes
# For laptops with hybrid graphics, it is recommended to also go to "Settings" -> "Power"
# Then, select  the "Performance" "Power Mode" and disable all "Power Saving Options"

sudo apt install -y mesa-utils
glxinfo | grep "OpenGL renderer"    # Check the GPU is the OpenGL renderer
```

## Install Docker Engine

```sh
# Based on https://docs.docker.com/engine/install/ubuntu/ and https://docs.docker.com/engine/install/linux-postinstall/

for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker Engine
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo docker run hello-world         # Test Docker is working
sudo docker version                 # Check version, 28.3.0 at the time of writing

# Remove the need to sudo the docker command
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker                       # Reboot

docker run hello-world              # Test Docker is working without sudo
```

## Install NVIDIA Container Toolkit

```sh
# Based on https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

sudo apt-get update && sudo apt-get install -y --no-install-recommends curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker info | grep -i runtime       # Check `nvidia` runtime is available

docker run --rm --gpus all nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi        # Test nvidia-smi works in a container with CUDA
```

## Optimize Memory Usage

**Optionally**, increase the swap size and, if you have an SSD, decrease swappiness

```sh
# Increase Ubuntu's default 2GB swap memory to 8GB
sudo swapon --show
sudo swapoff /swapfile
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show

# Decrease Ubuntu's default swappiness of 60 to 10 (to reduce SSD wear)
cat /proc/sys/vm/swappiness
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
cat /proc/sys/vm/swappiness
```

## Troubleshoot

To be able to pull the base Docker images frequently, you might have to log in to the NVIDIA Registry:

- Go to https://ngc.nvidia.com and login/create an account.
- Click on your account the top right, go to Setup -> Get API Key.
- Click "Generate API Key" -> "+ Generate Personal Key" for the "NCG Catalog" service, confirm, and copy the key.

```sh
docker login nvcr.io                # To be able to reliably pull NVIDIA base images
Username:                           # type $oauthtoken
Password:                           # copy and paste the API key and press enter to pull base images from nvcr.io/
```

<!--
> Major dependencies: [*Ubuntu 22.04*](https://ubuntu.com/about/release-cycle) (LTS, ESM 4/2032), [*`nvidia-driver-580`*](https://developer.nvidia.com/datacenter-driver-archive) (latest as of 9/2025), [*Docker Engine v28*](https://docs.docker.com/engine/release-notes/28/), [*NVIDIA Container Toolkit 1.18*](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) (latest as of 11/2025), [*ROS2 Humble*](https://docs.ros.org/en/rolling/Releases.html) (LTS, EOL 5/2027), [*Gazebo Sim Harmonic*](https://gazebosim.org/docs/latest/releases/) (LTS, EOL 9/2028), [*PX4 1.16*](https://github.com/PX4/PX4-Autopilot/releases) interfaced *via* [XRCE-DDS](https://github.com/eProsima/Micro-XRCE-DDS/releases), [*ArduPilot 4.6*](https://github.com/ArduPilot/ardupilot/releases) interfaced *via* [MAVROS](https://github.com/mavlink/mavros/releases), [*YOLOv8*](https://github.com/ultralytics/ultralytics/releases) on [*ONNX Runtime 1.22*](https://onnxruntime.ai/getting-started) (latest stable releases as of 8/2025), [*L4T 36* (Ubuntu 22-based)/*JetPack 6*](https://developer.nvidia.com/embedded/jetpack-archive) (for deployment only, latest major release as of 8/2025), [WSLg](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) (for simulation and development on Windows 11 only)
-->

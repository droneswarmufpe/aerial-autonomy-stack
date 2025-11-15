# Setup Avionics

## Flash JetPack 6 to Jetson Orin

Holybro Jetson baseboards normally ship with JetPack 5

To upgrade to JetPack 6, download NVIDIA SDK Manager on the Ubuntu 22 host computer from [here](https://developer.nvidia.com/sdk-manager#installation_get_started)

```sh
cd ~/Downloads
sudo apt install ./sdkmanager_2.3.0-12626_amd64.deb 
sdkmanager                          # Log in with your https://developer.nvidia.com account 
```

- Put the Holybro Jetson baseboard in recovery mode with the dedicated switch
- Connect the USB-C port closes to the fan to the computer running `sdkmanager` and power on the board
- On Step 1, select "Jetson", "Host Machine Ubuntu 22 x86_64", "Target Hardware Jetson Orin NX" (automatically detected), "SDK Version JetPack 6.2.1"
- On Step 2, under "Target Components", select all "Jetson Linux" (uncheck all others)
- Accept the "terms and conditions" and click "CONTINUE" (if prompted, click "Create" folder and input the password to `sudo`)
- Wait for `sdkmanager` to download the necessary software
- On the flash dialog after the download, choose "OEM Pre-config", username, password, and "Storage NVMe", click "Flash"
- On `sdkmanager` click "FINISH AND EXIT" once the process is completed
- With a screen, mouse, and keyboard connected to the Jetson basedboad, log in, finish the configuration, power-off, put the board out of recovery mode and power-on again
- Select an appropriate "Power Mode" (e.g. 25W)

<!-- 
- [PX4 documentation](https://github.com/PX4/PX4-Autopilot/blob/main/docs/en/companion_computer/holybro_pixhawk_jetson_baseboard.md#flashing-the-jetson-board)
-->

> [!WARNING]
> At the time of writing, **Snap is broken on JetPack 6**, a fix is suggested [here](https://forums.developer.nvidia.com/t/chromium-other-browsers-not-working-after-flashing-or-updating-heres-why-and-quick-fix/338891)
> ```sh
> snap download snapd --revision=24724
> sudo snap ack snapd_24724.assert
> sudo snap install snapd_24724.snap
> sudo snap refresh --hold snapd
> 
> snap install firefox
> ```

## Configure Jetson-IO for the CSI IMX219-200 Camera

```sh
sudo /opt/nvidia/jetson-io/jetson-io.py

# Follow these steps: 
#   "Configure Jetson 24pin CSI Connector"
#   -> "Configure for compatible hardware"
#   -> "Camera IMX219 Dual" (even if only using one)
#   -> "Save pin changes"
#   -> "Save and reboot to reconfigure pins"
#   -> Press any key to reboot

sudo dmesg | grep -i imx219         # After reboot, this will show at least one imx219 successfully bound

# Inspect device (e.g. /dev/video0) resolution and frame rate
sudo apt update && sudo apt install -y v4l-utils
v4l2-ctl --list-formats-ext -d /dev/video0 
```

## Install Docker Engine and NVIDIA Container Toolkit on Jetson Orin

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

Log in to the NVIDIA Registry:

- Go to https://ngc.nvidia.com and login/create an account.
- Click on your account the top right, go to Setup -> Get API Key.
- Click "Generate API Key" -> "+ Generate Personal Key" for the "NCG Catalog" service, confirm, and copy the key.

```sh
docker login nvcr.io                # To be able to reliably pull NVIDIA base images
Username:                           # type $oauthtoken
Password:                           # copy and paste the API key and press enter to pull base images from nvcr.io/
```

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

docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-base:r36.2.0 nvidia-smi    # Test nvidia-smi works in a container with Linux4Tegra
```

## Build and Flash PX4 or ArduPilot Firmware

Build the PX4 firmware in `simulation-image` for Pixhawk 6X

```sh
# Check available PX4 targets
docker run -it --rm --entrypoint bash simulation-image -c "cd /aas/github_apps/PX4-Autopilot && make list_config_targets"
```

```sh
# Build PX4 for Pixhawk 6X (saved in the ~/Downloads folder)
docker run -it --rm --entrypoint bash -v ~/Downloads:/temp simulation-image -c \
  "cd /aas/github_apps/PX4-Autopilot && make px4_fmu-v6x_default && cp build/px4_fmu-v6x_default/*.px4 /temp/"
```

Build the ArduPilot firmware in `simulation-image` for Pixhawk 6X

```sh
# Check available ArduPilot targets
docker run -it --rm --entrypoint bash simulation-image -c "cd /aas/github_apps/ardupilot && ./waf list_boards"
```

```sh
# Build ArduCopter (quads) for Pixhawk 6X (saved in the ~/Downloads folder)
docker run -it --rm --entrypoint bash -v ~/Downloads:/temp simulation-image -c \
  "cd /aas/github_apps/ardupilot && ./waf configure --board Pixhawk6X && ./waf copter && cp build/Pixhawk6X/bin/*.apj /temp/"
```

```sh
# Build ArduPlane (VTOLs) for Pixhawk 6X (saved in the ~/Downloads folder)
docker run -it --rm --entrypoint bash -v ~/Downloads:/temp simulation-image -c \
  "cd /aas/github_apps/ardupilot && ./waf configure --board Pixhawk6X && ./waf plane && cp build/Pixhawk6X/bin/*.apj /temp/"
```

To flash the newly created `.px4` or `.apj` file to your autopilot board, follow [QGC's User Guide](https://docs.qgroundcontrol.com/Stable_V5.0/en/qgc-user-guide/setup_view/firmware.html) 

## PX4: Configure 6x's Network and DDS Client

On the Jetson Baseboard's Orin NX, under "Settings" -> "Network", configure the "PCI Ethernet" connection to "Manual" with IPv4 with address 10.10.1.44 and netmask 255.255.255.0

Connect the Pixhawk 6X to the ground station with the USB-C port next to the RJ-45 port

- Access QGroundControl -> "Analyze Tools" -> "MAVLink console"
- Copy-and-paste the following commands (these will assign an IP to the PX4 autopilot (e.g., 10.10.1.33) and let the `uxrce_dds_client` connect to the Orin NX (e.g., on IP 10.10.1.44) using namespace `Drone1`)
- Re-start the autopilot 

```sh
# Configure DDS Client connection to the NX
mkdir /fs/microsd/etc
cd /fs/microsd/etc
echo "uxrce_dds_client stop" > extras.txt
echo "sleep 3" >> extras.txt
echo -n "uxrce_dds_client start -p 8888" >> extras.txt
echo " -h 10.10.1.44 -n Drone1" >> extras.txt

# Check the content of the file (remember to set the proper "DroneX" namespace)
cat /fs/microsd/etc/extras.txt

# Configure Autopilot network settings
echo DEVICE=eth0 > /fs/microsd/net.cfg
echo BOOTPROTO=static >> /fs/microsd/net.cfg
echo IPADDR=10.10.1.33 >> /fs/microsd/net.cfg
echo NETMASK=255.255.255.0 >> /fs/microsd/net.cfg
echo ROUTER=10.10.1.254 >> /fs/microsd/net.cfg
echo DNS=10.10.1.254 >> /fs/microsd/net.cfg

# Check the content of the file
cat /fs/microsd/net.cfg
# Check the current network configuration
ifconfig

# Apply the new configuration
netman update
# Reboot and check new network configuration
ifconfig

# Set vehicle/MAV_SYS_ID (for multiple QGC connections) and UXRCE_DDS_DOM_ID/ROS_DOMAIN_ID
param set MAV_SYS_ID 1
param set UXRCE_DDS_DOM_ID 1

# Optionally
param set MAV_2_CONFIG 0        # Disable MAVLINK on Ethernet (so Ethernet is used for XRCE-DDS only), if needed, also check params MAV_0_CONFIG, MAV_1_CONFIG
param set UXRCE_DDS_CFG 1000    # Use DDS over Ethernet
```

> [!CAUTION]
> Match `MAV_SYS_ID`, `UXRCE_DDS_DOM_ID`, and `uxrce_dds_client`'s namespace `-n Drone1`, with the `DRONE_ID` used to launch `./deploy_run.sh`: this is the `ROS_DOMAIN_ID` of the aircraft container

One should be able to `ping 10.10.1.44` (the Orin NX) from MAVLink Console on QGC; and `ping 10.10.1.33` (the autopilot) from a terminal on the Orin NX

<!-- 
- [PX4 documentation](https://github.com/PX4/PX4-Autopilot/blob/main/docs/en/companion_computer/holybro_pixhawk_jetson_baseboard.md#ethernet-setup-using-netplan)
-->

## ArduPilot: Configure 6X's Serial for MAVROS

MAVROS can be connected to MAVLink using the Pixhawk 6X's `TELEM2` serial port (this is ArduPilot's `SERIAL2` and Orin's `/dev/ttyTHS1`)

In QGroundControl -> "Vehicle Configuration" -> "Parameters" set:

```sh
SERIAL2_BAUD 921600
SERIAL2_OPTIONS 0
SERIAL2_PROTOCOL MAVLink2
```

> [!CAUTION]
> Match ArduPilot parameter `SYSID_THISMAV` with the `DRONE_ID` used to launch `./deploy_run.sh`: this is the `ROS_DOMAIN_ID` of the aircraft container

<!-- 
- [ArduPilot serial configuration](https://ardupilot.org/copter/docs/common-serial-options.html)
- [Jetson baseboard serial configuration](https://github.com/PX4/PX4-Autopilot/blob/main/docs/en/companion_computer/holybro_pixhawk_jetson_baseboard.md#serial-connection)
- [MAVROS connection](https://github.com/mavlink/mavros/blob/ros2/mavros/README.md)
- [Requesting MAVLink data from ArduPilot](https://ardupilot.org/dev/docs/mavlink-requesting-data.html)
-->

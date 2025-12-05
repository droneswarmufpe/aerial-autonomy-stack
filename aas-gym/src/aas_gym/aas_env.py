import numpy as np
import gymnasium as gym
import docker
import zmq
import time
import struct
import os
import subprocess
import shutil

from docker.types import NetworkingConfig, EndpointConfig, DeviceRequest


class AASEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.max_steps = 1000  # Max steps per episode
        self.dt = 0.05         # Time step
        # Observation Space: [position, velocity], position is in [-1, 1], velocity is in [-5, 5]
        self.obs_low = np.array([-1.0, -5.0], dtype=np.float32)
        self.obs_high = np.array([1.0, 5.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        # Action Space: [force] between -1.0 and 1.0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Internal state
        self.position = 0.0
        self.velocity = 0.0
        self.step_count = 0
        
        # Rendering
        self.render_mode = render_mode

        # Docker setup
        try:
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError("Could not connect to the Docker daemon. Ensure Docker is running.") from e
        
        self.AUTOPILOT = "px4"
        self.HEADLESS = True # Use False for debugging, set to True to disable GUI rendering
        self.CAMERA = True
        self.LIDAR = True
        #
        self.SIM_SUBNET = "10.42"
        self.AIR_SUBNET = "10.22"
        self.SIM_ID = "100"
        # self.GROUND_ID = "101"
        #
        self.NUM_QUADS = 1
        self.NUM_VTOLS = 0
        self.WORLD = "impalpable_greyness"
        #
        self.GND_CONTAINER = False
        self.RTF = 0.0
        self.START_AS_PAUSED = True
        self.INSTANCE = 0
        #
        sim_parts = self.SIM_SUBNET.split('.')
        self.SIM_SUBNET = f"{sim_parts[0]}.{int(sim_parts[1]) + self.INSTANCE}"
        # air_parts = self.AIR_SUBNET.split('.')
        # self.AIR_SUBNET = f"{air_parts[0]}.{int(air_parts[1]) + self.INSTANCE}"
        #
        self.SIM_NET_NAME = f"aas-sim-network-inst{self.INSTANCE}"
        # self.AIR_NET_NAME = f"aas-air-network-inst{self.INSTANCE}"
        self.SIM_CONT_NAME = f"simulation-container-inst{self.INSTANCE}"
        # self.GND_CONT_NAME = f"ground-container-inst{self.INSTANCE}"
        #
        if not self.HEADLESS:
            if shutil.which("xhost"):
                print("Granting X Server access to Docker containers...")
                try:
                    subprocess.run(["xhost", "+local:docker"], check=True)
                    print("X Server access granted.")
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Could not configure xhost: {e}")
            else:
                print("Error: 'xhost' command not found.")
        networks_config = [
            {"name": self.SIM_NET_NAME, "subnet_base": self.SIM_SUBNET},
            # {"name": self.AIR_NET_NAME, "subnet_base": self.AIR_SUBNET}
        ]
        self.networks = {}
        for net_config in networks_config:
            net_name = net_config["name"]
            base_ip = net_config["subnet_base"]
            print(f"Setting up Docker Network: {net_name}...")
            try:
                existing_network = self.client.networks.get(net_name)
                existing_network.remove()
                print(f"Existing network '{net_name}' removed.")
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as e:
                print(f"Warning: Could not remove {net_name}: {e}")
            ipam_pool = docker.types.IPAMPool(
                subnet=f"{base_ip}.0.0/16",
                gateway=f"{base_ip}.0.1"
            )
            ipam_config = docker.types.IPAMConfig(
                pool_configs=[ipam_pool]
            )
            new_network = self.client.networks.create(
                net_name,
                driver="bridge",
                ipam=ipam_config
            )
            self.networks[net_name] = new_network
            print(f"Network '{net_name}' created on subnet {base_ip}.0.0/16")
        #
        env_display = os.environ.get('DISPLAY', '')
        env_xdg = os.environ.get('XDG_RUNTIME_DIR', '')
        gpu_requests = [
            DeviceRequest(count=-1, capabilities=[['gpu']]) # Replaces "--gpus all"
        ]
        volume_binds = { # Replaces "--volume /tmp/.X11-unix:/tmp/.X11-unix:rw"
            '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'} # Format: {'host_path': {'bind': 'container_path', 'mode': 'rw'}}
        }
        device_binds = ['/dev/dri:/dev/dri:rwm'] # Replaces "--device /dev/dri"
        #
        print(f"Creating Simulation Container ({self.SIM_CONT_NAME})...")
        self.simulation_container = self.client.containers.create(
            "simulation-image:latest",
            name=self.SIM_CONT_NAME,
            tty=True, # Replaces -it
            detach=True,
            auto_remove=True,
            privileged=True, # Replaces --privileged
            volumes=volume_binds,
            devices=device_binds,
            device_requests=gpu_requests,
            environment={
                "DISPLAY": env_display,
                "QT_X11_NO_MITSHM": "1",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "XDG_RUNTIME_DIR": env_xdg,
                "AUTOPILOT": self.AUTOPILOT,
                "HEADLESS": str(self.HEADLESS).lower(),
                "CAMERA": str(self.CAMERA).lower(),
                "LIDAR": str(self.LIDAR).lower(),
                "NUM_QUADS": str(self.NUM_QUADS),
                "NUM_VTOLS": str(self.NUM_VTOLS),
                "WORLD": self.WORLD,
                "SIMULATED_TIME": "true",
                "RTF": str(self.RTF),
                "START_AS_PAUSED": str(self.START_AS_PAUSED).lower(),
                "SIM_SUBNET": self.SIM_SUBNET,
                # "GROUND_ID": self.GROUND_ID,
                "GND_CONTAINER": str(self.GND_CONTAINER).lower(),
                "ROS_DOMAIN_ID": self.SIM_ID,
                "GYMNASIUM" : "true",
            }
        )
        print(f"Connecting {self.SIM_CONT_NAME} to {self.SIM_NET_NAME}...")
        self.networks[self.SIM_NET_NAME].connect(
            self.simulation_container,
            ipv4_address=f"{self.SIM_SUBNET}.90.{self.SIM_ID}"
        )
        # print(f"Connecting {self.SIM_CONT_NAME} to {self.AIR_NET_NAME}...")
        # self.networks[self.AIR_NET_NAME].connect(
        #     self.simulation_container,
        #     ipv4_address=f"{self.AIR_SUBNET}.90.{self.SIM_ID}"
        # )
        self.simulation_container.start()
        #
        self.aircraft_containers = []
        for i in range(1, self.NUM_QUADS + self.NUM_VTOLS + 1):            
            air_cont_name = f"aircraft-container-inst{self.INSTANCE}_{i}"
            print(f"Creating Aircraft Container {air_cont_name}...")
            drone_type = "quad" if i <= self.NUM_QUADS else "vtol"
            air_cont = self.client.containers.create(
                "aircraft-image:latest",
                name=air_cont_name,
                tty=True, # Replaces -it
                detach=True,
                auto_remove=True,
                privileged=True, # Replaces --privileged
                volumes=volume_binds,
                devices=device_binds,
                device_requests=gpu_requests,
                environment={
                    "DISPLAY": env_display,
                    "QT_X11_NO_MITSHM": "1",
                    "NVIDIA_DRIVER_CAPABILITIES": "all",
                    "XDG_RUNTIME_DIR": env_xdg,
                    "AUTOPILOT": self.AUTOPILOT,
                    "HEADLESS": str(self.HEADLESS).lower(),
                    "CAMERA": str(self.CAMERA).lower(),
                    "LIDAR": str(self.LIDAR).lower(),
                    "DRONE_TYPE": drone_type,
                    "DRONE_ID": str(i),
                    "SIMULATED_TIME": "true",
                    "SIM_SUBNET": self.SIM_SUBNET,
                    # "AIR_SUBNET": self.AIR_SUBNET,
                    "SIM_ID": self.SIM_ID,
                    # "GROUND_ID": self.GROUND_ID,
                    "GND_CONTAINER": str(self.GND_CONTAINER).lower(),
                    "ROS_DOMAIN_ID": str(i),
                }
            )
            print(f"Connecting {air_cont_name} to {self.SIM_NET_NAME}...")
            self.networks[self.SIM_NET_NAME].connect(
                air_cont,
                ipv4_address=f"{self.SIM_SUBNET}.90.{i}"
            )
            # print(f"Connecting {air_cont_name} to {self.AIR_NET_NAME}...")
            # self.networks[self.AIR_NET_NAME].connect(
            #     air_cont,
            #     ipv4_address=f"{self.AIR_SUBNET}.90.{i}"
            # )
            air_cont.start()
            self.aircraft_containers.append(air_cont)
        print("Docker setup complete. All containers are running and connected.")

        # ZeroMQ Setup
        self.ZMQ_PORT = 5555
        self.ZMQ_IP = f"{self.SIM_SUBNET}.90.{self.SIM_ID}"
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 10 * 1000) # 1000 ms = 1 seconds
        self.socket.connect(f"tcp://{self.ZMQ_IP}:{self.ZMQ_PORT}")
        print(f"ZeroMQ socket connected to {self.ZMQ_IP}:{self.ZMQ_PORT}")

    def _get_obs(self):
        return np.array([self.position, self.velocity], dtype=np.float32)

    def _get_info(self):
        return {"position": self.position, "velocity": self.velocity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding

        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        try:
            reset = 9999.0 # A special action to reset the environment
            # Serialize the action and send the REQ
            action_payload = struct.pack('d', reset)
            self.socket.send(action_payload)
            # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
            reply_bytes = self.socket.recv()
            # Deserialize
            unpacked = struct.unpack('iI', reply_bytes) # i = int32 (sec), I = uint32 (nanosec)
            sec, nanosec = unpacked
            # print(f"Received clock update: {sec}.{nanosec}")
        except zmq.error.Again:
            print("ZMQ Error: Reply from container timed out.")
        except ValueError:
            print("ZMQ Error: Reply format error. Received garbage state.")
        ###########################################################################################
        # Reset state to a random position near the center ########################################
        ###########################################################################################
        self.position = self.np_random.uniform(low=-0.8, high=0.8)
        self.velocity = 0.0
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        self.step_count = 0
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        force = action[0]

        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        try:
            # Serialize the action and send the REQ
            action_payload = struct.pack('d', force)
            self.socket.send(action_payload)
            # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
            reply_bytes = self.socket.recv()
            # Deserialize
            unpacked = struct.unpack('iI', reply_bytes) # i = int32 (sec), I = uint32 (nanosec)
            sec, nanosec = unpacked
            # print(f"Received clock update: {sec}.{nanosec}")
        except zmq.error.Again:
            print("ZMQ Error: Reply from container timed out.")
        except ValueError:
            print("ZMQ Error: Reply format error. Received garbage state.")
        ###########################################################################################
        # Simple Euler integration ################################################################
        ###########################################################################################
        self.velocity += force * self.dt
        self.velocity *= 0.99  # Add some damping
        self.position += self.velocity * self.dt
        # Clip position to the bounds [-1.0, 1.0]
        self.position = np.clip(self.position, -1.0, 1.0)
        # If it hits a wall, dampen the velocity (like a bounce)
        if self.position == -1.0 or self.position == 1.0:
            self.velocity *= -0.5
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        self.step_count += 1
        
        # Calculate reward: Negative distance from the goal (position 0)
        reward = float(-np.abs(self.position))
        # Check for termination
        terminated = False  # This is a continuing task, never "terminates"
        # Check for truncation (episode ends due to time limit)
        truncated = self.step_count >= self.max_steps
        # Get obs and info
        obs = self._get_obs()
        info = self._get_info()
        
        # Handle rendering
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # Scale position from [-1, 1] to a 40-char width
        pos_int = int((self.position + 1.0) / 2.0 * 40)     
        # Create the display string
        display = ['-'] * 41
        display[pos_int] = 'O'  # The agent
        if pos_int != 20:
            display[20] = '|'       # The target (0.0)   
        # Print to console
        print(f"\r{''.join(display)}  Pos: {self.position:6.3f}, Vel: {self.velocity:6.3f}", end="")

    def close(self):
        if self.render_mode == "human":
            print()  # Add a newline after the final render
        
        # Docker clean-up (auto_remove=True in the creation step handles removal after stop)
        try:
            self.simulation_container.stop()
            print(f"Simulation container stopped.")
        except Exception:
            pass
        for container in self.aircraft_containers:
            try:
                container.stop()
                print(f"Aircraft container '{container.name}' stopped.")
            except Exception:
                pass
        for net_name, network_obj in self.networks.items():
            try:
                network_obj.remove()
                print(f"Network {net_name} removed.")
            except Exception as e:
                print(f"Warning: Could not remove {net_name}: {e}")

        # Close ZMQ resources
        if self.socket:
            self.socket.close(linger=0)
        if self.zmq_context:
            self.zmq_context.term()

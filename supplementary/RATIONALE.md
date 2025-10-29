# Rationale

## The 2 Facets of the Sim2real Gap

The *sim2real gap* is an euphemism for robotic projects that work well on a developer's laptop but not so much in the field.
Aerial sim2real research often focuses on modeling and simulation of complex aerodynamics effects.

Nonetheless, in applied robotics, an equally important component of *sim2real gap* lies in software tooling and engineering, where the dynamic range between ["average and the best is 50-to-1, maybe 100-to-1"](https://www.youtube.com/watch?v=wTgQ2PBiz-g&t=35s).

This is the challenge—and good sport—of **full-stack integration** among:

- the **many frameworks** that go into drone autonomy (a physics engine to simulate drone dynamics, a rendering engine to generate realistic imagery, a GPU-accelerated machine learning runtime for perception, one or more inter-process and inter-thread communication middleware, the interface to the microcontroller and autopilot software performing state-estimation and low-level control, the SDKs of the deployed embedded systems, etc.)
- emulated **inter-robot communication** (in aerial systems, this is heavily affected by the actual flight plans and available RF hardware)

## Related Work

A summary of existing multi-drone flight stacks can be found in [Table II of this paper](https://arxiv.org/pdf/2303.18237). Notable ones are:

- *Universidad Politécnica de Madrid (UPM)*'s [`aerostack2`](https://github.com/aerostack2/aerostack2) (multicopter-only)
- *Czech Technical University in Prague (CTU)*'s [`mrs_uav_system`](https://github.com/ctu-mrs/mrs_uav_system) (multicopter-only)
- *Technische Universität (TU) Berlin*'s [`crazyswarm2`](https://github.com/IMRCLab/crazyswarm2) (indoor, crazyflie-only)
- *Peking University*'s [`XTDrone`](https://github.com/robin-shaun/XTDrone) (PX4-only)

A summary of aerial robotics simulators can be found in [Table IV of this paper](https://arxiv.org/pdf/2311.02296), these include:

- *Norwegian University of Science and Technology (NTNU)*'s [`aerial_gym_simulator`](https://github.com/ntnu-arl/aerial_gym_simulator) (high-performance simulator for RL)
- *University of Pennsylvania (UPenn)*'s [`RotorPy`](https://github.com/spencerfolk/rotorpy) (high-fidelity simulator for control)
- *University of Toronto (UofT)*'s [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) (simple simulator for education, control, and RL)
- *UZH*'s [`flightmare`](https://github.com/uzh-rpg/flightmare), *ETH*'s [`RotorS`](https://github.com/ethz-asl/rotors_simulator), *NYU*'s [`RotorTM`](https://github.com/arplaboratory/RotorTM), *Microsoft*'s [`AirSim`](https://github.com/microsoft/AirSim), etc.

For even more resources, check out [`aerial_robotic_landscape`](https://github.com/ROS-Aerial/aerial_robotic_landscape).

## Design Manifesto

- **Simplicity** (["simple is better than complex"](https://peps.python.org/pep-0020/), ["worse is better"](https://www.dreamsongs.com/RiseOfWorseIsBetter.html), and ["no fat software"](https://people.inf.ethz.ch/wirth/Articles/LeanSoftware.pdf) are the 3 slogans of the AAS)
- [おまかせ](https://dhh.dk/2012/rails-is-omakase.html) **end-to-end**ness (from camera frames, to autopilot uORB/MAVLink commands)
- **Recentness** (break and fix, rather than carrying technical [debt](https://c2.com/doc/oopsla92.html))
- **Deployment** focus
    - Clear, Dockerized split between aircraft, ground, and simulation software
    - ROS2 intra-companion board messaging
    - XRCE-DDS (PX4), MAVROS (ArduPilot) autopilot-to-companion board ROS2 UDP bridge
    - GStreamer camera-to-companion board acquisition
    - Zenoh inter-vehicle ROS2 bridge, with networking over LAN (HITL) or emulated by `docker network` (SITL)

[![Star History Chart](https://api.star-history.com/svg?repos=JacopoPan/aerial-autonomy-stack,robin-shaun/XTDrone,KumarRobotics/kr_autonomous_flight,ctu-mrs/mrs_uav_system,uzh-rpg/agilicious,CopterExpress/clover,caochao39/aerial_navigation_development_environment,aerostack2/aerostack2,IMRCLab/crazyswarm2&type=Date)](https://www.star-history.com/#JacopoPan/aerial-autonomy-stack&robin-shaun/XTDrone&KumarRobotics/kr_autonomous_flight&ctu-mrs/mrs_uav_system&uzh-rpg/agilicious&CopterExpress/clover&caochao39/aerial_navigation_development_environment&aerostack2/aerostack2&IMRCLab/crazyswarm2&Date&LogScale)

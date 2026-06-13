# Commercial platforms & the open autonomy ecosystem

Descriptive, civil/commercial reference. Structured data:
[`../data/commercial_platforms.csv`](../data/commercial_platforms.csv).

## Commercial platforms

- **DJI** — the dominant commercial vendor. Mavic 3 Enterprise (compact survey/
  thermal/RTK), Matrice 350 RTK (flagship industrial multi-payload, ~55 min, IP55),
  Matrice 4 (newer compact-industrial), and Agras T50 (heavy-lift agricultural
  spray/spread). Developer access: Mobile SDK, Payload SDK (PSDK), Cloud API.
- **Skydio X10** — US AI-autonomy enterprise drone (360° vision obstacle
  avoidance, NightSense, onboard mapping). Targets public-safety / drone-as-first-
  responder; documented Skydio Cloud REST API.
- **Autel EVO Max 4T** — enterprise quad (wide/zoom/thermal/laser-rangefinder),
  optional RTK; Mobile/Payload/Cloud SDKs.
- **Parrot ANAFI Ai** — 4G-connected pro/photogrammetry drone with an unusually
  open stack (Ground SDK, onboard Air SDK, Olympe Python) and MAVLink/GUTMA support.
- **senseFly eBee X (AgEagle)** — fixed-wing mapping/survey drone flown via eMotion.

Use cases across these are civil: mapping/survey, agriculture, inspection,
delivery, search-and-rescue, cinematography, public safety.

## Open autonomy / flight ecosystem

| Project | What it is | License |
|---|---|---|
| **PX4 Autopilot** | Open flight stack (multirotor/fixed-wing/VTOL/rover); ROS 2 bridge; SITL | BSD-3-Clause |
| **ArduPilot** | Open autopilot (Copter/Plane/Rover/Sub/Tracker) | GPLv3 |
| **MAVLink** | De-facto drone↔GCS↔payload messaging protocol; codegen for many languages | MIT |
| **QGroundControl** | Cross-platform ground control station for PX4/ArduPilot | Apache-2.0 OR GPLv3 |
| **ROS 2** | Robotics middleware (rclcpp/rclpy) | Apache-2.0 |
| **Gazebo** | 3D robotics simulator; pairs with PX4 SITL | Apache-2.0 |
| **Pixhawk** | Open-hardware flight-controller reference standard | Open hardware |

These open stacks underpin a large civil and research ecosystem (mapping,
agriculture, inspection, delivery, SAR) and are commonly developed/tested in
simulation (Gazebo + PX4 SITL) before field use.

Full sources: [../SOURCES.md](../SOURCES.md).

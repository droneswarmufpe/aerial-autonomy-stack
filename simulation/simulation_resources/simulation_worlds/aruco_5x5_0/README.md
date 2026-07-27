## Spawn With CUSTOM_OBJECTS

Use `CUSTOM_OBJECTS=true` to generate ArUco markers before Gazebo starts. By default, the generated markers follow the same spawn positions used by `_create_ardupilot_world.sh`, creating one marker under each drone.

For a flat world:

```sh
CUSTOM_OBJECTS=true NUM_QUADS=1 NUM_VTOLS=0 X_OFFSET=5 Y_OFFSET=-110 ARUCO_Z=0.03 ./sim_run.sh
```

For `esefex_fbx`, use the local terrain height where the drone starts. Example near `X_OFFSET=5 Y_OFFSET=-110`:

```sh
CUSTOM_OBJECTS=true NUM_QUADS=1 NUM_VTOLS=0 WORLD=esefex_fbx X_OFFSET=5 Y_OFFSET=-110 ARUCO_Z=-1.43225 ./sim_run.sh
```

To place markers randomly in a rectangular region, set `ARUCO_PLACEMENT=random` and pass the bounds:

```sh
CUSTOM_OBJECTS=true ARUCO_PLACEMENT=random ARUCO_COUNT=5 ARUCO_X_MIN=-20 ARUCO_X_MAX=20 ARUCO_Y_MIN=-130 ARUCO_Y_MAX=-90 ARUCO_Z=-1.43225 ARUCO_SEED=42 ./sim_run.sh
```

This creates `/tmp/aruco_custom_objects_config.json` inside the simulation container and injects the configured `aruco_5x5_0` models into `populated_ardupilot.sdf`.

## Spawn One Marker

Run this inside the simulation container, remember to change $WORLD_NAME accordingly

```sh

gz service -s /world/$WORLD_NAME/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'sdf_filename: "/aas/simulation_resources/simulation_worlds/aruco_5x5_0/model.sdf", name: "aruco_0", allow_renaming: true, pose: {position: {x: 10, y: -20, z: 2.0}}'
```

## Spawn Several Markers

Run this inside the simulation container, remember to change $WORLD_NAME accordingly

```sh

for spec in \
  "aruco_0 10 -20 2.0" \
  "aruco_1 11 -20 2.0" \
  "aruco_2 10 -19 2.0" \
  "aruco_3 11 -19 2.0"
do
  set -- $spec
  gz service -s /world/$WORLD_NAME/create \
    --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean \
    --timeout 3000 \
    --req "sdf_filename: \"/aas/simulation_resources/simulation_worlds/aruco_5x5_0/model.sdf\", name: \"$1\", allow_renaming: true, pose: {position: {x: $2, y: $3, z: $4}}"
done
```

## List Or Remove

List spawned models:

```sh
gz model --list
```

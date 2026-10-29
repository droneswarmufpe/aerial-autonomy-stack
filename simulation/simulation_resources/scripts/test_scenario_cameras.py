"""Run with: python3 -m unittest discover -s simulation/simulation_resources/scripts -p 'test_*.py'."""

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import build_world
import scenario


class ScenarioCameraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aas_camera_test_')
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.resources = Path(scenario.RESOURCES_DIR)
        self.sensor_source = self.resources / 'aircraft/sensors/d435/model.sdf'
        self.sensor_bytes = self.sensor_source.read_bytes()

    def resolve(self, quads=None, vtols=None, **kwargs):
        path = self.output / 'scenario.json'
        path.write_text(json.dumps({'world': 'esefex_fbx', 'quads': quads or [], 'vtols': vtols or []}))
        return scenario.resolve(str(path), **kwargs)

    def generate(self, drone):
        model_dir = build_world.build_vehicle_model(drone, str(self.resources), str(self.output))
        vehicle = ET.parse(Path(model_dir) / 'model.sdf')
        include = vehicle.find('.//include[name="d435"]')
        if include is None:
            return vehicle, None
        name = include.findtext('uri').removeprefix('model://')
        camera = ET.parse(self.output / name / 'model.sdf')
        self.assertEqual(include.findtext('name'), 'd435')
        self.assertEqual(camera.getroot().find('model').get('name'), name)
        self.assertEqual(camera.findtext('.//mesh/uri'), 'model://d435/meshes/realsense_d435.stl')
        self.assertEqual(self.sensor_source.read_bytes(), self.sensor_bytes)
        return vehicle, camera

    def sensor_names(self, camera):
        return {sensor.get('name') for sensor in camera.iter('sensor')}

    def test_omitted_fields_keep_all_four_sensors_for_quads_and_vtols(self):
        resolved = self.resolve(quads=[{'x': 0, 'y': 0}], vtols=[{'x': 0, 'y': 3}])
        for drone in resolved['drones']:
            with self.subTest(vehicle=drone['type']):
                _, camera = self.generate(drone)
                self.assertEqual(self.sensor_names(camera), {'d435_color', 'd435_ired1', 'd435_ired2', 'd435_depth'})

    def test_each_camera_can_be_disabled_independently(self):
        fields = {'rgb_camera': 'd435_color', 'infrared_1_camera': 'd435_ired1',
                  'infrared_2_camera': 'd435_ired2', 'depth_camera': 'd435_depth'}
        for vehicle_type in ('quads', 'vtols'):
            for field, sensor in fields.items():
                with self.subTest(vehicle=vehicle_type, disabled=field):
                    drone = self.resolve(**{vehicle_type: [{'x': 0, 'y': 0, field: False}]})['drones'][0]
                    _, camera = self.generate(drone)
                    self.assertEqual(self.sensor_names(camera), set(fields.values()) - {sensor})

    def test_master_switches_override_individual_camera_settings(self):
        for master in ('camera', 'rgbd_camera'):
            with self.subTest(master=master):
                drone = self.resolve(quads=[{'x': 0, 'y': 0, master: False, 'thermal_camera': True}])['drones'][0]
                vehicle, camera = self.generate(drone)
                self.assertIsNone(camera)
                thermal = vehicle.find('.//include[name="simple_thermal_camera"]')
                self.assertEqual(thermal is not None, master == 'rgbd_camera')

    def test_drone_camera_choices_are_isolated(self):
        drones = self.resolve(quads=[
            {'x': 0, 'y': 0, 'infrared_1_camera': False, 'infrared_2_camera': False, 'depth_camera': False},
            {'x': 3, 'y': 0, 'rgb_camera': False},
        ])['drones']
        self.generate(drones[0])
        self.generate(drones[1])
        self.assertEqual(self.sensor_names(ET.parse(self.output / 'd435_1/model.sdf')), {'d435_color'})
        self.assertEqual(self.sensor_names(ET.parse(self.output / 'd435_2/model.sdf')),
                         {'d435_ired1', 'd435_ired2', 'd435_depth'})

    def test_camera_switches_require_booleans(self):
        for field in ('rgb_camera', 'infrared_1_camera', 'infrared_2_camera', 'depth_camera'):
            for value in ('false', 0, None):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(scenario.ScenarioError, f'{field} must be true or false'):
                        self.resolve(quads=[{'x': 0, 'y': 0, field: value}])

    def test_vio_requires_both_infrared_cameras(self):
        for field in ('infrared_1_camera', 'infrared_2_camera'):
            with self.subTest(disabled=field):
                with self.assertRaisesRegex(scenario.ScenarioError, 'vio_bridge requires infrared_1_camera and infrared_2_camera'):
                    self.resolve(quads=[{'x': 0, 'y': 0, 'vio_bridge': True, field: False}], rcpilot=True)
        drone = self.resolve(quads=[{'x': 0, 'y': 0, 'vio_bridge': True, 'rgb_camera': False,
                                    'depth_camera': False}], rcpilot=True)['drones'][0]
        _, camera = self.generate(drone)
        self.assertEqual(self.sensor_names(camera), {'d435_ired1', 'd435_ired2'})

    def test_all_cameras_off_removes_both_camera_models(self):
        drone = self.resolve(quads=[{'x': 0, 'y': 0, 'camera': False, 'rgbd_camera': False,
                                    'rgb_camera': False, 'infrared_1_camera': False,
                                    'infrared_2_camera': False, 'depth_camera': False,
                                    'thermal_camera': False}])['drones'][0]
        vehicle, camera = self.generate(drone)
        self.assertIsNone(camera)
        self.assertIsNone(vehicle.find('.//include[name="simple_thermal_camera"]'))


if __name__ == '__main__':
    unittest.main()

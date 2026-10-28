"""SITL estimator overrides stay in generated output; source vehicle parameters remain unchanged."""

import json
from pathlib import Path
import tempfile
import unittest

import build_world
import scenario


class SitlEstimatorTests(unittest.TestCase):
    def read_params(self, path):
        return {words[0]: float(words[1]) for line in path.read_text().splitlines()
                if (words := line.split()) and not words[0].startswith('#')}

    def test_sim_estimator_is_isolated_to_generated_params(self):
        resources = Path(scenario.RESOURCES_DIR)
        with tempfile.TemporaryDirectory(prefix='aas_sitl_estimator_') as temporary:
            root = Path(temporary)
            config = root / 'scenario.json'
            config.write_text(json.dumps({'world': 'esefex_fbx',
                                          'quads': [{'x': 0, 'y': 0, 'camera': False}],
                                          'vtols': [{'x': 3, 'y': 0, 'camera': False}]}))
            for drone in scenario.resolve(str(config))['drones']:
                with self.subTest(vehicle=drone['type']):
                    source = resources / 'aircraft/vehicles' / drone['base_model']
                    source_params = source / 'ardupilot-4.6.params'
                    original = source_params.read_bytes()
                    model = Path(build_world.build_vehicle_model(drone, str(resources), str(root)))
                    overlay = model / 'sitl-estimator.params'
                    params = self.read_params(overlay)
                    self.assertEqual(params['AHRS_EKF_TYPE'], 10)
                    self.assertEqual(params['GPS1_TYPE'], 100)
                    self.assertEqual(params['SIM_IMU_COUNT'], 1)
                    for index in (2, 3):
                        self.assertEqual(params[f'INS_ACC{index}_ID'], 0)
                        for axis in 'XYZ':
                            self.assertEqual(params[f'INS_ACC{index}OFFS_{axis}'], 0)
                            self.assertEqual(params[f'INS_ACC{index}SCAL_{axis}'], 1)
                    self.assertNotIn('ARMING_CHECK', params)
                    self.assertNotIn('SIM_SPEEDUP', params)
                    self.assertNotIn('INS_GYR_CAL', params)
                    self.assertEqual((model / 'ardupilot-4.6.params').read_bytes(), original)
                    self.assertEqual(source_params.read_bytes(), original)
                    self.assertFalse((source / 'sitl-estimator.params').exists())

    def test_vio_uses_ekf3_without_affecting_other_drones(self):
        with tempfile.TemporaryDirectory(prefix='aas_sitl_vio_') as temporary:
            root = Path(temporary)
            config = root / 'scenario.json'
            config.write_text(json.dumps({'world': 'esefex_fbx', 'quads': [
                {'x': 0, 'y': 0, 'vio_bridge': True},
                {'x': 3, 'y': 0, 'camera': False},
            ]}))
            for drone, expected in zip(scenario.resolve(str(config), rcpilot=True)['drones'], (3, 10)):
                model = Path(build_world.build_vehicle_model(drone, scenario.RESOURCES_DIR, str(root)))
                params = self.read_params(model / 'sitl-estimator.params')
                self.assertEqual(params['AHRS_EKF_TYPE'], expected)
                if drone['vio_bridge']:
                    self.assertEqual(params, {'AHRS_EKF_TYPE': 3})


if __name__ == '__main__':
    unittest.main()

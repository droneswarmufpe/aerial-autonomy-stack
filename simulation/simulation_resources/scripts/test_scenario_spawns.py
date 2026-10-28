"""Reject coincident vehicles before Gazebo starts its collision engine."""

import json
from pathlib import Path
import tempfile
import unittest

import scenario


class ScenarioSpawnTests(unittest.TestCase):
    def resolve(self, quads, vtols=None, **kwargs):
        with tempfile.TemporaryDirectory(prefix='aas_spawn_test_') as temporary:
            path = Path(temporary) / 'scenario.json'
            path.write_text(json.dumps({'world': 'esefex_fbx', 'quads': quads, 'vtols': vtols or []}))
            return scenario.resolve(str(path), **kwargs)

    def test_duplicate_positions_include_default_altitude(self):
        with self.assertRaisesRegex(scenario.ScenarioError, 'drone 2 and drone 1 share spawn position'):
            self.resolve([{'x': 0, 'y': 0}, {'x': 0, 'y': 0, 'z': 0.75}])

    def test_quad_and_vtol_cannot_share_a_position(self):
        with self.assertRaisesRegex(scenario.ScenarioError, 'share spawn position'):
            self.resolve([{'x': 0, 'y': 0, 'z': 1}], vtols=[{'x': 0, 'y': 0, 'z': 1}])

    def test_unselected_vehicles_do_not_trigger_rejection(self):
        resolved = self.resolve([{'x': 0, 'y': 0}, {'x': 0, 'y': 0}], num_quads=1)
        self.assertEqual(len(resolved['drones']), 1)

    def test_separated_vehicles_are_accepted(self):
        resolved = self.resolve([{'x': 0, 'y': y} for y in (0, 3, 6)])
        self.assertEqual([drone['y'] for drone in resolved['drones']], [0, 3, 6])


if __name__ == '__main__':
    unittest.main()

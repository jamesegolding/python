
import unittest
import numpy as np
from lib import utilities as utils


class UtilitiesTest(unittest.TestCase):

    def test_norm2(self):
        """
        Utilities:   Vector two norm
        """
        for i in range(10):
            x = 10. * np.random.rand(3)
            x_norm = utils.norm2(x)
            self.assertAlmostEqual(np.multiply(x, x).sum(), x_norm ** 2, 4)

    def test_normalize(self):
        """
        Utilities:   Vector normalize
        """
        for i in range(10):
            x = 10. * np.random.rand(3)
            x_normalized = utils.normalize(x)
            x_norm = utils.norm2(x_normalized)
            if x_norm > 0.001:
                self.assertAlmostEqual(x_norm, 1., 6)

    def test_clip(self):
        """
        Utilities:   Vector clipping
        """
        for i in range(10):
            x = 100. * (np.random.rand(100) - 0.5)
            x_limits = 50. * (np.random.rand(2) - 0.5)

            x_clipped = utils.clip(x, a_min=min(x_limits), a_max=max(x_limits))
            self.assertLessEqual(max(x_clipped), max(x_limits))
            self.assertGreaterEqual(min(x_clipped), min(x_limits))

    def test_scale(self):
        """
        Utilities:   Vector scaling
        """
        for i in range(10):
            x = 100. * (np.random.rand(100) - 0.5)
            x_limits = 50. * (np.random.rand(2) - 0.5)

            x_scaled = utils.scale(x, a_min=min(x_limits), a_max=max(x_limits))
            x_scaled_range = max(x_scaled) - min(x_scaled)

            self.assertAlmostEqual(x_scaled_range, max(x_limits) - min(x_limits), 6)
            self.assertAlmostEqual(np.corrcoef(sorted(x), sorted(x_scaled))[0, 1], 1., 4)

    def test_quadrant(self):
        """
        Utilities:   Scalar convert to quadrant -pi to pi
        """
        for i in range(10):
            x = np.random.rand()
            x_quadrant = utils.quadrant(x)
            self.assertLessEqual(x_quadrant, np.pi)
            self.assertGreaterEqual(x_quadrant, -np.pi)

    def test_cross_product(self):
        """
        Utilities:   Vector cross product
        """
        for i in range(10):
            x = 20. * np.random.rand(3)
            y = np.random.rand(3)
            x_cross_y = utils.cross(x, y)

            self.assertAlmostEqual(np.multiply(x_cross_y, x).sum(), 0., 4)
            self.assertAlmostEqual(np.multiply(x_cross_y, y).sum(), 0., 4)


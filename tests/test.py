from .context import lsh
import numpy as np 
import unittest

class TestLSH(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(0)
        fastLSH=lsh.FastLSH(4,10,10)
        self.data=np.random.randn(100,10)
        fastLSH[self.data]=np.arange(100).tolist()  
        self.lsh=fastLSH
        return super().setUp()

    def test_query(self):
        self.assertEqual(self.lsh[self.data[0]], [0, 132, 173, 377, 406, 623, 960, 3, 279, 413, 423, 456, 794, 25, 484, 505, 736, 876])

    def test_remove(self):
        self.lsh.remove_items([0,132,406,3])
        self.assertEqual(self.lsh[self.data[0]], [ 173, 377, 623, 960, 279, 413, 423, 456, 794, 25, 484, 505, 736, 876])

    def test_saveLoad(self):
        self.lsh.save_json("test_save")
        new_lsh=lsh.FastLSH(4,10,10)
        self.assertEqual(new_lsh[self.data[0]], [0, 132, 173, 377, 406, 623, 960, 3, 279, 413, 423, 456, 794, 25, 484, 505, 736, 876])


if __name__ == '__main__':
    unittest.main()


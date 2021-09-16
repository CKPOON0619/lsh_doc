from context import lsh
import numpy as np
import unittest
import os


class TestLSH(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(0)
        fastLSH = lsh.FastLSH(4, 10, 10)
        self.data = np.random.randn(100, 10)
        fastLSH[self.data] = np.arange(100).astype(str).tolist()
        self.lsh = fastLSH
        return super().setUp()

    def test_query(self):
        self.assertEqual(self.lsh[self.data[0]], ['0', '3', '25'])

    def test_remove(self):
        self.lsh.remove_items(['0', '3'])
        self.assertEqual(self.lsh[self.data[0]], ['25'])

    def test_saveLoad(self):
        self.lsh.save_json("contentQuery/tests/test_save")
        new_lsh = lsh.FastLSH(4, 10, 10)
        new_lsh.load_json("contentQuery/tests/test_save")
        os.remove("contentQuery/tests/test_save.json")
        self.assertEqual(new_lsh[self.data[0]], self.lsh[self.data[0]])


if __name__ == '__main__':
    unittest.main()

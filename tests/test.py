'''
Unit tests
'''
import unittest
import os
from .context import server_app
import numpy as np


LSH = server_app.lsh.FastLSH
create_LSH_dict = server_app.lsh.create_LSH_dict


class TestLSH(unittest.TestCase):

    '''
    Tests for lsh implementation
    '''

    def setUp(self) -> None:
        np.random.seed(0)
        hasher = LSH(4, 10, 10)
        self.data = np.random.randn(100, 10)
        hasher[self.data] = np.arange(100).astype(str).tolist()
        self.lsh = hasher
        return super().setUp()

    def test_query(self):
        '''
        Test result query
        '''
        self.assertEqual(self.lsh[self.data[0]], ['0', '3', '25'])

    def test_remove(self):
        '''
        Test remove item records
        '''
        self.lsh.remove_items(['0', '3'])
        self.assertEqual(self.lsh[self.data[0]], ['25'])

    def test_saveLoad(self):
        '''
        Test saving and loading results as json
        '''
        self.lsh.save_json("tests/test_save")
        new_lsh = LSH(4, 10, 10)
        new_lsh.load_json("tests/test_save")
        os.remove("tests/test_save.json")
        self.assertEqual(new_lsh[self.data[0]], self.lsh[self.data[0]])

    def test_lsh_dict(self):
        '''
        Test create_LSH_dict that creates lsh instances lookup
        '''
        data_types = ['a', 'b']
        contents = ['x', 'y']
        lsh_dict = create_LSH_dict(data_types, contents, 3, 2, 4)
        for data_type in data_types:
            for content in contents:
                self.assertTrue(isinstance(
                    lsh_dict[data_type][content], LSH))


if __name__ == '__main__':
    unittest.main()

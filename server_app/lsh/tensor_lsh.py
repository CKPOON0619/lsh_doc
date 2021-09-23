'''
Tensor implementation for lsh
'''
import json as json
import numpy as np
import tensorflow as tf
from typing import ByteString, Dict, TypedDict, List, Union


class HashTable(TypedDict):
    '''
    hash table records for lsh.
    '''
    hash: Dict[str, str]
    hash_bucket: Dict[str, str]


Input_Vector = Union[List[float], List[List[float]]]


def create_hashes(hashes: List[ByteString]) -> ByteString:
    '''
    To join a list of byteString into single byteString.
    '''
    return tf.strings.reduce_join(hashes, axis=-1)


def create_lsh_dict(
    types: List[str],
    contents: List[str],
    n_tables: int, hash_dim: int,
    embedding_dim: int
) -> Dict[str, Dict[str, HashTable]]:
    '''
    To create a list of hash tables look up with different types and contents.
    '''
    lsh_dict = dict()
    for typ in types:
        lsh_dict[typ] = dict()
        for con in contents:
            lsh_dict[typ][con] = FastLSH(n_tables, hash_dim, embedding_dim)
    return lsh_dict


class FastLSH:
    '''
    Create locality sensitive hashing with tensorflow matrix operations.
    '''

    def __init__(self, num_tables: int, hash_size: int, inp_dimensions: int, projections=None, directory="./") -> None:
        self.dir = directory
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = [{"hash_bucket": dict(), "hash": dict()}
                            for i in np.arange(num_tables)]
        if projections is None:
            self.projections = np.random.randn(
                self.num_tables, inp_dimensions, self.hash_size)
        else:
            if np.shape(projections) == [inp_dimensions, hash_size]:
                self.projections = projections
            else:
                raise Exception(
                    "Projections shape not matching hash size and input dimension provided.")

    def generate_hashes(self, inp_vector: Input_Vector) -> List[str]:
        '''
        Generate hashes with self projections for the input vector
        '''
        hashes_array = tf.map_fn(create_hashes, tf.where(
            tf.matmul(inp_vector, self.projections) > 0, "1", "0"))
        return hashes_array.numpy().astype("str")

    def add_items(self, inp_vector: Input_Vector, item_ids: List[str]) -> None:
        '''
        Add items to the array
        '''
        hashes_array = self.generate_hashes(inp_vector)
        casted_item_ids = np.array(item_ids).astype(str)
        for table_idx, hashes in enumerate(hashes_array):
            self.hash_tables[table_idx]["hash"].update(
                dict(zip(casted_item_ids, hashes)))
            hash_bucket = self.hash_tables[table_idx]["hash_bucket"]
            for idx, hash_code in enumerate(hashes):
                bucket = hash_bucket.get(hash_code, None)
                if bucket is None:
                    hash_bucket[hash_code] = {casted_item_ids[idx]: True}
                else:
                    bucket.update({casted_item_ids[idx]: True})

    def get_items(self, inp_vector: Input_Vector) -> List[str]:
        '''
        To get items with corresponding vector embeddings
        '''
        hashes_array = self.generate_hashes(inp_vector)
        res_lookup = dict()
        np.apply_along_axis(lambda input_hashes: [
            res_lookup.update(
                self.hash_tables[table_idx]["hash_bucket"].get(hash, {})
            )
            for table_idx, hash in enumerate(input_hashes)
        ], arr=hashes_array, axis=0)
        return list(res_lookup.keys())

    def remove_items(self, item_ids: List[str]) -> None:
        '''
        To remove items from the lsh tables
        '''
        for table in self.hash_tables:
            for itemId in item_ids:
                hash_code = table["hash"].get(itemId, None)
                if hash_code is not None:
                    table["hash_bucket"][hash_code].pop(itemId)
                    table["hash"].pop(itemId)

    def save_json(self, filepath: str) -> None:
        '''
        Save tables as a json file.(TODO: move towards using a database)
        '''
        with open(self.dir + filepath + ".json", 'w') as outfile:
            json.dump({"hash_tables": self.hash_tables,
                      "projections": self.projections.tolist()}, outfile)

    def load_json(self, filepath: str) -> None:
        '''
        Load tables as a json file.(TODO: move towards using a database)
        '''
        with open(self.dir + filepath + ".json", 'r') as outfile:
            data = json.load(outfile)
            self.hash_tables = data["hash_tables"]
            self.projections = np.array(data["projections"])

    def __setitem__(self, input_val: Input_Vector, item_id: Union[str, List[str]]) -> None:
        input_vec = input_val
        item_ids = item_id
        input_shape = np.shape(input_val)
        if not isinstance(item_id, list):
            item_ids = [item_id]
        if len(input_shape) == 1:
            input_vec = [input_val]
        if len(input_shape) > 2:
            raise Exception(
                '''
                Unexpected shape of input vector.
                Expecting input dimension to be 1 or 2, received {}
                '''.format(len(input_shape)))
        self.remove_items(item_ids)
        self.add_items(input_vec, item_ids)

    def __getitem__(self, input_val: Union[List[str], List[List[str]]]) -> List[str]:
        input_vec = input_val
        input_shape = np.shape(input_val)
        if len(input_shape) == 1:
            input_vec: List[List[str]] = [input_val]
        if len(input_shape) > 2:
            raise Exception(
                '''
                Unexpected shape of input vector.
                Expecting input dimension to be 1 or 2, received {}
                '''.format(len(input_shape)))
        return self.get_items(input_vec)

import numpy as np
import tensorflow as tf
import json as json

#TODO: implement typing
def create_hashes(hashes):
  return tf.strings.reduce_join(hashes,axis=-1)

class FastLSH:
  def __init__(self, num_tables, hash_size, inp_dimensions, projections=None, directory="./"):
      self.dir=directory
      self.num_tables = num_tables
      self.hash_size = hash_size
      self.inp_dimensions = inp_dimensions
      self.hash_tables=[{"hash_bucket":dict(),"hash":dict()} for i in np.arange(num_tables)]
      if projections==None:
        self.projections = np.random.randn(self.num_tables,inp_dimensions,self.hash_size)
      else:
        if np.shape(projections)==[inp_dimensions,hash_size]:
          self.projections=projections
        else:
          raise Exception("Projections shape not matching hash size and input dimension provided.")

  def generate_hashes(self,inp_vector):
    hashes_array=tf.map_fn(create_hashes,tf.where(tf.matmul(inp_vector,self.projections)>0,"1","0"))
    return hashes_array.numpy().astype("U13")


  def add_items(self, inp_vector, itemIds):
    hashes_array=self.generate_hashes(inp_vector)
    for table_idx,hashes in enumerate(hashes_array):
      self.hash_tables[table_idx]["hash"].update(dict(zip(itemIds,hashes)))
      hash_bucket=self.hash_tables[table_idx]["hash_bucket"]
      for idx,hash in enumerate(hashes):
        bucket=hash_bucket.get(hash,None)
        if bucket==None:
          hash_bucket[hash]={itemIds[idx]:True}
        else:
          bucket.update({itemIds[idx]:True})

  def get_items(self, inp_vector):
    hashes_array=self.generate_hashes(inp_vector)
    res_lookup=dict()
    np.apply_along_axis(lambda input_hashes:[res_lookup.update(self.hash_tables[table_idx]["hash_bucket"].get(hash,{})) for table_idx,hash in enumerate(input_hashes)],arr=hashes_array,axis=0)
    return list(res_lookup.keys())

  def remove_items(self,itemIds):
    for table in self.hash_tables: 
      for itemId in itemIds:
        hash=table["hash"].get(itemId,None)
        if hash!=None:
          table["hash_bucket"][hash].pop(itemId)
          table["hash"].pop(itemId)
  
  def save_json(self,filePath):
    with open(self.dir+filePath+".json", 'w') as outfile:
      json.dump(self.hash_tables,outfile)

  def load_json(self,filePath):
    with open(self.dir+filePath+".json", 'r') as outfile:
      self.hash_tables=json.load(outfile)

  def __setitem__(self, input, itemId):
    input_vec=input
    itemId_vec=itemId
    input_shape = np.shape(input)
    if not isinstance(itemId,list):
     itemId_vec=[itemId]
    if len(input_shape)==1:
      input_vec=[input]
    if len(input_shape)>2:
      raise Exception("Unexpected shape of input vector. Expecting input dimension to be 1 or 2, received {}".format(len(input_shape)))
    self.remove_items(itemId_vec)
    self.add_items(input_vec,itemId_vec)

  def __getitem__(self, input):
    input_vec=input
    input_shape = np.shape(input)
    if len(input_shape)==1:
      input_vec=[input]
    if len(input_shape)>2:
      raise Exception("Unexpected shape of input vector. Expecting input dimension to be 1 or 2, received {}".format(len(input_shape)))
    return self.get_items(input_vec)

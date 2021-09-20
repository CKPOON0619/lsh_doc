'''
Server app with fast api
'''
from typing import Union, Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from .embeddings import USE_embed
from .lsh import create_LSH_dict

TYPES = ["dimension", "temporal", "measure", "boolean"]
CONTENTS = ["name", "description"]
EMBEDDING_DIM = 512
HASH_DIM = 23  # @param {type:"slider", min:1, max:100, step:1}
N_TABLES = 300  # @param {type:"slider", min:1, max:300, step:1}
LSH_DICT = create_LSH_dict(TYPES, CONTENTS, N_TABLES, HASH_DIM, EMBEDDING_DIM)


class Register(BaseModel):
    '''
    Type for POST to register items
    '''
    type: str
    name: List[str]
    description: Optional[List[str]] = None
    id: List[str]


class ItemQuery(BaseModel):
    '''
    Type for GET query to fetch items
    '''
    type: str
    content: str
    embeddings: Union[List[float], List[List[float]]]


class TableQuery(BaseModel):
    '''
    Type for GET query to fetch table    
    '''
    type: str
    content: Optional[str] = None


app = FastAPI()


@app.post("/register/items")
async def register_items(register: Register):
    register_dict = dict(register)
    for c in CONTENTS:
        embeddings = USE_embed(register_dict[c])
        LSH_DICT[register.type][c][embeddings] = register.id
    return {"embeddings": embeddings.numpy().tolist()}


@app.get("/query/items")
async def get_items(query: ItemQuery):
    return LSH_DICT[query.type][query.content][query.embeddings]


@app.get("/query/tables")
async def get_tables(query: TableQuery):
    res = dict()
    type_tables = LSH_DICT[query.type]
    if query.content is None:
        for c in CONTENTS:
            res[c] = type_tables[c].hash_tables
    return res

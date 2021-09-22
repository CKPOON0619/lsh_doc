'''
Server app with fast api
'''
from typing import Union, Optional, List, TypedDict
from pydantic import BaseModel
from fastapi import FastAPI
from .embeddings import USE_embed
from .lsh import create_lsh_dict, HashTable

TYPES = ["dimension", "temporal", "measure", "boolean"]
CONTENTS = ["name", "description"]
EMBEDDING_DIM = 512
HASH_DIM = 23  # @param {type:"slider", min:1, max:100, step:1}
N_TABLES = 300  # @param {type:"slider", min:1, max:300, step:1}
LSH_DICT = create_lsh_dict(TYPES, CONTENTS, N_TABLES, HASH_DIM, EMBEDDING_DIM)

Embeddings = Union[List[float], List[List[float]]]


class Registration(BaseModel):
    '''
    Type for POST to register items
    '''
    type: str
    name: List[str]
    description: Optional[List[str]] = None
    id: List[str]


class RegistrationResponse(TypedDict):
    '''
    Response from registration result
    '''
    embeddings: Embeddings


class ItemQuery(BaseModel):
    '''
    Type for GET query to fetch items
    '''
    type: str
    content: str
    embeddings: Embeddings


ItemQueryResponse = List[str]


class TableQuery(BaseModel):
    '''
    Type for GET query to fetch table
    '''
    type: str
    content: Optional[str] = None


app = FastAPI()


@app.post("/register/items")
async def register_items(register: Registration) -> RegistrationResponse:
    '''
    Register new item to the table and also return the embeddings.
    '''
    register_dict = dict(register)
    for c in CONTENTS:
        embeddings = USE_embed(register_dict[c])
        LSH_DICT[register.type][c][embeddings] = register.id
    return {"embeddings": embeddings.numpy().tolist()}


@app.get("/query/items")
async def get_items(query: ItemQuery) -> ItemQueryResponse:
    '''
    API query for an item.
    '''
    return LSH_DICT[query.type][query.content][query.embeddings]


@app.get("/query/tables")
async def get_tables(query: TableQuery) -> List[HashTable]:
    '''
    API query for the hash tables
    '''
    res = dict()
    type_tables = LSH_DICT[query.type]
    if query.content is None:
        for content in CONTENTS:
            res[content] = type_tables[content].hash_tables
    return res

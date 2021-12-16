'''
Index of server app with fast api.
'''
from .lsh import create_lsh_dict, FastLSH, app
from .embeddings import USE_embed
from .mongodb_lsh import app as db_app

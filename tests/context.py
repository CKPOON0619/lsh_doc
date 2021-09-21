'''
To set up context for tests
'''
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from server_app import lsh, embeddings
from server_app import app

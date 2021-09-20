'''
Embeddings model
'''
import tensorflow_hub as hub
# The 16-language multilingual module is the default but feel free
# to pick others from the list and compare the results.
# @param ['https://tfhub.dev/google/universal-sentence-encoder-multilingual/3', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3']
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
USE_embed = hub.load(module_url)
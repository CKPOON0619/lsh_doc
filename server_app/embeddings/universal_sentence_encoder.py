'''
Embeddings model
'''
# pylint: disable=unused-import
# `tensorflow_text` is required for tensorflow hub model pre-processing.
import tensorflow_text
import tensorflow_hub as hub
# The 16-language multilingual module is the default but feel free
# to pick others from the list and compare the results.
# @param ['https://tfhub.dev/google/universal-sentence-encoder-multilingual/3', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3']
MODULE_URL = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
USE_embed = hub.load(MODULE_URL)
USE = {"encode": USE_embed, "embedding_dim": 512}

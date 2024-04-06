import os
import torch

import sys
import logging

import os, sys
from dotenv import find_dotenv, dotenv_values
import subprocess


# We will keep all global variables in an object to not pollute the global namespace.
class MyConfig(object):
    pass


MY_CONFIG = MyConfig()

## Atlas settings
MY_CONFIG.DB_NAME = "rag1"
MY_CONFIG.COLLECTION_NAME = "10k_local"
MY_CONFIG.EMBEDDING_ATTRIBUTE = "embedding_local"
MY_CONFIG.INDEX_NAME = "idx_embedding_local"

## Embedding settings
## Option 1 : small model - about 133 MB size
## Option 2 : large model - about 1.34 GB
## See Step-12 for more details

MY_CONFIG.EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

print("DB_NAME: ", MY_CONFIG.DB_NAME)

## Check if GPU is enabled

print("using CUDA/GPU: ", torch.cuda.is_available())

for i in range(torch.cuda.device_count()):
    print("device ", i, torch.cuda.get_device_properties(i).name)

## Setup logging.  To see more loging set the level to DEBUG


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


this_dir = os.path.abspath("")
parent_dir = os.path.dirname(this_dir)
sys.path.append(os.path.abspath(parent_dir))

config = dotenv_values(find_dotenv())
# debug
# print (config)
MY_CONFIG.ATLAS_URI = config.get("ATLAS_URI")

if MY_CONFIG.ATLAS_URI:
    print("✅ config ATLAS_URI found")
else:
    raise Exception("'❌ ATLAS_URI' is not set.  Please set it above to continue...")

import pymongo

mongodb_client = pymongo.MongoClient(MY_CONFIG.ATLAS_URI)
print("✅ Connected to Atlas instance!")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name=MY_CONFIG.EMBEDDING_MODEL)

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext


vector_store = MongoDBAtlasVectorSearch(
    mongodb_client=mongodb_client,
    db_name=MY_CONFIG.DB_NAME,
    collection_name=MY_CONFIG.COLLECTION_NAME,
    index_name=MY_CONFIG.INDEX_NAME,
    embedding_key=MY_CONFIG.EMBEDDING_ATTRIBUTE,
    ## the following columns are set to default values
    # text_key = 'text', metadata_= 'metadata',
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

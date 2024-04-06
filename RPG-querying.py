## Check if GPU is enabled
import os
import torch

## To disable GPU and experiment, uncomment the following line
## Normally, you would want to use GPU, if one is available.
# os.environ["CUDA_VISIBLE_DEVICES"]=""

print("using CUDA/GPU: ", torch.cuda.is_available())

for i in range(torch.cuda.device_count()):
    print("device ", i, torch.cuda.get_device_properties(i).name)
## Setup logging.  To see more loging set the level to DEBUG

import sys
import logging

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os, sys

this_dir = os.path.abspath("")
parent_dir = os.path.dirname(this_dir)
sys.path.append(os.path.abspath(parent_dir))

import os, sys

## Load Settings from .env file
from dotenv import find_dotenv, dotenv_values

# _ = load_dotenv(find_dotenv()) # read local .env file
config = dotenv_values(find_dotenv())

# debug
# print (config)

ATLAS_URI = config.get("ATLAS_URI")

if not ATLAS_URI:
    raise Exception("'ATLAS_URI' is not set.  Please set it above to continue...")

## Only need this if we are using OpenAI for Embeddings
OPENAI_API_KEY = config.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("'OPENAI_API_KEY' is not set.  Please set it above to continue...")

## Atlas settings
DB_NAME = "LORT_Bert"
COLLECTION_NAME = "LORT_Bert"
EMBEDDING_ATTRIBUTE = "embedding_local"
INDEX_NAME = "idx_embedding_local"

import os

## LlamaIndex will download embeddings models as needed.
## Set llamaindex cache dir to ./cache dir here (Default is system tmp)
## This way, we can easily see downloaded artifacts
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.path.join(
    os.path.abspath(""), "..", "llama-index-cache"
)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from pymongo import MongoClient

mongodb_client = MongoClient(ATLAS_URI)

print("Atlas client initialized")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="bert-base-uncased")

from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI

# The LLM used to generate natural language responses to queries.
# If not provided, defaults to gpt-3.5-turbo from OpenAI
# If your OpenAI key is not set, defaults to llama2-chat-13B from Llama.cpp

## Here are the models available : https://platform.openai.com/docs/models
##  gpt-3.5-turbo  |   gpt-4  | gpt-4-turbo

#  | model         | context window |
#  |---------------|----------------|
#  | gpt-3.5-turbo | 4,096          |
#  | gpt-4         | 8,192          |
#  | gpt-4-turbo   | 128,000        |


## set temperature=0 for predictable results

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
## setup embed model

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext, VectorStoreIndex


vector_store = MongoDBAtlasVectorSearch(
    mongodb_client=mongodb_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name=INDEX_NAME,
    embedding_key=EMBEDDING_ATTRIBUTE,
    ## the following columns are set to default values
    # embedding_key = 'embedding', text_key = 'text', metadata_= 'metadata',
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, service_context=service_context
)


from IPython.display import Markdown
from pprint import pprint

response = index.as_query_engine().query("Who is Gandalf?")
print(response)
print()
pprint(response, indent=4)

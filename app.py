import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openapi_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key ,model_name="text-embedding-s-small")

# initialize the chroma client with persistense
chroma_client = chromadb.PersistentClient(path="chroma_persistence_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name , embedding_function=openapi_ef
)
client = OpenAI(api_key=openai_key)

resp = client.chat.completions.create(model="gpt-3.5-turbo",messages=
                                      [
                                          {"role":"system", "content":"You are a Helpful Assistence "}
                                          ,{"role":"user","content":"What do you think of WORLD WAR 3 ?"}
                                      ])

print(resp.choices[0].message.content)

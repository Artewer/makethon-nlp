from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext
from llama_index.core.query_pipeline import QueryPipeline
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.agent.openai import OpenAIAgent
#from llama_index.llms.openai_like import OpenAILike
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class LlamaIndex:
    def __init__(self):
        # Initialization code can go here
        pass
    
    def read_data(self, input_dir, num_workers=4):
        reader = SimpleDirectoryReader(input_dir=input_dir)
        documents = reader.load_data()
        return documents

    def build_index(self, documents):
        # create the pipeline with transformations
        #embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embed_model = OpenAIEmbedding()
        #index = VectorStoreIndex.from_documents(documents=documents, transformations=[embed_model])
        pipeline = IngestionPipeline(
            transformations=[
                #SentenceSplitter(chunk_size=25, chunk_overlap=0),
                #TitleExtractor(),
                #OpenAIEmbedding(),
                embed_model
            ],
        )
        
        nodes = pipeline.run(documents=documents)
        index = VectorStoreIndex(nodes, embed_model=embed_model)

        return index
    
    
    def save_emb(self, documents, persist_dir):
        index = self.build_index(documents)
        
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )
        index.storage_context.persist(persist_dir=persist_dir)
    
    def load_emb_index(self, persist_dir):
        # storage_context = StorageContext.from_defaults(
        #     docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
        #     vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
        #     index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir)
        # )
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir)
        return index_store

    
llama = LlamaIndex()
documents = llama.read_data('data/')
#llama.save_emb(documents, 'storage/')
index = llama.build_index(documents)
#index = llama.load_emb_index('storage/')
#print(index)
#llm = OpenAILike(model="NousResearch/Hermes-2-Pro-Mistral-7B",api_base="http://localhost:8000/v1", api_key="fake")
llm = OpenAI(model="gpt-3.5-turbo-0613")

main = index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=main,
        metadata=ToolMetadata(
            name="main",
            description="Provides information the data "
            "Use a detailed plain text question as input to the tool.",
        ),
    )
]

agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)

print(agent.chat('What do you know?'))
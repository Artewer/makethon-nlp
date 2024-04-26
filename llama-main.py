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
        nodes = self.build_index(documents)
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )
        storage_context.persist(persist_dir=persist_dir)
    
    def load_emb(self, persist_dir):
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir)
        )
        return storage_context
    
    def build_agent(self, llm, engine):
        
        tools = [
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name="lecture",
                    description=(
                        "Provides information about lectures."
                    ),
                ),
            )
        ]

        agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)
        return agent
        
llama = LlamaIndex()
documents = llama.read_data('data/')
llama.save_emb(documents, 'data/')
index = llama.build_index(documents)


#llm = OpenAILike(model="NousResearch/Hermes-2-Pro-Mistral-7B",api_base="http://localhost:8000/v1", api_key="fake")

llm = OpenAIAgent()

engine = index.as_query_engine(llm = llm)


#response = llm.complete("Hello World!")

agent = llama.build_agent(llm=llm, engine=engine)

response = agent.chat("Tell me about two celebrities from the United States. ")
print(str(response))

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
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
import os
from tqdm.notebook import tqdm
import pickle
from pathlib import Path





# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
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
    

async def build_agent_per_doc(nodes, file_base):
    print(file_base)

    vi_out_path = f"./data/llamaindex_docs/{file_base}"
    summary_out_path = f"./data/llamaindex_docs/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data/llamaindex_docs/").mkdir(parents=True, exist_ok=True)
        # build vector index
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=vi_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
        )

    # build summary index
    summary_index = SummaryIndex(nodes)

    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", llm=llm
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            await summary_query_engine.aquery(
                "Extract a concise 1-2 line summary of this document"
            )
        )
        pickle.dump(summary, open(summary_out_path, "wb"))
    else:
        summary = pickle.load(open(summary_out_path, "rb"))

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-4")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the `{file_base}.html` part of the LlamaIndex docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent, summary


async def build_agents(docs):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    # # this is for the baseline
    # all_nodes = []

    for idx, doc in enumerate(tqdm(docs)):
        nodes = node_parser.get_nodes_from_documents([doc])
        # all_nodes.extend(nodes)

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = await build_agent_per_doc(nodes, file_base)

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict
        
    
documents = read_data('data/')
#llama.save_emb(documents, 'storage/')
index = build_index(documents)
#llm = OpenAILike(model="NousResearch/Hermes-2-Pro-Mistral-7B",api_base="http://localhost:8000/v1", api_key="fake")
llm = OpenAI(model="gpt-3.5-turbo-0613")

# main = index.as_query_engine(similarity_top_k=3)

# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=main,
#         metadata=ToolMetadata(
#             name="main",
#             description="Provides information the data "
#             "Use a detailed plain text question as input to the tool.",
#         ),
#     )
# ]
agents_dict, extra_info_dict = await build_agents(docs)


#agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)

#print(agent.chat('What do you know?'))
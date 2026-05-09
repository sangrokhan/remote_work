import os
from pathlib import Path
from typing import Optional
import openai
from fastapi import FastAPI
from pydantic import BaseModel

from causality_graph.store.graph import CausalityGraph
from causality_graph.store.embeddings import EmbeddingStore
from causality_graph.agent.retriever import Retriever
from causality_graph.agent.reasoner import Reasoner

GRAPH_PATH = Path("data/graph.json")
CHROMA_DIR = "data/chroma"

app = FastAPI(title="Causality Graph Query API")

_retriever: Optional[Retriever] = None
_reasoner: Optional[Reasoner] = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        graph = CausalityGraph.load(GRAPH_PATH) if GRAPH_PATH.exists() else CausalityGraph()
        embeddings = EmbeddingStore(persist_dir=CHROMA_DIR)
        _retriever = Retriever(graph=graph, embeddings=embeddings)
    return _retriever


def _get_reasoner() -> Reasoner:
    global _reasoner
    if _reasoner is None:
        client = openai.OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ.get("LLM_API_KEY", "local"),
        )
        _reasoner = Reasoner(client=client, model=os.environ.get("LLM_MODEL", "llama3"))
    return _reasoner


class QueryRequest(BaseModel):
    question: str
    gen_filter: Optional[str] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    reasoning_chain: list[str]
    source_nodes: list[str]
    subgraph_node_count: int
    subgraph_edge_count: int


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    subgraph = _get_retriever().retrieve(
        req.question, top_k=req.top_k, hops=2, gen_filter=req.gen_filter
    )
    result = _get_reasoner().answer(req.question, subgraph)
    return QueryResponse(
        answer=result.answer,
        reasoning_chain=result.reasoning_chain,
        source_nodes=result.source_nodes,
        subgraph_node_count=len(subgraph["nodes"]),
        subgraph_edge_count=len(subgraph["edges"]),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

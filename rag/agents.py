"""
agents.py
---------
Implements the Agentic RAG workflow using LangGraph.


OUR GRAPH LOOKS LIKE THIS:

    [User Query]
         │
     [ROUTER]  ──── decides action ────┐
         │                              │
    RETRIEVE?               CLARIFY?   DIRECT_ANSWER?
         │                    │               │
    [RETRIEVER]          [CLARIFY]       [ANSWER]
         │                    │               │
    [VERIFIER]              END            END
         │
   sufficient?  ── yes ──► [ANSWER] ──► END
         │
        no (retry once)
         │
    [QUERY REWRITER]
         │
    [RETRIEVER] (second attempt)
         │
    [VERIFIER] ──► [ANSWER] ──► END  (best-effort even if still insufficient)

WHY THIS DESIGN?
  • Router   : avoids unnecessary database lookups for simple greetings
  • Verifier : catches cases where retrieval found nothing useful
  • Rewriter : expands the query with synonyms/broader terms for a second try
  • Answer   : always cites sources so users can verify information
"""

import json
import logging
from typing import TypedDict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# ─── STATE DEFINITION ─────────────────────────────────────────────────────────
# This TypedDict describes every piece of information that flows through the
# graph. Every node receives the full state and returns the fields it changed.

class RAGState(TypedDict):
    query:                  str          # Original user question
    action:                 str          # RETRIEVE | CLARIFY | DIRECT_ANSWER
    rewritten_query:        str          # Reformulated query for retry
    retrieved_chunks:       List[dict]   # Chunks found in the vector store
    is_sufficient:          bool         # Verifier's verdict
    retry_count:            int          # How many retrieval retries happened
    final_answer:           str          # The response shown to the user
    citations:              List[str]    # "filename, Chunk N" strings
    clarification_question: str          # Filled when action == CLARIFY
    agent_log:              List[str]    # Human-readable trace of decisions


# ─── WORKFLOW CLASS ────────────────────────────────────────────────────────────

class AgenticRAGWorkflow:
    """
    Encapsulates the LangGraph agentic RAG workflow.
    Call .run(query) to process a user question end-to-end.
    """

    def __init__(self, api_key: str, vector_store, model: str = "gpt-4o"):
        # The LLM used by all nodes.  temperature=0 means deterministic output.
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        self.vector_store = vector_store
        self.graph = self._build_graph()

    # ═══════════════════════════════════════════════════════════════════════
    # NODE FUNCTIONS
    # Each function receives the current RAGState and returns a *partial*
    # dict — only the keys it wants to update.  LangGraph merges these back.
    # ═══════════════════════════════════════════════════════════════════════

    def _router_node(self, state: RAGState) -> dict:
        """
        NODE 1 — ROUTER
        Looks at the query and decides:
          • RETRIEVE      → needs information from our documents
          • CLARIFY       → question is too vague to answer
          • DIRECT_ANSWER → general knowledge / greeting, no documents needed
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query router for a company policy and HR document assistant.

Analyse the user query and choose exactly ONE action:

RETRIEVE        — the user is asking about company policies, HR rules, leave, benefits,
                  IT security, expenses, onboarding, training, performance reviews,
                  remote work, incidents, data privacy, or any internal process.
CLARIFY         — the query is genuinely ambiguous (e.g., "tell me about it").
DIRECT_ANSWER   — greeting, small talk, or pure general knowledge that has nothing
                  to do with company policies.

Respond with ONLY valid JSON — no extra text:
{{"action": "RETRIEVE|CLARIFY|DIRECT_ANSWER", "reason": "one-sentence explanation"}}"""),
            ("human", "User query: {query}"),
        ])

        try:
            response = self.llm.invoke(prompt.format_messages(query=state["query"]))
            result = json.loads(response.content.strip())
            action = result.get("action", "RETRIEVE")
            reason = result.get("reason", "")
        except Exception:
            action, reason = "RETRIEVE", "Defaulted to RETRIEVE (parse error)"

        log = f"🔀 Router → {action}  |  {reason}"
        logging.info(log)

        return {
            "action": action,
            "agent_log": state.get("agent_log", []) + [log],
        }

    def _retriever_node(self, state: RAGState) -> dict:
        """
        NODE 2 — RETRIEVER
        Queries the ChromaDB vector store with either the original query or
        the rewritten query (on retry).  Returns the top-6 relevant chunks.
        """
        search_query = state.get("rewritten_query") or state["query"]
        chunks = self.vector_store.search(search_query, k=6)

        log = (
            f"🔍 Retriever → found {len(chunks)} chunks"
            f"  (query: \"{search_query[:60]}{'...' if len(search_query)>60 else ''}\")"
        )
        logging.info(log)

        return {
            "retrieved_chunks": chunks,
            "agent_log": state.get("agent_log", []) + [log],
        }

    def _verifier_node(self, state: RAGState) -> dict:
        """
        NODE 3 — VERIFIER / GROUNDING AGENT
        Checks whether the retrieved chunks actually contain useful information
        to answer the query.  If not, the graph will rewrite the query and retry.
        """
        chunks = state.get("retrieved_chunks", [])
        context = "\n\n".join(c["content"] for c in chunks)[:3000]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict content verifier.

Decide whether the provided document excerpts contain SUFFICIENT information
to answer the user's question — at least partially.

Rules:
• Sufficient = the documents mention the topic and provide some specific details.
• Insufficient = the documents are completely off-topic or entirely empty.
• If you see even partial relevant information, say sufficient.

Respond with ONLY valid JSON:
{{"is_sufficient": true|false, "reason": "one-sentence explanation"}}"""),
            ("human", "Question: {query}\n\nDocument excerpts:\n{context}"),
        ])

        try:
            response = self.llm.invoke(
                prompt.format_messages(query=state["query"], context=context)
            )
            result = json.loads(response.content.strip())
            is_sufficient = bool(result.get("is_sufficient", True))
            reason = result.get("reason", "")
        except Exception:
            is_sufficient = True
            reason = "Defaulted to sufficient (parse error)"

        verdict = "✅ Sufficient" if is_sufficient else "⚠️  Insufficient"
        log = f"{verdict}  |  {reason}"
        logging.info(f"Verifier → {log}")

        return {
            "is_sufficient": is_sufficient,
            "agent_log": state.get("agent_log", []) + [f"🔎 Verifier → {log}"],
        }

    def _query_rewriter_node(self, state: RAGState) -> dict:
        """
        NODE 4 — QUERY REWRITER
        Rewrites the original query using different terminology / broader scope
        so the retriever gets a second chance to find relevant material.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a search query optimizer.

The initial search did not find sufficient information. Rewrite the query
using DIFFERENT keywords, synonyms, or a slightly broader scope.
Return ONLY the rewritten query — no explanation, no quotes.
"""),
            ("human", "Original query: {query}"),
        ])

        try:
            response = self.llm.invoke(prompt.format_messages(query=state["query"]))
            rewritten = response.content.strip()
        except Exception:
            rewritten = state["query"]  # Fall back to original

        log = f"✏️  Query Rewriter → \"{rewritten}\""
        logging.info(log)

        return {
            "rewritten_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
            "agent_log": state.get("agent_log", []) + [log],
        }

    def _answer_node(self, state: RAGState) -> dict:
        """
        NODE 5 — ANSWER GENERATOR
        Generates the final response using the LLM.

        For RETRIEVE / DIRECT_ANSWER the answer is grounded in retrieved chunks.
        Citations are built from the source filenames of every chunk used.
        """
        action = state.get("action", "RETRIEVE")
        chunks = state.get("retrieved_chunks", [])

        # ── Direct answer path (no documents needed) ────────────────────────
        if action == "DIRECT_ANSWER" or not chunks:
            direct_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a friendly and helpful company assistant. "
                 "Answer concisely. If the question might relate to a company "
                 "policy, suggest the user ask more specifically."),
                ("human", "{query}"),
            ])
            response = self.llm.invoke(
                direct_prompt.format_messages(query=state["query"])
            )
            log = "💬 Answer → DIRECT (no documents)"
            logging.info(log)
            return {
                "final_answer": response.content,
                "citations": [],
                "agent_log": state.get("agent_log", []) + [log],
            }

        # ── Document-grounded answer path ────────────────────────────────────
        context_parts = []
        citation_set: dict = {}  # source → display label

        for i, chunk in enumerate(chunks, start=1):
            source   = chunk.get("source", "Unknown")
            chunk_id = chunk.get("chunk_id", "?")
            page     = chunk.get("page", "")

            context_parts.append(
                f"[Source {i} — {source}"
                + (f", Page {page}" if page else "")
                + f", Chunk {chunk_id}]\n{chunk['content']}"
            )

            citation_label = source
            if page:
                citation_label += f" (Page {page})"
            citation_set[source] = citation_label

        context_text = "\n\n---\n\n".join(context_parts)
        is_grounded  = state.get("is_sufficient", True)

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert company policy assistant.

Your ONLY job is to answer questions using the document excerpts provided below.

Rules you MUST follow:
1. Base every statement on the provided sources. Do NOT add information from
   general knowledge that is not in the documents.
2. Cite sources inline using [Source N] notation. Example: "Employees are
   entitled to 21 days of annual leave [Source 1]."
3. If the documents cover the topic but lack a specific detail, say so clearly.
4. If the documents do NOT contain the answer at all, say:
   "I could not find this information in the available policy documents."
5. Be thorough, specific, and well-structured. Use bullet points for lists.
6. Never guess or fabricate numbers, dates, or procedures.
"""),
            ("human", "Question: {query}\n\nDocument Sources:\n\n{context}"),
        ])

        response = self.llm.invoke(
            rag_prompt.format_messages(query=state["query"], context=context_text)
        )

        citations = list(citation_set.values())
        log = f"💬 Answer → RAG  |  {len(citations)} document(s) cited"
        logging.info(log)

        return {
            "final_answer": response.content,
            "citations": citations,
            "agent_log": state.get("agent_log", []) + [log],
        }

    def _clarify_node(self, state: RAGState) -> dict:
        """
        NODE 6 — CLARIFY
        Returns a clarifying question instead of an answer when the query is
        too vague to determine what document to search.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. The user's question is ambiguous. "
             "Ask one clear, specific clarifying question to narrow it down."),
            ("human", "Vague query: {query}"),
        ])

        response = self.llm.invoke(prompt.format_messages(query=state["query"]))
        question = response.content.strip()

        log = "❓ Clarification requested"
        logging.info(log)

        return {
            "clarification_question": question,
            "final_answer": question,
            "citations": [],
            "agent_log": state.get("agent_log", []) + [log],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # CONDITIONAL ROUTING FUNCTIONS
    # These are NOT nodes — they just look at the state and return a string
    # that tells LangGraph which node to go to next.
    # ═══════════════════════════════════════════════════════════════════════

    def _route_after_router(self, state: RAGState) -> str:
        action = state.get("action", "RETRIEVE")
        if action == "RETRIEVE":
            return "retrieve"
        elif action == "CLARIFY":
            return "clarify"
        else:
            return "answer"

    def _route_after_verifier(self, state: RAGState) -> str:
        if state.get("is_sufficient", True):
            return "answer"
        elif state.get("retry_count", 0) < 1:
            return "rewrite"   # One retry allowed
        else:
            return "answer"    # Best-effort answer even if still insufficient

    # ═══════════════════════════════════════════════════════════════════════
    # GRAPH CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════

    def _build_graph(self):
        """Assemble all nodes and edges into a compiled LangGraph workflow."""
        wf = StateGraph(RAGState)

        # ── Register nodes ────────────────────────────────────────────────
        wf.add_node("router",   self._router_node)
        wf.add_node("retrieve", self._retriever_node)
        wf.add_node("verify",   self._verifier_node)
        wf.add_node("rewrite",  self._query_rewriter_node)
        wf.add_node("answer",   self._answer_node)
        wf.add_node("clarify",  self._clarify_node)

        # ── Entry point ───────────────────────────────────────────────────
        wf.set_entry_point("router")

        # ── Edges from router (conditional) ──────────────────────────────
        wf.add_conditional_edges(
            "router",
            self._route_after_router,
            {"retrieve": "retrieve", "clarify": "clarify", "answer": "answer"},
        )

        # ── After retrieval → always verify ──────────────────────────────
        wf.add_edge("retrieve", "verify")

        # ── After verification (conditional) ─────────────────────────────
        wf.add_conditional_edges(
            "verify",
            self._route_after_verifier,
            {"answer": "answer", "rewrite": "rewrite"},
        )

        # ── After rewrite → retrieve again ───────────────────────────────
        wf.add_edge("rewrite", "retrieve")

        # ── Terminal nodes → END ──────────────────────────────────────────
        wf.add_edge("answer",  END)
        wf.add_edge("clarify", END)

        return wf.compile()

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC RUN METHOD
    # ═══════════════════════════════════════════════════════════════════════

    def run(self, query: str) -> dict:
        """
        Process a user query through the full agentic RAG workflow.
        Returns the final RAGState dict containing the answer, citations,
        action taken, and the agent's reasoning log.
        """
        logging.info(f"=== New Query: {query} ===")

        initial_state: RAGState = {
            "query":                  query,
            "action":                 "",
            "rewritten_query":        "",
            "retrieved_chunks":       [],
            "is_sufficient":          False,
            "retry_count":            0,
            "final_answer":           "",
            "citations":              [],
            "clarification_question": "",
            "agent_log":              [],
        }

        result = self.graph.invoke(initial_state)

        logging.info(f"Action taken : {result.get('action')}")
        logging.info(f"Documents used: {result.get('citations')}")
        logging.info(f"=== Query complete ===")

        return result

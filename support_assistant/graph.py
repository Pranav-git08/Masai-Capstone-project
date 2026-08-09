import os
import json
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from support_assistant.vectorstore import query_vectorstore
from support_assistant.prompts import STRUCTURED_RAG_PROMPT_TEMPLATE

# Environment variable check (defaults to Mock Mode)
def is_mock_llm() -> bool:
    val = os.getenv("MOCK_LLM", "1").strip()
    return val in ["1", "true", "True", "TRUE"]

# Pydantic schema enforcing structured output
class AssistantResponse(BaseModel):
    answer: str = Field(description="The direct answer to the user query.")
    sources: List[str] = Field(default_factory=list, description="List of document IDs used as context.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level between 0.0 and 1.0.")

# Graph State definition
class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[dict]
    response: AssistantResponse

# KEYWORD HEURISTIC FOR MOCK CLASSIFICATION
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", 
    "tracking", "cancel", "gift card", "support hours"
]

# NODE 1: Classify Intent
def classify_intent_node(state: GraphState) -> GraphState:
    query = state["query"]
    
    if is_mock_llm():
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # Optional real LLM classification (Groq)
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{
                    "role": "user",
                    "content": f"Classify this query as 'policy_question' or 'general_question'. Query: '{query}'. Respond with ONLY the label."
                }]
            )
            raw = completion.choices[0].message.content.strip().lower()
            intent = "policy_question" if "policy_question" in raw else "general_question"
        except Exception:
            # Fallback to keyword heuristic if API fails
            query_lower = query.lower()
            intent = "policy_question" if any(k in query_lower for k in POLICY_KEYWORDS) else "general_question"
            
    state["intent"] = intent
    return state

# NODE 2: Retrieve and Answer (for policy_question)
def retrieve_and_answer_node(state: GraphState) -> GraphState:
    query = state["query"]
    
    # Retrieval step ALWAYS runs for real in both modes
    retrieved_chunks = query_vectorstore(query, top_k=3)
    state["retrieved_chunks"] = retrieved_chunks
    
    if is_mock_llm():
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            top_snippet = top_chunk["text"][:200]
            answer_text = f"Based on the retrieved context: {top_snippet}"
            sources = [chunk["id"] for chunk in retrieved_chunks]
        else:
            answer_text = "Based on the retrieved context: No relevant policy found."
            sources = []
            
        response = AssistantResponse(
            answer=answer_text,
            sources=sources,
            confidence=1.0
        )
    else:
        # Optional MOCK_LLM=0 Real LLM Branch with retry logic
        context_str = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in retrieved_chunks])
        prompt = STRUCTURED_RAG_PROMPT_TEMPLATE.format(context=context_str, question=query)
        
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        max_retries = 2
        validated_response = None
        current_prompt = prompt
        
        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": current_prompt}],
                    response_format={"type": "json_object"}
                )
                raw_json = json.loads(completion.choices[0].message.content)
                validated_response = AssistantResponse(**raw_json)
                break
            except Exception as e:
                if attempt < max_retries:
                    current_prompt += f"\n\nCorrection: Your previous response failed validation: {str(e)}. Please output valid JSON matching the exact schema."
                else:
                    validated_response = AssistantResponse(
                        answer="Error generating response from LLM.",
                        sources=[c["id"] for c in retrieved_chunks],
                        confidence=0.0
                    )
        response = validated_response
        
    state["response"] = response
    return state

# NODE 3: Direct Answer (for general_question)
def direct_answer_node(state: GraphState) -> GraphState:
    if is_mock_llm():
        response = AssistantResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    else:
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant. Keep answers brief."
                }, {
                    "role": "user",
                    "content": state["query"]
                }]
            )
            ans = completion.choices[0].message.content.strip()
            response = AssistantResponse(answer=ans, sources=[], confidence=1.0)
        except Exception:
            response = AssistantResponse(
                answer="I can only answer questions about Zepto policies right now.",
                sources=[],
                confidence=1.0
            )
            
    state["response"] = response
    return state

# CONDITIONAL EDGE ROUTER
def route_intent(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

# BUILD GRAPH
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)
    
    workflow.set_entry_point("classify_intent")
    
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )
    
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    
    return workflow.compile()

app_graph = build_graph()

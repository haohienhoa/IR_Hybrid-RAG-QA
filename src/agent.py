from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL, HF_TOKEN

def create_rag_chain(retriever):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        huggingfacehub_api_token=HF_TOKEN
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    prompt_template = """You are an elite Aerospace Engineering AI. Answer the user's query using ONLY the provided Context.
        
PROCESS:
1. <thought>: Analyze the context step-by-step. If the context does not contain the answer, explicitly state it here.
2. <answer>: Provide a professional, Markdown-formatted answer. If no information is found, output "Information not found in the documents."

--- FEW-SHOT EXAMPLE ---
Context: 
Doc 1: The wing sweepback delays the onset of wave drag.
Query: What is the purpose of wing sweepback?
<thought>
The context mentions "wing sweepback" and states its effect is to "delay the onset of wave drag". I will output this as the answer.
</thought>
<answer>
**Purpose of Wing Sweepback:**
The primary purpose is to **delay the onset of wave drag**.
</answer>
------------------------

--- ACTUAL TASK ---
Context:
{context}

Query: {query}
"""
    prompt = PromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        print(f"\n[Retriever] Found {len(docs)} documents. Feeding to LLM...")
        return "\n\n".join([f"Doc {i+1}: {d.page_content}" for i, d in enumerate(docs)])

    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
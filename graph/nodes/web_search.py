from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from ..state import GraphState
from dotenv import load_dotenv

load_dotenv(".env.adv",override=True)
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})
    #additional changes
    # print("TYPE:", type(tavily_results))
    # print("RESULT:")
    # print(tavily_results)
    joined_tavily_result = "\n".join(
        [result["content"] for result in tavily_results["results"]]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


# if __name__ == "__main__":
#     web_search(state={"question": "agent memory", "documents": None})
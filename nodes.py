import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from typing import List, Dict
import json

# Import API key from config
from config import TAVILY_API_KEY, gemeni_key

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize Sentence Transformer model for embeddings
# This model will be downloaded the first time it's used.

# We'll no longer do direct dictionary lookups for retrieval.
initial_knowledge_base_data = {
    "Solve for x: 2x + 5 = 11": "Step 1: Subtract 5 from both sides: 2x = 6.\nStep 2: Divide by 2: x = 3.\nFinal Answer: x = 3",
    "Simplify the expression: (a+b)^2": "Step 1: Use the formula (x+y)^2 = x^2 + 2xy + y^2.\nStep 2: Apply the formula: (a+b)^2 = a^2 + 2ab + b^2.\nFinal Answer: a^2 + 2ab + b^2",
    "Find the slope of the line passing through points (1, 2) and (3, 6)": "Step 1: Use the slope formula: m = (y2 - y1) / (x2 - x1).\nStep 2: Substitute the points: m = (6 - 2) / (3 - 1) = 4 / 2.\nStep 3: Simplify: m = 2.\nFinal Answer: 2",
    "What is the derivative of x^2?": "Step 1: Use the power rule: d/dx (x^n) = n*x^(n-1).\nStep 2: Apply the rule with n=2: d/dx (x^2) = 2*x^(2-1) = 2x^1.\nFinal Answer: 2x",
    "What is the integral of x dx?": "Step 1: Use the power rule for integrals: ∫x^n dx = (x^(n+1))/(n+1) + C.\nStep 2: Apply the rule with n=1: ∫x^1 dx = (x^(1+1))/(1+1) + C = x^2/2 + C.\nFinal Answer: x^2/2 + C"
}


# --- Node Functions ---

def check_math_related_node(state: Dict):
    """
    Input guardrail node. Checks if the query is math-related.
    Returns: "math_related" or "not_math_related" for conditional routing.
    """
    query = state["query"]
    math_keywords = ["solve", "simplify", "find", "derivative", "integral", "equation", "function", "slope", "probability", "calculate", "+", "-", "*", "/", "^", "="]
    is_math = False
    for keyword in math_keywords:
        if keyword in query.lower():
            is_math = True
            break

    if is_math:
        print(f"[{check_math_related_node.__name__}] Query is math-related.")
        return {"query": query}
    else:
        print(f"[{check_math_related_node.__name__}] Query is NOT math-related.")
        return {"error": "This agent is designed for mathematical questions. Please ask a math-related query."}


def retrieve_from_knowledge_base_node(state: Dict):
    """
    Retrieves an answer directly from the knowledge base dictionary if the query is an exact match.
    """
    query = state["query"]
    kb_result = initial_knowledge_base_data.get(query, "")  # Returns "" if no exact match
    print(f"[retrieve_from_knowledge_base_node] Direct dictionary lookup for: {query}. Result: {kb_result}")
    return {"kb_result": kb_result}


def perform_web_search_node(state: Dict):
    """
    Performs a web search using Tavily.
    """
    query = state["query"]
    print(f"[{perform_web_search_node.__name__}] Performing web search using Tavily for: {query}")
    try:
        response_dict = tavily_client.search(query=query, max_results=3)
        web_results = response_dict.get('results', [])

        print(f"[{perform_web_search_node.__name__}] Tavily search results (URLs and snippets):")
        for res in web_results:
            print(f"- URL: {res.get('url', 'N/A')}\n  Snippet: {res.get('content', 'N/A')[:100]}...")

        return {"web_results": web_results}
    except Exception as e:
        print(f"[{perform_web_search_node.__name__}] Error during Tavily web search: {e}. Make sure your API key is correct and you have internet access.")
        return {"error": f"Web search failed: {e}"}


def extract_from_web_results_node(state: Dict):
    """
    Extracts relevant information from Tavily search results.
    """
    web_results = state["web_results"]
    extracted_text = ""
    if web_results:
        for result in web_results:
            extracted_text += f"Source URL: {result.get('url', 'N/A')}\n"
            extracted_text += f"Content Snippet: {result.get('content', 'N/A')}\n---\n"

    print(f"[{extract_from_web_results_node.__name__}] Extracted Text from Tavily (Snippet):\n", extracted_text[:700])
    return {"extracted_content": extracted_text}


async def generate_solution_node(state: Dict):
    """
    Generates a solution using an LLM (gemini-2.0-flash).
    If kb_result is present, uses that. Otherwise, uses extracted_content from web.
    Includes explicit output guardrails in the prompt.
    """
    query = state["query"]
    kb_result = state.get("kb_result")
    extracted_content = state.get("extracted_content")
    feedback = state.get("feedback")  # Get feedback from state if available

    solution_text = ""

    if kb_result:  # kb_result will be a non-empty string if found
        print(f"[{generate_solution_node.__name__}] Generating solution from Knowledge Base.")
        solution_text = kb_result
    elif extracted_content:
        print(f"[{generate_solution_node.__name__}] Generating solution from Web Content using LLM.")

        llm_prompt = (
            f"You are a highly accurate and responsible mathematical professor. Your primary goal is to provide "
            f"a step-by-step, simplified, and **mathematically correct** solution to a mathematical problem. "
            f"Ensure the solution is clear, concise, and easy for a student to understand, using standard mathematical notation where appropriate.\n\n"
            f"**IMPORTANT OUTPUT GUIDELINES:**\n"
            f"- **Accuracy is paramount:** Double-check all calculations and logical steps.\n"
            f"- **Step-by-step format:** Break down the solution into distinct, numbered steps.\n"
            f"- **Simplification:** Explain complex concepts in a way that is accessible to a student.\n"
            f"- **No incorrect information:** If you are unsure or the provided information is insufficient to derive a correct solution, "
            f"you **MUST** state that you cannot provide a complete or accurate solution and explain why.\n"
            f"- **Stay on topic:** Only provide educational content related to the mathematical problem. Avoid irrelevant information.\n\n"
            f"Problem: {query}\n\n"
            f"Here is some relevant information found online:\n{extracted_content}\n\n"
        )

        # Only add refinement feedback if it's not one of the simple 'yes'/'no' responses
        if feedback and feedback not in ["yes", "no", ""]:
            llm_prompt += f"The user provided the following feedback for refinement: '{feedback}'. Please incorporate this feedback into your revised solution.\n\n"

        llm_prompt += "Please provide the solution now:"

        chatHistory = []
        chatHistory.append({"role": "user", "parts": [{"text": llm_prompt}]})
        payload = {"contents": chatHistory}
        apiKey = gemeni_key  # Canvas will provide this at runtime
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

        try:
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()

            if result.get('candidates') and result['candidates'][0].get('content') and \
                    result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text'):
                solution_text = result['candidates'][0]['content']['parts'][0]['text']
            else:
                solution_text = "LLM could not generate a solution from the provided information due to an unexpected response format."
                print(f"[{generate_solution_node.__name__}] LLM response structure unexpected: {result}")

        except requests.exceptions.RequestException as e:
            solution_text = f"Error communicating with LLM: {e}. Please check your internet connection or API service status."
            print(f"[{generate_solution_node.__name__}] LLM API call failed: {e}")
        except json.JSONDecodeError as e:
            solution_text = f"Error decoding LLM response: {e}. The LLM might have returned malformed JSON."
            print(f"[{generate_solution_node.__name__}] LLM response not valid JSON: {e}")

    else:
        print(f"[{generate_solution_node.__name__}] No information available to generate solution.")
        solution_text = "Could not generate a solution due to lack of relevant information from either knowledge base or web search."

    # The feedback is now handled by the Streamlit UI, so we just return the solution
    return {"solution": solution_text}

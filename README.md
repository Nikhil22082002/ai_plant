Math Agent: An Agentic-RAG Mathematical Professor This repository contains the source code for the Math Agent, an Agentic-RAG (Retrieval-Augmented Generation) system designed to act as a mathematical professor. It provides step-by-step, simplified solutions to mathematical questions by leveraging a knowledge base and real-time web search, incorporating a human-in-the-loop feedback mechanism for continuous improvement.

Features Intelligent Routing: Dynamically decides whether to retrieve from an internal knowledge base or perform a web search.

Knowledge Base Integration: Utilizes Qdrant (a vector database) for efficient semantic search and retrieval of pre-solved problems.

Web Search Capabilities: Integrates with Tavily API to fetch up-to-date information for novel queries.

LLM-Powered Solution Generation: Uses Google's Gemini 2.0 Flash model to generate clear, step-by-step mathematical solutions.

Input & Output Guardrails: Ensures queries are math-related and generated solutions are accurate and on-topic.

Human-in-the-Loop Feedback: Allows users to provide feedback on solutions, enabling potential future refinements and learning.

Streamlit Web Interface: Provides an easy-to-use graphical interface for interaction.

Architecture The agent's core logic is built using LangGraph, enabling a stateful, multi-step workflow. The general flow is as follows:

User Query: A mathematical question is submitted.

Input Guardrail: Checks if the query is math-related.

Knowledge Base Retrieval: Attempts to find a relevant solution in Qdrant.

Web Search (if KB miss): If no relevant solution is found in the KB, Tavily performs a web search.

Content Extraction: Relevant snippets are extracted from web search results.

Solution Generation: An LLM generates a step-by-step solution using either the KB result or extracted web content.

Present Solution & Feedback: The solution is presented to the user, who can provide feedback ("Yes", "No", "Improve").

Refinement (Future): Feedback can be used for iterative refinement or long-term model improvement.

Setup and Installation Clone the repository:

git clone https://github.com/YOUR_USERNAME/ai_plant.git cd ai_plant

Create a virtual environment (recommended):

python -m venv venv

On Windows:
.\venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Set up API Keys:

Create a config.py file in the root directory of the project.

Add your Tavily API key to config.py:

config.py
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"

Ensure your environment has access to the Gemini API (e.g., via GOOGLE_API_KEY environment variable if running locally, or handled by your deployment environment).

Run Qdrant (Vector Database):

The application uses Qdrant. You can run it as a Docker container:

docker pull qdrant/qdrant docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

The Qdrant knowledge base will be initialized and populated automatically the first time you run the app.py script.

Running the Application To start the Streamlit web application:

streamlit run app.py

This will open the Math Agent interface in your web browser, typically at http://localhost:8501.

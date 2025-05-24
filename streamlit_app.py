# This is a simple Streamlit app that uses the LangGraph framework to create a math problem-solving agent.
import streamlit as st
import asyncio
from graph_builder import app  # LangGraph app with compiled workflow

st.set_page_config(page_title="Math Agent", layout="centered")

st.title("Math Problem Solver")
st.markdown("Ask any math-related question. The agent will try to solve it using its knowledge base or the web.")

query = st.text_input("Enter your math question:")

if st.button("Get Solution"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Solving your problem..."):
            inputs = {
                "query": query,
                "kb_result": "",
                "web_results": [],
                "extracted_content": "",
                "solution": "",
                "feedback": "",
                "error": ""
            }

            async def run_agent():
                return await app.ainvoke(inputs)

            final_output = asyncio.run(run_agent())

            if final_output.get("error"):
                st.error(f"Error: {final_output['error']}")
            elif final_output.get("solution"):
                st.session_state["last_solution"] = final_output["solution"]
                st.session_state["last_query"] = query
                st.success("Here's the solution:")
                st.markdown(final_output["solution"])
                st.session_state["feedback_stage"] = True
            else:
                st.info("No solution generated.")

# Show feedback UI if we have a solution
if st.session_state.get("feedback_stage", False):
    st.markdown("###  Was the solution helpful?")
    feedback = st.radio("Your feedback:", ["Yes", "No", "Improve"])

    if feedback == "Improve":
        improvement = st.text_area("Tell us what to improve:")
    else:
        improvement = ""

    if st.button("Submit Feedback"):
        with st.spinner("Incorporating your feedback..."):
            refined_inputs = {
                "query": st.session_state["last_query"],
                "kb_result": "",
                "web_results": [],
                "extracted_content": "",
                "solution": "",
                "feedback": improvement if feedback == "Improve" else feedback.lower(),
                "error": ""
            }

            async def run_refined():
                return await app.ainvoke(refined_inputs)

            refined_output = asyncio.run(run_refined())

            if refined_output.get("solution"):
                st.success("Here's is final answer:")
                st.markdown(refined_output["solution"])
            else:
                st.info("No refined solution was generated.")
        st.session_state["feedback_stage"] = False

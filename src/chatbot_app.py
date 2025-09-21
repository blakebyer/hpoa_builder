import streamlit as st
from typing import Optional
from aurelian.agents.hpoa.hpoa_config import get_config
from aurelian.agents.hpoa.hpoa_agent import hpoa_simple_agent, hpoa_agent, hpoa_reasoning_agent, call_agent_with_retry, call_agent
import time, json

# initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}
if "pending_msg" not in st.session_state:
    st.session_state.pending_msg = None
if "run_example" not in st.session_state:
    st.session_state.run_example = False

st.title("HPOA Assistant")

# examples that can auto-run
examples = [
    "List the phenotypes and source studies for OMIM:300615",
    "What neurological phenotypes are associated with Coffin-Lowry syndrome?",
    "Propose new annotations for Fabry disease based on PMID:21092187",
    "Which organ system does HP:0004322 (Short stature) belong to?"
]

cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex, key=f"ex_{i}"):
        st.session_state.pending_msg = ex
        st.session_state.run_example = True

# render full chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# collect user input or run example if clicked
user_msg = st.chat_input("Ask the agent…")

if st.session_state.run_example:
    user_msg = st.session_state.pending_msg
    st.session_state.run_example = False
    st.session_state.pending_msg = None

# process new message
if user_msg:
    # save and render the user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
            st.markdown(user_msg)
    # create assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = ""
            try:
                result = call_agent_with_retry(input=user_msg, agent=hpoa_agent)
                data = getattr(result, "output", None) or getattr(result, "data", None)

                if hasattr(data, "model_dump"):
                    dd = data.model_dump()
                    text = (dd.get("text") or "").strip()
                    ann = dd.get("annotations") or []
                    if ann:  # annotations present → copyable JSON only
                        block = json.dumps({
                            "explanation": text,
                            "annotations": ann,
                        }, indent=2)
                        reply = f"```json\n{block}\n```"
                    else:    # no annotations → just plain text
                        reply = text
                else:
                    reply = str(data) if data is not None else "Error: no content returned from model. Please try again later."
            except Exception as e:
                reply = f"Error: {e}"

            st.markdown(reply)

    # save assistant reply and rerun to render full history statically
    st.session_state.messages.append({"role": "assistant", "content": reply})
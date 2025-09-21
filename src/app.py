# Components
# 1. Hierarchy browser of both MONDO and HPO
# 2. HPOA builder
# 3. Chatbot interface that can query pubmed for articles, return their HPs and auto-populate HPOA file
import numpy as np
import pandas as pd
import streamlit as st
from contextlib import contextmanager
from streamlit_tree_select import tree_select
import streamlit_antd_components as sac
from st_ant_tree import st_ant_tree
from hpotk.model import TermId, MinimalTerm
from hpotk.ontology import MinimalOntology, create_minimal_ontology
from hpotk.graph import CsrIndexedGraphFactory
from hpotk.util import open_text_io_handle_for_reading
from hpotk.ontology.load.obographs._model import create_node, create_edge, NodeType
from hpotk.ontology.load.obographs._factory import MinimalTermFactory
from pydantic_ai import Agent
import json, re, typing, requests, io
from aurelian.agents.hpoa.hpoa_agent import (
    hpoa_simple_agent,
    hpoa_agent,
    hpoa_reasoning_agent,
    call_agent_with_retry,
    call_agent,
)

PURL_PATTERN = re.compile(
    r"http://purl\.obolibrary\.org/obo/(?P<curie>(?P<prefix>\w+)_(?P<id>\w+))"
)

def extract_curie_from_purl(purl: str) -> typing.Optional[str]:
    matcher = PURL_PATTERN.match(purl)
    return matcher.group("curie") if matcher else None

def extract_terms_ontology(nodes, prefixes_of_interest={"MONDO"}):
    curie_to_term: dict[str, TermId] = {}
    terms: list[MinimalTerm] = []
    term_factory = MinimalTermFactory()

    for data in nodes:
        node = create_node(data)
        if not node or node.type != NodeType.CLASS:
            continue
        curie = extract_curie_from_purl(node.id)
        if not curie:
            continue
        term_id = TermId.from_curie(curie)
        if term_id.prefix not in prefixes_of_interest:
            continue
        curie_to_term[curie] = term_id
        term = term_factory.create_term(term_id, node)
        if term:
            terms.append(term)

    return curie_to_term, terms

def create_edge_list(edges, curie_to_termid):
    edge_list = []
    for data in edges:
        edge = create_edge(data)
        if edge.pred != "is_a":
            continue
        src_curie = extract_curie_from_purl(edge.sub)
        dest_curie = extract_curie_from_purl(edge.obj)
        if not src_curie or not dest_curie:
            continue
        try:
            src = curie_to_termid[src_curie]
            dest = curie_to_termid[dest_curie]
        except KeyError:
            continue
        edge_list.append((src, dest))
    return edge_list

def load_minimal_ontology(url: str, prefix: str = "MONDO") -> MinimalOntology:
    with open_text_io_handle_for_reading(url) as fh:
        doc = json.load(fh)

    obograph = doc["graphs"][0]
    id_to_term_id, terms = extract_terms_ontology(
        obograph["nodes"], prefixes_of_interest={prefix}
    )
    edges = create_edge_list(obograph["edges"], id_to_term_id)
    graph = CsrIndexedGraphFactory().create_graph(edges)
    return create_minimal_ontology(graph, terms, version=None)

# Build HPO Tree Recursively
def build_tree(ontology, term_id="HP:0000001", path=""):
    term = ontology.get_term(term_id)
    term_name = term.name
    current_path = f"{path}/{term_name+term_id}" if path else term_name + term_id

    children = ontology.graph.get_children(term) or []
    children_nodes = []
    for child in children:
        node = build_tree(ontology, child.value, current_path)
        if node:
            children_nodes.append(node)

    return {
        "label": f"{term.identifier} ({term.name})",
        "value": current_path,
        "children": children_nodes,
    }

# ------------------------------
# Agent annotation parsing + styling
# ------------------------------
def annotations_to_df(result) -> pd.DataFrame:
    """Flatten agent annotations into a DataFrame with status + rationale."""
    if not hasattr(result, "output"):
        return pd.DataFrame()

    data = result.output
    if hasattr(data, "model_dump"):
        annotations = data.model_dump().get("annotations") or []
    elif isinstance(data, dict):
        annotations = data.get("annotations", [])
    else:
        annotations = []

    rows = []
    for item in annotations:
        if isinstance(item, dict) and "annotation" in item:
            row = item["annotation"].copy()
            row["status"] = "removed"
            row["rationale"] = item.get("rationale", "")
        else:
            row = item.copy()
            row["status"] = row.get("status", "added")
            row["rationale"] = row.get("rationale", "")
        rows.append(row)

    return pd.DataFrame(rows)

def style_agent_edits(df: pd.DataFrame):
    def highlight(row):
        if row["status"] == "added":
            return ["background-color: #d4f4dd"] * len(row)  # green
        elif row["status"] == "changed":
            return ["background-color: #fff3cd"] * len(row)  # yellow
        elif row["status"] == "removed":
            return ["background-color: #f8d7da"] * len(row)  # red
        return [""] * len(row)
    return df.style.apply(highlight, axis=1)

# ------------------------------
# Style
# ------------------------------
st.set_page_config(layout="wide")
st.title("HPOA Builder")

@st.cache_resource(show_spinner="Loading HPO...")
def load_minimal_hpo():
    url = "https://purl.obolibrary.org/obo/hp.json"
    return load_minimal_ontology(url, prefix="HP")

hpo = load_minimal_hpo()
HPO_TREE = [build_tree(hpo)]

@st.cache_resource(show_spinner="Loading MONDO...")
def load_minimal_mondo():
    url = "https://purl.obolibrary.org/obo/mondo.json"
    return load_minimal_ontology(url, prefix="MONDO")

mondo = load_minimal_mondo()
MONDO_TREE = [build_tree(mondo, term_id="MONDO:0700096")]

@st.cache_resource(show_spinner="Loading HPO Annotations...")
def load_hpoa_df() -> pd.DataFrame:
    r = requests.get(
        "https://api.github.com/repos/obophenotype/human-phenotype-ontology/releases/latest",
        timeout=20,
    )
    r.raise_for_status()
    url = next(
        a["browser_download_url"]
        for a in r.json().get("assets", [])
        if "phenotype.hpoa" in a.get("browser_download_url", "")
    )
    f = requests.get(url, timeout=60)
    f.raise_for_status()
    return pd.read_csv(io.StringIO(f.text), sep="\t", comment="#", dtype=str, keep_default_na=False)

hpoa_df = load_hpoa_df()

col1, col2 = st.columns([1, 3])

# -------------------
# Left column with tabs
# -------------------
with col1:
    tab1, tab2 = st.tabs(["Ontology Browser", "Agent"])

    with tab1:
        options = ["HPO", "MONDO"]
        ontology_choice = st.segmented_control("Select ontology:", options, selection_mode="single")

        if ontology_choice == "HPO":
            q = st.text_input("Search phenotypic abnormality:", placeholder="Search").lower()
            filtered = [HPO_TREE[0]] if not q else []
            selected_hp = sac.tree(items=filtered, height=500, open_all=True, checkbox=False, show_line=True)

        elif ontology_choice == "MONDO":
            q = st.text_input("Search disease:", placeholder="Search").lower()
            filtered = [MONDO_TREE[0]] if not q else []
            selected_mondo = sac.tree(items=filtered, height=500, open_all=True, checkbox=False, show_line=True)

    with tab2:
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for role, content in st.session_state.chat_messages:
            with st.chat_message(role):
                st.markdown(content)

        user_msg = st.chat_input("Ask the agent…")
        if user_msg:
            st.session_state.chat_messages.append(("user", user_msg))
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        reply = call_agent_with_retry(user_msg)
                        st.session_state.last_agent_result = reply
                    except Exception as e:
                        reply = f"Error: {e}"
                    st.markdown(getattr(reply, "output", str(reply)))
                    st.session_state.chat_messages.append(("assistant", str(reply)))

# -------------------
# Right column
# -------------------
with col2:
    st.header("HPO Annotations")
    opts = sorted(hpoa_df["disease_name"].dropna().unique())
    picked = st.multiselect("Select diseases to edit:", options=opts)

    if picked:
        copy_df = hpoa_df[hpoa_df["disease_name"].isin(picked)].copy().reset_index(drop=True)

        if "last_agent_result" in st.session_state:
            agent_df = annotations_to_df(st.session_state.last_agent_result)
        else:
            agent_df = pd.DataFrame()

        if not agent_df.empty:
            st.subheader("Agent-suggested edits")
            edited = st.data_editor(agent_df, hide_index=True, num_rows="dynamic", key="agent_edits")

            st.dataframe(style_agent_edits(agent_df), width="stretch")

            if st.button("Approve Edits"):
                for _, row in edited.iterrows():
                    if row["status"] == "removed":
                        hpoa_df = hpoa_df[
                            ~(
                                (hpoa_df["database_id"] == row["database_id"]) &
                                (hpoa_df["hpo_id"] == row["hpo_id"])
                            )
                        ]
                    else:
                        hpoa_df = pd.concat(
                            [hpoa_df, row.drop(["status", "rationale"]).to_frame().T],
                            ignore_index=True,
                        )
                st.success("Approved agent edits applied!")
        else:
            edited = st.data_editor(copy_df, hide_index=True, num_rows="dynamic", key="manual_edits")
            if st.button("Approve Edits"):
                st.session_state.edited_copy = edited
                st.success("Edits approved")

    st.page_link(
        "https://hpo-annotation-qc.readthedocs.io/en/latest/annotationFormat.html",
        label="HPOA Format", icon="ℹ️"
    )
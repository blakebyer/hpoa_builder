# app1.py

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
# from st_ant_tree import st_ant_tree
from hpotk.model import TermId, MinimalTerm
from hpotk.ontology import MinimalOntology, create_minimal_ontology
from hpotk.graph import CsrIndexedGraphFactory
from hpotk.util import open_text_io_handle_for_reading
from hpotk.ontology.load.obographs._model import create_node, create_edge, NodeType
from hpotk.ontology.load.obographs._factory import MinimalTermFactory
import json
import re
import typing

PURL_PATTERN = re.compile(r"http://purl\.obolibrary\.org/obo/(?P<curie>(?P<prefix>\w+)_(?P<id>\w+))")

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
    id_to_term_id, terms = extract_terms_ontology(obograph["nodes"], prefixes_of_interest={prefix})
    edges = create_edge_list(obograph["edges"], id_to_term_id)
    graph = CsrIndexedGraphFactory().create_graph(edges)
    return create_minimal_ontology(graph, terms, version=None)

# Build HPO Tree Recursively
def build_tree(ontology, term_id="HP:0000001", path=""):
    term = ontology.get_term(term_id)
    term_name = term.name
    current_path = f"{path}/{term_name+term_id}" if path else term_name + term_id # make value path-unique

    children = ontology.graph.get_children(term) or []
    children_nodes = []
    for child in children:
        node = build_tree(ontology, child.value, current_path)
        if node:
            children_nodes.append(node)

    return {
        "label": f"{term.identifier} | {term.name}",
        "value": current_path,  # unique per path (allows multiple parenthood)
        "children": children_nodes
    }

# Style
st.set_page_config(layout="wide")

st.markdown("""
<style>
/* ------------------------------
   GLOBAL THEME
------------------------------ */
html, body, [class*="css"] {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  color: #1F2937; /* dark gray text */
  background-color: #FFFFFF; /* light bg like Monarch pages */
}

/* Hover/focus state for better a11y */
.rct-tree .rct-node:hover .rct-icon { 
  color: #009191 !important; 
}
.rct-tree .rct-node:focus-within .rct-icon { 
  outline: 2px solid rgba(0,168,168,0.35); 
  outline-offset: 2px; 
}
            
/* ------------------------------
   SLIDERS
------------------------------ */
.stSlider > div[data-baseweb="slider"] [role="slider"] {
  background-color: #00A8A8 !important; /* teal handle */
  border: 2px solid #00A8A8 !important;
}
.stSlider > div[data-baseweb="slider"] > div > div {
  background: #00A8A8 !important; /* teal track */
}

/* ------------------------------
   MULTISELECT TAGS / PILLS
------------------------------ */
div[data-baseweb="tag"] {
  border-radius: 999px !important;
  padding: 2px 10px !important;
  background: #F1F5F9 !important;   /* light gray */
  border: 1px solid #E2E8F0 !important;
  color: #374151 !important;        /* dark gray text */
}
div[data-baseweb="tag"] span {
  max-width: none !important;
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
}

/* ------------------------------
   BUTTONS
------------------------------ */
.stButton > button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  color: white !important;
  border: none !important;
  padding: 0.5rem 1rem !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.08);
  background-color: #00A8A8 !important; /* default teal */
}

/* Secondary-style button */
.stButton > button[kind="secondary"] {
  background-color: #E6F6F6 !important;
  color: #007777 !important;
  border: 1px solid #89D7D7 !important;
}

/* ------------------------------
   DATAFRAMES & EDITORS
------------------------------ */
.stDataFrame, .stDataEditor {
  border-radius: 12px !important;
  box-shadow: 0 3px 12px rgba(0,0,0,0.05);
  background-color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

## Pretty tree viewer
st.set_page_config(layout="wide")
st.title("HPOA Builder")

@st.cache_resource(show_spinner = "Loading HPO...")
def load_minimal_hpo():
    url = "https://purl.obolibrary.org/obo/hp.json"
    return load_minimal_ontology(url, prefix="HP")

hpo = load_minimal_hpo()
HPO_TREE = [ build_tree(hpo) ] # must be a list

@st.cache_resource(show_spinner = "Loading MONDO...")
def load_minimal_mondo():
    url = "https://purl.obolibrary.org/obo/mondo.json"
    return load_minimal_ontology(url, prefix="MONDO")

mondo = load_minimal_mondo()
MONDO_TREE = [ build_tree(mondo, term_id = "MONDO:0700096") ] # only human diseases

# Get HPOA
@st.cache_resource(show_spinner = False)
def load_hpoa_df():
    url = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-05-06/phenotype.hpoa?raw=true"
    return pd.read_csv(url, sep="\t", comment="#", dtype=str, index_col=False)

hpoa_df = load_hpoa_df()

# Filter function
def filter_nodes(nodes, q):
    if not q: return nodes
    def keep(n):
        kids = n.get("children", [])
        kept = [c for c in (keep(k) for k in kids) if c]
        match = q in n["label"].lower() or any(kept)
        if not match: return None
        out = {"label": n["label"], "value": n["value"]}
        if kept: out["children"] = kept
        return out
    return [n for n in (keep(x) for x in nodes) if n]

def expanded_values(nodes):
    vals = []
    for n in nodes:
        if "children" in n:
            vals.append(n["value"])
            vals.extend(expanded_values(n["children"]))
    return vals

col1, col2 = st.columns([1, 3]) 

with col1:
    st.header("Ontology Browser")

    options = ["HPO", "MONDO"]
    ontology_choice = st.segmented_control(
    "Select ontology:", options, selection_mode="single"
    )
    
    if ontology_choice == "HPO":
        q = st.text_input("Search phenotypic abnormality:", placeholder="Search").lower()
        filtered = filter_nodes(HPO_TREE, q)
        expanded = expanded_values(filtered) if q else []
    
        with st.container(height=500, border=False):
            selected = tree_select(
                nodes=filtered,
                expanded=expanded,
                check_model="leaf",
                only_leaf_checkboxes=True,
            )
       
    elif ontology_choice == "MONDO":
        q = st.text_input("Search disease:", placeholder="Search").lower()
        filtered = filter_nodes(MONDO_TREE, q)
        expanded = expanded_values(filtered) if q else []
    
        with st.container(height=500, border=False):
            selected = tree_select(
                nodes=filtered,
                expanded=expanded,
                check_model="leaf",
                only_leaf_checkboxes=True,
            )
    
    
with col2:
    st.header("HPO Annotations")
    name_series = hpoa_df["disease_name"].astype(str)
    opts = sorted(hpoa_df["disease_name"].dropna().unique())

    # max width of disease names
    st.markdown("""
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 300px;
            font-size: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    picked = st.multiselect("Select diseases to edit:", options=opts)

    if picked:
        copy_df = hpoa_df[hpoa_df["disease_name"].isin(picked)].copy().reset_index(drop = True)
        edited = st.data_editor(
            copy_df,
            hide_index=True,
            num_rows="dynamic",
            key="edit_copy"
        )

    st.page_link("https://hpo-annotation-qc.readthedocs.io/en/latest/annotationFormat.html", label="HPOA Format", icon="ℹ️")

    if st.button("Save edits"):
        st.session_state.edited_copy = edited
        st.success("Edits saved")




# st.sidebar.write(return_select)
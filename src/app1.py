# app1.py

# Components
# 1. Hierarchy browser of both MONDO and HPO
# 2. HPOA builder
# 3. Chatbot interface that can query pubmed for articles, return their HPs and auto-populate HPOA file
import numpy as np
import pandas as pd
import streamlit as st
#from streamlit_tree_select import tree_select
from st_ant_tree import st_ant_tree
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

## Pretty tree viewer
st.set_page_config(layout="wide")
st.title("HPOA Builder")

@st.cache_resource()
def load_minimal_hpo():
    url = "https://purl.obolibrary.org/obo/hp.json"
    return load_minimal_ontology(url, prefix="HP")

hpo = load_minimal_hpo()
hpo_tree = [ build_tree(hpo) ] # must be a list

@st.cache_resource()
def load_minimal_mondo():
    url = "https://purl.obolibrary.org/obo/mondo.json"
    return load_minimal_ontology(url, prefix="MONDO")

mondo = load_minimal_mondo()
mondo_tree = [ build_tree(mondo, term_id = "MONDO:0000001") ]

# Get HPOA
@st.cache_resource()
def load_hpoa_df():
    url = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-05-06/phenotype.hpoa?raw=true"
    return pd.read_csv(url, sep="\t", comment="#", dtype=str, index_col=False)

hpoa_df = load_hpoa_df()

col1, col2 = st.columns([1, 3]) 

with col1:
    st.header("Ontology Browser")

    options = ["HPO", "MONDO"]
    ontology_choice = st.segmented_control(
    "Select ontology:", options, selection_mode="single"
    )
    
    if ontology_choice == "HPO":
        # selected = tree_select(
        #     nodes=hpo_tree,
        #     check_model="leaf"
        # )
        selected = st_ant_tree(
            treeData=hpo_tree,
            treeCheckable=False,
            placeholder="Search",
            showSearch=True,
            overall_css="""
        .ant-select-arrow {
            display: none !important;
        }
    """
        )
    elif ontology_choice == "MONDO":
        # selected = tree_select(
        #     nodes=mondo_tree,
        #     check_model="leaf"
        # )
        selected = st_ant_tree(
            treeData=mondo_tree,
            treeCheckable=False,
            placeholder="Search",
            showSearch=True,
            overall_css="""
        .ant-select-arrow {
            display: none !important;
        }
    """
        )

with col2:
    st.header("HPO Annotations")
    query = st.text_input("Search disease name:")
    if query:
        filtered_df = hpoa_df[hpoa_df["disease_name"].str.contains(query, case=False, na=False)]
    else:
        filtered_df = hpoa_df
    
    # st.data_editor(filtered_df, hide_index=True)
    selection_event = st.dataframe(
        filtered_df,
        selection_mode="multi-row",
        on_select="rerun",
        use_container_width=True,
        key="search_table",
        hide_index=True
    )

    st.page_link("https://hpo-annotation-qc.readthedocs.io/en/latest/annotationFormat.html", label="HPOA Format", icon="ℹ️")

    if selection_event.selection:
        selected_indices = selection_event.selection["rows"]
        selected_rows = filtered_df.iloc[selected_indices]

        if st.button("Edit selected rows"):
            st.session_state.editing_rows = selected_rows.copy()
    
        
    if "editing_rows" in st.session_state:
        st.subheader("Edit HPOA")
        edited = st.data_editor(st.session_state.editing_rows, num_rows="dynamic")

# st.sidebar.write(return_select)
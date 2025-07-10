# app.py

# Components
# 1. Hierarchy browser of both MONDO and HPO
# 2. HPOA builder
# 3. Chatbot interface that can query pubmed for articles, return their HPs and auto-populate HPOA file
import numpy as np
import pandas as pd
import json
import hpotk
from hpotk.annotations.load.hpoa import SimpleHpoaDiseaseLoader
import streamlit as st
from st_ant_tree import st_ant_tree
from st_btn_group import st_btn_group

# Build HPO Tree Recursively
def build_tree(hpo, term_id="HP:0000001", path=""):
    term = hpo.get_term(term_id)
    current_path = f"{path}/{term_id}" if path else term_id  # make value path-unique

    children = hpo.graph.get_children(term) or []
    children_nodes = []
    for child in children:
        node = build_tree(hpo, child.value, current_path)
        if node:
            children_nodes.append(node)

    return {
        "label": f"{term.identifier} | {term.name}",
        "value": current_path,  # unique per path (allows multiple parenthood)
        "children": children_nodes
    }

## Pretty tree viewer
st.set_page_config(layout="wide")
st.title("HPO Hierarchy Viewer")

@st.cache_resource()
def load_minimal_hpo():
    import hpotk
    store = hpotk.configure_ontology_store()
    return store.load_minimal_hpo()

hpo = load_minimal_hpo()
tree = [ build_tree(hpo) ] # must be a list

return_select = st_ant_tree(
    treeData=tree,
    treeCheckable=True
)
st.write(return_select)
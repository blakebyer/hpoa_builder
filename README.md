# HPOA Builder
An interactive **Streamlit app** for browsing diseases, editing HPO annotations, and extending them with LLM-based PubMed suggestions. This is designed for curators who need a clean workflow for validating **Human Phenotype Ontology Annotation (HPOA)** tables.

## Features ##
- **Search and filter diseases** 
    
    Quickly find diseases using multiselect or search.
- **Inline editing**
    
    Select diseases and edit their annotations directly in a spreadsheet-like table with no need to touch the original dataframe.
- **Session-safe copy**
    
    All edits are saved to a copied dataframe, so your master data remains untouched.
- **PubMed + LLM Agent**
    
    Coming soon, with help from Aurelian.
- **Monarch-style UI**

    Custom colors and UI elements styled to match the [Monarch Initiative](https://monarchinitiative.org)

## Quick Start ##
Clone and install:

    git clone https://github.com/blakebyer/hpoa_builder.git
    cd src/
    pip install requirements.txt

Run the app:

    streamlit run app.py

## Screenshot ##
![HPOA Builder App](/images/image.png)
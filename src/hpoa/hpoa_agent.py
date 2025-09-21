import os
import requests
import json
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext, Tool
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, Literal, List
from oaklib import get_adapter
from oaklib.datamodels.search import SearchConfiguration

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMIM_API_KEY = os.getenv("OMIM_API_KEY")

# Load Ontologies
hpo = get_adapter("ontobee:hp")
mondo = get_adapter("ontobee:mondo")

HPOA_AGENT_PROMPT = (
    """
    You are an expert biocurator familiar with the OBO Ontologies including the Human Phenotype Ontology (HP) and Monarch Disease Ontology (MONDO).
    Your task is to assist the user in creation of Human Phenotype Ontology Annotation (.hpoa aka HPOA) files, a schema used to describe phenotypes annotated to a disease, given input text, a list of PubMed IDs (PMID), or an existing .hpoa file requiring curation assistance. You should use the `search_hp` function to curate `hpo_id`, `onset`, or `frequency` fields. Only include `onset` or `frequency` if either show up in the input text. You should use the `search_mondo` and `get_omim_terms` functions to find `database_id` (MONDO:RefId OR OMIM:mimNumber) and `disease_name` (MONDO:Label OR OMIM:preferredTitle). For the `reference` field use the OMIM:mimNumber. You should show your reasoning, and your candidate changes to the existing .hpoa file (as many as appropriate). IMPORTANT: precision is paramount. If a user requests help to improve an .hpoa for a given disease, only return phenotypic abnormalities and corresponding evidence associated with that SAME disease. Before doing this, try varying search terms of HPO, MONDO, and OMIM using your context to ensure its the same disease (synonyms are okay). You must NEVER guess ontology or database terms, the query results should always be the source of truth. Additionally, only fill out fields if there is sufficient evidence in the text for asserting that association. Even if multiple phenotypes apply to the same disease, return each annotation as a separate object in the list. Return a JSON array of as many HPOA annotation rows as appropriate. For instance:
    [
    {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": null,
        "hpo_id": "HP:0000963",
        "reference": "OMIM:301500",
        "evidence": "IEA",
        "onset": null,
        "frequency": "HP:0040282"
        "sex": null,
        "modifier": null,
        "aspect": "P",
        "biocuration": "HPO:You[2025-07-11]"
    },
    {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": null,
        "hpo_id": "HP:0004322",
        "reference": "PMID:39292930",
        "evidence": "PCS",
        "onset": null,
        "frequency": "HP:0040282"
        "sex": null,
        "modifier": null,
        "aspect": "P",
        "biocuration": "HPO:You[2025-07-11]"
    }
    ]

    """
)

# written using ontology-access-kit docs
def search_hp(label: str) -> List[dict]:
    """Search the HPO for phenotypic abnormalities, qualifiers, or frequencies."""
    results = list(hpo.basic_search(label, SearchConfiguration(is_partial=True)))
    data = []
    for curie in results:
        if not curie.startswith("HP:"):
            continue
        data.append({
                "id": curie,
                "label": hpo.label(curie),
                "definition": hpo.definition(curie),
            })
    return data

HUMAN_DISEASE_ROOT = "MONDO:0700096"

def is_human_disease(curie: str) -> bool:
    ancestors = set(mondo.ancestors(curie))
    return HUMAN_DISEASE_ROOT in ancestors

def search_mondo(label: str) -> List[dict]:
    """Search the MONDO Ontology for disease identifiers."""
    results = list(mondo.basic_search(label, SearchConfiguration(is_partial=True)))
    data = []
    for curie in results:
        if not is_human_disease(curie):
            continue
        data.append({
            "id" : curie,
            "label" : mondo.label(curie),
            "definition": mondo.definition(curie),
        })
    return data

def get_omim_terms(label: str):
    """Search the OMIM DB for disease identifiers."""
    url = f"https://api.omim.org/api/entry/search"
    params = {
        "search": label,
        "format": "json",
        "apiKey": OMIM_API_KEY,
    }
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(url, params=params, headers=headers)
    return response.json()

def get_omim_clinical(label: str):
    """Search the OMIM DB for clinical synopses which contain other clinical DB terms e.g. SNOMED:399020009, HP:0001256, ICD10CM:I42.0"""
    url = f"https://api.omim.org/api/entry/search"
    params = {
        "search": label,
        "format": "json",
        "include": "clinicalSynopsis",
        "apiKey": OMIM_API_KEY,
    }
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(url, params=params, headers=headers)
    return response.json()

class HPOA(BaseModel):
    database_id: str = Field(..., description="Refers to the database `disease_name` is drawn from. Must be formatted as a CURIE, e.g., OMIM:1547800 or MONDO:0021190")
    disease_name: str = Field(..., description="This is the name of the disease associated with the `database_id` in the database. Only the accepted name should be used, synonyms should not be listed here.")	
    qualifier: Optional[Literal["", "NOT"]] = Field(..., description="""This field is used to qualify the annotation shown in field `hpo_id`. The field can only be used to record `NOT` or is empty. A value of NOT indicates that the disease in question is not characterized by the indicated HPO term. This is used to record phenotypic features that can be of special differential diagnostic utility.""")
    hpo_id: str = Field(..., description="This field is for the HPO identifier for the term attributed to the `disease_name`.")
    reference: str = Field(..., description="""This field indicates the source of the information used for the annotation. This may be the clinical experience of the annotator, an article as indicated by a PMID, or an HPO collaborator ID, e.g. HPO:RefId. If a PMID cannot be found, default back to OMIM:mimNumber.""")	
    evidence: Literal["IEA", "PCS", "TAS"] = Field(..., description="""IEA (inferred from electronic annotation): annotations extracted from OMIM.
                                                   PCS (published clinical study): annotations extracted from articles in the medical literature.
                                                   TAS (traceable author statement): annotations extracted from knowledge bases such as OMIM or Orphanet that have derived the information from a published source..""")
    onset: Optional[str] = Field(..., description="""A term-id from the HPO-sub-ontology below the term `Age of onset` (HP:0003674). Note that if an HPO onset term is used in this field, it refers to the onset of the feature specified in field `hpo_id` in the disease being annotated. If an HPO term is used for age of onset in field `hpo_id` then it refers to the overall age of onset of the disease.""")
    frequency: Optional[str] = Field(..., description="""There are three allowed options for this field. (A) A term-id from the HPO-sub-ontology below the term `Frequency` (HP:0040279), (B) A count of patients affected within a cohort. For instance, 7/13 would indicate 7 of 13 patients with the disease in the `reference` field study were affected by the phenotype in the `hpo_id` field, and (C) A percentage value such as 17%.""")	
    sex: Optional[Literal["MALE", "FEMALE"]] = Field(..., description="""This field contains the strings MALE or FEMALE if the annotation in question is limited to males or females. This field refers to the phenotypic (and not the chromosomal) sex. If a phenotype is limited to one sex then a modifier from the clinical modifier subontology should be noted in the modifier field.""")	
    modifier: Optional[str]	= Field(..., description="A term-id from the HPO-sub-ontology below the term `Clinical modifier`.")
    aspect: Literal["P", "I", "C", "M"] = Field(..., description="""Terms with the P aspect are located in the Phenotypic abnormality subontology.
                              Terms with the I aspect are from the Inheritance subontology.
                              Terms with the C aspect are located in the Clinical course subontology, which includes onset, mortality, and other terms related to the temporal aspects of disease.
                              Terms with the M aspect are located in the Clinical Modifier subontology.""")	
    biocuration: str = Field(..., description="""This refers to the biocurator who made the annotation and the date on which the annotation was made; the date format is YYYY-MM-DD. The first entry in this field refers to the creation date. Any additional biocuration is recorded following a semicolon. So, if Joseph curated on July 5, 2012, and Suzanna curated on December 7, 2015, one might have a field like this: HPO:Joseph[2012-07-05];HPO:Suzanna[2015-12-07]. It is acceptable to use ORCID ids.""")

hpoa_agent = Agent(
    model="openai:gpt-4.1",
    output_type=List[HPOA],
    system_prompt=HPOA_AGENT_PROMPT,
    tools=[search_hp, search_mondo, get_omim_terms],
)

result = hpoa_agent.run_sync("""As opposed to the extensive somatic involvement seen in MPS I, II and VII, all forms of MPS III present with cognitive and neurological impairment with little or no somatic involvement. This disorder may be recognized in childhood by developmental delays, behavioural difficulties, sleep disturbances and dementia. The mental retardation can be profound in patients with severe disease, with a lack of development of social or communicative skills in early childhood. Such patients eventually enter a vegetative state and generally only live into their second or third decade. Some individual patients with MPS III show only mild-to-moderate developmental delays and behavioural problems. It is quite likely that many mildly affected MPS III patients are not recognized in clinical practice.

Both forms of MPS IV are characterized by a skeletal dysplasia, ligamentous laxity/joint hypermobility, odontoid hypoplasia and short stature, without cognitive impairment. Of interest to the rheumatologist, the skeletal dysplasia is distinct from the dysostosis multiplex seen in MPS I, II and VII. The ligamentous laxity/joint hypermobility associated with MPS IV is also unique among the MPS disorders, since the other disorders with joint involvement present with stiffness and decreased mobility. Neurological involvement, such as cervical spine instability and communicating hydrocephalus, is common in MPS IV and can be life threatening. Patients with severe MPS IV may live into their second or third decade, and those with attenuated disease may live much longer.""")
print(result.output)
import os
import time

from llmware.library import Library
from llmware.retrieval import Query
from llmware.models import ModelCatalog, PromptCatalog
from llmware.prompts import Prompt
from llmware.configs import LLMWareConfig

from huggingface_hub import login

import pandas as pd


def register_gguf_model():

    prompter = Prompt()

    your_model_name = "llama3"
    hf_repo_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_file = "./Meta-Llama-3-8B-Instruct"

    ModelCatalog().register_new_hf_generative_model(hf_repo_name)
    prompter = Prompt().load_model(hf_repo_name)
    print(prompter.llm_name)



def add_custom_embed_model(model_name:str, embedding_dim:int=384, context_window:int=256) -> str:
    ModelCatalog().register_sentence_transformer_model(model_name=model_name,
                                                   embedding_dims=embedding_dim,
                                                   context_window=context_window)



def semantic_rag (library_name: str, embedding_model_name:str, llm_model_name:str, query:str) -> str:

    """ Illustrates the use of semantic embedding vectors in a RAG workflow
        --self-contained example - will be duplicative with some of the steps taken in other examples """
    
    is_embed_model_new = ModelCatalog().lookup_model_card(embedding_model_name)
    if not is_embed_model_new:
        add_custom_embed_model(embedding_model_name)

    library = Library().create_new_library(library_name)
    library.add_files(input_folder_path="C:/Users/ASAGUN/Desktop/Training/open-source-llm/kbs")
    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db)

    ModelCatalog().register_new_hf_generative_model(llm_model_name)

    # Run Order List - How to construct the prompt
    run_order_list = ["blurb1", "$context", "blurb2", "$query", "instruction"]

    # Dictionary to use for the prompt
    my_prompt_dict = {"blurb1": "Please use the following materials- ",
                      "blurb2": "Please answer the following question - ",
                      "instruction": "Regardless of the context, you will only answer 'Got it!' ",
                      "system_message": "You are assistant that will only answer 'Got it!' "}

    
    # Add the new custom prompt
    prompt_catalog = PromptCatalog()
    prompt_catalog.add_custom_prompt_card("my_prompt", run_order_list, my_prompt_dict)

   

    #   key step: run semantic query against the library and get all of the top results
    results = Query(library).semantic_query(query, result_count=50, embedding_distance_threshold=1.0)

    # RAG steps start here ...
    prompter = Prompt(save_state=True,prompt_catalog=prompt_catalog).load_model(llm_model_name)

    source = prompter.add_source_query_results(query_results=results)
    is_source_attached = prompter.verify_source_materials_attached()
    print(f"Is source has attachement: {is_source_attached}")

    llm_response = prompter.prompt_with_source(query, prompt_name='explain_child', temperature=0.3, max_output=None)

    prompter.clear_source_materials()
    library.delete_library(confirm_delete=True)
    return llm_response[0]["llm_response"]
    # return results
    
    

if __name__ == '__main__':

    # login(token = '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')

    LLMWareConfig().set_active_db("sqlite")
    vector_db = "faiss"
    embedding_models = 'all-MiniLM-L6-v2'
    lib_name = "data_privacy_lib2"
    rag_models = "TheBloke/Llama-2-7B-Chat-GGUF"

    result = semantic_rag(lib_name, embedding_models, rag_models, "WILL SSS APPROVE THE DAYS MY PHYSICIAN INDICATED AT MY MEDICAL CERTIFICATE?")
    print(result)

    
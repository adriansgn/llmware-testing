import os
import time

from llmware.library import Library
from llmware.retrieval import Query
from llmware.models import ModelCatalog
from llmware.prompts import Prompt
from llmware.configs import LLMWareConfig

from huggingface_hub import login

import pandas as pd



def add_custom_embed_model(model_name:str, embedding_dim:int=384, context_window:int=256) -> str:
    ModelCatalog().register_sentence_transformer_model(model_name=model_name,
                                                   embedding_dims=embedding_dim,
                                                   context_window=context_window)
    


def semantic_rag (library_name: str, llm_model_name:str, query:str) -> str:

    """ Illustrates the use of semantic embedding vectors in a RAG workflow
        --self-contained example - will be duplicative with some of the steps taken in other examples """
    
    library = Library().load_library(library_name)
    ModelCatalog().register_new_hf_generative_model(llm_model_name)

    #   key step: run semantic query against the library and get all of the top results
    results = Query(library).semantic_query(query, result_count=50, embedding_distance_threshold=1.0)

    # RAG steps start here ...
    prompter = Prompt().load_model(llm_model_name)

    source = prompter.add_source_query_results(query_results=results)
    is_source_attached = prompter.verify_source_materials_attached()
    print(f"Is source has attachement: {is_source_attached}")
    llm_response = prompter.prompt_with_source(query, prompt_name="explain_child", temperature=0.3, max_output=None)

    # prompter.clear_source_materials()
    return [llm_response[0]["llm_response"], is_source_attached]
    
def save_llm_response(rag_model: str, embedding_model: str) -> None:
    df: pd.DataFrame = pd.read_excel('testing\questions.xlsx', sheet_name='Questions', names=['Question', 'Expected Answer']).dropna()
    result_file_path = "./testing/automated_test.xlsx"


    for index, question in df.iterrows():
        start_time = time.time()
        result = semantic_rag(library_name, rag_model, question['Question'])
        llm_response = result[0]
        end_time = time.time()
        df.loc[index, f'LLM Response'] = llm_response
        df.loc[index, 'Execution Time'] = str(end_time - start_time)
        df.loc[index, 'Is Question Get Source'] = result[1]
        print(f"Question: {question['Question']}", f"LLM Response: {llm_response}")
        
    try:
        if not os.path.exists(result_file_path):
            df.to_excel(result_file_path, sheet_name=f"{rag_model.split('/')[-1]}_{embedding_model.replace('/','-')[:10]}", index=False)
        else:
            with pd.ExcelWriter(result_file_path, mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=f"{rag_model.split('/')[-1][:18]}_{embedding_model.replace('/','-')[:10]}", index=False)
    except Exception as e:
        print(e)
    
    print(f"{rag_model} {embedding_model} DONE!!!!")

def create_library(library_name: str, embedding_model: str) -> None:
    library = Library().create_new_library(library_name)
    library.add_files(input_folder_path="C:/Users/ASAGUN/Desktop/Training/open-source-llm/kbs")
    is_embed_model_new = ModelCatalog().lookup_model_card(embedding_model)
    if not is_embed_model_new:
        add_custom_embed_model(embedding_model)
    library.install_new_embedding(embedding_model_name=embedding_model, vector_db=vector_db)


if __name__ == "__main__":
    # login(token = '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')
    
    LLMWareConfig().set_active_db("sqlite")
    vector_db = "faiss"
    library_name = "docs_77"
    # embedding_models = ['multi-qa-MiniLM-L6-cos-v1', 'all-mpnet-base-v2', 'all-MiniLM-L6-v2']

    # llm_models = ["dragon-llama-answer-tool", "dragon-mistral-answer-tool", "llmware/bling-1.4b-0.1"] #"llmware/dragon-yi-6b-v0" "meta-llama/Meta-Llama-Guard-2-8B",

    embedding_models = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']
    llm_models = ["dragon-llama-answer-tool"]
    
    total_combi = len(embedding_models) * len(llm_models)
    progress = 0
    print('AUTOMATION START ....')

    for llm_model in llm_models:
        for embedding_model in embedding_models:
            create_library(library_name, embedding_model)
            save_llm_response(llm_model, embedding_model)
            progress += 1
            print(f"PROGRESS: {progress} / {total_combi} .....")

            library = Library().load_library(library_name)
            library.delete_library(confirm_delete=True)
    print("AUTOMATION COMPLETED!!!!")


# add_custom_embed_model('BAAI/bge-small-en-v1.5')
# model_card = ModelCatalog().lookup_model_card('BAAI/bge-small-en-v1.5')
# print(model_card)


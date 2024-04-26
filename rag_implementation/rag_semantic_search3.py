
"""     Fast Start Example #5 - RAG with Semantic Query

    This example illustrates the most common RAG retrieval pattern, which is using a semantic query, e.g.,
    a natural language query, as the basis for retrieving relevant text chunks, and then using as
    the context material in a prompt to ask the same question to a LLM.

    In this example, we will show the following:

    1.  Create library and install embeddings (feel free to skip / substitute a library created in an earlier step).
    2.  Ask a general semantic query to the entire library collection.
    3.  Select the most relevant results by document.
    4.  Loop through all of the documents - packaging the context and asking our questions to the LLM.

"""

import os
import time

from llmware.library import Library
from llmware.retrieval import Query
from llmware.prompts import Prompt
from llmware.configs import LLMWareConfig

import pandas as pd


def semantic_rag (library_name: str, embedding_model_name:str, llm_model_name:str, query:str) -> str:

    """ Illustrates the use of semantic embedding vectors in a RAG workflow
        --self-contained example - will be duplicative with some of the steps taken in other examples """

    library = Library().create_new_library(library_name)



    library.add_files(input_folder_path="C:/Users/ASAGUN/Desktop/Training/open-source-llm/kbs")
    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db)

    # RAG steps start here ...
    prompter = Prompt().load_model(llm_model_name)

    #   key step: run semantic query against the library and get all of the top results
    results = Query(library).semantic_query(query, result_count=50, embedding_distance_threshold=1.0)
    # closest_result = min(results, key=lambda d: d['distance'])
    # print(closest_result)

    
    for result in results:
        source = prompter.add_source_query_results(query_results=[result])
    llm_response = prompter.prompt_with_source(query, prompt_name="default_with_context", temperature=0.3, max_output=None)

    prompter.clear_source_materials()
    return llm_response[0]["llm_response"]
    
def save_llm_response( rag_model: str, embedding_model:str) -> None:
    df:pd.DataFrame = pd.read_excel('testing\questions_data_privacy.xlsx', sheet_name='Questions', names=['Question', 'Expected Answer']).dropna()
    result_file_path = "./testing/automated_test_data_priv.xlsx"

    total_items = len(df)
    processed_item = 0
    for index, question in df.iterrows():
        start_time = time.time()
        llm_response = semantic_rag(lib_name, embedding_model, rag_model, question['Question'])
        end_time = time.time()
        df.loc[index, f'LLM Response'] = llm_response
        df.loc[index, 'Execution Time'] = str(end_time - start_time)
        processed_item += 1
        print(f"Progress: {processed_item}/{total_items}")

    try:
        if not os.path.exists(result_file_path):
            df.to_excel(result_file_path, sheet_name=f"{rag_model.split('/')[-1]}_{embedding_model.replace('/','-')[:10]}", index=False)
        else:
            with pd.ExcelWriter(result_file_path, mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=f"{rag_model.split('/')[-1]}_{embedding_model.replace('/','-')[:10]}", index=False)
    except Exception as e:
        print(e)
    
    print(f"{rag_model} {embedding_model} DONE!!!!")


if __name__ == "__main__":

    LLMWareConfig().set_active_db("sqlite")
    vector_db = "faiss"
    embedding_models = ['mini-lm-sbert', 'all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5', 'jinaai/jina-embeddings-v2-small-en', 'thenlper/gte-small','WhereIsAI/UAE-Large-V1', 'llmrails/ember-v1']
    lib_name = "data_privacy_lib"
    rag_models = ["llmware/bling-1b-0.1", "llmware/bling-tiny-llama-v0", "llmware/dragon-yi-6b-gguf"] #"llmware/dragon-yi-6b-v0"

    total_combi = len(embedding_models) * len(rag_models)
    progress = 0
    print('AUTOMATION START ....')

    for rag_model in rag_models:
        for embedding_model in embedding_models:
            save_llm_response(rag_model, embedding_model)
            progress += 1
            print(f"PROGRESS: {progress} / {total_combi} .....")

    # from llmware.models import ModelCatalog
    # embedding_models = ModelCatalog().list_embedding_models()

    # for i, models in enumerate(embedding_models):
    #     print("embedding models: ", i, models)
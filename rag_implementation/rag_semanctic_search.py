
from llmware.library import Library
from llmware.retrieval import Query
from llmware.models import ModelCatalog, PromptCatalog
from llmware.prompts import Prompt
from llmware.configs import LLMWareConfig






def add_custom_embed_model(model_name:str, embedding_dim:int=384, context_window:int=256) -> str:
    ModelCatalog().register_sentence_transformer_model(model_name=model_name,
                                                   embedding_dims=embedding_dim,
                                                   context_window=context_window)



def semantic_rag (library: Library, llm_model: str, query:str) -> str:

    """ 
    Illustrates the use of semantic embedding vectors in a RAG workflow 
    > Make sure LLL model already installed in ollama
    > ollama server must be running
    """

    
    # Run semantic query against the library and get all of the top results
    results = Query(library).semantic_query(query, result_count=3, embedding_distance_threshold=1.0)


    # Construct custom prompt
    run_order_list = ["blurb1", "$context", "blurb2", "$query", "instruction"]
    instruction = (
                   "In answering the question, please only use information contained in the provided materials." 
                   "Do not add unnecessary phrases." 
                   "Do not use the words 'According to the material', 'Based on the provided material'." 
                   "Do not include the document section where you got the answer." 
                   "Answer the question like the answer is coming from you." 
                   "Answer all question in a friendly manner. Do not ask follow-up question to user."
                )

    my_prompt_dict = {"blurb1": "Please use the following materials- ",
                      "blurb2": "Please answer the following question - ",
                      "instruction": instruction,
                      "system_message": "You are a helpful HR assistant in 77GSI company. All materials fed into you are coming from 77GSI company documents. "}

    # Add the new custom prompt
    prompt_catalog = PromptCatalog()
    prompt_catalog.add_custom_prompt_card("custom_prompt", run_order_list, my_prompt_dict)

    # Register ollama model
    ModelCatalog().register_ollama_model(model_name=llm_model,model_type="chat",temperature=0.5, host="localhost",port=11434)
    prompter = Prompt().load_model(llm_model)

    prompter.add_source_query_results(query_results=results)
    is_source_attached = prompter.verify_source_materials_attached()
    print(f"Is source has attachement: {is_source_attached}") # if False, meaning the query does not have sematic similarity in semantic search results

    llm_response = prompter.prompt_with_source(query, prompt_name='custom_prompt', temperature=0.3, max_output=None)

    return llm_response[0]["llm_response"]

    
    

if __name__ == '__main__':

    LLMWareConfig().set_active_db("sqlite")
    vector_db = "faiss"
    lib_name = "seven_seven_library"
    embedding_model = "multi-qa-MiniLM-L6-cos-v1"
    llm_model = "llama3"

    add_custom_embed_model(embedding_model)

    if not Library().check_if_library_exists(library_name=lib_name):
        library = Library().create_new_library(lib_name)
        library.add_files(input_folder_path="./kbs")
        library.install_new_embedding(embedding_model_name=embedding_model, vector_db=vector_db)
    else:
        library = Library().load_library(lib_name)

    query = "How to apply loan in SSS?"

    result = semantic_rag(library, llm_model, query)
    print(result)

    
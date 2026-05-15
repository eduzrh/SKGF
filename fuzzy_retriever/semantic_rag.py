import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import numpy as np


def load_ents(path):
    """
    Load entity file
    Args:
        path: Path to the entity file
    Returns:
        data: Dictionary of entities
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            data[line[0]] = line[1]
    print(f'load {path} {len(data)}')
    return data


def retrieve_top_k_entities(query, retriever, k=10):
    """
    Use FAISS to retrieve TOP-K entities for given query
    Args:
        query: Query entity name
        retriever: Retriever instance for searching
        k: Number of candidate entities to return
    Returns:
        top_k_answers: TOP-K most relevant entities
    """
    query = query + ", what information may be potentially related to this incident?"
    answers = retriever.invoke(query)

    answers_all = {}
    for doc in answers:
        doc1 = doc.page_content.strip().split('\t')
        answers_all[doc1[0]] = doc1[1].replace(' ', '')

    top_k_answers = sorted(answers_all.items(), key=lambda item: item[1], reverse=True)[:k]
    return top_k_answers


def setup_retriever(api_base, api_key, retriever_document_path, faiss_index_path):
    """
    Setup and configure the retriever
    Args:
        api_base: OpenAI API base URL
        api_key: OpenAI API key
        retriever_document_path: Path to the retriever document
        faiss_index_path: Path to save/load FAISS index
    Returns:
        retriever: Configured retriever instance
    """
    # Configure OpenAI API
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_API_KEY"] = api_key

    # Initialize OpenAI embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load FAISS vector store
    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"score_threshold": 0.01})

    return retriever


def init_process(api_base, api_key, retriever_document_path, faiss_index_path):
    """
    Initialize process with a retriever
    This function will be called once per worker process
    """
    global process_retriever
    process_retriever = setup_retriever(api_base, api_key, retriever_document_path, faiss_index_path)


def process_entity_batch(batch, top_k=5):
    """
    Process a batch of entities using the process-local retriever
    Args:
        batch: List of (entity_id, entity_name) tuples
        top_k: Number of top entities to retrieve
    Returns:
        outputs: List of retrieval results
    """
    global process_retriever
    outputs = []
    for ent_id, ent_name in batch:
        try:
            top_k_answers = retrieve_top_k_entities(ent_name, process_retriever, k=top_k)
            for idx, (top_answer_id, top_answer_name) in enumerate(top_k_answers):
                outputs.append(f"{ent_id}\t{top_answer_id}\n")
        except Exception as e:
            print(f"Error with entity {ent_name}: {str(e)}")
    return outputs


def prepare_faiss_index(retriever_document_path, faiss_index_path):
    """
    Prepare FAISS index if it doesn't exist
    """
    if not os.path.exists(faiss_index_path):
        # Load documents
        loader = TextLoader(retriever_document_path)
        raw_documents = loader.load()

        # Initialize OpenAI embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create FAISS index
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        print('start embeddings', len(documents))
        db = FAISS.from_documents(documents, embeddings)
        print('end embeddings')
        db.save_local(faiss_index_path)


def process_entities_parallel(config):
    """
    Process entities in parallel and generate retriever outputs
    Args:
        config: Dictionary containing all configuration parameters
    """
    start_time = time.time()
    os.makedirs(os.path.dirname(config['retriever_output_file']), exist_ok=True)

    # Prepare FAISS index if needed
    prepare_faiss_index(config['retriever_document_path'], config['faiss_index'])

    # Load entities
    ents_1 = load_ents(config['ents_path_1'])
    ents_2 = load_ents(config['ents_path_2'])

    # Create batches
    entity_items = list(ents_1.items())
    num_batches = (len(entity_items) + config['batch_size'] - 1) // config['batch_size']
    batches = np.array_split(entity_items, num_batches)

    # Set up multiprocessing
    if config['num_processes'] is None:
        config['num_processes'] = mp.cpu_count() - 1

    # Initialize pool with retriever setup
    pool = mp.Pool(
        processes=config['num_processes'],
        initializer=init_process,
        initargs=(
            config['api_base'],
            config['api_key'],
            config['retriever_document_path'],
            config['faiss_index']
        )
    )

    process_batch_partial = partial(process_entity_batch, top_k=config['top_k'])

    # Process batches in parallel with progress bar
    outputs = []
    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch_results in pool.imap(process_batch_partial, batches):
            outputs.extend(batch_results)
            # Write batch results to file immediately
            with open(config['retriever_output_file'], 'a+') as file:
                file.writelines(batch_results)
            pbar.update(1)

    pool.close()
    pool.join()

    end_time = time.time()
    print(f"Parallel Retriever Execution time: {end_time - start_time:.2f} seconds")


def semantic_rag_all(data_dir):
    """Main RAG retrieval function"""
    S1_PRIVATE_MESSAGE_POOL = {
        'top_k_candidate_entities': os.path.join(data_dir, "message_pool", "retriever_outputs.txt"),
    }

    config_top_k = 5

    config = {
        'api_base': 'yours',
        'api_key': 'yours',
        'retriever_document_path': data_dir + "/message_pool/line_triples_name_2",
        'faiss_index': data_dir + "/index/faiss_index",
        'retriever_output_file': S1_PRIVATE_MESSAGE_POOL['top_k_candidate_entities'],
        'ents_path_1': data_dir + '/message_pool/line_triples_name_1',
        'ents_path_2': data_dir + '/message_pool/line_triples_name_2',
        'top_k': config_top_k,
        'batch_size': 10,
        'num_processes': None,
    }

    # Clear output file if it exists
    if os.path.exists(config['retriever_output_file']):
        os.remove(config['retriever_output_file'])

    # Process entities in parallel
    process_entities_parallel(config)


if __name__ == "__main__":
    data_dir = "/home/dex/Desktop/entity_sy/AdaCoAgent_backup/data/icews_yago/"
    semantic_rag_all(data_dir)

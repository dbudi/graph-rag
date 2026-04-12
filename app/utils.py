import hashlib

def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_doc_id_from_source(source: str):
    return hash_text(source)

def get_chunk_id(doc_id, chunk_index):
    return f"{doc_id}_{chunk_index}"
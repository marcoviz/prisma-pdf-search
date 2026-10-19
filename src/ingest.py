"""Indexa o PDF com IDs estáveis e chunks de 1000/150 caracteres."""
import sys
import time
from hashlib import sha256
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Settings, vector_store

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, add_start_index=True,
    )
    return [chunk for chunk in splitter.split_documents(pages) if chunk.page_content.strip()]

def ingest_pdf(settings=None, store=None):
    settings = settings or Settings.load()
    chunks = split_pages(PyPDFLoader(str(settings.pdf_path)).load())
    if not chunks:
        raise ValueError('PDF sem texto extraível. Este projeto não executa OCR.')
    digest = settings.fingerprint()
    ids = []
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(document_sha256=digest, chunk_index=index, source=settings.pdf_path.name)
        ids.append(sha256(f'{settings.collection}:{digest}:{index}'.encode()).hexdigest())
    store = store if store is not None else vector_store(settings)
    batch_size = settings.ingest_batch_size
    for offset in range(0, len(chunks), batch_size):
        if offset and settings.ingest_batch_delay_seconds:
            time.sleep(settings.ingest_batch_delay_seconds)
        store.add_documents(chunks[offset:offset + batch_size], ids=ids[offset:offset + batch_size])
    return len(chunks)

def main():
    try:
        print(f'Ingestão concluída: {ingest_pdf()} chunks indexados.')
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Falha na ingestão ({type(exc).__name__}). Confira banco, chave e modelos.', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())

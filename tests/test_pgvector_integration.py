"""Persistência real; vetores determinísticos são apenas um dublê de teste."""
import hashlib
import os
import uuid
from pathlib import Path

import psycopg
import pytest
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from config import Settings
from ingest import ingest_pdf


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [float(value + 1) for value in digest[:8]]


@pytest.mark.skipif(not os.getenv('TEST_DATABASE_URL'), reason='Requer PostgreSQL de teste')
def test_real_vector_persistence_is_idempotent_and_filters_document():
    url = os.environ['TEST_DATABASE_URL']
    collection = 'prisma_test_' + uuid.uuid4().hex
    settings = Settings(url, collection, Path(__file__).parents[1] / 'document.pdf', 'test-only', 'test-only')
    store = PGVector(embeddings=LocalEmbeddings(), connection=url, collection_name=collection, use_jsonb=True)
    try:
        count = ingest_pdf(settings, store)
        assert count > 0
        assert ingest_pdf(settings, store) == count
        with psycopg.connect(url.replace('postgresql+psycopg://', 'postgresql://', 1)) as connection:
            actual = connection.execute(
                'SELECT count(*) FROM langchain_pg_embedding e JOIN langchain_pg_collection c '
                'ON c.uuid = e.collection_id WHERE c.name = %s', (collection,)
            ).fetchone()[0]
        assert actual == count
        hits = store.similarity_search_with_score('faturamento', k=10, filter={'document_sha256': settings.fingerprint()})
        assert len(hits) == min(10, count)
        assert all(doc.metadata['document_sha256'] == settings.fingerprint() for doc, _ in hits)
        assert store.similarity_search_with_score('faturamento', k=10, filter={'document_sha256': 'outro-pdf'}) == []
    finally:
        store.delete_collection()

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ingest import ingest_pdf, split_pages
from search import create_answerer, FALLBACK
from config import Settings

def test_splitter_keeps_exact_limits_and_page_metadata():
    chunks = split_pages([Document(page_content='A' * 2150, metadata={'page': 2})])
    assert [len(x.page_content) for x in chunks] == [1000, 1000, 450]
    assert [x.metadata['start_index'] for x in chunks] == [0, 850, 1700]
    assert all(x.metadata['page'] == 2 for x in chunks)

def test_ingestion_real_pdf_is_idempotent_by_content():
    settings = Settings('unused', 'unit_collection', ROOT / 'document.pdf', 'embedding', 'chat')
    store = MagicMock()
    count = ingest_pdf(settings, store)
    first_ids = store.add_documents.call_args.kwargs['ids']
    assert count > 0
    assert len(first_ids) == len(set(first_ids))
    ingest_pdf(settings, store)
    assert first_ids == store.add_documents.call_args.kwargs['ids']
    assert all(len(d.page_content) <= 1000 for d in store.add_documents.call_args.args[0])

def test_search_requests_ten_only_for_selected_document():
    store, model = MagicMock(), MagicMock()
    store.similarity_search_with_score.return_value = [(Document(page_content='Receita: 10 milhões.'), 0.1)]
    model.invoke.return_value = AIMessage(content='10 milhões.')
    assert create_answerer(store, model, 'fingerprint')('Receita?') == '10 milhões.'
    store.similarity_search_with_score.assert_called_once_with('Receita?', k=10, filter={'document_sha256': 'fingerprint'})
    assert 'Receita: 10 milhões.' in model.invoke.call_args.args[0][-1].content

def test_empty_index_refuses_without_calling_model():
    store, model = MagicMock(), MagicMock()
    store.similarity_search_with_score.return_value = []
    assert create_answerer(store, model, 'hash')('Qual é a capital?') == FALLBACK
    model.invoke.assert_not_called()

def test_empty_question_is_rejected():
    store, model = MagicMock(), MagicMock()
    with pytest.raises(ValueError):
        create_answerer(store, model, 'hash')('  ')
    store.similarity_search_with_score.assert_not_called()

def test_ingestion_batches_all_documents_and_waits_between_batches(monkeypatch):
    import ingest
    settings = Settings('unused', 'paced_collection', ROOT / 'document.pdf', 'embedding', 'chat', 16, 30)
    store, sleep = MagicMock(), MagicMock()
    monkeypatch.setattr(ingest.time, 'sleep', sleep)
    count = ingest_pdf(settings, store)
    calls = store.add_documents.call_args_list
    all_ids = [value for call in calls for value in call.kwargs['ids']]
    assert len(calls) == (count + 15) // 16
    assert all(1 <= len(call.args[0]) <= 16 for call in calls)
    assert len(all_ids) == len(set(all_ids)) == count
    assert sleep.call_count == len(calls) - 1
    assert all(call.args == (30,) for call in sleep.call_args_list)

@pytest.mark.parametrize(('name', 'value'), [
    ('INGEST_BATCH_SIZE', '0'), ('INGEST_BATCH_SIZE', '65'), ('INGEST_BATCH_SIZE', 'not-a-number'),
    ('INGEST_BATCH_DELAY_SECONDS', '-1'), ('INGEST_BATCH_DELAY_SECONDS', '301'),
    ('INGEST_BATCH_DELAY_SECONDS', 'nan'), ('INGEST_BATCH_DELAY_SECONDS', 'inf'),
])
def test_settings_rejects_invalid_ingestion_rate_configuration(monkeypatch, name, value):
    for key, val in {
        'GOOGLE_API_KEY': 'unit-test-only',
        'DATABASE_URL': 'postgresql+psycopg://postgres:postgres@localhost:5432/rag',
        'GOOGLE_EMBEDDING_MODEL': 'embedding', 'GOOGLE_CHAT_MODEL': 'chat',
        'PDF_PATH': str(ROOT / 'document.pdf'),
        'INGEST_BATCH_SIZE': '16', 'INGEST_BATCH_DELAY_SECONDS': '30',
    }.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match='INGEST_BATCH'):
        Settings.load()

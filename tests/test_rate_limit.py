"""429 transitório, cota esgotada e falhas não recuperáveis sem chamadas pagas."""
from unittest.mock import MagicMock

import pytest
from google.genai.errors import ClientError
from langchain_google_genai._common import GoogleGenerativeAIError

from ingest import add_batch, is_rate_limit


def wrapped_error(code):
    try:
        raise ClientError(code, {'error': {'code': code, 'message': 'test-only'}})
    except ClientError as cause:
        try:
            raise GoogleGenerativeAIError('embedding test failure') from cause
        except GoogleGenerativeAIError as error:
            return error


def test_retries_wrapped_gemini_429_with_same_ids_and_bounded_backoff(monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr('ingest.time.sleep', sleep)
    store = MagicMock()
    store.add_documents.side_effect = [wrapped_error(429), wrapped_error(429), None]
    chunks, ids = ['first', 'second'], ['stable-1', 'stable-2']
    add_batch(store, chunks, ids)
    assert store.add_documents.call_count == 3
    assert all(call.args == (chunks,) and call.kwargs == {'ids': ids}
               for call in store.add_documents.call_args_list)
    assert [call.args[0] for call in sleep.call_args_list] == [30, 60]


def test_exhausted_quota_stops_after_three_retries(monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr('ingest.time.sleep', sleep)
    error = wrapped_error(429)
    store = MagicMock()
    store.add_documents.side_effect = error
    with pytest.raises(GoogleGenerativeAIError) as caught:
        add_batch(store, ['document'], ['stable-id'])
    assert caught.value is error
    assert store.add_documents.call_count == 4
    assert [call.args[0] for call in sleep.call_args_list] == [30, 60, 120]


@pytest.mark.parametrize('error', [wrapped_error(400), wrapped_error(403), RuntimeError('database unavailable')])
def test_does_not_retry_authentication_invalid_input_or_database_failure(monkeypatch, error):
    sleep = MagicMock()
    monkeypatch.setattr('ingest.time.sleep', sleep)
    store = MagicMock()
    store.add_documents.side_effect = error
    with pytest.raises(type(error)):
        add_batch(store, ['document'], ['id'])
    store.add_documents.assert_called_once()
    sleep.assert_not_called()


def test_exception_message_alone_cannot_trigger_retry():
    assert not is_rate_limit(RuntimeError('invalid data with 429 in its text'))
    error = RuntimeError('cycle')
    error.__cause__ = error
    assert not is_rate_limit(error)

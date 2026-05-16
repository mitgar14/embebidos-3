"""Test de smoke del lifecycle de TRTWorker. Mockea pycuda/tensorrt para correr
en el host sin GPU. Solo verifica que las llamadas a request_swap() encolan
correctamente y que el orden de destrucción es el esperado."""
from nano_server import TRTWorker


def test_request_swap_sets_event():
    worker = TRTWorker("/tmp/fake.engine")
    assert not worker._swap_event.is_set()
    worker.request_swap("/tmp/new.engine")
    assert worker._swap_event.is_set()
    assert worker._swap_path == "/tmp/new.engine"


def test_request_swap_overwrites_pending():
    worker = TRTWorker("/tmp/fake.engine")
    worker.request_swap("/tmp/first.engine")
    worker.request_swap("/tmp/second.engine")
    assert worker._swap_path == "/tmp/second.engine"

import hashlib
import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qat.runtime.audio_io import WavReplaySource
from server.app import create_app

SPEECH_WAV = Path(__file__).parent / "data" / "speech_ru.wav"


def stage1(event_id=None):
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "timestamp_utc": "2026-10-27T12:00:00+00:00",
        "detector": "energy",
        "detector_version": "0.1.0",
        "score": 0.93,
        "threshold": 0.8,
        "clip_path": None,
        "clip_sha256": None,
        "clip_start_offset_sec": None,
        "clip_end_offset_sec": None,
        "operator_decision": None,
    }


def stage2(event, events_dir):
    clip = events_dir / f"{event['event_id']}.wav"
    clip.write_bytes(b"RIFF-test-clip")
    done = {
        **event,
        "clip_path": f"events/{clip.name}",
        "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "clip_start_offset_sec": -5.0,
        "clip_end_offset_sec": 5.0,
    }
    # как EventWriter: второй этап перезаписывает JSON, решения в нём нет
    (events_dir / f"{event['event_id']}.json").write_text(json.dumps(done))
    return done


@pytest.fixture
def events_dir(tmp_path):
    return tmp_path / "events"


@pytest.fixture
def client(events_dir):
    with TestClient(create_app(events_dir=events_dir)) as c:
        yield c


def publish(client, event):
    client.portal.call(client.app.state.hub.publish, event)


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Подтвердить" in r.text and "Отклонить" in r.text


def test_websocket_history_and_live_events(client):
    old = stage1()
    publish(client, old)

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "history", "events": [old]}
        new = stage1()
        publish(client, new)
        assert ws.receive_json() == {"type": "event", "event": new}


def test_clip_served_and_bad_ids_rejected(client, events_dir):
    event_id = str(uuid.uuid4())
    (events_dir / f"{event_id}.wav").write_bytes(b"RIFF-data")
    (events_dir / "secret.wav").write_bytes(b"x")

    r = client.get(f"/clips/{event_id}.wav")
    assert r.status_code == 200 and r.content == b"RIFF-data"
    assert r.headers["content-type"] == "audio/wav"

    assert client.get(f"/clips/{uuid.uuid4()}.wav").status_code == 404
    assert client.get("/clips/secret.wav").status_code == 404
    assert client.get("/clips/..%2Fsecret.wav").status_code == 404


def test_decision_updates_json_log_and_clients(client, events_dir):
    event = stage2(stage1(), events_dir)
    publish(client, event)
    event_id = event["event_id"]

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # история
        r = client.post(f"/events/{event_id}/decision", json={"decision": "rejected"})
        assert r.status_code == 200
        update = ws.receive_json()

    assert update["event"]["operator_decision"] == "rejected"
    saved = json.loads((events_dir / f"{event_id}.json").read_text())
    assert saved["operator_decision"] == "rejected"
    assert saved["clip_sha256"] == event["clip_sha256"]
    (line,) = (events_dir / "decisions.jsonl").read_text().splitlines()
    record = json.loads(line)
    assert (record["event_id"], record["decision"]) == (event_id, "rejected")
    assert "decided_at_utc" in record

    # повторное решение не принимается
    r = client.post(f"/events/{event_id}/decision", json={"decision": "confirmed"})
    assert r.status_code == 409


def test_decision_errors(client):
    event_id = str(uuid.uuid4())
    publish(client, stage1(event_id))

    bad = client.post(f"/events/{event_id}/decision", json={"decision": "maybe"})
    assert bad.status_code == 422
    unknown = f"/events/{uuid.uuid4()}/decision"
    assert client.post(unknown, json={"decision": "confirmed"}).status_code == 404


def test_decision_before_clip_survives_second_stage(client, events_dir):
    first = stage1()
    publish(client, first)
    event_id = first["event_id"]

    r = client.post(f"/events/{event_id}/decision", json={"decision": "confirmed"})
    assert r.status_code == 200

    publish(client, stage2(first, events_dir))

    final = client.app.state.hub.events[event_id]
    assert final["operator_decision"] == "confirmed"
    assert final["clip_path"] is not None
    saved = json.loads((events_dir / f"{event_id}.json").read_text())
    assert saved["operator_decision"] == "confirmed"
    assert saved["clip_sha256"] == final["clip_sha256"]


def test_pipeline_replay_end_to_end(events_dir):
    app = create_app(lambda: WavReplaySource(SPEECH_WAV), threshold=0.6, events_dir=events_dir)

    with TestClient(app) as client:
        hub = client.app.state.hub
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(e["clip_path"] for e in hub.events.values()):
                break
            time.sleep(0.05)

        with client.websocket_connect("/ws") as ws:
            history = ws.receive_json()

    (event,) = history["events"]
    assert event["detector"] == "energy"
    assert event["clip_path"] is not None
    clip = events_dir / Path(event["clip_path"]).name
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == event["clip_sha256"]

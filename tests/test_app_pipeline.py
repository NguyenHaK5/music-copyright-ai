"""Test Phase 5: lõi demo, hiển thị, cache chỉ mục, hiệu chuẩn, mẫu/smoke, và giao diện Streamlit (khởi động an toàn)."""
import json
import os

import numpy as np
import pytest
import soundfile as sf
import yaml

torch = pytest.importorskip("torch")

from src.app import pipeline as P
from src.app.calibrate_demo import compute_thresholds
from src.app.engine import get_or_build_index
from src.app.make_demo_samples import write_samples
from src.app.render import BANNER, explain, label_caption, rights_lines, top_rows
from src.app.smoke_demo import run_smoke
from src.features.logmel import logmel, peak_normalize
from src.models.m3_cnn import build_model
from src.models.m3_encoder import M3Encoder
from src.models.m3_retrieval import M3Retriever

SR = 22050
RULES = yaml.safe_load(open("configs/license_rules.yaml", encoding="utf-8"))
FORBIDDEN = ["hợp lệ", "không có bản quyền", "được phép dùng"]
CFG = {"data": {"n_mels": 128, "sr": SR, "hop_length": 512},
       "model": {"channels": [4, 8], "blocks_per_stage": 1, "embed_dim": 16, "pool": "freq_flatten", "dropout": 0.0}}


def _audio(i, seconds=12.0):
    t = np.arange(int(seconds * SR)) / SR
    base = 180 * (1 + 0.37 * i)
    y = sum(np.sin(2 * np.pi * base * (k + 1) * t) / (k + 1) for k in range(5)) * (1 + 0.5 * np.sin(2 * np.pi * (1.5 + i) * t))
    return (y / np.abs(y).max()).astype(np.float32)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    torch.manual_seed(0)
    m = build_model(CFG).train()
    for _ in range(4):
        m(torch.randn(8, 1, 128, 80))
    torch.save({"model": m.state_dict(), "cfg": CFG, "norm_std": 12.0, "seed": 1}, d / "best.pt")
    retr = M3Retriever(M3Encoder.from_checkpoint(d / "best.pt", device="cpu"), 215, 108)
    ys = {10 + i: _audio(i) for i in range(5)}
    mels = {t: logmel(peak_normalize(y), SR, 2048, 512, 128) for t, y in ys.items()}
    retr.build_index(list(ys), lambda t: mels[t])
    return {"dir": d, "retr": retr, "ys": ys, "ckpt": d / "best.pt"}


def _engine(world, tau_high=0.0, tau_low=-1.0, meta=None, **cfgkw):
    return P.Engine(world["retr"], P.DemoConfig(**cfgkw), RULES, tau_high, tau_low, meta or {}, {})


def test_decide_three_states_and_boundaries():
    assert P.decide(0.9, 0.5, 0.8) == P.IN and P.decide(0.8, 0.5, 0.8) == P.IN
    assert P.decide(0.6, 0.5, 0.8) == P.UNSURE and P.decide(0.5, 0.5, 0.8) == P.UNSURE
    assert P.decide(0.49, 0.5, 0.8) == P.OUT


def test_crop_query_middle_and_short():
    cfg = P.DemoConfig()
    y = np.arange(30 * SR, dtype=np.float32)
    c = P.crop_query(y, cfg)
    assert len(c) == 10 * SR and c[0] == (len(y) - 10 * SR) // 2
    assert P.crop_query(y[:5 * SR], cfg).shape == (5 * SR,)


def test_validate_upload_codes():
    cfg = P.DemoConfig(max_mb=1)
    assert P.validate_upload("a.mp3", 0, cfg) == "empty"
    assert P.validate_upload("a.mp3", 2 * 1024 * 1024, cfg) == "too_large"
    assert P.validate_upload("a.exe", 100, cfg) == "bad_ext"
    assert P.validate_upload("A.MP3", 100, cfg) is None
    for code in ("too_large", "bad_ext", "empty", "decode_error", "too_short", "silent", "missing_file", "nonfinite"):
        assert P.error_text(code, cfg)


def test_analyze_file_known_query_matches_and_structure(world, tmp_path):
    p = tmp_path / "q.wav"
    sf.write(p, world["ys"][12][:5 * SR], SR)
    eng = _engine(world, meta={12: {"title": "T", "artist": "A", "genre": "G", "license_raw": "Attribution-NonCommercial 3.0 International"}})
    r = P.analyze_file(p, "q.wav", p.stat().st_size, eng)
    assert r["ok"] and r["top"][0]["track_id"] == 12 and r["top"][0]["score"] > 0.98 and len(r["top"]) == 3
    assert r["label"] == P.IN and r["rights"]["license"] == "CC BY-NC 3.0" and r["disclaimer"] == RULES["disclaimer"]
    assert set(r["timing_ms"]) == {"logmel", "match", "decode"} and r["analyzed_seconds"] == pytest.approx(5.0, abs=0.01)


def test_rights_only_for_in_catalog_and_branches(world, tmp_path):
    p = tmp_path / "q.wav"
    sf.write(p, world["ys"][11][:5 * SR], SR)
    meta = {11: {"license_raw": "Attribution 3.0 International"}}
    inn = P.analyze_file(p, "q.wav", 1000, _engine(world, 0.0, -1.0, meta))
    uns = P.analyze_file(p, "q.wav", 1000, _engine(world, 2.0, -1.0, meta))
    out = P.analyze_file(p, "q.wav", 1000, _engine(world, 2.0, 2.0, meta))
    assert (inn["label"], uns["label"], out["label"]) == (P.IN, P.UNSURE, P.OUT)
    assert inn["rights"] is not None and uns["rights"] is None and out["rights"] is None
    text = json.dumps([inn, uns, out], ensure_ascii=False).lower()
    assert not [w for w in FORBIDDEN if w in text]


def test_error_branches(world, tmp_path):
    eng = _engine(world)
    sf.write(tmp_path / "short.wav", world["ys"][10][:SR], SR)
    sf.write(tmp_path / "silent.wav", np.zeros(5 * SR, np.float32), SR)
    (tmp_path / "broken.mp3").write_bytes(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8).tobytes())
    (tmp_path / "x.txt").write_text("hello")
    get = lambda n: P.analyze_file(tmp_path / n, n, (tmp_path / n).stat().st_size, eng)
    assert get("short.wav")["error"] == "too_short" and get("silent.wav")["error"] == "silent"
    assert get("broken.mp3")["error"] in ("decode_error", "too_short") and get("x.txt")["error"] == "bad_ext"
    assert P.analyze_file(tmp_path / "nope.wav", "nope.wav", 100, eng)["error"] == "missing_file"
    r = get("short.wav")
    assert r["ok"] is False and r["message"] and r["disclaimer"]


def test_analyze_bytes_removes_temp_file_even_on_error(world, monkeypatch):
    eng = _engine(world)
    created = []
    real = P.tempfile.mkstemp
    monkeypatch.setattr(P.tempfile, "mkstemp", lambda **kw: (lambda fd_p: (created.append(fd_p[1]), fd_p)[1])(real(**kw)))
    buf = __import__("io").BytesIO()
    sf.write(buf, world["ys"][10][:5 * SR], SR, format="WAV")
    assert P.analyze_bytes(buf.getvalue(), "q.wav", eng)["ok"]
    assert created and not os.path.exists(created[-1])
    monkeypatch.setattr(P, "analyze_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        P.analyze_bytes(buf.getvalue(), "q.wav", eng)
    assert not os.path.exists(created[-1])
    assert P.analyze_bytes(b"", "q.wav", eng)["error"] == "empty" and len(created) == 2          # lỗi sớm: không tạo tệp tạm


def test_index_export_import_and_cache(world, tmp_path):
    r = world["retr"]
    q = [world["ys"][11][:5 * SR]]
    mel = [logmel(peak_normalize(q[0]), SR, 2048, 512, 128)]
    s0 = r.score_matrix(mel)
    r2 = M3Retriever(r.enc, 215, 108)
    r2.import_index(r.export_index())
    assert np.allclose(r2.score_matrix(mel), s0) and r2.track_ids == r.track_ids
    with pytest.raises(ValueError):
        M3Retriever(r.enc, 100, 50).import_index(r.export_index())
    ids, loader = r.track_ids, (lambda t: logmel(peak_normalize(world["ys"][t]), SR, 2048, 512, 128))
    cp = tmp_path / "idx.npz"
    r3 = M3Retriever(r.enc, 215, 108)
    assert get_or_build_index(r3, ids, loader, cp, "sha1") == "built"
    r4 = M3Retriever(r.enc, 215, 108)
    assert get_or_build_index(r4, ids, loader, cp, "sha1") == "cache" and np.allclose(r4.score_matrix(mel), s0, atol=1e-5)
    assert get_or_build_index(M3Retriever(r.enc, 215, 108), ids, loader, cp, "sha2") == "built"        # checkpoint đổi
    assert get_or_build_index(M3Retriever(r.enc, 215, 108), ids[:3], loader, cp, "sha2") == "built"    # danh sách bài đổi


def test_compute_thresholds():
    kc = np.linspace(0.5, 1.0, 101)
    un = np.array([0.1, 0.3, 0.6, 0.85, 0.95])
    t = compute_thresholds(kc, un, tau_high=0.8)
    assert t["tau_low"] == pytest.approx(np.percentile(kc, 5)) and t["tau_low"] < t["tau_high"] and t["note"] is None
    assert t["unknown"]["frac_ge_tau_high"] == pytest.approx(0.4) and t["known_correct"]["frac_lt_tau_low"] <= 0.06
    hi = compute_thresholds(np.full(10, 0.99), un, tau_high=0.8)
    assert hi["tau_low"] == 0.8 and hi["note"]


def test_render_strings_and_no_forbidden_words():
    for lab in (P.IN, P.OUT, P.UNSURE):
        kind, cap = label_caption(lab)
        assert kind in ("success", "info", "warning") and cap
        assert explain(lab, 0.7, 0.5, 0.8)
    from src.rights.license_rules import describe
    lines = rights_lines(describe("Attribution-NonCommercial-ShareAlike 3.0 International", P.IN, RULES))
    assert any("phi thương mại" in l for l in lines) and any("KHÔNG có nghĩa" in l for l in lines)
    assert rights_lines(None) == [] and rights_lines(describe("weird", P.IN, RULES))[0].startswith("**UNKNOWN")
    rows = top_rows([{"rank": 1, "track_id": 5, "score": 0.91234, "title": None, "artist": None, "genre": None}])
    assert rows[0]["Bài"] == "track 5" and rows[0]["Điểm giống"] == 0.912
    blob = (BANNER + " ".join(lines) + " ".join(explain(l, 0.7, 0.5, 0.8) for l in (P.IN, P.OUT, P.UNSURE))).lower()
    assert not [w for w in FORBIDDEN if w in blob]


def test_samples_and_smoke_logic(world, tmp_path):
    exp = write_samples(tmp_path, (10, world["ys"][10]), (11, world["ys"][11]))
    assert (tmp_path / "expected.json").exists() and exp["silent.wav"] == "silent" and exp["empty.wav"] == "empty"
    assert (tmp_path / "broken.mp3").stat().st_size == 4096 and (tmp_path / "empty.wav").stat().st_size == 0
    k, _ = sf.read(tmp_path / "known_10.wav"); assert len(k) == 10 * SR
    ok = lambda label: {"ok": True, "label": label, "top": [{"score": 0.9}], "rights": None, "timing_ms": {}}
    answers = {"known_10.wav": ok(P.IN), "known_10_snr10.wav": ok(P.UNSURE), "unknown_11.wav": ok(P.OUT),
               "too_short.wav": {"ok": False, "error": "too_short"}, "silent.wav": {"ok": False, "error": "silent"},
               "broken.mp3": {"ok": False, "error": "decode_error"}, "empty.wav": {"ok": False, "error": "empty"}}
    analyze = lambda path, name, size, engine: answers[name]
    res = run_smoke(None, tmp_path, analyze)
    assert res["all_pass"] and len(res["rows"]) == len(exp)
    bad = run_smoke(None, tmp_path, lambda *a: {"ok": True, "label": P.OUT, "top": [{"score": 0.1}], "rights": None, "timing_ms": {}})
    assert not bad["all_pass"]


def test_streamlit_app_starts_and_reports_missing_setup_without_crash():
    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1 and any("phân loại kỹ thuật" in w.value for w in at.warning)
    assert len(at.error) == 1 and "Không khởi động được hệ thống" in at.error[0].value


def test_streamlit_app_renders_with_engine_stub(world, monkeypatch):
    import pathlib
    import src.app.engine as E
    eng = _engine(world, 0.8, 0.5)
    eng.info = {"ckpt": "x/best.pt", "ckpt_sha256": "abc123", "n_index_tracks": 5, "index_windows": world["retr"].index_size}
    monkeypatch.setattr(E, "build_engine", lambda log=print: eng)
    path = str(pathlib.Path(__file__).resolve().parents[1] / "src" / "app" / "streamlit_app.py")
    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(path, default_timeout=60)
    at.run()
    assert not at.exception and len(at.error) == 0
    assert "Nhận diện nhạc" in at.title[0].value
    assert any("phân loại kỹ thuật" in w.value for w in at.warning)
    assert len(at.expander) == 1 and "Thông tin hệ thống" in at.expander[0].label

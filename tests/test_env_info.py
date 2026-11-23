"""実行環境の記録(src/env_info.py)と、結果ファイルへの書き込みのテスト。"""

import json
from pathlib import Path

from src.analysis.yolo_threshold import capture_detections
from src.env_info import TRACKED_PACKAGES, runtime_environment
from src.providers.yolo import YoloProvider
from src.runner import run_eval


def test_runtime_environment_lists_tracked_packages():
    env = runtime_environment(device="cpu")
    assert env["device"] == "cpu"
    assert env["python"].count(".") == 2
    assert set(env["packages"]) == set(TRACKED_PACKAGES)
    # テスト環境には numpy が必ずある。入っていないパッケージは None で残る
    assert env["packages"]["numpy"]


def test_missing_package_is_none(monkeypatch):
    monkeypatch.setattr("src.env_info.TRACKED_PACKAGES", ("no-such-package-xyz",))
    assert runtime_environment()["packages"] == {"no-such-package-xyz": None}


def test_runner_result_has_environment(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps([
        {"image_id": 1, "file_name": "a.jpg", "checklist": ["dog"],
         "ground_truth": {"dog": "present"}}]), encoding="utf-8")
    out = run_eval(str(eval_path), {"provider": "mock"}, cache_dir=str(tmp_path / "c"),
                   results_dir=str(tmp_path / "r"), images_dir=str(tmp_path / "i"))
    env = json.loads(Path(out).read_text(encoding="utf-8"))["environment"]
    assert env["device"] is None  # mock はデバイスを持たない
    assert set(env["packages"]) == set(TRACKED_PACKAGES)


class _FakeTensor:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return self._vals


class _FakeBoxes:
    cls = _FakeTensor([0])
    conf = _FakeTensor([0.9])


class _FakeResult:
    def __init__(self):
        self.names = {0: "dog"}
        self.boxes = _FakeBoxes()


class _FakeModel:
    device = "cpu"  # ultralytics の YOLO.device は torch.device。str にして記録する

    def predict(self, source, conf, verbose):
        return [_FakeResult()]


def test_yolo_device_and_capture_environment(monkeypatch):
    assert YoloProvider(weights="fake.pt").device is None  # 読み込む前は取れない

    import src.providers.yolo as yolo_mod
    orig_init = yolo_mod.YoloProvider.__init__

    def init_with_fake(self, weights="fake.pt", conf_threshold=0.25, model=None):
        orig_init(self, weights=weights, conf_threshold=conf_threshold, model=_FakeModel())

    monkeypatch.setattr(yolo_mod.YoloProvider, "__init__", init_with_fake)
    raw = capture_detections([{"image_id": 1, "file_name": "a.jpg", "checklist": ["dog"]}],
                             weights="fake.pt", images_dir="imgs")
    assert raw["detections"] == {"1": {"dog": [0.9]}}
    assert raw["environment"]["device"] == "cpu"

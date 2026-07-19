"""YoloProvider の単体テスト。torch/ultralytics 不要の fake モデルで還元ロジックを検証。"""

from pathlib import Path

from src.providers.yolo import YoloProvider


class _FakeBoxes:
    def __init__(self, cls_ids):
        self.cls = _FakeTensor(cls_ids)


class _FakeTensor:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return self._vals


class _FakeResult:
    def __init__(self, names, cls_ids):
        self.names = names
        self.boxes = _FakeBoxes(cls_ids)


class _FakeModel:
    """predict() が固定の検出を返す。dog×2, cat×1 を検出した想定。"""
    def predict(self, source, conf, verbose):
        names = {0: "dog", 1: "cat", 2: "car"}
        return [_FakeResult(names, cls_ids=[0, 0, 1])]


def test_detection_reduced_to_presence_and_count():
    prov = YoloProvider(weights="fake.pt", conf_threshold=0.25, model=_FakeModel())
    resp = prov.judge(Path("x.jpg"), ["dog", "cat", "car"])
    by_cat = {j.category: j for j in resp.judgements}

    assert by_cat["dog"].judgement == "present" and by_cat["dog"].count == 2
    assert by_cat["cat"].judgement == "present" and by_cat["cat"].count == 1
    assert by_cat["car"].judgement == "absent" and by_cat["car"].count == 0
    # YOLO は uncertain を返さない
    assert all(j.judgement in {"present", "absent"} for j in resp.judgements)
    # ローカル推論=コスト 0
    assert resp.usage.input_tokens == 0 and resp.usage.output_tokens == 0
    assert resp.model_version == "yolo-fake"


def test_empty_detection_all_absent():
    class _Empty(_FakeModel):
        def predict(self, source, conf, verbose):
            return [_FakeResult({0: "dog"}, cls_ids=[])]

    prov = YoloProvider(weights="y.pt", model=_Empty())
    resp = prov.judge(Path("x.jpg"), ["dog", "cat"])
    assert all(j.judgement == "absent" and j.count == 0 for j in resp.judgements)

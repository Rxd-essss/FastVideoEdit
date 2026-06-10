"""Feature C — vertical 9:16 auto face-crop: pure crop math + cv2-less fallback."""
import vpipe.facecrop as fc


# --- parse_target ------------------------------------------------------------
def test_parse_target_default():
    assert fc.parse_target(None) == (1080, 1920)
    assert fc.parse_target("") == (1080, 1920)
    assert fc.parse_target("garbage") == (1080, 1920)
    assert fc.parse_target("0x0") == (1080, 1920)


def test_parse_target_valid():
    assert fc.parse_target("1080x1920") == (1080, 1920)
    assert fc.parse_target("720x1280") == (720, 1280)
    assert fc.parse_target("1080X1920") == (1080, 1920)   # case-insensitive


# --- crop_rect: clamping / boundaries (concrete integer rectangle) -----------
def test_crop_rect_center():
    # 1920x1080 landscape, center crop -> 9:16 column of width 1080*9/16 = 607.5
    r = fc.crop_rect(1920, 1080, 0.5)
    assert r is not None
    x, y, w, h = r
    assert h == 1080
    assert w == round(1080 * 1080 / 1920)        # 607 (height-driven crop width)
    # centered: x = (1920 - w) / 2
    assert x == round((1920 - w) / 2)
    assert y == 0


def test_crop_rect_left_clamped():
    # Face at far left -> x clamps to 0, never negative.
    r = fc.crop_rect(1920, 1080, 0.0)
    assert r is not None
    assert r[0] == 0


def test_crop_rect_right_clamped():
    # Face at far right -> x clamps to (W - crop_w), crop stays inside the frame.
    r = fc.crop_rect(1920, 1080, 1.0)
    assert r is not None
    x, _y, w, _h = r
    assert x + w <= 1920
    assert x == 1920 - w


def test_crop_rect_offcenter_within_bounds():
    r = fc.crop_rect(1920, 1080, 0.25)
    assert r is not None
    x, _y, w, _h = r
    assert 0 <= x <= 1920 - w


def test_crop_rect_out_of_range_center_clamped():
    # center_x outside [0,1] is clamped, not propagated into a bad rect.
    assert fc.crop_rect(1920, 1080, -3.0) == fc.crop_rect(1920, 1080, 0.0)
    assert fc.crop_rect(1920, 1080, 9.0) == fc.crop_rect(1920, 1080, 1.0)


def test_crop_rect_none_when_already_vertical():
    # Source already 9:16 (width:height == 0.5625) or narrower -> no crop needed.
    assert fc.crop_rect(1080, 1920, 0.5) is None       # exactly 9:16
    assert fc.crop_rect(720, 1280, 0.5) is None        # portrait (narrower)
    assert fc.crop_rect(500, 1080, 0.5) is None         # tall portrait


def test_crop_rect_square_source_is_cropped():
    # A square (AR 1.0) is WIDER than 9:16 (0.5625), so it still gets cropped.
    r = fc.crop_rect(1080, 1080, 0.5)
    assert r is not None
    assert r[2] == round(1080 * 1080 / 1920)           # 608


def test_crop_rect_invalid_dims():
    assert fc.crop_rect(0, 0, 0.5) is None
    assert fc.crop_rect(1920, 0, 0.5) is None


def test_crop_rect_custom_target_4_5():
    # A 4:5 target (1080x1350) on a 1920x1080 source: width = 1080*1080/1350 = 864.
    r = fc.crop_rect(1920, 1080, 0.5, target=(1080, 1350))
    assert r is not None
    assert r[2] == round(1080 * 1080 / 1350)


# --- crop_filter: ffmpeg expression string -----------------------------------
def test_crop_filter_contains_crop_and_scale():
    f = fc.crop_filter(1920, 1080, 0.5)
    assert f is not None
    assert f.startswith("crop=")
    assert "scale=1080:1920" in f
    # crop runs BEFORE scale in the compound string (order crop -> scale).
    assert f.index("crop=") < f.index("scale=")
    # height-driven crop using ffmpeg iw/ih expressions, clamped with max(0,min()).
    assert "ih*1080/1920" in f
    assert "max(0,min(" in f


def test_crop_filter_center_value_embedded():
    f = fc.crop_filter(1920, 1080, 0.25)
    assert "0.2500" in f          # CX substituted as a 4-decimal float literal


def test_crop_filter_clamps_center_literal():
    f = fc.crop_filter(1920, 1080, 5.0)
    assert "1.0000" in f          # out-of-range center clamped before embedding


def test_crop_filter_none_when_already_vertical():
    assert fc.crop_filter(1080, 1920, 0.5) is None
    assert fc.crop_filter(720, 1280, 0.5) is None


def test_crop_filter_unknown_dims_falls_back_to_expression():
    # Unknown source dims (0): still produce a valid center-crop expression.
    f = fc.crop_filter(0, 0, 0.5)
    assert f is not None
    assert f.startswith("crop=")
    assert "scale=1080:1920" in f


# --- vertical_filter: always returns an exact-target filter ------------------
def test_vertical_filter_landscape_delegates_to_crop():
    # Landscape -> face-aware crop branch (same string as crop_filter).
    assert fc.vertical_filter(1920, 1080, 0.5) == fc.crop_filter(1920, 1080, 0.5)


def test_vertical_filter_portrait_covers_and_crops():
    # Already-vertical source -> cover+center-crop to exact target, never None.
    f = fc.vertical_filter(720, 1280, 0.5)
    assert f is not None
    assert "force_original_aspect_ratio=increase" in f
    assert "scale=1080:1920" in f
    assert "crop=1080:1920" in f


def test_vertical_filter_never_none():
    for dims in [(1920, 1080), (720, 1280), (1080, 1080), (0, 0), (3840, 2160)]:
        assert fc.vertical_filter(*dims, 0.5) is not None


# --- detect_center: cv2-less / failure fallback ------------------------------
def test_detect_center_returns_half_without_cv2(monkeypatch):
    # Force the cv2-unavailable path: must return 0.5 and never raise.
    monkeypatch.setattr(fc, "_CV2_OK", False)
    assert fc.detect_center("nonexistent.mp4", None, 10.0) == 0.5


def test_detect_center_zero_duration():
    # Zero/negative duration -> center, regardless of cv2.
    assert fc.detect_center("whatever.mp4", None, 0.0) == 0.5
    assert fc.detect_center("whatever.mp4", None, -5.0) == 0.5


def test_detect_center_bad_path_returns_half():
    # A path cv2 cannot open must degrade to 0.5 (no exception escapes).
    assert fc.detect_center("D:/definitely/not/a/real/file.mp4", None, 10.0) == 0.5


def test_detect_center_in_unit_interval():
    val = fc.detect_center("D:/definitely/not/a/real/file.mp4", None, 10.0)
    assert 0.0 <= val <= 1.0


# --- YuNet engine (plan 2.3): fallback cascade YuNet -> Haar -> 0.5 -----------
class _FakeFrame:
    """Just enough of an ndarray for the detector helpers (.shape only)."""
    shape = (1080, 1920, 3)


def _yunet_row(cx_frac, score, w=200.0, frame_w=1920.0):
    """One YuNet detection row: [x, y, w, h, 10 landmarks, score]."""
    x = cx_frac * frame_w - w / 2.0
    return [x, 100.0, w, 220.0] + [0.0] * 10 + [score]


class _FakeCap:
    """Deterministic VideoCapture stand-in: always open, always yields a frame."""
    def __init__(self, path):
        self.path = path

    def isOpened(self):
        return True

    def get(self, prop):
        return {1: 25.0, 3: 1920.0, 4: 1080.0, 7: 250.0}.get(prop, 0.0)

    def set(self, prop, val):
        pass

    def read(self):
        return True, _FakeFrame()

    def release(self):
        pass


class _FakeYuNetDetector:
    def __init__(self, faces):
        self._faces = faces
        self.detect_calls = 0

    def setInputSize(self, size):
        assert size == (1920, 1080)

    def detect(self, frame):
        self.detect_calls += 1
        return 1, self._faces


def _fake_cv2(monkeypatch, *, yunet_faces=None, yunet_create_raises=False,
              haar_faces=()):
    """Install a fake _cv2 into facecrop and return a call-recorder dict."""
    calls = {"haar_built": 0, "yunet_created": 0}

    class _FakeCascade:
        def empty(self):
            return False

        def detectMultiScale(self, gray, **kw):
            return list(haar_faces)

    class _FakeData:
        haarcascades = "fake/haarcascades/"

    class _FakeFaceDetectorYN:
        @staticmethod
        def create(model, config, size, **kw):
            calls["yunet_created"] += 1
            if yunet_create_raises:
                raise RuntimeError("broken onnx")
            assert kw.get("score_threshold") is not None     # sane threshold set
            return _FakeYuNetDetector(yunet_faces)

    class _FakeCV2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FRAME_COUNT = 7
        CAP_PROP_POS_FRAMES = 2
        COLOR_BGR2GRAY = 6
        data = _FakeData()
        FaceDetectorYN = _FakeFaceDetectorYN

        @staticmethod
        def VideoCapture(path):
            return _FakeCap(path)

        @staticmethod
        def CascadeClassifier(path):
            calls["haar_built"] += 1
            return _FakeCascade()

        @staticmethod
        def cvtColor(frame, code):
            return frame

    monkeypatch.setattr(fc, "_cv2", _FakeCV2)
    monkeypatch.setattr(fc, "_CV2_OK", True)
    return calls


def test_detect_center_uses_yunet_when_model_present(monkeypatch, tmp_path):
    # Model file exists -> YuNet path; Haar cascade must NOT be constructed.
    model = tmp_path / "face_detection_yunet_2023mar.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(fc, "yunet_model_path", lambda: model)
    faces = [_yunet_row(0.75, 0.9)]
    calls = _fake_cv2(monkeypatch, yunet_faces=faces)
    logged = []
    val = fc.detect_center("fake.mp4", None, 10.0, samples=4, log=logged.append)
    assert abs(val - 0.75) < 1e-6
    assert calls["yunet_created"] == 1
    assert calls["haar_built"] == 0
    assert any("YuNet" in m for m in logged)


def test_detect_center_yunet_picks_most_confident_face(monkeypatch, tmp_path):
    # Two faces: center 0.25 @ 0.95 confidence beats center 0.8 @ 0.62.
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(fc, "yunet_model_path", lambda: model)
    faces = [_yunet_row(0.8, 0.62), _yunet_row(0.25, 0.95)]
    _fake_cv2(monkeypatch, yunet_faces=faces)
    val = fc.detect_center("fake.mp4", None, 10.0, samples=4, log=lambda m: None)
    assert abs(val - 0.25) < 1e-6


def test_detect_center_yunet_no_faces_returns_half(monkeypatch, tmp_path):
    # YuNet runs but finds nothing (detect -> None) -> 0.5, exactly like before.
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(fc, "yunet_model_path", lambda: model)
    _fake_cv2(monkeypatch, yunet_faces=None)
    assert fc.detect_center("fake.mp4", None, 10.0, samples=4,
                            log=lambda m: None) == 0.5


def test_detect_center_falls_back_to_haar_without_model(monkeypatch, tmp_path):
    # No model file -> the ORIGINAL Haar branch must be used (cascade built,
    # detectMultiScale faces aggregated with the old median math).
    monkeypatch.setattr(fc, "yunet_model_path",
                        lambda: tmp_path / "missing.onnx")
    calls = _fake_cv2(monkeypatch, haar_faces=[(400, 100, 200, 200)])
    logged = []
    val = fc.detect_center("fake.mp4", None, 10.0, samples=4, log=logged.append)
    assert calls["haar_built"] == 1
    assert abs(val - (400 + 200 / 2.0) / 1920.0) < 1e-6
    assert any("Haar" in m for m in logged)


def test_detect_center_falls_back_to_haar_when_yunet_create_fails(monkeypatch,
                                                                  tmp_path):
    # Model present but FaceDetectorYN.create raises (corrupt onnx / old cv2
    # build) -> graceful Haar fallback, never a crash.
    model = tmp_path / "m.onnx"
    model.write_bytes(b"bad")
    monkeypatch.setattr(fc, "yunet_model_path", lambda: model)
    calls = _fake_cv2(monkeypatch, yunet_create_raises=True,
                      haar_faces=[(860, 100, 200, 200)])
    val = fc.detect_center("fake.mp4", None, 10.0, samples=4, log=lambda m: None)
    assert calls["haar_built"] == 1
    assert abs(val - (860 + 100) / 1920.0) < 1e-6


def test_yunet_center_helper_most_confident_and_empty():
    # Direct helper contract: max-score face wins; None/empty -> None.
    det = _FakeYuNetDetector([_yunet_row(0.7, 0.61), _yunet_row(0.3, 0.99)])
    assert abs(fc._yunet_center(det, _FakeFrame()) - 0.3) < 1e-6
    assert fc._yunet_center(_FakeYuNetDetector(None), _FakeFrame()) is None
    assert fc._yunet_center(_FakeYuNetDetector([]), _FakeFrame()) is None


def test_create_yunet_none_without_model(monkeypatch, tmp_path):
    # Missing model file -> None (Haar branch chosen), regardless of cv2 state.
    monkeypatch.setattr(fc, "yunet_model_path", lambda: tmp_path / "no.onnx")
    _fake_cv2(monkeypatch)
    assert fc._create_yunet(1920, 1080) is None


def test_create_yunet_none_when_api_missing(monkeypatch, tmp_path):
    # opencv too old (no FaceDetectorYN attribute) -> None -> Haar branch.
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(fc, "yunet_model_path", lambda: model)
    _fake_cv2(monkeypatch)
    monkeypatch.setattr(fc._cv2, "FaceDetectorYN", None)
    assert fc._create_yunet(1920, 1080) is None


def test_vendored_yunet_model_is_real():
    # The downloaded artifact must be the actual onnx (~230 KB), not an
    # LFS pointer stub (~130 bytes) and not missing.
    p = fc.yunet_model_path()
    assert p.is_file()
    assert p.stat().st_size > 200_000

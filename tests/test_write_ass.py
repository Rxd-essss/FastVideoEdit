"""Feature A — unit tests for vpipe.subtitles.write_ass (burn-in ASS/libass)."""
import re

from vpipe.config import AssStyleCfg, MaskingCfg, ProfanityLists
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Word
from vpipe.render import _ass_path_for_filter
from vpipe.subtitles import Cue, _ass_text_escape, _ass_ts, write_ass


def _read(path):
    # write_ass writes UTF-8 with BOM; utf-8-sig strips it back transparently.
    return path.read_text(encoding="utf-8-sig")


def test_ass_ts_centiseconds():
    assert _ass_ts(0) == "0:00:00.00"
    assert _ass_ts(1.5) == "0:00:01.50"
    assert _ass_ts(3661.23) == "1:01:01.23"
    assert _ass_ts(-1) == "0:00:00.00"


def test_write_ass_has_required_sections(tmp_path):
    cues = [Cue(0.0, 2.0, "привет мир"), Cue(2.5, 4.0, "вторая реплика")]
    out = tmp_path / "burn.ass"
    write_ass(cues, out, AssStyleCfg(), karaoke=False, play_res=(1920, 1080))
    txt = _read(out)
    assert "[Script Info]" in txt
    assert "PlayResX: 1920" in txt
    assert "PlayResY: 1080" in txt
    assert "[V4+ Styles]" in txt
    assert "Style: Default," in txt
    assert "[Events]" in txt
    # One Dialogue line per cue, with the cyrillic text intact.
    dialogues = [l for l in txt.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogues) == 2
    assert "привет мир" in txt
    assert "вторая реплика" in txt


def test_write_ass_alignment_from_position(tmp_path):
    out = tmp_path / "b.ass"
    for pos, align in (("bottom", "2"), ("top", "8"), ("center", "5")):
        write_ass([Cue(0, 1, "x")], out, AssStyleCfg(position=pos), karaoke=False)
        style = next(l for l in _read(out).splitlines() if l.startswith("Style: Default,"))
        # Alignment is the field right before MarginL/MarginR/MarginV (…,Align,40,40,MV,1).
        fields = style.split(",")
        assert fields[-5] == align, f"{pos} -> {style}"


def test_write_ass_karaoke_k_sum_equals_cue_duration(tmp_path):
    # Two words filling a 2.00s cue: sum of \k must equal 200 centiseconds.
    words = [Word("раз", 0.0, 0.8), Word("два", 0.8, 2.0)]
    cue = Cue(0.0, 2.0, "раз два")
    out = tmp_path / "k.ass"
    m = ProfanityMatcher(ProfanityLists())
    write_ass([cue], out, AssStyleCfg(), karaoke=True, words=words,
              matcher=m, mask=MaskingCfg(), play_res=(1280, 720))
    dia = next(l for l in _read(out).splitlines() if l.startswith("Dialogue:"))
    ks = [int(x) for x in re.findall(r"\\k(\d+)", dia)]
    assert len(ks) == 2                       # one \k per word
    assert sum(ks) == 200                     # exactly the cue duration (cs)


def test_write_ass_karaoke_sum_with_lead_in_gap(tmp_path):
    # Lead-in (cue starts 0.0 but first word at 0.3) and an inter-word gap must
    # be folded so the \k sum still equals the cue duration in centiseconds.
    words = [Word("один", 0.3, 1.0), Word("два", 1.4, 2.5)]
    cue = Cue(0.0, 2.5, "один два")
    out = tmp_path / "k2.ass"
    write_ass([cue], out, AssStyleCfg(), karaoke=True, words=words,
              matcher=ProfanityMatcher(ProfanityLists()), mask=MaskingCfg())
    dia = next(l for l in _read(out).splitlines() if l.startswith("Dialogue:"))
    ks = [int(x) for x in re.findall(r"\\k(\d+)", dia)]
    assert sum(ks) == 250                     # 2.50s -> 250 cs, all gaps folded
    assert all(k >= 1 for k in ks)


def test_ass_text_escape():
    # Newlines become the ASS hard break \N.
    assert _ass_text_escape("a\nb") == "a\\Nb"
    assert _ass_text_escape("a\r\nb") == "a\\Nb"
    # Braces are swapped for fullwidth lookalikes (safe in libass AND VSFilter),
    # so no real '{'/'}' survives to open an override block.
    out = _ass_text_escape("x {y} z")
    assert "{" not in out and "}" not in out
    assert "｛" in out and "｝" in out
    # A literal backslash must NOT be doubled -- ASS has no general '\' escape.
    assert "\\\\" not in _ass_text_escape("a\\b")


def test_write_ass_masks_profanity_in_karaoke(tmp_path):
    # Karaoke is re-derived from raw words, so it must re-apply profanity masking.
    words = [Word("ну", 0.0, 0.4), Word("блядь", 0.4, 1.0)]
    cue = Cue(0.0, 1.0, "ну б***ь")
    out = tmp_path / "mask.ass"
    m = ProfanityMatcher(ProfanityLists(roots=["бля"], allow=[]))
    write_ass([cue], out, AssStyleCfg(), karaoke=True, words=words,
              matcher=m, mask=MaskingCfg())
    txt = _read(out)
    assert "блядь" not in txt                  # raw profanity must not leak
    assert "б***ь" in txt


def test_ass_path_for_filter_windows():
    # Backslashes -> forward slashes; the drive colon is escaped (C: -> C\:),
    # so the value can live inside subtitles='...'.
    got = _ass_path_for_filter(r"C:\work\sess\burn.ass")
    assert got == "C\\:/work/sess/burn.ass"
    # POSIX-style absolute path is left as-is (no drive colon to escape).
    assert _ass_path_for_filter("/tmp/x.ass") == "/tmp/x.ass"
    # Filtergraph delimiters inside the path get escaped (quote idiom: below).
    assert _ass_path_for_filter(r"D:\a,b[1];x.ass") == "D\\:/a\\,b\\[1\\]\\;x.ass"


def _unquote_single(s: str) -> str:
    """Minimal shell/ffmpeg single-quote unescaper: '...' spans literal text; a
    backslash OUTSIDE quotes escapes the next char, so the close-escape-reopen
    idiom '\\'' collapses back to a literal apostrophe."""
    out, i, in_q = [], 0, False
    while i < len(s):
        c = s[i]
        if in_q:
            if c == "'":
                in_q = False
            else:
                out.append(c)
        elif c == "'":
            in_q = True
        elif c == "\\" and i + 1 < len(s):
            out.append(s[i + 1])
            i += 1
        else:
            out.append(c)
        i += 1
    return "".join(out)


def test_ass_path_quote_idiom():
    # #59: an apostrophe inside a single-quoted subtitles='...' value must be the
    # close-escape-reopen idiom '\'' -- a bare backslash does NOT escape it, so the
    # old \\' left the quote OPEN and mangled the path, failing the whole burn.
    got = _ass_path_for_filter("C:/a/Don't panic/burn.ass")
    assert got == "C\\:/a/Don'\\''t panic/burn.ass"
    # Wrapped in single quotes it round-trips: the quote layer restores the ' ,
    # then the subtitles-option layer unescapes the drive colon \: -> : .
    unq = _unquote_single("'" + got + "'").replace("\\:", ":")
    assert unq == "C:/a/Don't panic/burn.ass"
    # A path with no apostrophe is inert under the idiom (no ' introduced).
    assert "'" not in _ass_path_for_filter("C:/plain/burn.ass")


def test_write_ass_karaoke_colors_in_style(tmp_path):
    # With karaoke on, PrimaryColour (sung) must be the karaoke colour.
    out = tmp_path / "c.ass"
    st = AssStyleCfg(primary_color="&H00FFFFFF", karaoke_color="&H0000FFFF")
    write_ass([Cue(0, 1, "x")], out, st, karaoke=True,
              words=[Word("x", 0.0, 1.0)],
              matcher=ProfanityMatcher(ProfanityLists()), mask=MaskingCfg())
    style = next(l for l in _read(out).splitlines() if l.startswith("Style: Default,"))
    # Style: Default,<font>,<size>,<Primary>,<Secondary>,...
    fields = style.split(",")
    assert fields[3] == "&H0000FFFF"           # PrimaryColour = karaoke highlight
    assert fields[4] == "&H00FFFFFF"           # SecondaryColour = base text colour


def test_ass_size_scales_with_playres(tmp_path):
    # Presets are defined @1080; write_ass scales Fontsize/MarginV by the SHORT
    # side (min(PlayResX, PlayResY)/1080): 4K doubles, while the hand-tuned
    # vertical 1080x1920 Shorts frame keeps k=1 (W6 #4).
    st = AssStyleCfg(size=52, margin_v=40)

    def _style_fields(pr):
        out = tmp_path / f"s_{pr[1]}.ass"
        write_ass([Cue(0, 1, "x")], out, st, karaoke=False, play_res=pr)
        line = next(l for l in _read(out).splitlines()
                    if l.startswith("Style: Default,"))
        return line.split(",")

    f1080 = _style_fields((1920, 1080))
    assert f1080[2] == "52"                    # k=1.0 -> unchanged @1080
    assert f1080[-2] == "40"                   # MarginV unchanged @1080
    f4k = _style_fields((3840, 2160))
    assert f4k[2] == "104"                     # 52 * 2.0 (k = min side / 1080)
    assert f4k[-2] == "80"                     # 40 * 2.0
    f720 = _style_fields((1280, 720))
    assert f720[2] == "35"                     # round(52 * 0.6667)
    # W6 #4: portrait 9:16 scales by the SHORT side -> k=1.0, NOT 1.78 — the
    # presets were tuned against the 1080-wide Shorts frame; pr_y/1080 blew the
    # fonts past the 42-char wrap budget and overflowed the frame width.
    fvert = _style_fields((1080, 1920))
    assert fvert[2] == "52"
    assert fvert[-2] == "40"

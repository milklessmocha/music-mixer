"""Self-check: python test_mixer.py"""
import asyncio
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

import bot
import mixer
import suggest
from bot import default_fade, formats_keyboard, parse_name, parse_period, probe_duration


def rejects(text, duration=200):
    try:
        parse_period(text, duration)
    except ValueError:
        return True
    return False


def click_track(path, bpm, seconds=30):
    audio = np.zeros(seconds * mixer.SR, np.float32)
    click = np.sin(np.arange(2000) * 2 * np.pi * 60 / mixer.SR) * np.exp(-np.arange(2000) / 400)
    for t in np.arange(0, seconds, 60 / bpm):
        i = int(t * mixer.SR)
        audio[i:i + 2000] += click[:len(audio) - i]
    audio += np.random.default_rng(0).normal(0, 0.01, audio.size).astype(np.float32)  # some highs
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(mixer.SR), "-ac", "1", "-i", "-", str(path)],
                   input=audio.tobytes(), check=True)


# period parsing
assert parse_period("1:2 - 3:00", 200) == (62, 180, None)
assert parse_period("start - end", 200) == (0, 200, None)
assert parse_period("00:00 - 1:00", 200) == parse_period("start - 1:00", 200) == (0, 60, None)
assert parse_period("start-end-0:10", 200) == (0, 200, 10)
assert parse_period("START - END", 200) == (0, 200, None)
for bad in ["2:00 - 1:00", "0:10 - 3:21", "0:61 - 1:00", "start - end - 0:02", "0:10 - 0:30 - 0:11",
            "0:10 - 0:15", "end - start", "hello", "1:00", "0:10 - 0:30 - 0:05 - 0:01", "90 - 120"]:
    assert rejects(bad), bad
assert not rejects("0:10 - 0:30 - 0:10")

# mix name
assert parse_name("  Friday set vol. 2 ") == "Friday set vol. 2"
assert parse_name("mix v1.10") == "mix v1.10"
for bad in ["", " ", "x" * 65, "set.mp3", "set.flac", "a/b", "a\\b", "what?", ".hidden"]:
    try:
        parse_name(bad)
        raise AssertionError(bad)
    except ValueError:
        pass

# format checklist: one button per format, then All, then Submit
rows = formats_keyboard({"mp3"}).inline_keyboard
labels = [b.text for row in rows for b in row]
assert labels[0] == "✅ mp3" and labels.count("⬜ All extensions") == 1 and labels[-1] == "Submit"
assert all(b.callback_data.startswith("fmt:") for row in rows for b in row)
assert "✅ All extensions" in [b.text for row in formats_keyboard(set(mixer.FORMATS)).inline_keyboard for b in row]
assert "⬜ wav" not in [b.text for row in formats_keyboard(set(), ["mp3", "ogg"]).inline_keyboard for b in row]

# default fade
assert default_fade(120, 200) == 6
assert default_fade(40, 200) == 3
assert default_fade(50, 200) == 3
assert default_fade(130, 200) == 7
assert default_fade(200) == 10

# tempo
assert mixer.match_octave(124, 120) == 124
assert mixer.match_octave(62, 120) == 124   # half-time reading
assert mixer.match_octave(250, 120) == 125  # double-time reading

# ramp: 120 -> 128 BPM over the overlap, every click lands where the ramp math puts it
SR = mixer.SR
a = np.random.default_rng(0).normal(0, 0.01, (SR * 8, 2)).astype(np.float32)
for k in range(16):
    a[int(k * 0.5 * SR):int(k * 0.5 * SR) + 200] = 1
T = 2 * 120 * 8 / (120 + 128)
y = mixer.ramp(a, 120, 128, 120, T)
assert len(y) == int(T * SR)
hot = np.abs(y[:, 0]) > 0.5
clicks = np.flatnonzero(hot[1:] & ~hot[:-1]) / SR
clicks = clicks[np.r_[True, np.diff(clicks) > 0.2]]
acc = 8 / (2 * T)  # input beat k*0.5 s is reached at the root of 120 t + acc t^2 = 60 k
predicted = (-120 + np.sqrt(120 ** 2 + 4 * acc * 60 * np.arange(16))) / (2 * acc)
assert max(np.abs(predicted - t).min() for t in clicks) < 0.008
assert np.diff(clicks)[0] > 0.49 and np.diff(clicks)[-1] < 0.48  # spacing shrinks from 0.500 toward 0.469
assert np.abs(mixer.ramp(a, 120, 120, 120, 8.0) - a).max() < 1e-6  # steady tempo is an exact copy

# crossfade starts as pure outgoing and ends as pure incoming
rng = np.random.default_rng(1)
a, b = rng.normal(size=(44100, 2)).astype(np.float32), rng.normal(size=(44100, 2)).astype(np.float32)
x = mixer.crossfade(a, b, 0.5)
assert np.allclose(x[0], a[0], atol=1e-4) and np.allclose(x[-1], b[-1], atol=1e-4)

# end to end
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    click_track(d / "a.wav", 120)
    click_track(d / "b.wav", 128)
    assert abs(probe_duration(d / "a.wav") - 30) < 0.1
    for name, bpm in [("a.wav", 120), ("b.wav", 128)]:
        found, beats = mixer.analyze(d / name, 0, 30)
        assert abs(found - bpm) < 0.5, (name, found)
        assert np.abs(beats - np.round(beats * bpm / 60) * 60 / bpm).max() < 0.03, name  # grid sits on the clicks
    fade = 8
    mix = mixer.render([(d / "a.wav", 0, 30, fade), (d / "b.wav", 0, 30, 5)], progress=lambda s: None)
    seconds = len(mix) / SR
    for ext in mixer.FORMATS:
        mixer.encode(mix, d / f"mix.{ext}")
        assert abs(probe_duration(d / f"mix.{ext}") - seconds) < 0.2, ext
    # a plays 22 s, then the 8 s overlap ramps 120 -> 128 BPM and takes T, then b's remaining 30 - 7.5 s
    expected = 22 + T + 30 - 8 * 120 / 128
    assert abs(seconds - expected) < 0.3, (seconds, expected)
    assert abs(mixer.analyze(d / "mix.mp3", 2, 20)[0] - 120) < 0.5   # before the overlap: a's tempo
    assert abs(mixer.analyze(d / "mix.mp3", 32, 50)[0] - 128) < 0.5  # after it: b's tempo


# finish(): a format over the upload limit keeps the session and brings the picker back without it
class FakeChat:
    def __init__(self):
        self.chat, self.texts, self.markups, self.files = type("C", (), {"id": 1}), [], [], []

    async def answer(self, text, reply_markup=None, **_):
        self.texts.append(text)
        self.markups.append(reply_markup)
        return self

    async def edit_text(self, text, **_):
        self.texts.append(text)

    async def send_audio(self, chat_id, file, **_):
        self.files.append(file.filename)

    send_document = send_audio


async def finish_twice():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        click_track(d / "a.wav", 120)
        click_track(d / "b.wav", 128)
        session = d / "session"
        session.mkdir()
        state = FSMContext(MemoryStorage(), StorageKey(bot_id=1, chat_id=1, user_id=1))
        tracks = [{"path": str(d / "a.wav"), "begin": 0, "end": 30, "fade": 8},
                  {"path": str(d / "b.wav"), "begin": 0, "end": 30, "fade": None}]
        await state.set_data({"dir": str(session), "tracks": tracks, "name": "set", "formats": ["mp3", "wav"]})
        chat = FakeChat()
        bot.TG_UPLOAD_LIMIT = 3 * 1024 * 1024  # ~52 s mix: mp3 ~1.2 MB fits, wav ~9 MB doesn't
        await bot.finish(chat, state, chat)
        assert chat.files == ["set.mp3"], chat.files
        assert await state.get_state() == bot.MixFlow.choosing_formats
        assert (await state.get_data())["blocked"] == ["wav"] and session.exists()
        picker = [b.text for row in chat.markups[-1].inline_keyboard for b in row]
        assert "⬜ wav" not in picker and "⬜ mp3" in picker

        await state.update_data(formats=["ogg"])
        await bot.finish(chat, state, chat)
        assert chat.files == ["set.mp3", "set.ogg"] and chat.texts[-1] == "Done!"
        assert await state.get_state() is None and not session.exists()


asyncio.run(finish_twice())


# idle sessions expire; any action restarts the clock
async def idle_timeout():
    with tempfile.TemporaryDirectory() as d:
        session = Path(d) / "session"
        session.mkdir()
        state = FSMContext(MemoryStorage(), StorageKey(bot_id=1, chat_id=1, user_id=1))
        chat = FakeChat()
        chat.send_message = lambda chat_id, text, **kw: chat.answer(text, **kw)
        data = {"event_chat": chat.chat, "event_from_user": chat.chat, "state": state, "bot": chat}

        async def start_mix(event, data):
            await state.set_state(bot.MixFlow.waiting_audio)
            await state.set_data({"dir": str(session)})

        bot.SESSION_TIMEOUT = 0.3
        await bot.session_timer(start_mix, None, data)
        await asyncio.sleep(0.2)
        await bot.session_timer(lambda e, d: asyncio.sleep(0), None, data)  # activity at 0.2 s restarts the clock
        await asyncio.sleep(0.2)
        assert await state.get_state() is not None and session.exists()  # 0.4 s total, but only 0.2 s idle
        await asyncio.sleep(0.2)
        assert await state.get_state() is None and not session.exists()
        assert "expired" in chat.texts[-1]
        assert chat.markups[-1] is bot.DECK  # the command buttons come back with the expiry notice


asyncio.run(idle_timeout())


# /download: the link becomes the whole track, the same as /mix + "start - end" + /done
async def download_flow():
    with tempfile.TemporaryDirectory() as d:
        click_track(Path(d) / "a.wav", 120)
        state = FSMContext(MemoryStorage(), StorageKey(bot_id=1, chat_id=1, user_id=1))
        await state.set_data({"dir": d, "tracks": [], "pending": None, "single": True})
        chat = FakeChat()
        await bot.accept_audio(chat, state, Path(d) / "a.wav")
        track = (await state.get_data())["tracks"][0]
        assert (track["begin"], track["fade"]) == (0, None) and abs(track["end"] - 30) < 0.1
        assert track["end"] == parse_period("start - end", track["duration"])[1]
        assert await state.get_state() == bot.MixFlow.waiting_name


asyncio.run(download_flow())
# /suggest: keys, ranking on songs with a known structure, valid copy-paste output, bot flow
def tones(*freqs):
    chroma = np.zeros(12)
    for f in freqs:
        chroma[int(round(12 * np.log2(f / 440) + 9)) % 12] += 1
    return chroma


C_MAJOR, A_MINOR, FS_MAJOR = (261.6, 329.6, 392.0), (220.0, 261.6, 329.6), (370.0, 466.2, 554.4)
assert [suggest.camelot(tones(*c)) for c in (C_MAJOR, A_MINOR, FS_MAJOR)] == ["8B", "8A", "2B"]
assert suggest.key_match("8B", "8A") == suggest.key_match("8A", "9A") == 1.0
assert suggest.key_match("8A", "10A") == 0.5 and suggest.key_match("8B", "2B") == 0.0
assert suggest.key_match("12B", "1B") == 1.0  # the wheel wraps


def song(path, bpm, sections, chord):
    """sections: (bars, loud). Loud: kick on every beat (accent on 1), bass and chord. Quiet: soft chord only."""
    beat = 60 / bpm
    t = np.arange(int(sum(n for n, _ in sections) * 4 * beat * SR)) / SR
    y, bar = np.zeros_like(t), 0
    for n_bars, loud in sections:
        lo, hi = int(bar * 4 * beat * SR), int((bar + n_bars) * 4 * beat * SR)
        y[lo:hi] += (0.3 if loud else 0.08) * sum(np.sin(2 * np.pi * f * t[lo:hi]) for f in chord) / len(chord)
        if loud:
            y[lo:hi] += 0.2 * np.sin(2 * np.pi * chord[0] / 4 * t[lo:hi])
            for k in range(n_bars * 4):
                i = lo + int(k * beat * SR)
                n = min(4000, len(y) - i)
                y[i:i + n] += (1.0 if k % 4 == 0 else 0.6) * np.sin(2 * np.pi * 55 * np.arange(n) / SR) * np.exp(-np.arange(n) / 800)
        bar += n_bars
    y += np.random.default_rng(0).normal(0, 0.003, y.size)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", str(path)],
                   input=y.astype(np.float32).tobytes(), check=True)
    return len(t) / SR


class FakeCallback:
    def __init__(self, data, message):
        self.data, self.message = data, message

    async def answer(self, *_, **__):
        pass


async def suggest_flow():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        dur_a = song(d / "a.wav", 124, [(48, True), (16, False)], C_MAJOR)  # loud, then a 16-bar outro
        dur_b = song(d / "b.wav", 126, [(16, False), (48, True)], A_MINOR)  # 16-bar intro, then loud
        state = FSMContext(MemoryStorage(), StorageKey(bot_id=1, chat_id=1, user_id=1))
        await state.set_state(bot.MixFlow.waiting_audio)
        await state.set_data({"dir": str(d), "tracks": [], "pending": None, "suggest": True})
        chat = FakeChat()
        chat.delete = lambda: asyncio.sleep(0)
        chat.edit_reply_markup = lambda **_: asyncio.sleep(0)
        await bot.accept_audio(chat, state, d / "a.wav")
        assert await state.get_state() == bot.MixFlow.waiting_audio and "second" in chat.texts[-1]
        await bot.accept_audio(chat, state, d / "b.wav")
        assert await state.get_state() == bot.MixFlow.choosing_suggestion

        found = (await state.get_data())["suggestions"]
        assert 1 <= len(found) <= suggest.TOP
        assert [s["score"] for s in found] == sorted((s["score"] for s in found), reverse=True)
        best = found[0]
        outro = 48 * 4 * 60 / 124
        assert best["end"] - best["fade"] > outro - 1 and best["begin"] == 0, best  # leave in the outro, enter at the top
        for s in found:  # what's shown is valid /mix input
            parse_period(s["a"], dur_a)
            parse_period(s["b"], dur_b)
        assert len(chat.markups[-1].inline_keyboard[0]) == len(found)

        await bot.got_suggestion(FakeCallback("sug:0", chat), state)
        a, b = (await state.get_data())["tracks"]
        assert (a["begin"], a["end"], a["fade"]) == (0, best["end"], best["fade"])
        assert (b["begin"], b["end"], b["fade"]) == (best["begin"], b["duration"], None)
        assert await state.get_state() == bot.MixFlow.waiting_name


asyncio.run(suggest_flow())
# /analyze: report per track and the best order; a quiet outro into a matching quiet intro goes first
async def analyze_flow():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        song(d / "c.wav", 100, [(8, True), (24, True)], FS_MAJOR)  # clashes in key and tempo with both
        song(d / "a.wav", 124, [(24, True), (8, False)], C_MAJOR)  # quiet outro
        song(d / "b.wav", 126, [(8, False), (24, True)], A_MINOR)  # quiet intro, relative key of a
        names = ["c", "a", "b"]
        profiles = [suggest.profile(d / f"{n}.wav", probe_duration(d / f"{n}.wav")) for n in names]
        order = suggest.best_order(profiles)
        assert "1 2" in " ".join(map(str, order)), order  # a (index 1) right before b (index 2)
        assert suggest.pair(profiles[1], profiles[2]) > suggest.pair(profiles[2], profiles[1])

        state = FSMContext(MemoryStorage(), StorageKey(bot_id=1, chat_id=1, user_id=1))
        await state.set_state(bot.MixFlow.waiting_audio)
        await state.set_data({"dir": str(d), "tracks": [], "pending": None, "analyze": True})
        chat = FakeChat()
        chat.delete = lambda: asyncio.sleep(0)
        for n in names:
            await bot.accept_audio(chat, state, d / f"{n}.wav", f"song {n}")
        assert "/done" in chat.texts[-1] and len((await state.get_data())["tracks"]) == 3
        await bot.cmd_done(chat, state)
        report = chat.texts[-1]
        assert "song a" in report and "8B" in report and "Best order:" in report, report
        assert "2 -> 3" in report  # a -> b, numbered as sent
        assert await state.get_state() is None and not d.exists()  # session and its downloads are gone


asyncio.run(analyze_flow())
# deck: buttons show readable labels, and pressing one (which sends the label) runs the command
assert bot.label("/done") == "Done" and bot.label("/three-word-command") == "Three word command"
assert [[b.text for b in row] for row in bot.DECK.keyboard] == [["Mix", "Download", "Suggest", "Analyze"],
                                                                 ["Done", "Cancel"]]


async def deck_press():
    from aiogram import Bot
    from aiogram.client.session.base import BaseSession
    from aiogram.types import Chat, Message, Update, User

    class OfflineSession(BaseSession):  # answers every API call locally, nothing reaches Telegram
        async def make_request(self, bot, method, timeout=None):
            return Message(message_id=1, date=0, chat=Chat(id=1, type="private"), text="x")

        async def stream_content(self, *a, **k):
            yield b""

        async def close(self):
            pass

    tg = Bot("1:x", session=OfflineSession())
    me = User(id=7, is_bot=False, first_name="me")
    state = bot.dp.fsm.get_context(tg, chat_id=7, user_id=7)
    for text, expected in [("Mix", bot.MixFlow.waiting_audio), ("Cancel", None), ("/mix", bot.MixFlow.waiting_audio)]:
        msg = Message(message_id=1, date=0, chat=Chat(id=7, type="private"), from_user=me, text=text)
        await bot.dp.feed_update(tg, Update(update_id=1, message=msg))
        assert await state.get_state() == expected, (text, await state.get_state())
    await bot.drop_session(state)
    for task in bot.timers.values():
        task.cancel()


asyncio.run(deck_press())
# links: yt-dlp gets a copy of the cookies file when there is one, and nothing when there isn't
class FakeYDL:
    seen = []

    def __init__(self, opts):
        FakeYDL.seen.append(opts)
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def extract_info(self, url, download):
        return {"title": "t", "duration": 60}

    def process_ie_result(self, info, download):
        Path(self.opts["outtmpl"].replace("%(ext)s", "m4a")).write_bytes(b"x")


with tempfile.TemporaryDirectory() as d:
    d, real = Path(d), bot.yt_dlp.YoutubeDL
    bot.yt_dlp.YoutubeDL, bot.COOKIES = FakeYDL, d / "youtube.txt"
    try:
        assert bot.fetch_url("https://x", d, 0)[1] == "t" and "cookiefile" not in FakeYDL.seen[-1]
        bot.COOKIES.write_text("# Netscape HTTP Cookie File\n")
        bot.fetch_url("https://x", d, 1)
        copy = Path(FakeYDL.seen[-1]["cookiefile"])
        assert copy != bot.COOKIES and copy.read_text() == bot.COOKIES.read_text()
        assert "extractor_args" not in FakeYDL.seen[-1]  # no helper configured, as in a local run
        bot.POT_PROVIDER = "http://pot-provider:4416"
        bot.fetch_url("https://x", d, 2)
        assert FakeYDL.seen[-1]["extractor_args"] == {"youtubepot-bgutilhttp": {"base_url": ["http://pot-provider:4416"]}}
    finally:
        bot.yt_dlp.YoutubeDL = real

print("ok")

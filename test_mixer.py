"""Self-check: python test_mixer.py"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import mixer
from bot import default_fade, parse_period, probe_duration


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
    seconds = mixer.render([(d / "a.wav", 0, 30, fade), (d / "b.wav", 0, 30, 5)], d / "mix.mp3", progress=lambda s: None)
    # a plays 22 s, then the 8 s overlap ramps 120 -> 128 BPM and takes T, then b's remaining 30 - 7.5 s
    expected = 22 + T + 30 - 8 * 120 / 128
    assert abs(seconds - expected) < 0.3, (seconds, expected)
    assert abs(probe_duration(d / "mix.mp3") - seconds) < 0.2
    assert abs(mixer.analyze(d / "mix.mp3", 2, 20)[0] - 120) < 0.5   # before the overlap: a's tempo
    assert abs(mixer.analyze(d / "mix.mp3", 32, 50)[0] - 128) < 0.5  # after it: b's tempo

print("ok")

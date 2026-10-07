"""Beat-matched, EQ-swapped DJ mix renderer. ffmpeg does decode/stretch/encode, numpy/scipy do the mixing."""
import math
import subprocess

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt, stft

SR = 44100
ANALYSIS_SR = 22050
HOP = 512             # analysis frame step, ~23 ms
BPM_WINDOW = 15        # seconds of audio used to measure tempo around a transition point
CROSSOVER_HZ = 200     # bass/rest split point for the EQ swap
# output formats: extension -> ffmpeg codec args. Lossy ones stay ~1.2-1.4 MB/min, under Telegram's 50 MB
# bot upload cap for a 30 min mix; FLAC (~5 MB/min) and WAV (~10 MB/min) only fit short mixes.
FORMATS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
    "m4a": ["-c:a", "aac", "-b:a", "192k"],
    "ogg": ["-c:a", "libopus", "-b:a", "160k"],
    "flac": ["-c:a", "flac"],
    "wav": ["-c:a", "pcm_s16le"],
}


def decode(path, start, end, sr=SR, channels=2, tempo=1.0):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)]
    if end is not None:
        cmd += ["-t", str(end - start)]
    cmd += ["-i", str(path)]
    if tempo != 1.0:
        cmd += ["-af", f"atempo={tempo:.5f}"]
    cmd += ["-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, "<f4")
    return audio.reshape(-1, channels) if channels > 1 else audio


def analyze(path, start, end):
    """Return (bpm, beat grid in seconds from segment start). Assumes a steady tempo, as in most DJ material."""
    y = decode(path, start, end, sr=ANALYSIS_SR, channels=1)
    if y.size < ANALYSIS_SR:
        raise ValueError(f"segment {start}-{end} of {path} is shorter than 1 s (start past end of file?)")
    fps = ANALYSIS_SR / HOP
    # onset strength: positive spectral flux on a log spectrogram
    spec = np.log1p(100 * np.abs(stft(y, nperseg=2048, noverlap=2048 - HOP)[2]))
    onset = np.maximum(0, np.diff(spec, axis=1)).sum(axis=0)
    # tempo: autocorrelation scored at the first 8 multiples of each candidate beat period
    x = onset - onset.mean()
    acf = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * len(x))) ** 2)[:len(x)]
    bpms = np.arange(60, 200, 0.05)
    periods = fps * 60 / bpms
    score = sum(np.interp(k * periods, np.arange(len(acf)), acf, right=0) for k in range(1, 9))
    score *= np.exp(-0.5 * np.log2(bpms / 120) ** 2)  # prefer readings near 120, settles half/double time
    bpm = float(bpms[np.argmax(score)])
    # phase: shift a fixed grid until it sits on the most onset energy
    period = fps * 60 / bpm
    grid = np.arange(0, len(onset) - period, period)
    phases = np.arange(0, period, 0.25)
    phase = phases[np.argmax([np.interp(p + grid, np.arange(len(onset)), onset).sum() for p in phases])]
    # refine: snap each grid beat to the strongest onset within +/-3 frames, then fit a straight line
    pos = np.clip(np.round(phase + grid).astype(int)[:, None] + np.arange(-3, 4), 0, len(onset) - 1)
    peaks = pos[np.arange(len(pos)), onset[pos].argmax(axis=1)]
    if len(peaks) >= 4:
        period, phase = np.polyfit(np.arange(len(peaks)), peaks, 1)
        bpm = float(fps * 60 / period)
    beats = (phase + period * np.arange(len(grid)) + 1) / fps  # +1: diff frame i is the onset at frame i + 1
    return bpm, beats


def match_octave(bpm, target):
    """bpm, halved or doubled if that reads closer to target (beat trackers often land on half/double time)."""
    return min((bpm * k for k in (0.5, 1, 2)), key=lambda b: abs(math.log(target / b)))


def local_tempo(path, start, end, center):
    """(bpm, beat times in absolute seconds) measured on a BPM_WINDOW slice of [start, end] around center."""
    ws = max(start, min(center - BPM_WINDOW / 2, end - BPM_WINDOW))
    we = min(end, ws + BPM_WINDOW)
    bpm, beats = analyze(path, ws, we)
    return bpm, ws + beats


def ramp(audio, from_bpm, to_bpm, own_bpm, seconds, frame=2048, tol=256):
    """WSOLA time-stretch (pitch kept): over `seconds` of output, play `audio` at a tempo going linearly
    from_bpm -> to_bpm, where own_bpm is the audio's native tempo. Returns exactly seconds * SR samples."""
    hop = frame // 2
    n_out = int(seconds * SR)
    starts = np.arange(-hop, n_out, hop)  # output start of each frame
    centre = (starts + hop) / SR
    tau = np.minimum(centre, seconds)
    # input time reached at output time tau: integral of ramp_bpm / own_bpm, then steady at to_bpm
    x = (from_bpm * tau + (to_bpm - from_bpm) * tau ** 2 / (2 * seconds) + (centre - tau) * to_bpm) / own_bpm
    pos = np.round(x * SR).astype(int) - hop  # input start of each frame
    pad = frame + tol
    src = np.pad(audio, ((pad, 2 * pad), (0, 0)))
    mono = src.mean(axis=1)
    win = np.hanning(frame + 2)[1:-1].astype(np.float32)  # no zero ends, so the edges normalize cleanly
    out = np.zeros((n_out + 2 * frame, 2), np.float32)
    wsum = np.zeros(n_out + 2 * frame, np.float32)
    prev = None
    for o, p in zip(starts + frame, pos + pad):
        # free search in the middle; the edges stay exact so the joins with untouched audio are seamless
        if prev is not None and frame <= o - frame and o + frame <= n_out:
            ref = mono[prev + hop:prev + hop + frame]
            seg = mono[p - tol:p + tol + frame]
            energy = np.convolve(seg ** 2, np.ones(frame), mode="valid")  # normalize so loud spots don't win
            score = correlate(seg, ref, mode="valid", method="fft") / np.sqrt(energy + 1e-9)
            p += int(np.argmax(score)) - tol
        out[o:o + frame] += win[:, None] * src[p:p + frame]
        wsum[o:o + frame] += win
        prev = p
    return (out / np.maximum(wsum, 1e-6)[:, None])[frame:frame + n_out]


def crossfade(a, b, beat_sec):
    """Mix outgoing tail a into incoming head b (same shape).
    Highs/mids: equal-power fade across the whole overlap.
    Bass: hard swap over one beat at the midpoint, so two kick drums never stack."""
    n = len(a)
    sos = butter(4, CROSSOVER_HZ, "low", fs=SR, output="sos")
    a_lo, b_lo = sosfiltfilt(sos, a, axis=0), sosfiltfilt(sos, b, axis=0)
    a_hi, b_hi = a - a_lo, b - b_lo  # complement split: lo + hi == input exactly
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    swap_width = max(beat_sec * SR / n, 1e-6)
    bass = np.clip((t - 0.5) / swap_width + 0.5, 0, 1)
    return (a_hi * np.cos(t * np.pi / 2) + b_hi * np.sin(t * np.pi / 2)
            + a_lo * (1 - bass) + b_lo * bass).astype(np.float32)


def normalize(x, rms=0.1):
    level = np.sqrt(np.mean(x ** 2))
    return x * (rms / level if level > 0 else 1.0)  # always a writable copy


def render(segments, progress=print):
    """segments: list of (path, start_sec, end_sec, fade_sec). A segment's fade is its outgoing overlap with the
    next one (or its fade-out to silence if last). Returns the mix as float32 stereo at SR. In an overlap both tracks follow one beat grid whose tempo
    ramps from the outgoing track's BPM to the incoming one's. Elsewhere every track plays at native tempo."""
    path, start, end, fade = segments[0]
    progress(f"Loading track 1/{len(segments)}...")
    done, cur, cur_head = [], normalize(decode(path, start, end)), 0.0  # cur_head: mixed seconds at cur's start
    for i, (n_path, n_start, n_end, n_fade) in enumerate(segments[1:], 2):
        progress(f"Analyzing transition {i - 1} -> {i}...")
        c, c_beats = local_tempo(path, start, end, end - fade)
        n, n_beats = local_tempo(n_path, n_start, n_end, n_start + fade)
        n_eff = match_octave(n, c)
        b = normalize(decode(n_path, n_start, n_end))
        b_len = len(b) / SR

        # outgoing: overlap starts on the cur beat nearest end - fade
        overlap_start = c_beats[np.argmin(np.abs(c_beats - (end - fade)))]
        L = min(end - overlap_start, len(cur) / SR - cur_head)
        # incoming: start on its first beat (extend the local grid back to n_start)
        period = 60 / n
        offset = max(0.0, (n_beats[0] - n_start + period / 4) % period - period / 4)  # a beat just before n_start counts
        L = min(L, (b_len / 2 - offset) * n_eff / c)  # incoming overlap must fit in half its part
        note = f", fade shortened to {L:.1f} s" if L < fade - 60 / c else ""
        progress(f"Transition {i - 1} -> {i}: {c:.1f} -> {n_eff:.1f} BPM{note}")
        T = 2 * c * L / (c + n_eff)  # overlap length in the mix
        L_n = L * c / n_eff          # seconds of the incoming track used during the overlap

        cut = len(cur) - int(L * SR)
        a_part = ramp(cur[cut:], c, n_eff, c, T)
        b_part = ramp(b[int(offset * SR):int((offset + L_n) * SR)], c, n_eff, n_eff, T)
        done.append(cur[:cut])
        cur = np.concatenate([crossfade(a_part, b_part, 120 / (c + n_eff)), b[int((offset + L_n) * SR):]])
        path, start, end, fade, cur_head = n_path, n_start, n_end, n_fade, T

    tail = min(int(fade * SR), len(cur))  # last track fades out to silence
    cur[-tail:] *= np.cos(np.linspace(0, np.pi / 2, tail, dtype=np.float32))[:, None]
    done.append(cur)
    mix = np.concatenate(done)
    mix /= max(1.0, float(np.abs(mix).max()))  # ponytail: peak normalize, swap for a limiter if quiet mixes bother you
    return mix


def encode(mix, out_path):
    """Write mix to out_path, codec picked by its extension (a FORMATS key)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                    *FORMATS[out_path.suffix[1:]], str(out_path)], input=mix.astype("<f4").tobytes(), check=True)

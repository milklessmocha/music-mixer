"""Beat-matched, EQ-swapped DJ mix renderer. ffmpeg does decode/stretch/encode, numpy/scipy do the mixing."""
import math
import subprocess

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft

SR = 44100
ANALYSIS_SR = 22050
HOP = 512             # analysis frame step, ~23 ms
MAX_STRETCH = 0.08     # beyond +/-8% tempo change the stretch sounds bad, so mix without locking
CROSSOVER_HZ = 200     # bass/rest split point for the EQ swap
BITRATE = "192k"       # ~1.4 MB/min, keeps a 30 min mix under Telegram's 50 MB bot upload cap


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


def stretch_ratio(bpm, target):
    """atempo ratio that puts bpm on target, allowing half/double-time readings. 1.0 if too far off."""
    r = min((target / (bpm * k) for k in (0.5, 1, 2)), key=lambda r: abs(math.log(r)))
    return r if abs(r - 1) <= MAX_STRETCH else 1.0


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


def render(segments, out_path, progress=print):
    """segments: list of (path, start_sec, end_sec, fade_sec). A segment's fade is its outgoing overlap
    with the next one (or its fade-out to silence if last). Writes an MP3 to out_path."""
    done, cur, cur_beats, cur_fade, target = [], None, None, None, None
    for i, (path, start, end, fade) in enumerate(segments, 1):
        progress(f"Analyzing beats {i}/{len(segments)}...")
        bpm, beats = analyze(path, start, end)
        if target is None:
            r, target = 1.0, bpm
        else:
            r = stretch_ratio(bpm, target)
            target = target if r != 1.0 else bpm
        b = normalize(decode(path, start, end, tempo=r))
        b_beats = beats / r
        if cur is None:
            cur, cur_beats, cur_fade = b, b_beats, fade / r
            continue

        cur_len, b_len = len(cur) / SR, len(b) / SR
        overlap = min(cur_fade, cur_len / 2, b_len / 2)
        note = f", fade shortened to {overlap:.1f} s to fit" if overlap < cur_fade - 0.05 else ""
        progress(f"Rendering transition {i - 1} -> {i} ({bpm:.0f} -> {target:.0f} BPM{note})...")
        s = cur_len - overlap
        if len(cur_beats) and len(b_beats):
            # land b's first beat on the nearest outgoing beat
            nearest = cur_beats[np.argmin(np.abs(cur_beats - (s + b_beats[0])))]
            s = nearest - b_beats[0]
        # keep transitions from colliding with the previous one or running past b's midpoint
        s = min(max(s, cur_len / 2, cur_len - b_len / 2), cur_len - 0.05)
        cut = int(s * SR)
        n = len(cur) - cut
        done.append(cur[:cut])
        cur = np.concatenate([crossfade(cur[cut:], b[:n], 60 / target), b[n:]])
        cur_beats, cur_fade = b_beats, fade / r

    n = min(int(cur_fade * SR), len(cur))  # last track fades out to silence
    cur[-n:] *= np.cos(np.linspace(0, np.pi / 2, n, dtype=np.float32))[:, None]
    done.append(cur)
    mix = np.concatenate(done)
    mix /= max(1.0, float(np.abs(mix).max()))  # ponytail: peak normalize, swap for a limiter if quiet mixes bother you
    progress("Encoding MP3...")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                    "-b:a", BITRATE, str(out_path)], input=mix.astype("<f4").tobytes(), check=True)
    return len(mix) / SR

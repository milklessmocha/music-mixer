"""Transition suggestions for two tracks (where to leave track 1, where to enter track 2, how long to fade) and
an overview of several tracks with the order that mixes best. Pure measurement (beat grid, phrases, energy,
key, tempo), no trained model, so it runs on a Pi."""
import itertools
import math

import numpy as np
from scipy.signal import stft

import mixer

SR = 11025            # analysis rate: enough for kick, bass and chroma, a quarter of the data of 44.1 kHz
HOP = 512             # ~46 ms frames
FADE_BARS = (8, 16, 32)
TOP = 5
TEMPO_LIMIT = 0.08    # tempo score is 0 at 8% apart or more
WEIGHTS = {"phrase": 30, "energy": 25, "key": 25, "tempo": 15, "length": 5}
MIN_SEC, MIN_FADE = 6, 3  # same rules as bot.parse_period, so every suggestion is valid /mix input

FREQS = np.fft.rfftfreq(2048, 1 / SR)
_pitch = np.round(12 * np.log2(np.maximum(FREQS, 1) / 440) + 9).astype(int) % 12  # 0 = C
CHROMA = np.array([(_pitch == pc) & (FREQS >= 55) & (FREQS <= 4000) for pc in range(12)], np.float32)
_edges = np.geomspace(50, 5000, 13)
BANDS = np.array([(FREQS >= lo) & (FREQS < hi) for lo, hi in zip(_edges[:-1], _edges[1:])], np.float32)

# Krumhansl-Kessler key profiles, index 0 = tonic
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def camelot(chroma):
    """Key of a 12-bin chroma (0 = C) as a Camelot code like '8B' (C major) or '8A' (A minor)."""
    scores = [(np.corrcoef(chroma, np.roll(prof, tonic))[0, 1], tonic, minor)
              for minor, prof in ((False, MAJOR), (True, MINOR)) for tonic in range(12)]
    _, tonic, minor = max(scores)
    major_tonic = (tonic + 3) % 12 if minor else tonic  # a minor key shares its number with its relative major
    return f"{(7 * major_tonic + 7) % 12 + 1}{'A' if minor else 'B'}"


def key_match(a, b):
    """1 for the same, adjacent or relative key on the Camelot wheel, 0.5 for two steps, else 0."""
    (na, la), (nb, lb) = (int(a[:-1]), a[-1]), (int(b[:-1]), b[-1])
    step = min((na - nb) % 12, (nb - na) % 12)
    if (step == 0) or (step == 1 and la == lb):
        return 1.0
    return 0.5 if step == 2 and la == lb else 0.0


def novelty(features, half=4):
    """Foote novelty on the bar self-similarity matrix: high where the music changes at the start of a bar."""
    x = (features - features.mean(0)) / (features.std(0) + 1e-9)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    sim = np.pad(x @ x.T, half, mode="edge")
    kernel = np.ones((2 * half, 2 * half))
    kernel[:half, half:] = kernel[half:, :half] = -1
    nov = np.array([(kernel * sim[i:i + 2 * half, i:i + 2 * half]).sum() for i in range(len(x))])
    nov = np.maximum(nov, 0) / (nov.max() + 1e-9)
    nov[0] = 1.0  # the very start of a song is always a clean entry point
    return nov


def profile(path, duration):
    """Bars, per-bar energy, phrase starts, novelty, tempo and key of one track."""
    y = mixer.decode(path, 0, None, sr=SR, channels=1)
    _, t, z = stft(y, fs=SR, nperseg=2048, noverlap=2048 - HOP)
    power = np.abs(z).astype(np.float32) ** 2
    frame = lambda times: np.clip(np.searchsorted(t, times), 0, len(t) - 1)

    # ponytail: one steady tempo per track, measured in the middle; tracks that speed up or slow down drift
    ws = max(0.0, min(duration / 2 - 30, duration - 60))
    bpm, beats = mixer.analyze(path, ws, min(duration, ws + 60))
    period = 60 / bpm
    grid = (ws + beats[0]) % period + period * np.arange(int(duration / period) + 1)
    grid = grid[grid < duration]
    # downbeat: of the 4 beat phases, the one where the low end hits hardest
    low = power[FREQS < 150].sum(axis=0)
    hits = np.maximum(0, np.diff(low, prepend=low[0]))
    bars = grid[max(range(4), key=lambda k: hits[frame(grid[k::4])].mean() if len(grid[k::4]) else 0)::4]

    chroma, bands = CHROMA @ np.sqrt(power), np.log(BANDS @ power + 1e-6)
    edges = frame(np.append(bars, duration))
    spans = [slice(lo, max(hi, lo + 1)) for lo, hi in zip(edges[:-1], edges[1:])]
    energy = np.array([math.sqrt(power[:, s].sum(axis=0).mean()) for s in spans])
    bar_chroma = np.array([chroma[:, s].mean(axis=1) for s in spans])
    bar_chroma /= bar_chroma.sum(axis=1, keepdims=True) + 1e-9
    nov = novelty(np.hstack([bar_chroma, np.array([bands[:, s].mean(axis=1) for s in spans])]))
    # phrases: every 8 bars, lined up with the biggest section change
    anchor = int(np.argmax(nov[1:])) + 1 if len(nov) > 1 else 0
    phrases = sorted(set(range(anchor % 8, len(bars), 8)) | {0})
    return {"bpm": bpm, "bars": bars, "energy": energy, "nov": nov, "phrases": phrases,
            "key": camelot(chroma.sum(axis=1)), "duration": duration}


def clock(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def rank(a, b, top=TOP):
    """Score every (overlap start in a, entry point in b, fade length) and return the best distinct ones."""
    c = a["bpm"]
    n = mixer.match_octave(b["bpm"], c)
    key = key_match(a["key"], b["key"])
    tempo = max(0.0, 1 - abs(math.log(c / n)) / math.log(1 + TEMPO_LIMIT))
    bar_a, bar_b = 240 / c, 240 / b["bpm"]
    med_a, med_b = np.median(a["energy"]), np.median(b["energy"])
    found = []
    for s in a["phrases"]:
        for n_bars in FADE_BARS:
            fade = n_bars * bar_a
            start = a["bars"][s]
            end = start + fade
            if end > a["duration"] + 60 / c:  # allow a beat of overshoot at the very end
                continue
            end = min(end, a["duration"])
            end_r, fade_r = int(end), round(fade)
            if fade_r < MIN_FADE or end_r < MIN_SEC or 2 * fade_r > end_r:
                continue
            out_level = a["energy"][s:s + n_bars].mean() / med_a  # < 1: leaving in a quieter part
            for i in b["phrases"]:
                begin = b["bars"][i] if i else 0.0
                used = fade * c / n  # seconds of b played during the overlap
                if b["duration"] - begin < max(MIN_SEC, 2 * used):
                    continue
                covered = max(1, round(used / bar_b))
                in_level = b["energy"][i:i + covered].mean() / med_b
                after = b["energy"][i + covered:i + covered + 8]
                after_level = after.mean() / med_b if len(after) else 1.0
                terms = {
                    "phrase": (a["nov"][s] + b["nov"][i]) / 2,
                    "energy": (np.clip(1.5 - out_level, 0, 1)
                               + np.clip(1.5 - in_level, 0, 1) / 2 + np.clip(after_level - 0.5, 0, 1) / 2) / 2,
                    "key": key,
                    "tempo": tempo,
                    "length": (end / a["duration"] + (b["duration"] - begin) / b["duration"]) / 2,
                }
                score = sum(WEIGHTS[k] * v for k, v in terms.items())
                found.append((score, start, end, fade, begin, n_bars, out_level, in_level, a["nov"][s], b["nov"][i]))

    picked = []
    for cand in sorted(found, key=lambda x: -x[0]):
        score, start, end, fade, begin, n_bars, out_level, in_level, nov_a, nov_b = cand
        if any(abs(start - p[1]) < 8 * bar_a and abs(begin - p[4]) < 8 * bar_b for p in picked):
            continue  # same spot as a better pick, maybe with another fade
        picked.append(cand)
        if len(picked) == top:
            break

    out = []
    for score, start, end, fade, begin, n_bars, out_level, in_level, nov_a, nov_b in picked:
        leave = ("leaves in a quieter part" if out_level < 0.75 else
                 "leaves at a section change" if nov_a >= 0.6 else "leaves mid-section")
        enter = ("enters at the very start" if begin == 0 else "enters on a quiet part" if in_level < 0.75 else
                 "enters at a section change" if nov_b >= 0.6 else "enters mid-section")
        clash = " (keys clash)" if key == 0 else ""
        out.append({
            "score": round(score), "end": float(end), "fade": float(fade), "begin": float(begin),
            "a": f"start - {clock(end)} - {clock(round(fade))}",
            "b": f"{'start' if int(begin) == 0 else clock(begin)} - end",
            "why": f"{c:.0f} -> {n:.0f} BPM, {a['key']} -> {b['key']}{clash}, {n_bars} bars; {leave}, {enter}",
        })
    return out


def suggest(path_a, duration_a, path_b, duration_b, progress=print):
    progress("Analyzing track 1...")
    a = profile(path_a, duration_a)
    progress("Analyzing track 2...")
    b = profile(path_b, duration_b)
    progress("Scoring transitions...")
    return rank(a, b)


def edges(p):
    """How loud the first and last 8 bars are next to the track's median: below ~0.75 is a quiet intro/outro."""
    e, med = p["energy"], np.median(p["energy"]) + 1e-9
    return e[:8].mean() / med, e[-8:].mean() / med


def pair(a, b):
    """How well b follows a, 0-100: key 40, tempo 40, a quiet outro into a quiet intro 20."""
    n = mixer.match_octave(b["bpm"], a["bpm"])
    tempo = max(0.0, 1 - abs(math.log(a["bpm"] / n)) / math.log(1 + TEMPO_LIMIT))
    quiet = (np.clip(1.5 - edges(a)[1], 0, 1) + np.clip(1.5 - edges(b)[0], 0, 1)) / 2
    return 40 * key_match(a["key"], b["key"]) + 40 * tempo + 20 * quiet


def best_order(profiles):
    """Order with the best total pair score. Brute force, fine for the 8 tracks the bot allows (40320 orders)."""
    n = len(profiles)
    score = [[pair(a, b) for b in profiles] for a in profiles]
    return max(itertools.permutations(range(n)), key=lambda o: sum(score[i][j] for i, j in zip(o, o[1:])))


def overview(paths, durations, names, progress=print):
    """Text report: one line per track, then the recommended order with how each neighbour pair fits."""
    profiles = []
    for i, (path, duration) in enumerate(zip(paths, durations), 1):
        progress(f"Analyzing track {i}/{len(paths)}...")
        profiles.append(profile(path, duration))

    lines = []
    for i, (p, name) in enumerate(zip(profiles, names), 1):
        intro, outro = edges(p)
        shape = ", ".join(x for x in ("quiet intro" if intro < 0.75 else "", "quiet outro" if outro < 0.75 else "") if x)
        lines.append(f"{i}. {name[:60]}\n   {clock(p['duration'])}, {p['bpm']:.0f} BPM, key {p['key']}"
                     + (f", {shape}" if shape else ""))
    if len(profiles) < 2:
        return "\n".join(lines)

    order = best_order(profiles)
    lines.append("\nBest order: " + " -> ".join(str(i + 1) for i in order))
    for i, j in zip(order, order[1:]):
        a, b = profiles[i], profiles[j]
        n = mixer.match_octave(b["bpm"], a["bpm"])
        keys = {1.0: "keys match", 0.5: "keys close", 0.0: "keys clash"}[key_match(a["key"], b["key"])]
        lines.append(f"   {i + 1} -> {j + 1}: {pair(a, b):.0f}/100, {a['key']} -> {b['key']} {keys}, "
                     f"{a['bpm']:.0f} -> {n:.0f} BPM ({(n / a['bpm'] - 1) * 100:+.1f}%)")
    lines.append("\nUse /suggest on a pair for exact mix points.")
    return "\n".join(lines)

# music-mixer

Telegram bot that turns a list of tracks into one DJ-style mix. Built to run on a Raspberry Pi 5.

## Deploy on the Pi

On a Raspberry Pi with 64-bit Raspberry Pi OS:

```bash
git clone https://github.com/milklessmocha/music-mixer.git
cd music-mixer
cp .env.example .env
nano .env
./deploy/install.sh
```

In `.env`, set `BOT_TOKEN` (from @BotFather) and `ALLOWED_USERS` (your id, from @userinfobot). `./deploy/install.sh` installs Docker if it is missing, starts the bot in Docker, and sets up a timer that pulls `main` and rebuilds within 5 minutes of every push. Running it again is safe.

| What | Command on the Pi |
| --- | --- |
| Bot logs | `docker compose logs -f` (inside the repo) |
| Deploy history | `journalctl -u music-mixer-deploy` |
| Deploy now | `./deploy/deploy.sh --force` |
| Stop auto-deploy | `sudo systemctl disable --now music-mixer-deploy.timer` |
| Stop the bot | `docker compose down` |

A deploy that would overwrite changes made by hand on the Pi refuses and shows up as failed in the deploy history.

## Setup by hand

Copy `.env.example` to `.env` next to `docker-compose.yml` and fill it in:

```
BOT_TOKEN=123456789:AAH...
ALLOWED_USERS=11111111,22222222
```

`ALLOWED_USERS` is required: the bot downloads any link it is sent, so only listed Telegram user ids can use it.

Run it (restarts on crash and on boot):

```bash
docker compose up -d --build
docker compose logs -f
```

Update yt-dlp when links stop working: `docker compose build --no-cache && docker compose up -d`.

Local run without Docker (needs ffmpeg):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
set -a && source .env && set +a && .venv/bin/python bot.py
```

## Usage

1. `/mix` starts a session.
2. Send an audio file or a link (6 s to 10 min).
3. Send the part to use: `beginning - end` or `beginning - end - fade`.
   - Times are `mm:ss` (`1:2` and `01:02` are the same). `start` and `end` mean the track edges.
   - `fade` is how long this track overlaps the next one (or fades out, if last). At least `00:03`, and it must fit twice into the part.
   - Without a fade: 5% of the shorter of this part and the next one, rounded to whole seconds, at least 3 s.
4. Repeat 2 and 3, then `/done`.
5. Send a name for the mix, without extension.
6. Tick the formats you want (mp3, m4a, ogg, flac, wav, or All extensions) and tap Submit. mp3 and m4a arrive as playable audio, the rest as files.

`/cancel` drops the session at any step.

`/download` does the same for a single whole track: send a link, a name, then pick formats. The result is identical to `/mix` with that link, `start - end` and `/done`.

`/suggest` takes two tracks (files or links) and replies with the 5 best ways to mix them, each written as `/mix` input, e.g. `Track 1: start - 03:12 - 00:30 | Track 2: start - end`. Tap a number to mix that one (name, then formats), or copy the lines into `/mix` yourself.

`/analyze` takes 2 to 8 tracks, then `/done`. It lists each track's length, BPM, Camelot key and whether it has a quiet intro or outro, then tries every order and shows the one where neighbours fit best (keys 40, tempo 40, quiet outro into quiet intro 20), with a score per pair. Use `/suggest` on a pair from that order for exact mix points.

The mix and all downloaded files are deleted once the mix is sent.

## How a transition works (`mixer.py`)

- Tempo and a beat grid are measured on 15 s of each track around the transition: the outgoing track at `end - fade`, the incoming one at `beginning + fade`.
- During the overlap both tracks follow one beat grid whose tempo ramps from the outgoing BPM to the incoming BPM (time-stretch with pitch kept). Outside overlaps every track plays at its own tempo, so the overlap lasts exactly `fade` only when both BPMs are equal.
- The overlap starts on an outgoing beat, and the incoming track starts on its first beat.
- During the overlap, mids/highs crossfade with an equal-power curve while the bass (below 200 Hz) swaps over one beat at the midpoint, so two kick drums never play at once.

Tuning knobs are at the top of `mixer.py`.

## How suggestions are picked (`suggest.py`)

No trained model, only measurements, so it runs on the Pi in seconds:

- Beat grid and bars (the downbeat is the beat where the low end hits hardest), phrases every 8 bars lined up with the biggest section change.
- Section changes from a self-similarity matrix of per-bar chroma and timbre.
- Key from chroma against the Krumhansl profiles, shown as a Camelot code (8A, 9B, ...).
- Every phrase start in track 1 × every phrase start in track 2 × fades of 8, 16 and 32 bars is scored: section changes at both cut points (30), leaving in a quieter part and entering on a quieter intro before a louder part (25), compatible keys (25), close tempos (15), keeping most of both songs (5).

## Limits

- Telegram bots can only download files up to 20 MB (send a link for larger ones) and upload up to 50 MB. A format over 50 MB is skipped and reported: about 35 min for mp3/m4a/ogg, 10 min for flac, 5 min for wav.
- Beats are aligned, bars and phrases are not (no downbeat detection).
- Tempo is assumed steady within the 15 s measuring window.
- One mix renders at a time; sessions live in memory and are lost on restart.

## Test

```bash
.venv/bin/python test_mixer.py
```

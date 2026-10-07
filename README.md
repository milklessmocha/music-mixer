# music-mixer

Telegram bot that turns a list of tracks into one DJ-style MP3 mix. Built to run on a Raspberry Pi 5.

## Setup

Create `.env` next to `docker-compose.yml`:

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
4. Repeat 2 and 3, then `/done`. `/cancel` drops the session.

The mix and all downloaded files are deleted once the mix is sent.

## How a transition works (`mixer.py`)

- Tempo and a beat grid are measured on 15 s of each track around the transition: the outgoing track at `end - fade`, the incoming one at `beginning + fade`.
- During the overlap both tracks follow one beat grid whose tempo ramps from the outgoing BPM to the incoming BPM (time-stretch with pitch kept). Outside overlaps every track plays at its own tempo, so the overlap lasts exactly `fade` only when both BPMs are equal.
- The overlap starts on an outgoing beat, and the incoming track starts on its first beat.
- During the overlap, mids/highs crossfade with an equal-power curve while the bass (below 200 Hz) swaps over one beat at the midpoint, so two kick drums never play at once.

Tuning knobs are at the top of `mixer.py`.

## Limits

- Telegram bots can only download files up to 20 MB and upload up to 50 MB (about 35 min at 192 kbps). Send a link for larger files.
- Beats are aligned, bars and phrases are not (no downbeat detection).
- Tempo is assumed steady within the 15 s measuring window.
- One mix renders at a time; sessions live in memory and are lost on restart.

## Test

```bash
.venv/bin/python test_mixer.py
```

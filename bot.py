"""Telegram front end: /mix -> audio -> time period -> audio -> ... -> /done -> MP3."""
import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yt_dlp
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile, Message

import mixer

MIN_SEC, MAX_SEC = 6, 600
MIN_FADE = 3
TG_DOWNLOAD_LIMIT = 20 * 1024 * 1024  # Bot API getFile cap
URL_DOWNLOAD_LIMIT = 100 * 1024 * 1024

PERIOD_HELP = ("Send the part to use as  beginning - end  or  beginning - end - fade\n"
               "Times are mm:ss (1:2 and 01:02 both work). Use start / end for the track edges.\n"
               "Examples:  start - end    0:54 - 2:13    1:00 - end - 0:08\n"
               "Fade is optional, at least 00:03, and must fit twice into the part.")

dp = Dispatcher()
render_lock = asyncio.Lock()  # ponytail: one render at a time keeps the Pi responsive; a real queue if users pile up


class MixFlow(StatesGroup):
    waiting_audio = State()
    waiting_period = State()


def fmt(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def parse_time(token, keyword=None, keyword_value=None):
    token = token.strip().lower()
    if keyword and token == keyword:
        return keyword_value
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2})", token)
    if not m or int(m[2]) >= 60:
        raise ValueError(f'"{token}" is not a valid time, use mm:ss')
    return int(m[1]) * 60 + int(m[2])


def parse_period(text, duration):
    """'beginning - end [- fade]' -> (begin_sec, end_sec, fade_sec or None). Raises ValueError with a reason."""
    parts = text.split("-")
    if len(parts) not in (2, 3):
        raise ValueError("Use the format  beginning - end  or  beginning - end - fade")
    begin = parse_time(parts[0], "start", 0)
    end = parse_time(parts[1], "end", duration)
    fade = parse_time(parts[2]) if len(parts) == 3 else None
    if end > duration:
        raise ValueError(f"End {fmt(end)} is past the track length {fmt(duration)}.")
    if begin >= end:
        raise ValueError("Beginning must be before end.")
    if end - begin < MIN_SEC:
        raise ValueError(f"The part must be at least {MIN_SEC} seconds long.")
    if fade is not None and fade < MIN_FADE:
        raise ValueError(f"Fade must be at least {fmt(MIN_FADE)}.")
    if fade is not None and 2 * fade > end - begin:
        raise ValueError(f"Fade must fit twice into the part, so at most {fmt((end - begin) / 2)} here.")
    return begin, end, fade


def default_fade(cur_len, next_len=None):
    """5% of the shorter of the two parts, whole seconds rounded half up, at least MIN_FADE."""
    pct = lambda x: int(x / 20 + 0.5)
    return max(MIN_FADE, min(pct(cur_len), pct(next_len if next_len is not None else cur_len)))


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=codec_type:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True)
    info = json.loads(out.stdout or "{}")
    if not info.get("streams"):
        raise ValueError("That file has no audio I can read.")
    return float(info["format"]["duration"])


def fetch_url(url, folder, idx):
    opts = {"format": "bestaudio/best", "noplaylist": True, "quiet": True, "no_warnings": True,
            "outtmpl": str(folder / f"{idx}-url.%(ext)s"), "max_filesize": URL_DOWNLOAD_LIMIT}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if "entries" in info:
            raise ValueError("Playlists are not supported, send a single track.")
        if (info.get("duration") or 0) > MAX_SEC:
            raise ValueError("That audio is longer than 10 minutes.")
        ydl.process_ie_result(info, download=True)
    files = list(folder.glob(f"{idx}-url.*"))
    if not files:
        raise ValueError("Download produced no file (too large?).")
    return files[0]


async def edit(msg, text):
    with contextlib.suppress(TelegramBadRequest):  # "message is not modified"
        await msg.edit_text(text)


async def drop_session(state):
    folder = (await state.get_data()).get("dir")
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
    await state.clear()


@dp.message(Command("start", "help"))
async def cmd_start(msg: Message):
    await msg.answer("Send /mix to start a mix. Then send audios or links one by one (6 s to 10 min each), "
                     "give each a time period, and send /done to render. /cancel drops the session.")


@dp.message(Command("mix"))
async def cmd_mix(msg: Message, state: FSMContext):
    await drop_session(state)
    await state.set_state(MixFlow.waiting_audio)
    await state.update_data(dir=tempfile.mkdtemp(prefix="mix-"), tracks=[], pending=None)
    await msg.answer("Send the first audio file or link (6 s to 10 min).")


@dp.message(Command("cancel"))
async def cmd_cancel(msg: Message, state: FSMContext):
    await drop_session(state)
    await msg.answer("Cancelled. Send /mix to start again.")


@dp.message(Command("done"), MixFlow.waiting_audio)
async def cmd_done(msg: Message, state: FSMContext, bot: Bot):
    await finish(msg, state, bot)


@dp.message(MixFlow.waiting_audio, F.audio | F.voice | F.document)
async def got_file(msg: Message, state: FSMContext, bot: Bot):
    media = msg.audio or msg.voice or msg.document
    if (getattr(media, "duration", None) or 0) > MAX_SEC:
        return await msg.answer("That audio is longer than 10 minutes. Send another one.")
    if (media.file_size or 0) > TG_DOWNLOAD_LIMIT:
        return await msg.answer("Telegram lets bots download files up to 20 MB only. Send a smaller file or a link.")
    data = await state.get_data()
    path = Path(data["dir"]) / f"{len(data['tracks'])}-{media.file_unique_id}"
    # ponytail: a second audio sent mid-download replaces the pending one; add a "downloading" state if that bites
    await bot.download(media, destination=path, timeout=300)
    await accept_audio(msg, state, path)


@dp.message(MixFlow.waiting_audio, F.text.regexp(r"^\s*https?://\S+\s*$"))
async def got_url(msg: Message, state: FSMContext):
    data = await state.get_data()
    status = await msg.answer("Downloading...")
    try:
        path = await asyncio.to_thread(fetch_url, msg.text.strip(), Path(data["dir"]), len(data["tracks"]))
    except Exception as e:
        return await edit(status, f"Couldn't download that: {e}\nSend another file or link.")
    await status.delete()
    await accept_audio(msg, state, path)


async def accept_audio(msg, state, path):
    try:
        duration = probe_duration(path)
    except (ValueError, KeyError) as e:
        path.unlink(missing_ok=True)
        return await msg.answer(f"{e} Send another file or link.")
    if not MIN_SEC <= duration <= MAX_SEC:
        path.unlink(missing_ok=True)
        return await msg.answer(f"That audio is {fmt(duration)}. It must be between 00:06 and 10:00. Send another one.")
    await state.update_data(pending={"path": str(path), "duration": duration})
    await state.set_state(MixFlow.waiting_period)
    await msg.answer(f"Got it, length {fmt(duration)}.\n{PERIOD_HELP}")


@dp.message(MixFlow.waiting_period, F.text)
async def got_period(msg: Message, state: FSMContext):
    data = await state.get_data()
    pending = data["pending"]
    try:
        begin, end, fade = parse_period(msg.text, pending["duration"])
    except ValueError as e:
        return await msg.answer(f"{e}\n\n{PERIOD_HELP}")
    tracks = data["tracks"] + [{**pending, "begin": begin, "end": end, "fade": fade}]
    await state.update_data(tracks=tracks, pending=None)
    await state.set_state(MixFlow.waiting_audio)
    await msg.answer(f"Track {len(tracks)} added ({fmt(begin)} - {fmt(end)}). "
                     "Send the next audio or link, or /done to render the mix.")


async def finish(msg, state, bot):
    """Render and send the session's mix. Kept separate so a future inline 'Render mix' button can call it."""
    data = await state.get_data()
    tracks = data["tracks"]
    if not tracks:
        return await msg.answer("Send at least one audio first.")
    await state.clear()
    lens = [t["end"] - t["begin"] for t in tracks]
    segments = [(t["path"], t["begin"], t["end"],
                 t["fade"] or default_fade(lens[i], lens[i + 1] if i + 1 < len(lens) else None))
                for i, t in enumerate(tracks)]

    status = await msg.answer("Waiting for another mix to finish..." if render_lock.locked() else "Starting...")
    loop = asyncio.get_running_loop()
    progress = lambda text: asyncio.run_coroutine_threadsafe(edit(status, text), loop)
    try:
        async with render_lock:
            out = Path(data["dir"]) / "mix.mp3"
            seconds = await asyncio.to_thread(mixer.render, segments, out, progress)
            await edit(status, "Sending...")
            await bot.send_audio(msg.chat.id, FSInputFile(out, filename="mix.mp3"), title="DJ mix",
                                 duration=int(seconds), request_timeout=600)
        await edit(status, "Done!")
    except Exception as e:
        logging.exception("mix failed")
        await edit(status, f"Mix failed: {e}\nSend /mix to try again.")
    finally:
        shutil.rmtree(data["dir"], ignore_errors=True)  # downloads and the sent MP3


@dp.message(MixFlow.waiting_audio)
async def expect_audio(msg: Message):
    await msg.answer("Send an audio file or an http(s) link, /done to render, or /cancel.")


@dp.message(MixFlow.waiting_period)
async def expect_period(msg: Message):
    await msg.answer(PERIOD_HELP)


@dp.message()
async def no_session(msg: Message):
    await msg.answer("Send /mix to start a mix.")


async def main():
    logging.basicConfig(level=logging.INFO)
    allowed = {int(x) for x in os.environ["ALLOWED_USERS"].split(",") if x.strip()}
    dp.message.filter(F.from_user.id.in_(allowed))  # URLs make the Pi fetch things, so keep strangers out
    await dp.start_polling(Bot(os.environ["BOT_TOKEN"]))


if __name__ == "__main__":
    asyncio.run(main())

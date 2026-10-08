"""Telegram front end: /mix -> audio -> time period -> ... -> /done -> name -> formats -> files.
/download is a one-track /mix of the whole track: link -> name -> formats -> files.
/suggest takes two tracks, lists ranked transitions, and a tapped one continues as /mix: name -> formats.
/analyze takes 2-8 tracks and reports tempo, key and intro/outro of each, plus the order that mixes best."""
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
from aiogram.filters import Command, or_f
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton,
                           Message, ReplyKeyboardMarkup)

import mixer
import suggest

MIN_SEC, MAX_SEC = 6, 600
MIN_FADE = 3
TG_DOWNLOAD_LIMIT = 20 * 1024 * 1024  # Bot API getFile cap
URL_DOWNLOAD_LIMIT = 100 * 1024 * 1024
TG_UPLOAD_LIMIT = 50 * 1024 * 1024  # Bot API sendDocument/sendAudio cap
SESSION_TIMEOUT = 600  # seconds without any action before a /mix session and its files are dropped
MAX_ANALYZE = 8  # tracks per /analyze; every order is tried, 8! = 40320
PLAYABLE = {"mp3", "m4a"}  # Telegram's audio player takes these; the rest go out as files

PERIOD_HELP = ("Send the part to use as  beginning - end  or  beginning - end - fade\n"
               "Times are mm:ss (1:2 and 01:02 both work). Use start / end for the track edges.\n"
               "Examples:  start - end    0:54 - 2:13    1:00 - end - 0:08\n"
               "Fade is optional, at least 00:03, and must fit twice into the part.")

def label(command):
    """Deck button text for a command: '/done' -> 'Done', '/three-word-command' -> 'Three word command'."""
    return command.lstrip("/").replace("-", " ").replace("_", " ").capitalize()


def cmd(*names):
    """Matches a typed /command or the deck button for it (a button sends its own text, e.g. 'Done')."""
    return or_f(Command(*names), F.text.in_({label(n) for n in names}))


def deck_button(command, style):
    return KeyboardButton(text=label(command), style=style)


# always-visible command buttons under the input field; colours show on recent Telegram apps
DECK = ReplyKeyboardMarkup(is_persistent=True, resize_keyboard=True, keyboard=[
    [deck_button("/mix", "primary"), deck_button("/download", "primary"),
     deck_button("/suggest", "primary"), deck_button("/analyze", "primary")],
    [deck_button("/done", "success"), deck_button("/cancel", "danger")],
])

dp = Dispatcher()
render_lock = asyncio.Lock()  # ponytail: one render at a time keeps the Pi responsive; a real queue if users pile up
timers = {}  # (chat id, user id) -> task that drops the session after SESSION_TIMEOUT idle seconds


class MixFlow(StatesGroup):
    waiting_audio = State()
    waiting_period = State()
    waiting_name = State()
    choosing_formats = State()
    choosing_suggestion = State()


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


def parse_name(text):
    """Mix name as typed, without extension. Raises ValueError with a reason."""
    name = text.strip()
    if not 1 <= len(name) <= 64:
        raise ValueError("The name must be 1 to 64 characters.")
    if re.search(r'[\\/:*?"<>|\x00-\x1f]', name) or name.startswith("."):
        raise ValueError('The name can\'t start with a dot or contain / \\ : * ? " < > |')
    if re.search(r"\.[A-Za-z][A-Za-z0-9]{1,3}$", name):
        raise ValueError("Send the name without an extension, you pick the formats next.")
    return name


def formats_keyboard(selected, available=tuple(mixer.FORMATS)):
    """Checklist: tap a format to tick or untick it, 'All extensions' toggles every one, Submit renders."""
    def button(on, label, key):
        return InlineKeyboardButton(text=("✅ " if on else "⬜ ") + label, callback_data=f"fmt:{key}")

    formats = [button(ext in selected, ext, ext) for ext in available]
    rows = [formats[i:i + 3] for i in range(0, len(formats), 3)]
    rows.append([button(set(selected) == set(available), "All extensions", "all")])
    rows.append([InlineKeyboardButton(text="Submit", callback_data="fmt:go", style="success")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


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
    return files[0], info.get("title")


async def edit(msg, text):
    with contextlib.suppress(TelegramBadRequest):  # "message is not modified"
        await msg.edit_text(text)


async def drop_session(state):
    folder = (await state.get_data()).get("dir")
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
    await state.clear()


async def expire(state, bot, chat_id):
    await asyncio.sleep(SESSION_TIMEOUT)
    await drop_session(state)
    await bot.send_message(chat_id, "The mix session expired after 10 minutes without activity and its files "
                                    "were deleted. Send /mix to start again.", reply_markup=DECK)


async def session_timer(handler, event, data):
    """Every message or button press restarts the idle timer of an open session."""
    if "event_chat" not in data or "state" not in data:  # no chat means no session to time
        return await handler(event, data)
    key = (data["event_chat"].id, data["event_from_user"].id)
    if task := timers.pop(key, None):
        task.cancel()
    try:
        return await handler(event, data)
    finally:
        if await data["state"].get_state() is not None:
            timers[key] = asyncio.create_task(expire(data["state"], data["bot"], key[0]))


dp.message.outer_middleware(session_timer)
dp.callback_query.outer_middleware(session_timer)


@dp.message(Command("start", "help"))
async def cmd_start(msg: Message):
    await msg.answer("Send /mix to start a mix. Then send audios or links one by one (6 s to 10 min each), "
                     "give each a time period, and send /done to render.\n"
                     "Send /download to get one whole track from a link in the format you pick.\n"
                     "Send /suggest with two tracks to get a ranked list of ways to mix them.\n"
                     "Send /analyze with up to 8 tracks to see their tempo and key and the best order to mix them.\n"
                     "/cancel drops the session.", reply_markup=DECK)


@dp.message(cmd("mix"))
async def cmd_mix(msg: Message, state: FSMContext):
    await drop_session(state)
    await state.set_state(MixFlow.waiting_audio)
    await state.update_data(dir=tempfile.mkdtemp(prefix="mix-"), tracks=[], pending=None)
    await msg.answer("Send the first audio file or link (6 s to 10 min).", reply_markup=DECK)


@dp.message(cmd("download"))
async def cmd_download(msg: Message, state: FSMContext):
    """Same as /mix with one link, period 'start - end', then /done."""
    await drop_session(state)
    await state.set_state(MixFlow.waiting_audio)
    await state.update_data(dir=tempfile.mkdtemp(prefix="mix-"), tracks=[], pending=None, single=True)
    await msg.answer("Send the link (6 s to 10 min).", reply_markup=DECK)


@dp.message(cmd("suggest"))
async def cmd_suggest(msg: Message, state: FSMContext):
    await drop_session(state)
    await state.set_state(MixFlow.waiting_audio)
    await state.update_data(dir=tempfile.mkdtemp(prefix="mix-"), tracks=[], pending=None, suggest=True)
    await msg.answer("Send the first audio file or link (6 s to 10 min).", reply_markup=DECK)


@dp.message(cmd("analyze"))
async def cmd_analyze(msg: Message, state: FSMContext):
    await drop_session(state)
    await state.set_state(MixFlow.waiting_audio)
    await state.update_data(dir=tempfile.mkdtemp(prefix="mix-"), tracks=[], pending=None, analyze=True)
    await msg.answer(f"Send the audio files or links to compare, one by one (up to {MAX_ANALYZE}), then /done.",
                     reply_markup=DECK)


@dp.message(cmd("cancel"))
async def cmd_cancel(msg: Message, state: FSMContext):
    await drop_session(state)
    await msg.answer("Cancelled. Send /mix to start again.", reply_markup=DECK)


@dp.message(cmd("done"), MixFlow.waiting_audio)
async def cmd_done(msg: Message, state: FSMContext):
    data = await state.get_data()
    if data.get("suggest"):
        return await msg.answer("/suggest needs two tracks. Send the next audio file or link.")
    if data.get("analyze"):
        if not data["tracks"]:
            return await msg.answer("Send at least one audio first.")
        return await show_overview(msg, state, data["tracks"])
    if not data["tracks"]:
        return await msg.answer("Send at least one audio first.")
    await state.set_state(MixFlow.waiting_name)
    await msg.answer("Send a name for the mix, without extension.")


@dp.message(MixFlow.waiting_name, F.text)
async def got_name(msg: Message, state: FSMContext):
    try:
        name = parse_name(msg.text)
    except ValueError as e:
        return await msg.answer(f"{e} Send another name.")
    await state.update_data(name=name, formats=[])
    await state.set_state(MixFlow.choosing_formats)
    await msg.answer(f"Name: {name}\nPick the formats you want, then Submit.", reply_markup=formats_keyboard([]))


@dp.callback_query(MixFlow.choosing_formats, F.data.startswith("fmt:"))
async def got_format(cb: CallbackQuery, state: FSMContext, bot: Bot):
    data = await state.get_data()
    selected, key = set(data["formats"]), cb.data[4:]
    available = [ext for ext in mixer.FORMATS if ext not in data.get("blocked", [])]
    if key == "go":
        if not selected:
            return await cb.answer("Pick at least one format.", show_alert=True)
        await cb.answer()
        chosen = [ext for ext in mixer.FORMATS if ext in selected]
        await cb.message.edit_text(f"Name: {data['name']}\nFormats: {', '.join(chosen)}")
        return await finish(cb.message, state, bot)
    if key == "all":
        selected = set() if selected == set(available) else set(available)
    elif key in available:
        selected ^= {key}
    await state.update_data(formats=sorted(selected))
    await cb.answer()
    with contextlib.suppress(TelegramBadRequest):
        await cb.message.edit_reply_markup(reply_markup=formats_keyboard(selected, available))


@dp.callback_query(F.data.startswith("fmt:"))
async def stale_format(cb: CallbackQuery):
    await cb.answer("This menu has expired. Send /mix to start a new mix.", show_alert=True)


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
    await accept_audio(msg, state, path, getattr(media, "title", None) or getattr(media, "file_name", None))


@dp.message(MixFlow.waiting_audio, F.text.regexp(r"^\s*https?://\S+\s*$"))
async def got_url(msg: Message, state: FSMContext):
    data = await state.get_data()
    status = await msg.answer("Downloading...")
    try:
        path, title = await asyncio.to_thread(fetch_url, msg.text.strip(), Path(data["dir"]), len(data["tracks"]))
    except Exception as e:
        return await edit(status, f"Couldn't download that: {e}\nSend another file or link.")
    await status.delete()
    await accept_audio(msg, state, path, title)


async def accept_audio(msg, state, path, title=None):
    try:
        duration = probe_duration(path)
    except (ValueError, KeyError) as e:
        path.unlink(missing_ok=True)
        return await msg.answer(f"{e} Send another file or link.")
    if not MIN_SEC <= duration <= MAX_SEC:
        path.unlink(missing_ok=True)
        return await msg.answer(f"That audio is {fmt(duration)}. It must be between 00:06 and 10:00. Send another one.")
    data = await state.get_data()
    if data.get("analyze"):  # /analyze: whole tracks, report on /done or at the limit
        tracks = data["tracks"] + [{"path": str(path), "duration": duration,
                                    "name": title or f"Track {len(data['tracks']) + 1}"}]
        await state.update_data(tracks=tracks)
        if len(tracks) == MAX_ANALYZE:
            return await show_overview(msg, state, tracks)
        return await msg.answer(f"Got track {len(tracks)}, length {fmt(duration)}. "
                                f"Send another (up to {MAX_ANALYZE}), or /done to analyze.")
    if data.get("suggest"):  # /suggest: whole tracks, analyze once there are two
        tracks = data["tracks"] + [{"path": str(path), "duration": duration}]
        await state.update_data(tracks=tracks)
        if len(tracks) == 1:
            return await msg.answer(f"Got track 1, length {fmt(duration)}. Send the second audio file or link.")
        return await show_suggestions(msg, state, tracks)
    if data.get("single"):  # /download: whole track, straight to naming
        track = {"path": str(path), "duration": duration, "begin": 0, "end": duration, "fade": None}
        await state.update_data(tracks=[track])
        await state.set_state(MixFlow.waiting_name)
        return await msg.answer(f"Got it, length {fmt(duration)}.\nSend a name for the file, without extension.")
    await state.update_data(pending={"path": str(path), "duration": duration})
    await state.set_state(MixFlow.waiting_period)
    await msg.answer(f"Got it, length {fmt(duration)}.\n{PERIOD_HELP}")


async def show_overview(msg, state, tracks):
    await state.set_state(MixFlow.choosing_suggestion)  # ignore further audio while analyzing
    status = await msg.answer("Waiting for another job to finish..." if render_lock.locked() else "Analyzing...")
    loop = asyncio.get_running_loop()
    progress = lambda text: asyncio.run_coroutine_threadsafe(edit(status, text), loop)
    try:
        async with render_lock:
            report = await asyncio.to_thread(suggest.overview, [t["path"] for t in tracks],
                                             [t["duration"] for t in tracks], [t["name"] for t in tracks], progress)
    except Exception as e:
        logging.exception("analyze failed")
        report = f"Couldn't analyze those tracks: {e}\nSend /analyze to try again."
    finally:
        await drop_session(state)  # the report is all there is; nothing to keep
    await status.delete()
    await msg.answer(report, reply_markup=DECK)


async def show_suggestions(msg, state, tracks):
    await state.set_state(MixFlow.choosing_suggestion)  # a third audio sent mid-analysis isn't taken as a track
    status = await msg.answer("Waiting for another job to finish..." if render_lock.locked() else "Analyzing...")
    loop = asyncio.get_running_loop()
    progress = lambda text: asyncio.run_coroutine_threadsafe(edit(status, text), loop)
    a, b = tracks
    try:
        async with render_lock:
            found = await asyncio.to_thread(suggest.suggest, a["path"], a["duration"], b["path"], b["duration"],
                                            progress)
    except Exception as e:
        logging.exception("suggest failed")
        await drop_session(state)
        return await edit(status, f"Couldn't analyze those tracks: {e}\nSend /suggest to try again.")
    if not found:
        await drop_session(state)
        return await edit(status, "No transition fits these two tracks: they are too short for an 8-bar fade. "
                                  "Use /mix to set the times yourself.")
    await state.update_data(suggestions=found)
    lines = [f"{i}. {s['score']}/100   Track 1: {s['a']}   |   Track 2: {s['b']}\n{s['why']}"
             for i, s in enumerate(found, 1)]
    buttons = [InlineKeyboardButton(text=str(i), callback_data=f"sug:{i - 1}") for i in range(1, len(found) + 1)]
    await status.delete()
    await msg.answer("Ways to mix these two, best first. Tap a number to mix it:\n\n" + "\n\n".join(lines),
                     reply_markup=InlineKeyboardMarkup(inline_keyboard=[buttons]))


@dp.callback_query(MixFlow.choosing_suggestion, F.data.startswith("sug:"))
async def got_suggestion(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    i = int(cb.data[4:])
    pick, (a, b) = data["suggestions"][i], data["tracks"]
    # exact times, not the rounded ones shown, so the overlap stays on the phrase boundaries
    tracks = [{**a, "begin": 0, "end": pick["end"], "fade": pick["fade"]},
              {**b, "begin": pick["begin"], "end": b["duration"], "fade": None}]
    await state.update_data(tracks=tracks)
    await state.set_state(MixFlow.waiting_name)
    await cb.answer()
    with contextlib.suppress(TelegramBadRequest):
        await cb.message.edit_reply_markup(reply_markup=None)
    await cb.message.answer(f"Picked {i + 1}: Track 1: {pick['a']}  |  Track 2: {pick['b']}\n"
                            "Send a name for the mix, without extension.")


@dp.callback_query(F.data.startswith("sug:"))
async def stale_suggestion(cb: CallbackQuery):
    await cb.answer("This list has expired. Send /suggest to start again.", show_alert=True)


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
    """Render the session's mix once, encode it to each chosen format and send the files.
    If anything wasn't delivered, the session is kept and the picker comes back without the too-big formats."""
    data = await state.get_data()
    tracks, name = data["tracks"], data["name"]
    formats = [ext for ext in mixer.FORMATS if ext in data["formats"]]
    await state.clear()
    lens = [t["end"] - t["begin"] for t in tracks]
    segments = [(t["path"], t["begin"], t["end"],
                 t["fade"] or default_fade(lens[i], lens[i + 1] if i + 1 < len(lens) else None))
                for i, t in enumerate(tracks)]

    status = await msg.answer("Waiting for another mix to finish..." if render_lock.locked() else "Starting...")
    loop = asyncio.get_running_loop()
    progress = lambda text: asyncio.run_coroutine_threadsafe(edit(status, text), loop)
    too_big, failed = {}, None
    try:
        async with render_lock:
            mix = await asyncio.to_thread(mixer.render, segments, progress)
            for ext in formats:
                out = Path(data["dir"]) / f"mix.{ext}"
                await edit(status, f"Encoding {ext}...")
                await asyncio.to_thread(mixer.encode, mix, out)
                size = out.stat().st_size
                if size > TG_UPLOAD_LIMIT:
                    too_big[ext] = size
                    continue
                await edit(status, f"Sending {ext}...")
                file = FSInputFile(out, filename=f"{name}.{ext}")
                if ext in PLAYABLE:
                    await bot.send_audio(msg.chat.id, file, title=name, duration=int(len(mix) / mixer.SR),
                                         request_timeout=600)
                else:
                    await bot.send_document(msg.chat.id, file, request_timeout=600)
                out.unlink()  # sent, so drop it now rather than holding every format until the end
    except Exception as e:
        logging.exception("mix failed")
        failed = e

    problems = []
    if too_big:
        problems.append("Skipped, over Telegram's 50 MB limit: "
                        + ", ".join(f"{ext} ({size / 1024 / 1024:.0f} MB)" for ext, size in too_big.items()))
    if failed:
        problems.append(f"Failed: {failed}")
    if not problems:
        shutil.rmtree(data["dir"], ignore_errors=True)  # downloads; sent files are already gone
        return await edit(status, "Done!")

    blocked = sorted(set(data.get("blocked", [])) | set(too_big))
    available = [ext for ext in mixer.FORMATS if ext not in blocked]
    await edit(status, "\n".join(problems))
    # keep the session for another pick, unless every format is too big or a new /mix already replaced it
    if available and await state.get_state() is None:
        await state.set_state(MixFlow.choosing_formats)
        await state.set_data({**data, "formats": [], "blocked": blocked})
        await msg.answer(f"Name: {name}\nPick other formats, then Submit. /cancel drops the mix.",
                         reply_markup=formats_keyboard([], available))
    else:
        shutil.rmtree(data["dir"], ignore_errors=True)
        await msg.answer("Send /mix to start again.")


@dp.message(MixFlow.waiting_audio)
async def expect_audio(msg: Message):
    await msg.answer("Send an audio file or an http(s) link, /done to render, or /cancel.")


@dp.message(MixFlow.waiting_period)
async def expect_period(msg: Message):
    await msg.answer(PERIOD_HELP)


@dp.message(MixFlow.waiting_name)
async def expect_name(msg: Message):
    await msg.answer("Send a name for the mix as text, without extension.")


@dp.message(MixFlow.choosing_formats)
async def expect_formats(msg: Message):
    await msg.answer("Tick the formats with the buttons above, then tap Submit. /cancel drops the mix.")


@dp.message(MixFlow.choosing_suggestion)
async def expect_suggestion(msg: Message):
    await msg.answer("Wait for the analysis to finish, then tap a number under the list if there is one. "
                     "/cancel stops.")


@dp.message()
async def no_session(msg: Message):
    await msg.answer("Send /mix to start a mix.", reply_markup=DECK)


async def main():
    logging.basicConfig(level=logging.INFO)
    allowed = {int(x) for x in os.environ["ALLOWED_USERS"].split(",") if x.strip()}
    dp.message.filter(F.from_user.id.in_(allowed))  # URLs make the Pi fetch things, so keep strangers out
    dp.callback_query.filter(F.from_user.id.in_(allowed))
    await dp.start_polling(Bot(os.environ["BOT_TOKEN"]))


if __name__ == "__main__":
    asyncio.run(main())

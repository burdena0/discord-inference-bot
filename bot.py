"""
self-hosted discord chat bot served by a local llm through ollama.

replies when @mentioned, replied to, dm'd, or in channels with /autoreply on, streaming the reply by editing
its message as tokens arrive. retrieval over indexed channel history (nomic-embed-text + sqlite), a member
directory for getting names right, slash-command games and utilities, and per-reply latency metrics
(time to first token, decode speed) logged to metrics.csv and shown by /stats.
"""

import os
import re
import json
import time
import random
import base64
import asyncio
import logging
import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict, deque, namedtuple

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN"]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
DEFAULT_SYSTEM = os.getenv(
    "SYSTEM_PROMPT",
    "You're a friendly, witty assistant hanging out in this Discord. Keep replies short and "
    "conversational unless someone asks for detail. Use Discord markdown.",
)
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "1h")          # how long ollama keeps the model in vram
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))         # context window in tokens
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))   # recent messages remembered per channel
AUTO_RESET_MINUTES = int(os.getenv("AUTO_RESET_MINUTES", "20"))   # chat memory clears after this long without messages
MAX_IMAGES = 4
EMPTY_REPLY = "(no response)"
UNAVAILABLE = "Couldn't come up with anything for that one. Try again?"
# messages rated at least this funny (1-10) get the reply drawn in big ascii letters. 11 turns it off.
FUNNY_MIN = int(os.getenv("FUNNY_MIN", "9"))
METRICS_PATH = Path(__file__).parent / "metrics.csv"
EDIT_EVERY = 1.2  # seconds between streaming edits (discord rate-limits edits)
CHUNK = 1900       # discord's hard limit is 2000 characters

SETTINGS_PATH = Path(__file__).parent / "settings.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")


# settings (persisted per server / dm)

def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


settings = load_settings()
settings.setdefault("scopes", {})
settings.setdefault("autoreply", [])
settings.setdefault("facts", {})       # {scope: {user_id: [facts]}}
settings.setdefault("trivia", {})      # {scope: {user_id: points}}
settings.setdefault("reminders", [])   # [{due, channel, user, what}]


def save_settings():
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def scope_key(channel):
    guild = getattr(channel, "guild", None)
    return f"g{guild.id}" if guild else f"c{channel.id}"


def cfg(channel):
    s = settings["scopes"].get(scope_key(channel), {})
    return {
        "model": s.get("model", DEFAULT_MODEL),
        "system": s.get("system", DEFAULT_SYSTEM),
        "think": s.get("think", False),
        "examples": s.get("examples", []),
        "likes": s.get("likes", []),
        "dislikes": s.get("dislikes", []),
    }


# keep it sounding like a group chat
STYLE = (
    "How you talk: like a real person in a group chat, never like an AI assistant. Casual, lowercase is fine. "
    "Fit the length to the message: a quick line for casual chat, but when someone asks you to explain, tell a "
    "story, list things, or go into detail, give a full, longer answer. No bullet lists, "
    "headers, or bold unless someone asks for them. No em dashes. Never say things like \"great question\", "
    "\"I'd be happy to help\", \"certainly\", \"as an AI\", \"it's important to note\", or \"let me know if you "
    "have any other questions\". Don't summarize, don't offer follow-ups, don't hedge. Have opinions and commit to them. "
    + "Like a real person texting, you can occasionally split a reply into 2 to 4 separate messages, but only "
    "when it has genuinely separate parts (a quick reaction, then a longer answer; or a thought, then an "
    "afterthought). Put [next] between messages. Only about one reply in four should be split; short replies "
    "never are, and never split mid-sentence."
)

# split replies at this marker
NEXT = "[next]"
NEXT_SPLIT = re.compile(r"\s*\[next\]\s*", re.I)


def split_reply(text, limit=5):
    """break a reply into the separate messages the model asked for. code blocks (and ascii art) never get split."""
    if "```" in text:
        return [NEXT_SPLIT.sub("\n", text).strip()]
    parts = [p.strip() for p in NEXT_SPLIT.split(text) if p.strip()]
    if len(parts) > 1 and sum(len(p) for p in parts) < 60:
        # keep short replies in one message
        return [" ".join(parts)]
    if len(parts) > limit:
        parts = parts[:limit - 1] + [" ".join(parts[limit - 1:])]
    return parts or [text]


async def send_followups(channel, parts):
    """send the rest of a multi-message reply one at a time, 'typing' for a moment before each like a person."""
    for part in parts:
        async with channel.typing():
            await asyncio.sleep(min(0.6 + len(part) / 60, 2.5))
        await channel.send(part)

# put ascii art in a code block so it lines up
ASCII_RULE = (
    "If asked for ASCII art: always put it inside a ``` code block, keep it under 40 characters wide and 20 "
    "lines tall, and draw a clear, simple outline of the subject rather than lots of detail."
)

# default examples; small models follow these better than rules
DEFAULT_EXAMPLES = [
    ["what should i eat tonight", "tacos. it's always tacos"],
    ["i failed my exam lol", "rip. study the stuff you missed and go get the next one"],
    ["who are you", "just a bot that knows way too much. what's up"],
    ["cats or dogs", "dogs. cats are just tiny landlords"],
]


def example_turns(conf):
    """example exchanges, sent as fake earlier chat turns."""
    turns = []
    # skip default examples when using a custom personality
    defaults = DEFAULT_EXAMPLES if conf["system"] == DEFAULT_SYSTEM else []
    for user_says, bot_says in conf["examples"] or defaults:
        turns += [{"role": "user", "content": f"Friend: {user_says}"}, {"role": "assistant", "content": bot_says}]
    return turns


def set_cfg(channel, **changes):
    s = settings["scopes"].setdefault(scope_key(channel), {})
    for k, v in changes.items():
        if v is None:
            s.pop(k, None)
        else:
            s[k] = v
    save_settings()


# ollama

http: Optional[aiohttp.ClientSession] = None
caps_cache: dict[str, list[str]] = {}


async def ollama_json(path, payload=None):
    method = "GET" if payload is None else "POST"
    async with http.request(method, OLLAMA_URL + path, json=payload) as r:
        r.raise_for_status()
        return await r.json()


async def list_models():
    data = await ollama_json("/api/tags")
    return sorted(m["name"] for m in data.get("models", []))


async def model_caps(model):
    if model not in caps_cache:
        try:
            info = await ollama_json("/api/show", {"model": model})
            caps_cache[model] = info.get("capabilities", [])
        except Exception:
            return []
    return caps_cache[model]


WARMUP_TEXT = ("Counting, arithmetic, geometry, algebra, probability, statistics, history, science, music, food, games, "
               "travel, sports, movies, weather, animals, space, and everyday conversation. ") * 60


async def warm(model):
    """load the model and run one long throwaway prompt, so the first real reply isn't slow.
    loading alone isn't enough for mixture-of-experts models too big for the gpu: the part kept in ram is
    only read from disk as experts get used, which made the first replies after a load take 10+ seconds."""
    try:
        t = time.monotonic()
        await ollama_json("/api/generate", {"model": model, "keep_alive": KEEP_ALIVE, "options": {"num_ctx": NUM_CTX}})
        await ollama_json("/api/generate", {"model": model, "prompt": WARMUP_TEXT + "\nSummarize that in one word.",
                                            "stream": False, "think": False, "keep_alive": KEEP_ALIVE,
                                            "options": {"num_ctx": NUM_CTX, "num_predict": 1}})
        log.info(f"{model} loaded and warmed up in {time.monotonic() - t:.0f}s")
    except Exception as e:
        log.warning(f"Couldn't preload {model}: {e}")


async def stream_chat(model, messages, think, timing=None):
    """yields reply text as it streams. if a `timing` dict with a "start" time is passed, it's filled in with
    time to first token and ollama's own prompt/generation counts and durations."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": NUM_CTX},
    }
    if think is not None:
        payload["think"] = think
    async with http.post(OLLAMA_URL + "/api/chat", json=payload) as r:
        if r.status >= 400:
            raise RuntimeError(f"Ollama {r.status}: {(await r.text())[:300]}")
        async for line in r.content:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            text = chunk.get("message", {}).get("content", "")
            if timing is not None and text and "ttft" not in timing:
                timing["ttft"] = time.perf_counter() - timing["start"]
            if timing is not None and chunk.get("done"):
                timing.update(total=time.perf_counter() - timing["start"],
                              prompt_tokens=chunk.get("prompt_eval_count", 0),
                              prompt_s=chunk.get("prompt_eval_duration", 0) / 1e9,
                              output_tokens=chunk.get("eval_count", 0),
                              output_s=chunk.get("eval_duration", 0) / 1e9,
                              load_s=chunk.get("load_duration", 0) / 1e9)
            yield text
            if chunk.get("done"):
                break


def record_metrics(model, timing):
    """append one reply's latency numbers to metrics.csv (read by /stats and handy for offline analysis)."""
    if "total" not in timing:
        return
    new = not METRICS_PATH.exists()
    with open(METRICS_PATH, "a", encoding="utf-8") as f:
        if new:
            f.write("time,model,ttft_s,total_s,prompt_tokens,prompt_s,output_tokens,output_s,load_s\n")
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')},{model},{timing.get('ttft', timing['total']):.3f},"
                f"{timing['total']:.3f},{timing['prompt_tokens']},{timing['prompt_s']:.3f},{timing['output_tokens']},"
                f"{timing['output_s']:.3f},{timing['load_s']:.3f}\n")


async def chat_once(model, messages, fmt=None, temperature=None, think=False, strip=True):
    """non-streaming chat call. fmt can be a json schema to force structured output.
    strip=false keeps leading spaces, which matter for ascii art."""
    options = {"num_ctx": NUM_CTX}
    if temperature is not None:
        options["temperature"] = temperature
    payload = {"model": model, "messages": messages, "stream": False, "keep_alive": KEEP_ALIVE, "options": options}
    if "thinking" in await model_caps(model):
        payload["think"] = think
    if fmt:
        payload["format"] = fmt
    data = await ollama_json("/api/chat", payload)
    content = data.get("message", {}).get("content", "")
    return content.strip() if strip else content.strip("\n")


FUNNY_PROMPT = (
    "You are a comedy judge rating how funny a Discord message is, from 1 to 10. Don't answer or reply to the "
    "message, only rate it. Scale: 1 = not trying to be funny (questions, greetings, requests, plain statements). "
    "3 = mildly amusing. 5 = decent joke. 7 = genuinely funny. 9 = extremely funny, clever or absurd enough to "
    "make a group chat cry laughing. 10 = legendary. Be strict."
)
FUNNY_SCHEMA = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 10}},
                "required": ["score"]}


async def funny_score(model, text):
    try:
        out = await chat_once(model, [{"role": "system", "content": FUNNY_PROMPT},
                                      {"role": "user", "content": f'Message to rate:\n"{text[:2000]}"'}],
                              fmt=FUNNY_SCHEMA, temperature=0)
        return int(json.loads(out)["score"])
    except Exception as e:
        log.warning(f"Funny check failed: {e}")
        return 0


def banner(text):
    """the reply in big ascii letters, in the largest font that fits one message, or none if it can't fit."""
    import pyfiglet
    for font in ("standard", "small", "mini"):
        art = pyfiglet.figlet_format(text, font=font, width=70).rstrip()
        if art.strip() and len(art) <= 1880:
            return f"```\n{art}\n```"
    return None


# discord output

IMAGE_TAG = re.compile(r"[ \t]*\[(?:image|attached \d+ image\(s\))\]", re.I)


def strip_image_tags(text):
    return IMAGE_TAG.sub("", text)


def split_text(text, size=CHUNK):
    parts = []
    while len(text) > size:
        cut = text.rfind("\n", 0, size)
        if cut < size // 2:
            cut = text.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        parts.append(text[:cut])
        text = text[cut:].lstrip()
    parts.append(text)
    # close code blocks before splitting the message
    for i in range(len(parts) - 1):
        if parts[i].count("```") % 2:
            parts[i] += "\n```"
            parts[i + 1] = "```\n" + parts[i + 1]
    return parts


class LiveReply:
    """streams text into one or more discord messages, editing them as the text grows."""

    def __init__(self, source: discord.Message):
        self.source = source
        self.msgs: list[discord.Message] = []
        self.shown: list[str] = []

    async def update(self, text):
        for i, part in enumerate(split_text(text)):
            part = part or "…"
            if i < len(self.msgs):
                if self.shown[i] != part:
                    await self.msgs[i].edit(content=part)
                    self.shown[i] = part
            else:
                if self.msgs:
                    msg = await self.source.channel.send(part)
                else:
                    msg = await self.source.reply(part)
                self.msgs.append(msg)
                self.shown.append(part)


# bot

class Bot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = MEMBERS_INTENT
        # allow user pings, keep mass pings off
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions(
            everyone=False, roles=False, users=True, replied_user=False))
        self.tree = SleepyTree(self)

    async def setup_hook(self):
        global http
        http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=900))
        await self.tree.sync()
        asyncio.create_task(warm(DEFAULT_MODEL))
        asyncio.create_task(reminder_loop())

    async def close(self):
        if http:
            await http.close()
        await super().close()


SLEEPING = False   # set by /shutdown, cleared by /startup; while asleep nothing loads a model


class SleepyTree(app_commands.CommandTree):
    async def interaction_check(self, inter: discord.Interaction):
        if SLEEPING and inter.command and inter.command.name not in ("startup", "shutdown", "status"):
            await inter.response.send_message("💤 The bot is asleep. Someone allowed to can wake it with `/startup`.", ephemeral=True)
            return False
        return True


def members_intent_enabled():
    """true if "server members intent" is on in the developer portal. asking for it when it's off stops the bot
    from connecting, so check first."""
    import urllib.request
    try:
        req = urllib.request.Request("https://discord.com/api/v10/applications/@me",
                                     headers={"Authorization": f"Bot {TOKEN}", "User-Agent": "DiscordBot (discord-inference-bot, 1.0)"})
        flags = json.load(urllib.request.urlopen(req, timeout=10)).get("flags", 0)
        return bool(flags & (1 << 14 | 1 << 15))
    except Exception:
        return False


MEMBERS_INTENT = members_intent_enabled()
client = Bot()
tree = client.tree
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


DETAIL_WORDS = re.compile(r"\b(explain|describe|detail|details|detailed|story|stories|list|steps|how do|how does|how to|"
                          r"why|compare|write|essay|paragraph|summar\w*|tell me about|in depth|long)\b", re.I)


def wants_detail(message):
    """whether a message is asking for a longer answer."""
    return bool(DETAIL_WORDS.search(message)) or len(message) > 200


def build_messages(channel, conf, people, memories, query, hist, turn):
    """the full message list for a chat reply, laid out for prompt caching: stable system prompt and examples
    first, then chat memory, then the new message with its per-message context note."""
    system, examples, extra = build_system(channel, conf, query)
    turn = with_context(turn, context_note(channel, people, memories, extra))
    if wants_detail(query):
        # ask for more detail when the message needs it
        turn = dict(turn, content=turn["content"] + "\n\n(This message asks for a longer answer: write a full reply, "
                                                    "several sentences or more, in your usual voice.)")
    if hist:
        # use recent chat as background
        turn = dict(turn, content=turn["content"] + "\n\n(Earlier messages are just background. Answer this "
                                                    "newest message on its own and don't repeat what you already said.)")
    recent = fit_history(channel, system, examples, hist, turn)
    return [{"role": "system", "content": system}, *examples, *recent, turn]
last_active = {}   # channel id -> when the bot last got a message there, for auto-reset


def fit_history(channel, system, examples, hist, turn):
    """drop the oldest chat until the whole prompt fits the context window with room left for a long reply.
    roughly 3.2 characters per token; about 2,000 tokens are kept free for the reply."""
    budget = int(NUM_CTX * 3.2) - 6500
    fixed = len(system) + sum(len(e["content"]) for e in examples) + len(turn["content"])
    kept = list(hist)
    while kept and fixed + sum(len(h["content"]) for h in kept) > budget:
        kept = kept[2:]   # oldest exchange first
    if len(kept) < len(hist):
        log.info(f"Trimmed chat memory in {channel} from {len(hist)} to {len(kept)} messages to fit the context")
        hist.clear()
        hist.extend(kept)
    return kept
recent_speakers = defaultdict(lambda: deque(maxlen=8))   # user ids of the latest people talking, per channel
locks = defaultdict(asyncio.Lock)


def me_in(channel):
    guild = getattr(channel, "guild", None)
    return guild.me if guild else client.user


def should_respond(m: discord.Message):
    if m.author.bot:
        return False
    if isinstance(m.channel, discord.DMChannel):
        return True
    if m.channel.id in settings["autoreply"]:
        return True
    if client.user in m.mentions:
        return True
    ref = m.reference.resolved if m.reference else None
    return isinstance(ref, discord.Message) and ref.author.id == client.user.id


def user_text(m: discord.Message):
    text = m.clean_content
    for name in {client.user.name, me_in(m.channel).display_name}:
        text = text.replace(f"@{name}", "")
    text = text.strip()
    # include the reply target unless it is already in bot history
    ref = m.reference.resolved if m.reference else None
    if isinstance(ref, discord.Message) and ref.author.id != client.user.id and ref.clean_content:
        text = f'(replying to {ref.author.display_name}: "{ref.clean_content[:500]}")\n{text}'
    return text


def facts_for(channel, user_id):
    return settings["facts"].get(scope_key(channel), {}).get(str(user_id), [])


def people_lines(channel, people):
    """one line per person: the name to use, other names they go by, and any saved facts."""
    lines = []
    for p in people:
        p = p if isinstance(p, Person) else as_person(p)
        aka = [a for a in dict.fromkeys(p.aliases) if a and a.lower() != p.display_name.lower()]
        line = f"- {p.display_name}" + (f" (also goes by {', '.join(aka)})" if aka else "")
        facts = facts_for(channel, p.id)
        if facts:
            line += ": " + "; ".join(facts)
        lines.append(line)
    return lines


def tastes(conf):
    parts = []
    if conf["likes"]:
        parts.append("You like: " + ", ".join(conf["likes"]) + ".")
    if conf["dislikes"]:
        parts.append("You dislike: " + ", ".join(conf["dislikes"]) + ".")
    return (" ".join(parts) + " Let these show in your opinions naturally; don't list them.\n\n") if parts else ""


# prompt layout, for prompt-prefix caching: ollama reuses the processed prompt as long as a request starts
# with the same tokens as the previous one. so the system prompt and examples hold only what stays the same
# from message to message, and everything that changes per message (who's talking, recalled memories, the date)
# goes in a context note at the very end, next to the new message.

def system_prompt(channel, conf):
    """the stable part of the prompt: identical for every message in a channel until settings change."""
    guild = getattr(channel, "guild", None)
    where = f"#{channel.name} in the {guild.name} server" if guild else "a direct message"
    # /system goes first and wins over the default style; it's repeated at the end because small models
    # weigh what they read last most heavily
    instructions = conf["system"]
    reminder = (f"\n\nMost important: follow your instructions from the top exactly:\n{instructions}"
                if len(instructions) <= 1500 else "\n\nMost important: follow your instructions from the top exactly.")
    # background is phrased so small models don't recite it or treat the account name as their identity
    return (
        "YOUR INSTRUCTIONS (these override every default below):\n"
        f"{instructions}\n\n"
        f"{tastes(conf)}"
        f"Default style, only where your instructions don't say otherwise: {STYLE}\n\n"
        "Keep things friendly and respectful. Don't insult people, and politely decline anything harmful, "
        "hateful, or explicit.\n\n"
        "Each new message comes with a short context note: the people in the conversation (use their names "
        "exactly as listed and don't mix them up), anything saved about them, and old chats found by search. "
        "Beyond that note, you know nothing about the people here except what they say; never make up facts, "
        "backstories, or records about them.\n\n"
        "To ping someone, write @ followed by their name exactly as listed in the context note (like @Mike). "
        "Only ping when someone asks you to or it really makes sense.\n\n"
        f"(Background only, never bring it up unless asked: this chat is {where}, and the bot account is "
        f"called {me_in(channel).display_name}. Messages arrive as \"Name: message\" so you know who is "
        "talking; never start your own reply with a name.)"
        f"{reminder}"
    )


def context_note(channel, people=(), memories=(), extra=""):
    """the per-message part of the prompt, placed right before the new message."""
    parts = [f"Today is {datetime.date.today().strftime('%A, %B %d, %Y')}."]
    lines = people_lines(channel, people)
    if lines:
        parts.append("People in this conversation:\n" + "\n".join(lines))
    if memories:
        parts.append("Old chats from this server found by search (may be old; only bring them up if relevant, "
                     "and quote them accurately):\n\n" + "\n\n".join(memories)[:3500])
    if extra:
        parts.append(extra)
    return "(Context for this message:\n" + "\n\n".join(parts) + ")"


def with_context(turn, note):
    """the new message with the context note in front; only sent to the model, never saved to chat memory."""
    return dict(turn, content=f"{note}\n\n{turn['content']}")


Person = namedtuple("Person", "id display_name aliases")


def as_person(u):
    return Person(u.id, u.display_name, (getattr(u, "nick", None), u.global_name, u.name))


def people_in(m: discord.Message):
    """everyone relevant to this message: the speaker, anyone mentioned, replied to, or named in the text,
    and whoever has been talking in the channel lately. used for names and saved facts."""
    people = {m.author.id: as_person(m.author)}
    for u in m.mentions:
        if not u.bot:
            people[u.id] = as_person(u)
    ref = m.reference.resolved if m.reference else None
    if isinstance(ref, discord.Message) and not ref.author.bot:
        people[ref.author.id] = as_person(ref.author)
    scope = scope_key(m.channel)
    for p in named_in(scope, m.clean_content) + [roster_person(scope, uid) for uid in recent_speakers[m.channel.id]]:
        if p and p.id not in people and len(people) < 15:
            people[p.id] = p
    return list(people.values())


@client.event
async def on_ready():
    log.info(f"Online as {client.user} | default model {DEFAULT_MODEL} | Ollama {OLLAMA_URL}")
    if MEMBERS_INTENT:
        total = sum(sync_guild_members(g) for g in client.guilds)
        log.info(f"Member directory: {total} members")
    else:
        log.info("Server Members Intent is off; learning names from people who talk")


@client.event
async def on_message(m: discord.Message):
    if SLEEPING:
        return   # no replies and no indexing while asleep; /index channel catches up on missed messages later
    await live_add(m)
    if not m.author.bot:
        note_member(scope_key(m.channel), m.author)
        speakers = recent_speakers[m.channel.id]
        if m.author.id in speakers:
            speakers.remove(m.author.id)
        speakers.append(m.author.id)
    if not should_respond(m):
        return

    conf = cfg(m.channel)
    caps = await model_caps(conf["model"])
    text = user_text(m)

    images = []
    image_atts = [a for a in m.attachments if (a.content_type or "").startswith("image/")]
    if image_atts and "vision" in caps:
        images = [base64.b64encode(await a.read()).decode() for a in image_atts[:MAX_IMAGES]]
    if not text and not images:
        text = "hi"

    content = f"{m.author.display_name}: {text}"
    turn = {"role": "user", "content": content}
    if re.search(r"\bascii\b|\bdraw\b|\bdrawing\b", text, re.I):
        # only add art instructions when asked
        turn["content"] += f"\n\n({ASCII_RULE})"
    if images:
        turn["images"] = images
    think = conf["think"] if "thinking" in caps else None

    log.info(f"[IN] {m.author} in {m.channel}: {text[:80]}")
    try:
        memories = await recall(m.channel, text)
    except Exception as e:
        log.warning(f"Recall failed: {e}")
        memories = []
    async with locks[m.channel.id]:
        hist = histories[m.channel.id]
        # reset chat memory after a quiet stretch
        now = time.time()
        if hist and now - last_active.get(m.channel.id, now) > AUTO_RESET_MINUTES * 60:
            hist.clear()
            log.info(f"Auto-reset chat memory in {m.channel} after {AUTO_RESET_MINUTES}+ quiet minutes")
        last_active[m.channel.id] = now
        # remove old blank replies so the model does not copy them
        for h in [h for h in hist if h["content"] == EMPTY_REPLY]:
            hist.remove(h)
        messages = build_messages(m.channel, conf, people_in(m), memories, text, hist, turn)
        reply = LiveReply(m)
        out, last_edit = "", 0.0
        try:
            async with m.channel.typing():
                score = await funny_score(conf["model"], text) if FUNNY_MIN <= 10 and not images else 0
                funny = score >= FUNNY_MIN
                if funny:
                    # big letters only fit a short reply, so ask for one (just for this message)
                    messages[-1] = dict(messages[-1], content=messages[-1]["content"] + "\n\n(This is hilarious. "
                                        "Reply the way you normally would, but in under 30 characters.)")
                # retry once if the reply is empty
                for attempt in range(2):
                    timing = {"start": time.perf_counter()}
                    async for tok in stream_chat(conf["model"], messages, think, timing):
                        out += tok
                        now = time.monotonic()
                        if not funny and out.strip() and now - last_edit >= EDIT_EVERY:
                            await reply.update(split_reply(out)[0])
                            last_edit = now
                    if strip_image_tags(out).strip():
                        break
                    out = ""
                    log.info("Empty reply, retrying" if attempt == 0 else "Empty reply twice")
                record_metrics(conf["model"], timing)
                # remove image tags copied from memory
                out = strip_image_tags(out).strip()
                if not out:
                    # keep blank replies out of memory
                    return await reply.update(EMPTY_REPLY)
            # save plain text, display the big letters
            big = banner(NEXT_SPLIT.sub(" ", out)) if funny else None
            parts = [big] if big else [pings(scope_key(m.channel), p) for p in split_reply(out)]
            await reply.update(parts[0])
            await send_followups(m.channel, parts[1:])
            if funny:
                log.info("Funny message, replied in ASCII letters")
            elif len(parts) > 1:
                log.info(f"Sent the reply as {len(parts)} messages")
        except aiohttp.ClientConnectorError:
            await reply.update(f"⚠️ Can't reach Ollama at {OLLAMA_URL}. Is it running?")
            return
        except discord.Forbidden:
            # mentioned in a channel the bot can read but not post in; nothing to send the reply to
            log.warning(f"No permission to reply in #{m.channel}")
            return
        except Exception as e:
            log.exception("Generation failed")
            partial = out.strip()
            await reply.update((partial + "\n\n" if partial else "") + f"⚠️ {e}")
            return

        # images aren't kept in memory, just a note that there was one
        note = f" [attached {len(images)} image(s)]" if images else ""
        hist.append({"role": "user", "content": content + note})
        hist.append({"role": "assistant", "content": out})
    log.info(f"[OUT] {out[:80]}")


# slash commands

# roles (ids or names, comma-separated) that can use the settings commands, on top of admins
COMMAND_ROLES = {r.strip().lower() for r in os.getenv("COMMAND_ROLES", "").split(",") if r.strip()}


def is_staff(inter: discord.Interaction, perm="manage_guild"):
    """people with the discord permission, or with one of command_roles. everyone counts in dms."""
    if inter.guild is None or getattr(inter.permissions, perm):
        return True
    return any(str(r.id) in COMMAND_ROLES or r.name.lower() in COMMAND_ROLES for r in getattr(inter.user, "roles", []))


def staff_only(perm="manage_guild"):
    async def predicate(inter: discord.Interaction):
        if is_staff(inter, perm):
            return True
        raise app_commands.CheckFailure("You need an admin permission or one of the bot's command roles for that.")
    return app_commands.check(predicate)


@tree.error
async def on_command_error(inter: discord.Interaction, error: app_commands.AppCommandError):
    msg = str(error) if isinstance(error, app_commands.CheckFailure) else f"⚠️ {error}"
    if not isinstance(error, app_commands.CheckFailure):
        log.error(f"Command error: {error}", exc_info=error)
    if inter.response.is_done():
        await inter.followup.send(msg, ephemeral=True)
    else:
        await inter.response.send_message(msg, ephemeral=True)


async def model_autocomplete(inter: discord.Interaction, current: str):
    try:
        names = await list_models()
    except Exception:
        return []
    return [app_commands.Choice(name=n, value=n) for n in names if current.lower() in n.lower()][:25]


@tree.command(description="Show or switch the local model")
@app_commands.describe(name="Ollama model to use")
@app_commands.autocomplete(name=model_autocomplete)
@staff_only()
async def model(inter: discord.Interaction, name: Optional[str] = None):
    try:
        available = await list_models()
    except Exception:
        return await inter.response.send_message(f"⚠️ Can't reach Ollama at {OLLAMA_URL}.", ephemeral=True)
    if not name:
        current = cfg(inter.channel)["model"]
        listing = "\n".join(f"{'▶' if n == current else '•'} `{n}`" for n in available)
        return await inter.response.send_message(f"**Current:** `{current}`\n{listing}", ephemeral=True)
    if name not in available:
        return await inter.response.send_message(f"`{name}` isn't installed. Try `ollama pull {name}`.", ephemeral=True)
    set_cfg(inter.channel, model=name)
    await inter.response.send_message(f"Switched to `{name}`. Loading it now…")
    await warm(name)


@tree.command(description="Show or change the bot's personality / instructions")
@app_commands.describe(prompt="New system prompt, or 'default' to reset")
@staff_only()
async def system(inter: discord.Interaction, prompt: Optional[str] = None):
    if not prompt:
        return await inter.response.send_message(f"**System prompt:**\n>>> {cfg(inter.channel)['system']}", ephemeral=True)
    # clear old replies when the personality changes
    histories.clear()
    if prompt.strip().lower() in ("default", "reset"):
        set_cfg(inter.channel, system=None)
        return await inter.response.send_message("System prompt reset to default.")
    set_cfg(inter.channel, system=prompt)
    await inter.response.send_message(f"System prompt updated:\n>>> {prompt}")


example = app_commands.Group(
    name="example",
    description="Example exchanges that teach the bot how to talk",
)


@example.command(name="add", description="Add an example of how the bot should reply")
@app_commands.describe(user_says="Something a person might say", bot_says="Exactly how the bot should answer it")
@staff_only()
async def example_add(inter: discord.Interaction, user_says: str, bot_says: str):
    examples = cfg(inter.channel)["examples"] + [[user_says, bot_says]]
    set_cfg(inter.channel, examples=examples)
    await inter.response.send_message(f"Example #{len(examples)} added. Run `/reset` so old replies don't override it.", ephemeral=True)


@example.command(name="list", description="Show the saved examples")
@staff_only()
async def example_list(inter: discord.Interaction):
    examples = cfg(inter.channel)["examples"]
    if not examples:
        return await inter.response.send_message("No examples yet. Add one with `/example add`.", ephemeral=True)
    lines = [f"**{i}.** {u}\n→ {b}" for i, (u, b) in enumerate(examples, 1)]
    await inter.response.send_message("\n".join(lines)[:2000], ephemeral=True)


@example.command(name="remove", description="Remove one example by its number from /example list")
@staff_only()
async def example_remove(inter: discord.Interaction, number: int):
    examples = cfg(inter.channel)["examples"]
    if not 1 <= number <= len(examples):
        return await inter.response.send_message(f"No example #{number}.", ephemeral=True)
    examples.pop(number - 1)
    set_cfg(inter.channel, examples=examples or None)
    await inter.response.send_message(f"Removed example #{number}.", ephemeral=True)


@example.command(name="clear", description="Remove all examples")
@staff_only()
async def example_clear(inter: discord.Interaction):
    set_cfg(inter.channel, examples=None)
    await inter.response.send_message("All examples removed.", ephemeral=True)


tree.add_command(example)


@tree.command(description="Turn step-by-step thinking on or off (slower, smarter)")
@staff_only()
async def think(inter: discord.Interaction, enabled: bool):
    set_cfg(inter.channel, think=enabled)
    await inter.response.send_message(f"Thinking {'on' if enabled else 'off'}.")


@tree.command(description="Reply to every message in this channel, not just mentions")
@staff_only("manage_channels")
async def autoreply(inter: discord.Interaction, enabled: bool):
    ids = settings["autoreply"]
    if enabled and inter.channel_id not in ids:
        ids.append(inter.channel_id)
    elif not enabled and inter.channel_id in ids:
        ids.remove(inter.channel_id)
    save_settings()
    await inter.response.send_message(f"Auto-reply {'on' if enabled else 'off'} in this channel.")


def pings(scope, text):
    """turn "@name" in the model's reply into a real ping for anyone in the member list."""
    if "@" not in text:
        return text
    names = []
    for uid, aliases in roster(scope).items():
        names += [(a, uid) for a in aliases if a and len(a) >= 2]
    for name, uid in sorted(names, key=lambda n: -len(n[0])):   # longest first, so "big mike" beats "big"
        text = re.sub(rf"@{re.escape(name)}(?![\w])", f"<@{uid}>", text, flags=re.I)
    return text


async def unload_models():
    """tell ollama to drop every loaded model from memory right away. returns their names."""
    loaded = [m["name"] for m in (await ollama_json("/api/ps")).get("models", [])]
    for name in loaded:
        await ollama_json("/api/generate", {"model": name, "keep_alive": 0})
    return loaded


# user ids (comma-separated) allowed to use /shutdown besides the bot's owner. ids, not names, so nobody
# can get access by renaming themselves.
SHUTDOWN_USERS = {int(u) for u in os.getenv("SHUTDOWN_USERS", "").replace(" ", "").split(",") if u.isdigit()}


async def can_power(inter: discord.Interaction):
    app = await client.application_info()
    allowed = {m.id for m in app.team.members} if app.team else {app.owner.id}
    return inter.user.id in allowed | SHUTDOWN_USERS


async def set_asleep(asleep: bool):
    global SLEEPING
    SLEEPING = asleep
    if asleep:
        await client.change_presence(status=discord.Status.idle, activity=discord.Game("asleep · /startup to wake"))
    else:
        await client.change_presence(status=discord.Status.online, activity=None)


@tree.command(description="Put the bot to sleep: frees the GPU completely until /startup")
@app_commands.describe(fully_exit="Turn the bot off completely instead. /startup can't wake it; restart it on the PC")
async def shutdown(inter: discord.Interaction, fully_exit: bool = False):
    if not await can_power(inter):
        return await inter.response.send_message("Only the bot's owner can do that.", ephemeral=True)
    await inter.response.defer(ephemeral=True, thinking=True)
    await set_asleep(True)   # first, so nothing reloads a model while it's unloading
    try:
        unloaded = await unload_models()
        freed = f"Unloaded {', '.join(f'`{n}`' for n in unloaded)}." if unloaded else "No models were loaded."
    except Exception as e:
        freed = f"Couldn't reach Ollama to unload models ({e})."
    log.info(f"{'Shut down' if fully_exit else 'Put to sleep'} by {inter.user}")
    if not fully_exit:
        return await inter.followup.send(f"💤 {freed} The bot is asleep and the GPU is free. `/startup` wakes it up.", ephemeral=True)
    await inter.followup.send(f"🔌 {freed} Shutting the bot down. Start it again on the PC with `start-bot.bat`.", ephemeral=True)
    await client.close()


@tree.command(description="Wake the bot up after /shutdown and reload the model")
async def startup(inter: discord.Interaction):
    if not await can_power(inter):
        return await inter.response.send_message("Only the bot's owner can do that.", ephemeral=True)
    if not SLEEPING:
        return await inter.response.send_message("Already awake.", ephemeral=True)
    await inter.response.defer(thinking=True)
    await warm(cfg(inter.channel)["model"])
    await set_asleep(False)
    log.info(f"Woken up by {inter.user}")
    await inter.followup.send("☀️ Awake and ready.")


@tree.command(description="Make the bot forget this channel's conversation")
async def reset(inter: discord.Interaction):
    histories.pop(inter.channel_id, None)
    await inter.response.send_message("🧹 Memory cleared for this channel.")


@tree.command(description="Show model, settings, and GPU status")
async def status(inter: discord.Interaction):
    conf = cfg(inter.channel)
    try:
        loaded = (await ollama_json("/api/ps")).get("models", [])
        running = ", ".join(
            f"`{m['name']}` ({m.get('size_vram', 0) / 1e9:.1f} GB VRAM)" for m in loaded
        ) or "none"
    except Exception:
        running = "⚠️ Ollama unreachable"
    await inter.response.send_message(
        f"**Model:** `{conf['model']}`\n"
        f"**Thinking:** {'on' if conf['think'] else 'off'}\n"
        f"**Auto-reply here:** {'on' if inter.channel_id in settings['autoreply'] else 'off'}\n"
        f"**Memory:** {len(histories.get(inter.channel_id, []))} messages\n"
        f"**Loaded in Ollama:** {running}",
        ephemeral=True,
    )


@tree.command(description="Reply latency: time to first token, speed, and prompt size over recent replies")
@app_commands.describe(last="How many recent replies to include (default 200)")
async def stats(inter: discord.Interaction, last: app_commands.Range[int, 10, 5000] = 200):
    if not METRICS_PATH.exists():
        return await inter.response.send_message("No replies logged yet.", ephemeral=True)
    import csv
    with open(METRICS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-last:]
    if not rows:
        return await inter.response.send_message("No replies logged yet.", ephemeral=True)

    def pct(values, p):
        values = sorted(values)
        return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]

    ttft = [float(r["ttft_s"]) for r in rows]
    total = [float(r["total_s"]) for r in rows]
    decode = [int(r["output_tokens"]) / float(r["output_s"]) for r in rows if float(r["output_s"]) > 0]
    prompt = [int(r["prompt_tokens"]) for r in rows]
    await inter.response.send_message(
        f"📊 **Last {len(rows)} replies** (`{rows[-1]['model']}`)\n"
        f"Time to first token: p50 **{pct(ttft, 50):.2f}s** · p95 {pct(ttft, 95):.2f}s\n"
        f"Full reply: p50 **{pct(total, 50):.2f}s** · p95 {pct(total, 95):.2f}s\n"
        f"Generation speed: median **{pct(decode, 50):.0f} tokens/s**\n"
        f"Prompt size: median {pct(prompt, 50):,} tokens · max {max(prompt):,}",
        ephemeral=True,
    )


# fun & utility commands

async def persona_say(channel, instruction, people=(), images=None, fmt=None, temperature=None):
    """one in-character reply, outside normal chat memory. returns none if the model gave nothing back."""
    conf = cfg(channel)
    turn = {"role": "user", "content": instruction}
    if images:
        turn["images"] = images
    messages = build_messages(channel, conf, people, (), instruction, [], turn)
    # join split replies for commands
    return NEXT_SPLIT.sub("\n", await chat_once(conf["model"], messages, fmt, temperature)).strip() or None


async def send_long(inter: discord.Interaction, text, ephemeral=False):
    for part in split_text(text):
        await inter.followup.send(part, ephemeral=ephemeral)


THEMES = ["food", "money", "superpowers", "embarrassing situations", "gross stuff", "school and work",
          "dating", "travel", "video games", "family", "the internet", "animals", "movies", "music"]


# would you rather / hot takes (vote with reactions)

async def add_reactions(msg, emojis):
    try:
        for e in emojis:
            await msg.add_reaction(e)
    except discord.HTTPException:
        pass  # missing add reactions permission; the post still works


@tree.command(description="A would-you-rather question; vote with reactions")
@app_commands.describe(topic="Optional topic, e.g. food")
async def wouldyourather(inter: discord.Interaction, topic: Optional[str] = None):
    await inter.response.defer(thinking=True)
    schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}
    out = await persona_say(
        inter.channel,
        f"Make up one funny would-you-rather question for this group chat about {topic or random.choice(THEMES)}. "
        "Both options should be really hard to choose between, and both should be equally funny. "
        "Each option is just the choice itself, under 15 words, without \"would you rather\". "
        'Reply as JSON like {"a": "sneeze every time someone says your name", '
        '"b": "have your mom read your search history out loud once a year"}.',
        fmt=schema, temperature=1.1,
    )
    if not out:
        return await inter.followup.send(UNAVAILABLE)
    q = json.loads(out)
    msg = await inter.followup.send(f"🤔 **Would you rather…**\n🅰️ {q['a']}\n🅱️ {q['b']}", wait=True)
    await add_reactions(msg, ["🅰️", "🅱️"])


@tree.command(description="The bot drops a hot take; vote agree or disagree")
@app_commands.describe(topic="Optional topic, e.g. pineapple pizza")
async def hottake(inter: discord.Interaction, topic: Optional[str] = None):
    await inter.response.defer(thinking=True)
    schema = {"type": "object", "properties": {"take": {"type": "string"}}, "required": ["take"]}
    out = await persona_say(
        inter.channel,
        f"Give one spicy but genuinely debatable hot take about {topic or random.choice(THEMES)}, in your voice, "
        "one or two sentences. Reply as JSON with it as take.",
        fmt=schema, temperature=1.1,
    )
    if not out:
        return await inter.followup.send(UNAVAILABLE)
    msg = await inter.followup.send(f"🌶️ **Hot take:** {json.loads(out)['take']}\n-# ✅ agree · ❌ disagree", wait=True)
    await add_reactions(msg, ["✅", "❌"])


# trivia with a leaderboard

recent_trivia = defaultdict(lambda: deque(maxlen=30))


def add_point(key, user_id):
    board = settings["trivia"].setdefault(key, {})
    board[str(user_id)] = board.get(str(user_id), 0) + 1
    save_settings()
    return board[str(user_id)]


class TriviaView(discord.ui.View):
    def __init__(self, key, header, q):
        super().__init__(timeout=45)
        self.key, self.header, self.q = key, header, q
        self.guessed, self.winner, self.message = set(), None, None
        for i, opt in enumerate(q["options"]):
            button = discord.ui.Button(label=f"{'ABCD'[i]}. {opt}"[:80], style=discord.ButtonStyle.secondary)
            button.callback = self._callback(i)
            self.add_item(button)

    def _callback(self, i):
        async def callback(inter: discord.Interaction):
            if self.winner:
                return await inter.response.send_message("Too late, someone already got it.", ephemeral=True)
            if inter.user.id in self.guessed:
                return await inter.response.send_message("You already guessed. One shot each.", ephemeral=True)
            self.guessed.add(inter.user.id)
            if i != self.q["answer"]:
                return await inter.response.send_message("❌ Nope.", ephemeral=True)
            self.winner = inter.user
            points = add_point(self.key, inter.user.id)
            await inter.response.edit_message(
                content=self._reveal(f"✅ **{inter.user.display_name}** got it! +1 ({points} total)"), view=self._lock())
            self.stop()
        return callback

    def _lock(self):
        for i, item in enumerate(self.children):
            item.disabled = True
            if i == self.q["answer"]:
                item.style = discord.ButtonStyle.success
        return self

    def _reveal(self, line):
        fact = f"\n-# {self.q['fact']}" if self.q.get("fact") else ""
        return f"{self.header}\n\n{line}{fact}"

    async def on_timeout(self):
        if self.winner or not self.message:
            return
        answer = f"{'ABCD'[self.q['answer']]}. {self.q['options'][self.q['answer']]}"
        await self.message.edit(content=self._reveal(f"⏰ Time's up! It was **{answer}**"), view=self._lock())


@tree.command(description="Trivia: first correct answer gets a point")
@app_commands.describe(topic="Optional topic, e.g. space")
@app_commands.choices(difficulty=[app_commands.Choice(name=d, value=d) for d in ("easy", "medium", "hard")])
async def trivia(inter: discord.Interaction, difficulty: str = "medium", topic: Optional[str] = None):
    await inter.response.defer(thinking=True)
    key = scope_key(inter.channel)
    schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "correct": {"type": "string"},
            "wrong": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
            "fact": {"type": "string"},
        },
        "required": ["question", "correct", "wrong", "fact"],
    }
    avoid = "\n".join(f"- {q}" for q in recent_trivia[key]) or "(none yet)"
    raw = await chat_once(cfg(inter.channel)["model"], [
        {"role": "system", "content": "You write accurate trivia. Only use facts you are completely sure of."},
        {"role": "user", "content":
            f"Write one {difficulty} trivia question about {topic or random.choice(['science', 'history', 'geography', 'movies', 'music', 'video games', 'animals', 'food', 'space', 'the human body', 'inventions', 'math', 'internet culture'])}. "
            f"Give the correct answer, three believable wrong answers, and a one-sentence fun fact. "
            f"Don't repeat these:\n{avoid}"},
    ], fmt=schema, temperature=1.0)
    q = json.loads(raw)
    options = [q["correct"], *q["wrong"][:3]]
    random.shuffle(options)
    q = {"options": options, "answer": options.index(q["correct"]), "fact": q["fact"], "question": q["question"]}
    recent_trivia[key].append(q["question"])
    header = f"🧠 **Trivia** ({difficulty})\n{q['question']}"
    view = TriviaView(key, header, q)
    view.message = await inter.followup.send(f"{header}\n-# 45 seconds, one guess each", view=view, wait=True)


@tree.command(description="Trivia points leaderboard")
async def leaderboard(inter: discord.Interaction):
    board = settings["trivia"].get(scope_key(inter.channel), {})
    if not board:
        return await inter.response.send_message("No points yet. Start with `/trivia`.")
    top = sorted(board.items(), key=lambda kv: -kv[1])[:10]
    medals = ["🥇", "🥈", "🥉"] + ["▫️"] * 7
    lines = [f"{medals[i]} <@{uid}>: **{pts}**" for i, (uid, pts) in enumerate(top)]
    await inter.response.send_message("🏆 **Trivia leaderboard**\n" + "\n".join(lines))


# rate my pic

@tree.command(description="The bot rates a picture out of 10")
async def ratepic(inter: discord.Interaction, image: discord.Attachment):
    if not (image.content_type or "").startswith("image/"):
        return await inter.response.send_message("That's not an image.", ephemeral=True)
    if "vision" not in await model_caps(cfg(inter.channel)["model"]):
        return await inter.response.send_message("The current model can't see images.", ephemeral=True)
    await inter.response.defer(thinking=True)
    b64 = base64.b64encode(await image.read()).decode()
    out = await persona_say(
        inter.channel,
        f"{inter.user.display_name} wants you to rate this picture out of 10. Be funny and honest, two or "
        "three sentences, in your voice. End with the score in bold, like **7/10**.",
        images=[b64], temperature=0.9,
    )
    embed = discord.Embed(description=out or UNAVAILABLE)
    if out:
        embed.set_image(url=image.url)
    await inter.followup.send(embed=embed)


# summarize

@tree.command(description="Recap what you missed in this channel")
@app_commands.describe(messages="How many recent messages to read (10-300)", private="Only you see the recap")
async def summarize(inter: discord.Interaction, messages: app_commands.Range[int, 10, 300] = 100, private: bool = False):
    await inter.response.defer(thinking=True, ephemeral=private)
    lines = []
    async for m in inter.channel.history(limit=messages):
        text = m.clean_content + (" [image]" if m.attachments else "")
        if text.strip():
            lines.append(f"{m.author.display_name}: {text[:400]}")
    transcript = "\n".join(reversed(lines))[-15000:]
    out = await persona_say(
        inter.channel,
        f"Here are the last {len(lines)} messages from this chat:\n\n{transcript}\n\n"
        "Recap what happened for someone who missed it: the main topics, who said what, and anything funny or "
        "dramatic. Under 150 words, in your voice. Don't make anything up.",
    )
    await send_long(inter, out or UNAVAILABLE, ephemeral=private)


# remembered facts about people

@tree.command(description="Teach the bot a fact about someone")
@app_commands.describe(user="Who it's about", fact="e.g. thinks the moon landing was faked")
async def remember(inter: discord.Interaction, user: discord.Member, fact: app_commands.Range[str, 1, 200]):
    if user.id != inter.user.id and not is_staff(inter):
        return await inter.response.send_message("You can only add facts about yourself.", ephemeral=True)
    people = settings["facts"].setdefault(scope_key(inter.channel), {})
    facts = people.setdefault(str(user.id), [])
    if len(facts) >= 10:
        return await inter.response.send_message(f"{user.display_name} already has 10 facts. Remove one with `/forget`.", ephemeral=True)
    facts.append(fact)
    save_settings()
    await inter.response.send_message(f"🧠 Got it: {user.mention} — {fact}")


@tree.command(description="Remove facts the bot remembers about someone")
@app_commands.describe(number="Which fact to remove (from /facts). Leave empty to remove all")
async def forget(inter: discord.Interaction, user: discord.Member, number: Optional[int] = None):
    if user.id != inter.user.id and not is_staff(inter):
        return await inter.response.send_message("You can only remove facts about yourself.", ephemeral=True)
    people = settings["facts"].setdefault(scope_key(inter.channel), {})
    facts = people.get(str(user.id), [])
    if number is None:
        people.pop(str(user.id), None)
        msg = f"Forgot everything about {user.display_name}."
    elif 1 <= number <= len(facts):
        removed = facts.pop(number - 1)
        msg = f"Forgot: {removed}"
    else:
        return await inter.response.send_message(f"No fact #{number}.", ephemeral=True)
    save_settings()
    await inter.response.send_message(msg, ephemeral=True)


@tree.command(description="See what the bot remembers about someone")
async def facts(inter: discord.Interaction, user: discord.Member):
    known = facts_for(inter.channel, user.id)
    if not known:
        return await inter.response.send_message(f"I don't know anything about {user.display_name} yet.", ephemeral=True)
    lines = "\n".join(f"**{i}.** {f}" for i, f in enumerate(known, 1))
    await inter.response.send_message(f"🧠 **{user.display_name}**\n{lines}", ephemeral=True)


# reminders (saved, so they survive restarts)

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    parts = re.findall(r"(\d+)\s*(d|h|m|s)[a-z]*", text.lower())
    return sum(int(n) * UNITS[u] for n, u in parts) if parts else None


@tree.command(description="Get reminded about something later, in character")
@app_commands.describe(when="e.g. 10m, 2h, 1d, 1h30m", what="What to remind you about")
async def remindme(inter: discord.Interaction, when: str, what: app_commands.Range[str, 1, 300]):
    secs = parse_duration(when)
    if not secs or secs < 10 or secs > 30 * 86400:
        return await inter.response.send_message("Use a time like `10m`, `2h`, `1d`, or `1h30m` (max 30 days).", ephemeral=True)
    due = int(time.time() + secs)
    settings["reminders"].append({"due": due, "channel": inter.channel_id, "user": inter.user.id, "what": what})
    save_settings()
    await inter.response.send_message(f"⏰ Got it. I'll remind you <t:{due}:R>.")


async def send_reminder(r):
    try:
        channel = client.get_channel(r["channel"]) or await client.fetch_channel(r["channel"])
    except discord.HTTPException as e:
        return log.warning(f"Reminder channel gone: {e}")
    line = None
    try:
        # reminders still go out while asleep, just without loading the model for an in-character line
        line = None if SLEEPING else await persona_say(channel, f'Remind your friend about this, in your voice, in one or two short '
                                          f'sentences. Don\'t start with their name: "{r["what"]}"', temperature=0.9)
    except Exception as e:
        log.warning(f"Reminder text failed: {e}")
    await channel.send(f"⏰ <@{r['user']}> {line or 'Reminder!'}\n> {r['what']}",
                       allowed_mentions=discord.AllowedMentions(users=[discord.Object(r["user"])]))


async def reminder_loop():
    await client.wait_until_ready()
    while not client.is_closed():
        now = time.time()
        due = [r for r in settings["reminders"] if r["due"] <= now]
        if due:
            settings["reminders"] = [r for r in settings["reminders"] if r["due"] > now]
            save_settings()
            for r in due:
                asyncio.create_task(send_reminder(r))
        await asyncio.sleep(15)


# vector memory (/index)
# past channel messages are grouped into chunks, embedded with a small embedding model, and stored in
# index.db. when someone talks to the bot, the closest chunks are added to its prompt.

import sqlite3
import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
INDEX_PATH = Path(__file__).parent / "index.db"
CHUNK_MSGS = 12        # messages per chunk
CHUNK_GAP = 3600       # an hour of silence starts a new chunk
RECALL_K = 4           # chunks added per reply
RECALL_MIN = float(os.getenv("RECALL_MIN", "0.5"))   # minimum similarity to count as relevant

db = sqlite3.connect(INDEX_PATH)
db.executescript("""
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, channel_id INTEGER,
                                   ts REAL, text TEXT, vec BLOB);
CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, scope TEXT, channel_id INTEGER, channel TEXT,
                                     author_id INTEGER, author TEXT, ts REAL, text TEXT, chunk_id INTEGER);
CREATE TABLE IF NOT EXISTS live (channel_id INTEGER PRIMARY KEY, scope TEXT);
CREATE INDEX IF NOT EXISTS messages_chunk ON messages(chunk_id);
CREATE INDEX IF NOT EXISTS chunks_scope ON chunks(scope);
""")
live_channels = {r[0] for r in db.execute("SELECT channel_id FROM live")}
pending = defaultdict(list)      # live messages waiting to fill a chunk, per channel
vec_cache = {}                   # scope -> (timestamps, texts, matrix, channel ids)
index_lock = asyncio.Lock()      # one /index job at a time


def msg_row(m: discord.Message, label=None):
    text = m.clean_content + (" [image]" if m.attachments else "")
    return {"id": m.id, "channel": getattr(m.channel, "name", "dm"), "author_id": m.author.id,
            "author": label or m.author.display_name, "ts": m.created_at.timestamp(), "text": text[:4000]}


def render_chunk(rows):
    day = datetime.datetime.fromtimestamp(rows[0]["ts"]).strftime("%b %d, %Y")
    return f"[{day}, #{rows[0]['channel']}]\n" + "\n".join(f"{r['author']}: {r['text']}" for r in rows)


def group_rows(rows):
    groups, cur = [], []
    for r in rows:
        if cur and (len(cur) >= CHUNK_MSGS or r["ts"] - cur[-1]["ts"] > CHUNK_GAP):
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    return groups


async def embed(texts, kind="document"):
    # num_gpu 0 keeps the small embedding model on the cpu, leaving the whole gpu for the chat model
    # (a big chat model fills it; sharing made every reply slow). cpu embedding a message takes milliseconds.
    data = await ollama_json("/api/embed", {"model": EMBED_MODEL, "keep_alive": KEEP_ALIVE, "options": {"num_gpu": 0},
                                            "input": [f"search_{kind}: {t[:2000]}" for t in texts]})
    v = np.array(data["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


async def store_chunks(scope, channel_id, groups):
    for i in range(0, len(groups), 32):
        batch = groups[i:i + 32]
        texts = [render_chunk(g) for g in batch]
        for g, text, vec in zip(batch, texts, await embed(texts)):
            cur = db.execute("INSERT INTO chunks (scope, channel_id, ts, text, vec) VALUES (?, ?, ?, ?, ?)",
                             (scope, channel_id, g[-1]["ts"], text, vec.tobytes()))
            db.executemany("INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [(r["id"], scope, channel_id, r["channel"], r["author_id"], r["author"], r["ts"],
                             r["text"], cur.lastrowid) for r in g])
    db.commit()
    vec_cache.pop(scope, None)


def load_vecs(scope):
    if scope not in vec_cache:
        rows = db.execute("SELECT ts, text, vec, channel_id FROM chunks WHERE scope = ?", (scope,)).fetchall()
        vec_cache[scope] = ([r[0] for r in rows], [r[1] for r in rows],
                            np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]),
                            np.array([r[3] for r in rows], dtype=np.int64)) if rows else None
    return vec_cache[scope]


async def recall(channel, query):
    data = load_vecs(scope_key(channel))
    if not data or not query.strip():
        return []
    stamps, texts, matrix, channel_ids = data
    sims = matrix @ (await embed([query], "query"))[0]
    recent = time.time() - 600   # skip the last 10 minutes; that's already in chat memory
    found = []
    # only recall messages from this channel
    candidates = np.flatnonzero(channel_ids == channel.id)
    for i in candidates[np.argsort(-sims[candidates])[:20]]:
        if sims[i] < RECALL_MIN:
            break
        if stamps[i] < recent:
            found.append(texts[i])
        if len(found) >= RECALL_K:
            break
    return found


def flush_pending(channel_id, scope):
    """take the waiting messages now (so new ones start a fresh chunk) and embed them in the background."""
    rows, pending[channel_id] = pending[channel_id], []

    async def store():
        try:
            await store_chunks(scope, channel_id, [rows])
        except Exception as e:
            log.warning(f"Live index failed: {e}")
    if rows:
        asyncio.create_task(store())


async def live_add(m: discord.Message):
    """keep indexed channels up to date as new messages arrive."""
    if m.channel.id not in live_channels or m.author.bot or not (m.clean_content or m.attachments):
        return
    row, scope = msg_row(m), scope_key(m.channel)
    buf = pending[m.channel.id]
    if buf and row["ts"] - buf[-1]["ts"] > CHUNK_GAP:
        flush_pending(m.channel.id, scope)
    pending[m.channel.id].append(row)
    if len(pending[m.channel.id]) >= CHUNK_MSGS:
        flush_pending(m.channel.id, scope)


async def forget_messages(ids):
    """deleted discord messages are removed from the index too."""
    ids = list(ids)
    for channel_id in list(pending):
        pending[channel_id] = [r for r in pending[channel_id] if r["id"] not in ids]
    marks = ",".join("?" * len(ids))
    chunks = db.execute(f"SELECT DISTINCT chunk_id, scope FROM messages WHERE id IN ({marks})", ids).fetchall()
    if not chunks:
        return
    db.execute(f"DELETE FROM messages WHERE id IN ({marks})", ids)
    for chunk_id, scope in chunks:
        left = db.execute("SELECT channel, author, ts, text FROM messages WHERE chunk_id = ? ORDER BY ts",
                          (chunk_id,)).fetchall()
        if not left:
            db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        else:
            rows = [{"channel": c, "author": a, "ts": t, "text": x} for c, a, t, x in left]
            text = render_chunk(rows)
            vec = (await embed([text]))[0]
            db.execute("UPDATE chunks SET text = ?, vec = ? WHERE id = ?", (text, vec.tobytes(), chunk_id))
        vec_cache.pop(scope, None)
    db.commit()
    log.info(f"Removed {len(ids)} deleted message(s) from the index")


@client.event
async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent):
    await forget_messages([payload.message_id])


@client.event
async def on_raw_bulk_message_delete(payload: discord.RawBulkMessageDeleteEvent):
    await forget_messages(payload.message_ids)


async def show_progress(progress: discord.Message, text):
    # keep indexing if the progress message cannot be edited
    try:
        await progress.edit(content=text)
    except discord.HTTPException:
        pass


async def run_index(channel, progress: discord.Message):
    """index a channel's history, then keep it updated live. saves every 150 messages, so retrieval improves
    while the job is still running. returns (messages read, messages saved)."""
    scope = scope_key(channel)
    existing = {r[0] for r in db.execute("SELECT id FROM messages WHERE channel_id = ?", (channel.id,))}
    rows, seen, added, last_update = [], 0, 0, time.monotonic()

    async def save():
        nonlocal rows, added
        if rows:
            await store_chunks(scope, channel.id, group_rows(rows))
            added += len(rows)
            rows = []

    async for m in channel.history(limit=None, oldest_first=True):
        seen += 1
        if not m.author.bot and m.id not in existing and (m.clean_content or m.attachments):
            rows.append(msg_row(m))
        if len(rows) >= 150:
            await save()
        if time.monotonic() - last_update > 5:
            await show_progress(progress, f"📚 Indexing {channel.mention}… read {seen:,} messages, saved {added + len(rows):,}")
            last_update = time.monotonic()
    await save()
    db.execute("INSERT OR REPLACE INTO live VALUES (?, ?)", (channel.id, scope))
    db.commit()
    live_channels.add(channel.id)
    await show_progress(progress, f"✅ Indexed {channel.mention}: read {seen:,} messages, saved {added:,} new ones. "
                                  "New messages here are added automatically.")
    return seen, added


index = app_commands.Group(name="index", description="Search memory built from past channel messages")


async def start_job(inter, channel, job):
    """channel=none means the job covers every channel (each is permission-checked as it goes)."""
    if index_lock.locked():
        return await inter.response.send_message("An index job is already running. Try again when it's done.", ephemeral=True)
    # check bot permissions in this channel
    here = inter.app_permissions
    if channel is not None:
        perms = here if channel.id == inter.channel_id else channel.permissions_for(channel.guild.me)
        missing = [name for name, ok in (("View Channel", perms.view_channel),
                                         ("Read Message History", perms.read_message_history)) if not ok]
        if missing:
            return await inter.response.send_message(
                f"I can't read <#{channel.id}>. Give the bot **{'** and **'.join(missing)}** in that channel's "
                "permissions (Edit Channel → Permissions), then try again.", ephemeral=True)
    where = f"<#{channel.id}>" if channel else "every channel"
    # show progress in the channel, or privately if needed
    if here.view_channel and here.send_messages:
        await inter.response.send_message("Starting…", ephemeral=True)
        progress = await inter.channel.send(f"📚 Starting on {where}…")
    else:
        await inter.response.send_message(f"📚 Starting on {where}…", ephemeral=True)
        progress = await inter.original_response()

    async def run():
        async with index_lock:
            try:
                await job(progress)
            except Exception as e:
                log.exception("Index job failed")
                await progress.edit(content=f"⚠️ Indexing stopped: {e}")
    asyncio.create_task(run())


@index.command(name="channel", description="Index every message in a channel so the bot can remember it")
@app_commands.describe(channel="Channel to index (defaults to this one)")
@staff_only()
async def index_channel(inter: discord.Interaction, channel: Optional[discord.TextChannel] = None):
    channel = channel or inter.channel
    await start_job(inter, channel, lambda progress: run_index(channel, progress))


@index.command(name="status", description="How much is indexed in this server")
async def index_status(inter: discord.Interaction):
    scope = scope_key(inter.channel)
    chunks, msgs = db.execute("SELECT COUNT(*), (SELECT COUNT(*) FROM messages WHERE scope = ?) FROM chunks WHERE scope = ?",
                              (scope, scope)).fetchone()
    live = [f"<#{r[0]}>" for r in db.execute("SELECT channel_id FROM live WHERE scope = ?", (scope,))]
    await inter.response.send_message(
        f"📚 **{msgs:,}** messages in **{chunks:,}** chunks\n**Auto-updating:** {', '.join(live) or 'none'}", ephemeral=True)


@index.command(name="clear", description="Delete a channel's index (or the whole server's) and stop auto-updating")
@app_commands.describe(channel="Channel to clear. Leave empty to clear the whole server")
@staff_only()
async def index_clear(inter: discord.Interaction, channel: Optional[discord.TextChannel] = None):
    scope = scope_key(inter.channel)
    where, args = ("channel_id = ?", (channel.id,)) if channel else ("scope = ?", (scope,))
    for table in ("chunks", "messages", "live"):
        db.execute(f"DELETE FROM {table} WHERE {where}", args)
    db.commit()
    vec_cache.pop(scope, None)
    live_channels.clear()
    live_channels.update(r[0] for r in db.execute("SELECT channel_id FROM live"))
    await inter.response.send_message(f"🧹 Cleared the index for {channel.mention if channel else 'this server'}.")


tree.add_command(index)


# member directory
# everyone's usernames, display names, and server nicknames, so the bot gets names right. filled from
# everyone who talks, and from the whole member list when "server members intent" is on.

db.execute("""CREATE TABLE IF NOT EXISTS members (scope TEXT, user_id INTEGER, username TEXT, display_name TEXT,
              nick TEXT, joined REAL, gone INTEGER DEFAULT 0, PRIMARY KEY (scope, user_id))""")
roster_cache = {}      # scope -> {user_id: (username, display_name, nick)}
member_seen = {}       # (scope, user_id) -> names last saved, to skip redundant writes


def note_member(scope, u, gone=False, commit=True):
    names = (u.name, u.global_name, getattr(u, "nick", None), gone)
    if member_seen.get((scope, u.id)) == names:
        return
    member_seen[(scope, u.id)] = names
    joined = u.joined_at.timestamp() if getattr(u, "joined_at", None) else None
    db.execute("""INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?)
                  ON CONFLICT (scope, user_id) DO UPDATE SET username = excluded.username,
                  display_name = excluded.display_name, nick = COALESCE(excluded.nick, members.nick),
                  joined = COALESCE(excluded.joined, members.joined), gone = excluded.gone""",
               (scope, u.id, u.name, u.global_name, getattr(u, "nick", None), joined, int(gone)))
    if commit:
        db.commit()
    roster_cache.pop(scope, None)


def roster(scope):
    if scope not in roster_cache:
        roster_cache[scope] = {uid: (un, dn, nick) for uid, un, dn, nick in
                               db.execute("SELECT user_id, username, display_name, nick FROM members "
                                          "WHERE scope = ? AND gone = 0", (scope,))}
    return roster_cache[scope]


def roster_person(scope, user_id):
    names = roster(scope).get(user_id)
    if not names:
        return None
    username, display_name, nick = names
    return Person(user_id, nick or display_name or username, (nick, display_name, username))


def named_in(scope, text, limit=8):
    """people whose username, display name, nickname, or first name appears in the text."""
    text = text.lower()
    found = []
    for uid, (username, display_name, nick) in roster(scope).items():
        aliases = {a.lower() for a in (username, display_name, nick) if a}
        aliases |= {a.split()[0] for a in aliases if " " in a and len(a.split()[0]) >= 4}
        if any(len(a) >= 3 and re.search(rf"(?<![\w]){re.escape(a)}(?![\w])", text) for a in aliases):
            found.append(roster_person(scope, uid))
            if len(found) >= limit:
                break
    return found


def sync_guild_members(guild):
    count = 0
    for member in guild.members:
        if not member.bot:
            note_member(f"g{guild.id}", member, commit=False)
            count += 1
    db.commit()
    return count


@client.event
async def on_member_join(member: discord.Member):
    if not member.bot:
        note_member(f"g{member.guild.id}", member)


@client.event
async def on_member_update(before: discord.Member, after: discord.Member):
    if not after.bot:
        note_member(f"g{after.guild.id}", after)


@client.event
async def on_member_remove(member: discord.Member):
    if not member.bot:
        note_member(f"g{member.guild.id}", member, gone=True)


@client.event
async def on_user_update(before: discord.User, after: discord.User):
    for guild in after.mutual_guilds:
        member = guild.get_member(after.id)
        if member and not member.bot:
            note_member(f"g{guild.id}", member)


@index.command(name="members", description="Save everyone's names in this server so the bot gets them right")
@staff_only()
async def index_members(inter: discord.Interaction):
    if not inter.guild:
        return await inter.response.send_message("Only works in a server.", ephemeral=True)
    if not MEMBERS_INTENT:
        known = len(roster(scope_key(inter.channel)))
        return await inter.response.send_message(
            "To read the full member list, turn on **Server Members Intent**: Discord Developer Portal → your app → "
            f"**Bot** → Privileged Gateway Intents → save, then restart the bot. Until then it learns names from "
            f"people who talk ({known:,} so far).", ephemeral=True)
    await inter.response.defer(thinking=True)
    count = 0
    async for member in inter.guild.fetch_members(limit=None):
        if not member.bot:
            note_member(scope_key(inter.channel), member, commit=False)
            count += 1
    db.commit()
    await inter.followup.send(f"👥 Saved names for {count:,} members. New members and name changes are added automatically.")


def build_system(channel, conf, query=""):
    """returns (system prompt, example turns, extra context). the first two are stable between messages so
    the server can reuse them from its prompt cache."""
    return system_prompt(channel, conf), example_turns(conf), ""


# ascii art
# text banners and image conversion are done by code (always clean); /ascii draw uses the model with a tuned prompt.

import io
import pyfiglet
from PIL import Image, ImageOps

# use fonts with explicit upstream license notices
FIGLET_FONTS = sorted({"standard", "small", "mini", "slant", "big", "banner"}
                      & set(pyfiglet.FigletFont.getFonts()))
ascii_cmd = app_commands.Group(name="ascii", description="ASCII art")


def fenced(art):
    return f"```\n{art.rstrip()}\n```"


async def font_autocomplete(inter: discord.Interaction, current: str):
    picks = [f for f in FIGLET_FONTS if current.lower() in f.lower()]
    return [app_commands.Choice(name=f, value=f) for f in picks[:25]]


@ascii_cmd.command(name="text", description="Big ASCII lettering from text")
@app_commands.describe(words="What to write", font="Lettering style: standard, small, mini, slant, big, or banner")
@app_commands.autocomplete(font=font_autocomplete)
async def ascii_text(inter: discord.Interaction, words: app_commands.Range[str, 1, 40], font: str = "standard"):
    if font not in FIGLET_FONTS:
        return await inter.response.send_message(f"No font called `{font}`. Pick one from the list.", ephemeral=True)
    art = pyfiglet.figlet_format(words, font=font, width=70)
    if len(art) > 1900:
        return await inter.response.send_message("Too big for one Discord message. Try fewer words or `font: small`.", ephemeral=True)
    await inter.response.send_message(fenced(art))


def image_to_ascii(data, width, invert):
    ramp = " .:-=+*#%@"   # dim to bright; bright pixels get dense characters, which suits discord's dark theme
    if invert:
        ramp = ramp[::-1]
    img = ImageOps.autocontrast(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L"))
    height = max(1, round(img.height / img.width * width * 0.5))   # characters are about twice as tall as wide
    height = min(height, 1880 // (width + 1))                         # fit in one message
    img = img.resize((width, height))
    px = img.load()
    return "\n".join("".join(ramp[px[x, y] * (len(ramp) - 1) // 255] for x in range(width)) for y in range(height))


@ascii_cmd.command(name="image", description="Turn a picture into ASCII art")
@app_commands.describe(image="The picture", width="Characters wide (20-70, default 50)",
                       light_mode="Turn on if you use Discord's light theme")
async def ascii_image(inter: discord.Interaction, image: discord.Attachment,
                      width: app_commands.Range[int, 20, 70] = 50, light_mode: bool = False):
    if not (image.content_type or "").startswith("image/"):
        return await inter.response.send_message("That's not an image.", ephemeral=True)
    await inter.response.defer(thinking=True)
    art = await asyncio.to_thread(image_to_ascii, await image.read(), width, light_mode)
    await inter.followup.send(fenced(art))


ASCII_ARTIST = """You are an expert ASCII artist. Draw what's asked using only plain keyboard characters.
Rules: at most 40 characters wide and 20 lines tall. Draw a clear, recognizable outline of the subject with simple
shapes; don't fill areas with noise. Keep lines aligned, since it's shown in a monospace font. Output only the art:
no code fences, no title, no explanation."""

ASCII_EXAMPLES = [
    ("a cat", " /\\_/\\\n( o.o )\n > ^ <"),
    ("a house", "    /\\\n   /  \\\n  /____\\\n  |    |\n  | [] |\n  |____|"),
    ("a tree", "     &&&\n   &&&&&&&\n  &&&&&&&&&\n   &&&&&&&\n     | |\n     | |\n   __|_|__"),
]


@ascii_cmd.command(name="draw", description="The bot draws something as ASCII art")
@app_commands.describe(thing="What to draw, e.g. a cat, a rocket, a skull")
async def ascii_draw(inter: discord.Interaction, thing: app_commands.Range[str, 1, 100]):
    await inter.response.defer(thinking=True)
    model = cfg(inter.channel)["model"]
    messages = [{"role": "system", "content": ASCII_ARTIST}]
    for ask, art in ASCII_EXAMPLES:
        messages += [{"role": "user", "content": f"Draw {ask}"}, {"role": "assistant", "content": art}]
    messages.append({"role": "user", "content": f"Draw {thing}"})
    art = (await chat_once(model, messages, temperature=0.6, strip=False)).replace("```", "").strip("\n")
    if not art.strip():
        return await inter.followup.send("Couldn't draw that one. Try again?")
    await send_long(inter, fenced(art))


tree.add_command(ascii_cmd)


# likes and dislikes

def taste_group(kind):
    group = app_commands.Group(name=kind, description=f"Things the bot {kind}")

    @group.command(name="add", description=f"Add something the bot {kind}")
    @staff_only()
    async def add(inter: discord.Interaction, thing: app_commands.Range[str, 1, 100]):
        items = cfg(inter.channel)[kind] + [thing]
        if len(items) > 25:
            return await inter.response.send_message(f"That's 25 already. Remove one with `/{kind} remove`.", ephemeral=True)
        set_cfg(inter.channel, **{kind: items})
        await inter.response.send_message(f"{'💚' if kind == 'likes' else '💢'} Now {kind}: **{thing}**")

    @group.command(name="remove", description=f"Remove one by its number from /{kind} list")
    @staff_only()
    async def remove(inter: discord.Interaction, number: int):
        items = cfg(inter.channel)[kind]
        if not 1 <= number <= len(items):
            return await inter.response.send_message(f"No #{number}.", ephemeral=True)
        removed = items.pop(number - 1)
        set_cfg(inter.channel, **{kind: items or None})
        await inter.response.send_message(f"Removed: {removed}", ephemeral=True)

    @group.command(name="list", description=f"Show what the bot {kind}")
    async def show(inter: discord.Interaction):
        items = cfg(inter.channel)[kind]
        text = "\n".join(f"**{i}.** {t}" for i, t in enumerate(items, 1)) or f"Nothing yet. Add with `/{kind} add`."
        await inter.response.send_message(f"**The bot {kind}:**\n{text}", ephemeral=True)

    tree.add_command(group)


taste_group("likes")
taste_group("dislikes")


if __name__ == "__main__":
    client.run(TOKEN, log_handler=None)

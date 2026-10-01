# discord-inference-bot

a self-hosted discord bot that uses [ollama](https://ollama.com) for local inference. it streams replies, remembers recent chat, and can search indexed history from the channel it is replying in.

no paid inference api is required. you still need a discord bot token and a machine that can run your chosen models.

intended for private discord servers only, with informed consent from all participants whose messages, images, or member information the bot processes. do not use it in public servers or to collect data from people who have not consented.

## features

- streaming replies when mentioned, replied to, messaged directly, or enabled with `/autoreply`
- recent conversation memory per channel
- channel history retrieval using local embeddings and sqlite
- member names and nicknames
- image understanding with a vision model
- trivia, summaries, reminders, saved facts, and ascii art
- latency and generation speed metrics with `/stats`
- model unloading and reloading with `/shutdown` and `/startup`

## setup

you need python 3.10 or newer, ollama, and a discord application with a bot. model speed and memory requirements depend on your hardware.

```bash
git clone https://github.com/burdena0/discord-inference-bot.git
cd discord-inference-bot
```

on windows powershell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

on linux or macos:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

install ollama and pull the models:

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

you can use another installed chat model by changing `OLLAMA_MODEL` in `.env`. vision commands need a model that accepts images.

create an application and bot in the [discord developer portal](https://discord.com/developers/applications). enable **message content intent**. enable **server members intent** if you want the full member directory.

invite the bot with the `bot` and `applications.commands` scopes. give it the channel permissions needed for your features: view channels, send messages, read message history, embed links, attach files, and add reactions. administrator access is not needed.

put your bot token in `.env`, then run it from the repo folder. on windows:

```powershell
.\.venv\Scripts\python.exe bot.py
```

on linux or macos:

```bash
.venv/bin/python bot.py
```

windows also has `start-bot.bat`. it opens a minimized window and restarts the bot after a crash. it uses the repo's virtual environment if present, otherwise `python` from your path. logs go to `bot.log`.

## configuration

edit `.env` locally. keep environment variable names uppercase.

| variable | purpose |
|---|---|
| `DISCORD_TOKEN` | discord bot token |
| `OLLAMA_MODEL` | chat model name |
| `OLLAMA_URL` | ollama address; defaults to `http://127.0.0.1:11434` |
| `EMBED_MODEL` | embedding model for indexed history |
| `NUM_CTX` | context window size |
| `MAX_HISTORY` | recent messages kept per channel |
| `AUTO_RESET_MINUTES` | idle time before chat memory resets |
| `KEEP_ALIVE` | how long ollama keeps the model loaded |
| `COMMAND_ROLES` | extra role names or ids allowed to change settings, comma separated |
| `SHUTDOWN_USERS` | extra user ids allowed to start or shut down the bot, comma separated |
| `SYSTEM_PROMPT` | default personality |
| `FUNNY_MIN` | funny score for big ascii replies; `11` disables them |
| `RECALL_MIN` | retrieval similarity threshold; defaults to `0.5` |

server settings changed through commands are saved in `settings.json`.

## commands

| command | purpose |
|---|---|
| `/system`, `/example`, `/likes`, `/dislikes` | personality and example replies |
| `/model`, `/think`, `/autoreply` | model and chat behavior |
| `/index channel`, `/index members`, `/index status`, `/index clear` | indexed history and member directory |
| `/remember`, `/facts`, `/forget` | saved facts about people |
| `/trivia`, `/leaderboard`, `/wouldyourather`, `/hottake` | games |
| `/summarize`, `/remindme`, `/ratepic`, `/ascii` | utilities |
| `/stats`, `/status`, `/reset` | metrics, model status, recent chat reset |
| `/shutdown`, `/startup` | unload or reload the model |

settings and indexing changes require server management permissions or a configured command role. shutdown and startup require the application owner, application team, or a configured user id. users can add or remove their own saved facts; staff can manage facts about others.

## data and privacy

the bot connects to discord and sends prompts to the configured ollama address. with the default loopback address, inference stays on the bot's machine. using a remote ollama server sends that context to the remote machine.

`index.db` contains indexed messages, embeddings, author information, and member names. retrieval uses only messages indexed from the channel receiving the reply. member names and saved facts are shared within a server.

`settings.json` contains settings, facts, trivia scores, and reminders. `metrics.csv` contains timing and model metrics. logs can contain operational details and errors. these files and `.env` are excluded from git.

use the bot only in private servers with informed consent from the participants. explain what it processes and stores, including messages, images, member information, saved facts, and indexed history. get consent before enabling it or indexing existing history, and stop processing a participant's data if they withdraw consent. this is an operator requirement; the bot does not enforce or track consent automatically.

avoid indexing sensitive channels. deletion events received while the bot is running remove messages from the index. deletions missed while it is offline are not reconciled automatically.

use `/index clear` to clear indexed messages and stop live indexing, `/forget` to remove facts, and `/reset` to clear recent chat memory. member records are separate. to remove all local data, stop the bot and delete `index.db`, any `index.db-*` sidecar files, `settings.json`, `metrics.csv`, and logs. this also removes settings and reminders.

## limitations

this is a hobby project. responses can be wrong, and chat messages are untrusted model input. do not put secrets in prompts. keep ollama on a trusted network. throughput, vision support, and reply quality depend on the selected model and hardware.

## contributing

bug reports are welcome. contact the owner before submitting code contributions so contribution permissions can be agreed on. keep comments lowercase and direct. do not include bot tokens, chat exports, logs, or runtime databases.

## license

this is public source code, with no open-source or general reuse license. see [COPYRIGHT](COPYRIGHT). rights granted by github's terms and applicable law still apply, including viewing and forking through github.

dependencies and models retain their own licenses. see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the licensing review. ascii text fonts are limited to `standard`, `small`, `mini`, `slant`, `big`, and `banner`, which have explicit upstream license notices.

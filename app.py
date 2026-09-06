# -*- coding: utf-8 -*-
"""
app.py
UAI - Streamlit Webアプリ（自由チャット版・社会的推理ゲーム）

役職構成（合計5名 / AI-01〜AI-05）:
  - 人間（プレイヤー）    × 1
  - エミュレーター(特殊AI) × 1  … 人間に擬態。人間が生き残れば同時勝利。
  - 占い師AI(特殊AI)      × 1  … 夜に1名の正体を密かに調査できる（本人にも他人にも非公開）。
  - 一般AI                × 2  … 人間を見つけ出すのが目的。

昼フェーズ: 固定の議題は無く、制限時間内は自由にチャットができる。
            人間はいつでも発言可能。AIは人間の発言に反応したり、
            一定間隔で自発的に発言に加わったりする。
夜フェーズ: 全員が「人間だと思う相手」に投票 → 最多票を追放
  - 追放者が「人間」        → 一般AI側の勝利でゲーム終了
  - 追放者が「AI」          → 役職非公開のまま人数減、残り2名(人間1+AI1)で
                              人間＆エミュレーター側の勝利としてゲーム終了
  - 最多票が同数(2名以上)   → 誰も追放されず、そのまま次の日の昼フェーズへ
"""

import base64
import io
import itertools
import os
import random
import re
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

from tts import (
    synthesize_sentence,
    concat_mp3_clips,
    get_mp3_duration_seconds,
)
from prompts import (
    ROLE_HUMAN,
    ROLE_EMULATOR,
    ROLE_GENERAL_AI,
    ROLE_SEER_AI,
    ROLE_LABEL_JP,
    SKIP_VOTE,
    SKIP_LABEL_JP,
    PERSONALITY_POOL,
    build_chat_reply_messages,
    build_vote_messages,
    build_seer_investigation_messages,
    build_vote_reveal_text,
    try_parse_vote,
    try_parse_seer_target,
)

# ======================================================================
# 基本設定
# ======================================================================
load_dotenv()  # ローカル実行時: .env を読み込む

MODEL_NAME = "openrouter/free"
BASE_URL = "https://openrouter.ai/api/v1"

# ---- 音声入力(STT / OpenRouter Whisper) 関連設定 ----
# OpenRouterの /audio/transcriptions エンドポイント用モデル。
# LLM呼び出しと同じ base_url・APIキーで叩けるため、専用のAPIキーは不要。
DEFAULT_STT_MODEL_NAME = "openai/whisper-large-v3"

SEATS = [f"AI-{i:02d}" for i in range(1, 6)]

# 画面上に「現在のテーマ」として表示する呼び水（アイスブレイク用）。
# ゲームの進行やAIの判定ロジックには一切影響しない、純粋な表示上の演出。
# 実際の会話内容はこれまで通り完全に自由（テーマに沿う必要はない）。
DISCUSSION_TOPICS = [
    "AIは人間の仕事を奪うべきか？",
    "もし明日から記憶が無くなるとしたら、何を最初にする？",
    "AIに「心」は存在しうるか？",
    "理想の1日の過ごし方とは？",
    "人間らしさとは、結局何なのか？",
    "もし1つだけ超能力が使えるなら？",
]

DAY_PHASE_SECONDS = 180          # 昼フェーズ（自由チャット）の制限時間（秒）
AI_SPEAK_MIN_INTERVAL = 6        # AIが自発的に発言する最短間隔（秒）
AI_SPEAK_MAX_INTERVAL = 12       # AIが自発的に発言する最長間隔（秒）
IDLE_CHECK_INTERVAL_MS = 4000    # 待機中にサーバー側で状態確認する間隔（ミリ秒）
MULTI_SPEAK_CHANCE = 0.35        # 複数のAIが同時に発言を考え始める確率
MAX_SIMULTANEOUS_SPEAKERS = 2    # 同時に発言を考えるAIの最大数
LLM_TIMEOUT_SECONDS = 15         # OpenRouterへの通信タイムアウト（秒）
                                  # これが無いと、応答が返ってこない場合に「考え中」のまま
                                  # 永遠に待ち続けてしまう。

# ---- 音声読み上げ(TTS / Fish Audio) 関連設定 ----
FISH_MODEL_NAME = "s2.1-pro-free"   # Fish Audioのモデル（無料枠。品質保証は無いが検証用途には十分）
TTS_MAX_WORKERS = 4                 # TTSリクエスト用スレッドプールの同時実行数
AIZUCHI_LINES = [                   # 複数AIが連続で話す際に、間に挟む短い相槌
    "うんうん。",
    "なるほどね。",
    "へえ、そうなんだ。",
    "ふむ……。",
    "そっか。",
]
AIZUCHI_INSERT_CHANCE = 0.6         # 相槌を挟む確率（連続発言が2件以上あるとき）
AUDIO_SLEEP_BUFFER_SECONDS = 0.2    # 音声の実長に足す余裕（レンダリング遅延などの吸収用）

st.set_page_config(
    page_title="UAI",
    page_icon="◈",
    layout="centered",
)


# ======================================================================
# APIキー取得（ローカル.env / Streamlit Cloud st.secrets 両対応）
# ======================================================================
def get_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENROUTER_API_KEY", "")
    return key


@st.cache_resource(show_spinner=False)
def get_client():
    api_key = get_api_key()
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
        default_headers={
            "HTTP-Referer": "https://github.com/",
            "X-Title": "Reverse Werewolf Game",
        },
    )


def get_stt_model_name() -> str:
    """音声入力(文字起こし)に使うOpenRouterのモデル名。未設定ならデフォルトを使う。"""
    name = ""
    try:
        name = st.secrets.get("OPENROUTER_STT_MODEL", "")
    except Exception:
        name = ""
    if not name:
        name = os.getenv("OPENROUTER_STT_MODEL", "")
    return name or DEFAULT_STT_MODEL_NAME


def transcribe_audio(audio_bytes: bytes, fmt: str = "wav") -> str:
    """
    録音した音声バイト列を、OpenRouterの音声文字起こしエンドポイント
    (POST /audio/transcriptions、Whisper系モデル)でテキストに変換する。

    重要（「必ず聞き取り失敗になる」バグの原因と対策）:
    以前はOpenAI Python SDKの client.audio.transcriptions.create(file=...) を
    使っていたが、これは内部的に multipart/form-data 形式でアップロードする。
    ところがOpenRouterの /audio/transcriptions エンドポイントは、
    現状このmultipart経由のアップロードがゲートウェイ側で壊れており
    （境界文字列のパースに失敗する既知の不具合）、毎回失敗していた。

    OpenRouterが案内している、正しく動作する形式は「音声をbase64にして
    JSONボディのinput_audioフィールドに乗せる」方式のため、ここでは
    OpenAI SDKを経由せず、requestsで直接そのJSON形式のリクエストを送る。

    fmt: 録音側（streamlit-mic-recorder）が実際に出力した形式
    （"wav"または"webm"）。input_audio.formatフィールドにそのまま渡す。

    失敗した場合は空文字列を返す（呼び出し側で「うまく聞き取れませんでした」
    という案内を出し、手入力へフォールバックする想定）。
    """
    if not audio_bytes:
        return ""
    api_key = get_api_key()
    if not api_key:
        return ""
    try:
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
        resp = requests.post(
            f"{BASE_URL}/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/",
                "X-Title": "Reverse Werewolf Game",
            },
            json={
                "model": get_stt_model_name(),
                "input_audio": {"data": b64_audio, "format": fmt or "wav"},
                "language": "ja",
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("text") or "").strip()
    except Exception as e:
        _record_debug_error("音声入力の文字起こし", e)
        return ""


def record_voice_input():
    """
    streamlit-mic-recorderの「🎤」/「⏹」ボタンを描画する。
    波形表示は無く、アイコンだけのシンプルなUI（発言欄の左に収まるよう、
    テキストラベルは付けずアイコンのみにしている）。

    just_once=True のため、録音が完了した直後の1回だけ音声データを返し、
    以降のrerunでは（再録音するまで）Noneを返す。これにより、呼び出し側で
    「同じ録音を何度も文字起こししてしまう」対策の重複チェックが不要になる。

    format引数（wav指定）は比較的新しいバージョンのstreamlit-mic-recorderに
    のみ存在するため、無い場合（古いバージョン）でも動くようフォールバックする。

    戻り値: (音声バイト列, フォーマット文字列("wav"など)) または (None, None)。
    """
    kwargs = dict(
        start_prompt="🎤",
        stop_prompt="⏹",
        just_once=True,
        use_container_width=True,
        key="voice_mic_recorder",
    )
    try:
        audio_dict = mic_recorder(format="wav", **kwargs)
    except TypeError:
        audio_dict = mic_recorder(**kwargs)
    if not audio_dict or not audio_dict.get("bytes"):
        return None, None
    return audio_dict["bytes"], audio_dict.get("format", "wav")



# ======================================================================
# 音声読み上げ(TTS / Fish Audio) 用のAPIキー・実行環境
# ======================================================================
def get_fish_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("FISH_AUDIO_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("FISH_AUDIO_API_KEY", "")
    return key


def get_fish_reference_id() -> str:
    """任意の声(reference_id)。未設定ならFish Audio側のデフォルト音声が使われる。"""
    ref = ""
    try:
        ref = st.secrets.get("FISH_AUDIO_REFERENCE_ID", "")
    except Exception:
        ref = ""
    if not ref:
        ref = os.getenv("FISH_AUDIO_REFERENCE_ID", "")
    return ref


def get_fish_reference_ids() -> list:
    """
    AIごとに声を変えるための、reference_id(声のID)のプール。
    FISH_AUDIO_REFERENCE_IDS（複数形）にカンマまたは改行区切りで
    最大5つ（座席数ぶん）まで設定できる。例:
        FISH_AUDIO_REFERENCE_IDS = "id1,id2,id3,id4,id5"
    各IDは https://fish.audio 上の音声モデルのIDで、URLや「コピー」ボタンから
    取得できる（詳しくは docs.fish.audio 参照）。
    未設定の場合は、従来の単一のFISH_AUDIO_REFERENCE_IDにフォールバックする
    （その場合、全AIが同じ声になる。さらにそれも未設定ならFish Audioの
    デフォルト音声になる）。
    """
    raw = ""
    try:
        raw = st.secrets.get("FISH_AUDIO_REFERENCE_IDS", "")
    except Exception:
        raw = ""
    if not raw:
        raw = os.getenv("FISH_AUDIO_REFERENCE_IDS", "")
    ids = [x.strip() for x in re.split(r"[,\n]+", raw) if x.strip()]
    if ids:
        return ids
    single = get_fish_reference_id()
    return [single] if single else []


def get_seat_reference_id(seat: str) -> str:
    """
    この座席(AI)に割り当てられた声(reference_id)を返す。
    initialize_game() でゲーム開始時に座席ごとの声を固定で割り振っており、
    同じ座席は日をまたいでも常に同じ声で喋る（＝「AIごとに声を持たせる」）。
    座席ごとの割り当てが無い場合（ゲーム未初期化・声プール未設定など）は、
    従来通りの単一reference_idにフォールバックする。
    """
    seat_voice_ids = st.session_state.get("seat_voice_ids") or {}
    return seat_voice_ids.get(seat) or get_fish_reference_id()


@st.cache_resource(show_spinner=False)
def get_tts_executor():
    """
    TTSリクエスト専用のスレッドプール。LLM呼び出し用のThreadPoolExecutorとは
    完全に分離し、アプリのプロセス寿命を通じて使い回す（セッションごとに
    毎回作り直さない）。文単位で非同期にFish Audioへ投げるための土台。
    """
    return ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS)


_JAPANESE_CHAR_RE = re.compile(r"[぀-ヿ一-鿿]")  # ひらがな・カタカナ・漢字の文字コード範囲

# 何度リトライしても失敗した場合の、キャラクター性を壊さない自然なフォールバック発言
_FALLBACK_LINES = [
    "……少し考えがまとまりません。",
    "うーん、うまく言葉にできませんでした。",
    "今は静観します。",
    "……ノイズが多くて、うまく聞き取れませんでした。",
]


def looks_japanese(text: str) -> bool:
    """テキストに日本語の文字(ひらがな/カタカナ/漢字)が含まれているかを簡易判定する。"""
    return bool(text) and bool(_JAPANESE_CHAR_RE.search(text))


def call_llm(messages, max_tokens=300, temperature=0.9):
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        # LLM_TIMEOUT_SECONDS を超えると、ここで例外として捕捉され、
        # 「考え中」のまま止まることなく処理が先に進む。
        _record_debug_error("LLM呼び出し", e)
        return f"[通信エラー: {e}]"


def call_llm_stream(messages, max_tokens=300, temperature=0.9, on_delta=None):
    """
    call_llm のストリーミング版。OpenRouterからテキストが断片(delta)で
    届くたびに on_delta(delta) を呼び出す。
    現在は音声合成が発言単位の一括生成に変わったため直接は使っていないが、
    ストリーミングが必要になった場合のために残してある。
    通信エラー時は例外を送出せず、call_llm と同じ形式の
    "[通信エラー: ...]" という文字列を返す。
    """
    client = get_client()
    text_parts = []
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            stream=True,
        )
        for event in stream:
            delta = ""
            try:
                delta = event.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                text_parts.append(delta)
                if on_delta:
                    on_delta(delta)
        return "".join(text_parts).strip()
    except Exception as e:
        _record_debug_error("LLM呼び出し(streaming)", e)
        return f"[通信エラー: {e}]"


def call_ai_chat_reply(role, seat_name, day, chat_log, alive_seats, personality=None, known_facts=None):
    """
    AIの発言を生成する。応答が空、または日本語になっていない場合は
    最大2回まで「必ず日本語で」と念押しして再試行し、それでも駄目な場合は
    「(応答なし)」のような不自然な文言ではなく、キャラクター性を保った
    フォールバック発言を返す。
    personality: このAIに割り当てられた個性辞書。発言のトーンに反映される。
    known_facts: 占い師AI自身が既に掴んでいる調査結果（占い師AI以外は通常None）。
    """
    messages = build_chat_reply_messages(
        role, seat_name, day, chat_log, alive_seats, personality=personality, known_facts=known_facts
    )
    text = ""

    for attempt in range(3):
        raw = call_llm(messages, max_tokens=220, temperature=0.95)
        candidate = raw.strip()

        is_comm_error = candidate.startswith("[通信エラー")
        if candidate and not is_comm_error and looks_japanese(candidate):
            text = candidate
            break

        # 失敗した場合、次の試行では日本語出力を強く念押しする
        messages = messages + [{
            "role": "user",
            "content": "重要: 必ず日本語のみで、150文字以内の1文を出力してください。英語や他の言語、空の返答は禁止です。",
        }]

    if not text:
        text = random.choice(_FALLBACK_LINES)

    if len(text) > 150:
        text = text[:148] + "…"
    return text


def call_ai_chat_reply_with_audio(
    role, seat_name, day, chat_log, alive_seats, personality=None, known_facts=None,
    tts_api_key=None, tts_reference_id=None, tts_executor=None,
):
    """
    call_ai_chat_reply と同じ検証・リトライ・フォールバックのロジックを保ちつつ、
    テキストが確定したら、その発言"全体"を1回のFish Audioリクエストで音声化する。

    以前は文単位でストリーミングTTSを行っていたが、reference_id（声のID）を
    指定していても/していなくても、Fish Audioへのリクエストを文ごとに分けて
    投げると発話ごとに声の抑揚や質感がわずかに変わって聞こえることがあり、
    「1文ごとに声が違う」という体感になっていた。1つの発言をまとめて
    1回のリクエストで音声化することで、その発言中の声は常に一貫する。

    tts_reference_id は呼び出し側（generate_replies_concurrently）で
    座席ごとに固定の声を渡す想定（get_seat_reference_id参照）。これにより
    「AIごとに違う声を持つ」が実現される。

    tts_api_key が無い（音声OFF / キー未設定）場合は音声合成自体を行わず、
    テキスト表示のテンポには一切影響しない。

    戻り値: (text, audio_clips)
      audio_clips は、その発言ぶんの音声バイト列を1件だけ含むリスト
      （音声OFF、または合成に失敗した場合は空リスト）。
    """
    messages = build_chat_reply_messages(
        role, seat_name, day, chat_log, alive_seats, personality=personality, known_facts=known_facts
    )
    text = ""
    voice_on = bool(tts_api_key and tts_executor)

    for attempt in range(3):
        raw = call_llm(messages, max_tokens=220, temperature=0.95)
        candidate = raw.strip()

        is_comm_error = candidate.startswith("[通信エラー")
        if candidate and not is_comm_error and looks_japanese(candidate):
            text = candidate
            break

        messages = messages + [{
            "role": "user",
            "content": "重要: 必ず日本語のみで、150文字以内の1文を出力してください。英語や他の言語、空の返答は禁止です。",
        }]

    if not text:
        text = random.choice(_FALLBACK_LINES)

    if len(text) > 150:
        text = text[:148] + "…"

    audio_clips = []
    if voice_on:
        # 発言全体を1回のリクエストでまとめて音声化する（文ごとの声のブレを防ぐ）。
        # 既に専用スレッド(generate_replies_concurrently)の中で呼ばれているため、
        # ここではさらにtts_executor経由に投げ直さず、素直に同期呼び出しでよい。
        audio = synthesize_sentence(text, tts_api_key, FISH_MODEL_NAME, tts_reference_id)
        if audio:
            audio_clips = [audio]

    return text, audio_clips


def call_ai_vote(role, seat_name, day, chat_log, candidates, known_facts=None, personality=None):
    """
    candidates: 投票先として選べる座席名のリスト（自分を除く生存者。SKIP_VOTEは含めない）
    known_facts: 占い師AIなど、確定情報を持つ役職のために渡す追加事実（文字列のリスト）。
    personality: このAIに割り当てられた個性辞書。
    戻り値は座席名、または SKIP_VOTE（スキップ）。
    """
    messages = build_vote_messages(
        role, seat_name, day, chat_log, candidates, known_facts=known_facts, personality=personality
    )
    raw = call_llm(messages, max_tokens=80, temperature=0.7)
    valid_choices = candidates + [SKIP_VOTE]
    vote = try_parse_vote(raw, valid_choices)
    if vote is None:
        vote = random.choice(valid_choices)
    return vote


def call_seer_choose_target(seat_name, day, chat_log, candidates, known_facts=None, personality=None):
    """
    占い師AI自身に、今夜調査する相手を1名選ばせる。
    candidates: 調査対象として選べる座席名のリスト（占い師自身を除く生存者）。
    known_facts: これまでの調査で判明済みの事実（文字列のリスト）。
    personality: このAIに割り当てられた個性辞書。
    戻り値は座席名。パース失敗時は候補からランダムに選ぶ（フォールバック）。
    """
    messages = build_seer_investigation_messages(
        seat_name, day, chat_log, candidates, known_facts=known_facts, personality=personality
    )
    raw = call_llm(messages, max_tokens=80, temperature=0.7)
    target = try_parse_seer_target(raw, candidates)
    if target is None:
        target = random.choice(candidates)
    return target


def pick_speaker(candidates, exclude_last=None):
    """候補の中から発言者を1名選ぶ。可能なら直前の発言者は避ける。"""
    if not candidates:
        return None
    pool = candidates
    if exclude_last and len(candidates) > 1:
        filtered = [s for s in candidates if s != exclude_last]
        if filtered:
            pool = filtered
    return random.choice(pool)


def decide_speakers(candidates, exclude_last=None):
    """
    次に発言するAIを1〜複数名選ぶ。一定確率で複数名が同時に選ばれ、
    「複数のAIが同時に考えて発言する」演出になる。
    """
    if not candidates:
        return []
    pool = candidates
    if exclude_last and len(candidates) > 1:
        filtered = [s for s in candidates if s != exclude_last]
        if filtered:
            pool = filtered

    n = 1
    if len(pool) >= 2 and random.random() < MULTI_SPEAK_CHANCE:
        n = min(MAX_SIMULTANEOUS_SPEAKERS, len(pool))

    return random.sample(pool, n)


def generate_replies_concurrently(
    speakers, day, chat_log, alive_seats, seat_roles,
    seat_personalities=None, seer_seat=None, seer_known_facts=None,
    tts_api_key=None, seat_reference_ids=None, tts_executor=None,
):
    """
    複数のAIの発言を、実際に並行して(同時に)通信・生成する。
    完了した順に (seat, text, audio_clips) のタプルを"その場で"yieldする
    ジェネレータ（以前は全員分をリストにまとめてから一括で返していた）。

    重要（「誰かが喋っている間も他のAIは裏で考え続ける」ようにするための変更）:
    呼び出し側は、1件受け取るたびにすぐテキストを表示し、その音声の再生
    （time.sleep()を含む）を行ってよい。まだ完了していない他の座席の
    LLM生成・TTS合成は、このジェネレータの中のThreadPoolExecutorの
    別スレッド上でそのまま並行して進み続けるため、呼び出し側が
    1件の再生に時間をかけている間も、他のAIの「考え中」は止まらない
    （＝あるAIがFish Audioで喋っている間、他のAIは裏でLLM生成・音声合成を
    進めておける）。

    audio_clips はその発言ぶんの音声バイト列を1件だけ含むリスト（TTS無効/失敗時は []）。
    seat_personalities: {座席名: 個性辞書} の対応表。各AIの発言トーンに反映する。
    seer_seat / seer_known_facts: 占い師AIが話者に含まれる場合、自身の調査結果を
        会話の判断材料として渡すために使う（占い師AI以外には渡さない）。
    tts_api_key が渡された場合、各AIのテキスト生成後に、その座席専用の声
    (seat_reference_ids[seat]、get_seat_reference_id参照)でFish Audioへ
    1回だけTTSリクエストを投げる（call_ai_chat_reply_with_audio参照）。

    重要: ThreadPoolExecutorを `with` 文で使うと、ブロック終了時に
    「まだ終わっていないスレッドの完了を待つ」処理(shutdown(wait=True))が
    暗黙に走ってしまい、たとえ以下のタイムアウト処理が正しく働いていても、
    結局そこで固まってしまう（これが「永遠に考え中」の真因だった）。
    そのため、ここでは `with` を使わず、shutdown(wait=False) で
    「返事が来なくても待たずに関数を抜ける」ようにしている
    （ジェネレータなので、呼び出し側が最後まで受け取り終えた時、または
    　途中で受け取るのをやめた時のどちらでも、このfinallyは実行される）。
    """
    seat_personalities = seat_personalities or {}
    seat_reference_ids = seat_reference_ids or {}
    hard_timeout = LLM_TIMEOUT_SECONDS * 3 + 10  # リトライ3回分+余裕を持った絶対上限
    executor = ThreadPoolExecutor(max_workers=max(1, len(speakers)))
    try:
        future_to_seat = {
            executor.submit(
                call_ai_chat_reply_with_audio, seat_roles[seat], seat, day, chat_log, alive_seats,
                personality=seat_personalities.get(seat),
                known_facts=(seer_known_facts if seat == seer_seat else None),
                tts_api_key=tts_api_key, tts_reference_id=seat_reference_ids.get(seat), tts_executor=tts_executor,
            ): seat
            for seat in speakers
        }
        done_seats = set()
        try:
            for future in as_completed(future_to_seat, timeout=hard_timeout + 5):
                seat = future_to_seat[future]
                try:
                    text, audio_clips = future.result(timeout=hard_timeout)
                except Exception as e:
                    text, audio_clips = random.choice(_FALLBACK_LINES), []
                    _record_debug_error(seat, e)
                done_seats.add(seat)
                yield seat, text, audio_clips
        except Exception as e:
            # as_completed自体が全体タイムアウトした場合もここで捕捉し、必ず先に進める
            _record_debug_error("全体", e)

        # 何らかの理由で結果が得られなかった座席は、必ずフォールバックで埋める
        for future, seat in future_to_seat.items():
            if seat not in done_seats:
                _record_debug_error(seat, "全体タイムアウトにより打ち切り")
                yield seat, random.choice(_FALLBACK_LINES), []
    finally:
        # wait=False: 未完了のスレッドを待たずに、ここで即座に関数を抜ける
        executor.shutdown(wait=False, cancel_futures=True)


def _record_debug_error(seat, error):
    """画面下の「通信の状態」から確認できるよう、直近のエラーを保持しておく。"""
    if "debug_errors" not in st.session_state:
        st.session_state.debug_errors = []
    st.session_state.debug_errors.append(f"{seat}: {error}")
    st.session_state.debug_errors = st.session_state.debug_errors[-10:]


# ======================================================================
# 無機質UI用 CSS
# ======================================================================
CUSTOM_CSS = """
<style>
html, body, [class*="css"]  {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
.stApp {
    background-color: #0b0d10;
    color: #d7dbe0;
}
h1, h2, h3 {
    letter-spacing: 2px;
    color: #e7ecf0 !important;
    font-weight: 700 !important;
}

/* ---- 発言カード（チャット吹き出し） ---- */
.seat-row {
    display: flex;
    margin-bottom: 10px;
}
.seat-row.self { justify-content: flex-end; }
.seat-row.ai { justify-content: flex-start; }

.seat-card {
    border: 1px solid #2a2f36;
    background-color: #101317;
    border-radius: 10px;
    padding: 10px 14px;
    max-width: 78%;
}
.seat-card.self {
    border-color: #f0c67466;
    background-color: #14140f;
    border-top-right-radius: 2px;
}
.seat-card.ai {
    border-top-left-radius: 2px;
}
.seat-name {
    font-size: 11px;
    letter-spacing: 2.5px;
    color: #7fd3c7;
    font-weight: 700;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.seat-name.self {
    color: #f0c674;
    justify-content: flex-end;
}
.seat-avatar {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #1c2126;
    border: 1px solid #33393f;
    font-size: 10px;
}
.seat-text {
    font-size: 15px;
    color: #d7dbe0;
    line-height: 1.6;
}

/* ---- ステータスバー（生存状況） ---- */
.status-bar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid #2a2f36;
    background-color: #101317;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    letter-spacing: 1px;
    color: #b6bec6;
}
.status-chip .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4fd6a8;
    box-shadow: 0 0 6px #4fd6a8aa;
}
.status-chip.self {
    border-color: #f0c67488;
    color: #f0c674;
}
.status-chip.self .dot { background: #f0c674; box-shadow: 0 0 6px #f0c674aa; }
.status-chip.dead {
    color: #4a4f55;
    border-color: #23272c;
    background-color: #0c0e10;
}
.status-chip.dead .dot { background: #4a4f55; box-shadow: none; }
.status-chip.dead .chip-label { text-decoration: line-through; }

/* ---- フェーズバッジ ---- */
.phase-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid #23272c;
    background-color: #101317;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
}
.phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    letter-spacing: 2px;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
}
.phase-badge.day { color: #f0c674; border: 1px solid #f0c67466; background: #1a1610; }
.phase-badge.night { color: #b98ce6; border: 1px solid #b98ce666; background: #16121a; }
.phase-meta {
    font-size: 12px;
    color: #8a939b;
    letter-spacing: 1px;
}

/* ---- ボタン全般 ---- */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #14181d;
    color: #e7ecf0;
    border: 1px solid #3a4149;
    border-radius: 6px;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    transition: all 0.15s ease;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    border-color: #7fd3c7;
    color: #7fd3c7;
}

/* ---- 発言入力欄 ---- */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background-color: #101317;
    color: #e7ecf0;
    border: 1px solid #2a2f36;
    border-radius: 10px;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    resize: none;
}
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #7fd3c7;
    box-shadow: 0 0 0 1px #7fd3c766;
}
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px;
}

/* ---- 投票カード ---- */
.vote-card-selected {
    border: 1px solid #f0c674 !important;
    color: #f0c674 !important;
    background-color: #1a1610 !important;
}

/* ---- 得票バー ---- */
.tally-row {
    margin-bottom: 10px;
}
.tally-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #d7dbe0;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.tally-track {
    height: 8px;
    background: #1a1d21;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #23272c;
}
.tally-fill {
    height: 100%;
    background: linear-gradient(90deg, #7fd3c7, #4fd6a8);
    border-radius: 4px;
}
.tally-fill.max {
    background: linear-gradient(90deg, #f0c674, #e0a94b);
}
.tally-fill.skip {
    background: linear-gradient(90deg, #4a4f55, #6a7178);
}

/* ---- 役職公開カード ---- */
.reveal-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid #2a2f36;
    background-color: #101317;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 8px;
}
.reveal-card.you { border-color: #f0c67488; }
.reveal-seat {
    font-size: 13px;
    letter-spacing: 2px;
    color: #e7ecf0;
    font-weight: 700;
}
.reveal-role {
    font-size: 12px;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 20px;
}
.reveal-role.human { color: #f0c674; border: 1px solid #f0c67466; background: #1a1610; }
.reveal-role.emulator { color: #b98ce6; border: 1px solid #b98ce666; background: #16121a; }
.reveal-role.general_ai { color: #7fd3c7; border: 1px solid #7fd3c766; background: #0f1614; }
.reveal-role.seer_ai { color: #8ec6f0; border: 1px solid #8ec6f066; background: #0f1620; }

/* ---- タイトル画面 ---- */
.title-wrap {
    text-align: center;
    padding: 56px 0 12px;
    position: relative;
}
.title-eyebrow {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 12px;
    letter-spacing: 6px;
    color: #7fd3c7;
    margin-bottom: 14px;
    opacity: 0.85;
}
.title-main {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 62px;
    font-weight: 700;
    letter-spacing: 12px;
    color: #e7ecf0;
    text-shadow: 0 0 18px rgba(127, 211, 199, 0.45), 0 0 40px rgba(127, 211, 199, 0.15);
    animation: uai-pulse 3.2s ease-in-out infinite;
}
.title-sub {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 12px;
    letter-spacing: 4px;
    color: #8a939b;
    margin-top: 10px;
}
@keyframes uai-pulse {
    0%, 100% { text-shadow: 0 0 18px rgba(127, 211, 199, 0.45), 0 0 40px rgba(127, 211, 199, 0.15); }
    50% { text-shadow: 0 0 28px rgba(127, 211, 199, 0.75), 0 0 60px rgba(127, 211, 199, 0.3); }
}
.role-card {
    border: 1px solid #2a2f36;
    background-color: #101317;
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 12px;
    text-align: left;
}
.role-card.you { border-color: #f0c674aa; }
.role-card.emulator { border-color: #b98ce6aa; }
.role-card.seer { border-color: #8ec6f0aa; }
.role-card.ai { border-color: #7fd3c7aa; }
.role-card-title {
    font-size: 13px;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 6px;
}
.role-card.you .role-card-title { color: #f0c674; }
.role-card.emulator .role-card-title { color: #b98ce6; }
.role-card.seer .role-card-title { color: #8ec6f0; }
.role-card.ai .role-card-title { color: #7fd3c7; }
.role-card-desc {
    font-size: 13px;
    color: #9aa4ad;
    line-height: 1.6;
}
.flow-step {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    border: 1px solid #23272c;
    background-color: #0e1114;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.flow-step-num {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 18px;
    font-weight: 700;
    color: #7fd3c7;
    min-width: 26px;
}
.flow-step-body-title {
    font-size: 13px;
    letter-spacing: 1px;
    color: #e7ecf0;
    font-weight: 700;
    margin-bottom: 3px;
}
.flow-step-body-desc {
    font-size: 13px;
    color: #9aa4ad;
    line-height: 1.6;
}

/* ---- 小見出し ---- */
.section-label {
    font-size: 12px;
    letter-spacing: 2px;
    color: #7fd3c7;
    font-weight: 700;
    margin: 4px 0 10px;
}

/* ---- AIの読み上げ音声プレイヤーは画面に表示しない（自動再生のみ行う）。
       再生の仕組み自体はst.audio()のまま（表示/非表示はCSSだけの問題で、
       再生の安定性には影響しない）。
       st.container(key="ai_voice_dock")で囲んだ枠ごと消すことで、
       Streamlitのバージョンによる内部DOM構造の違いに左右されないようにする。
       data-testid直指定の行は、それでも拾いきれない場合の保険。 ---- */
div.st-key-ai_voice_dock {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stAudio"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ---- マイク録音ボタン(streamlit-mic-recorder)の外枠を、発言欄と
       高さを揃えて1つの入力バーのように見せる ---- */
div.st-key-mic_recorder_wrap {
    display: flex;
    align-items: center;
    height: 40px;
}
div.st-key-mic_recorder_wrap iframe {
    height: 40px !important;
    width: 100% !important;
}

/* ============================================================
   追加: サイバーパンク強化テーマ / 新UIパーツ
   ============================================================ */

/* ---- グロー・グラデーションのメインタイトル ---- */
.uai-glow-title {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: 10px;
    text-align: center;
    background: linear-gradient(90deg, #6ea8fe, #a084ee, #e879c9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 22px rgba(126, 132, 255, 0.35));
    animation: uai-pulse 3.2s ease-in-out infinite;
    margin-bottom: 4px;
}

/* ---- 5体シルエット行（タイトル画面） ---- */
.silhouette-row {
    display: flex;
    justify-content: center;
    gap: 18px;
    margin: 28px 0;
    flex-wrap: wrap;
}
.silhouette-slot {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}
.silhouette-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: radial-gradient(circle at 50% 30%, #232a3a, #10131a);
    border: 1px solid #3a4a6e;
    box-shadow: 0 0 14px rgba(110, 168, 254, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    color: #6ea8fe;
}
.silhouette-label {
    font-size: 11px;
    letter-spacing: 2px;
    color: #7d8aa0;
}

/* ---- 概要カード（タイトル画面） ---- */
.overview-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 18px 0;
}
.overview-card {
    border: 1px solid #2a2f4a;
    background: linear-gradient(160deg, #121527, #0b0d16);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.overview-card-label {
    font-size: 11px;
    letter-spacing: 2px;
    color: #7d8aa0;
    margin-bottom: 6px;
}
.overview-card-value {
    font-size: 16px;
    font-weight: 700;
    color: #cfe0ff;
}

/* ---- ゲーム画面ヘッダーカード ---- */
.header-card {
    border: 1px solid #2a2f4a;
    background: linear-gradient(160deg, #121527, #0b0d16);
    border-radius: 12px;
    padding: 14px 18px;
    height: 100%;
}
.header-card-label {
    font-size: 11px;
    letter-spacing: 2px;
    color: #7d8aa0;
    margin-bottom: 6px;
}
.header-theme-badge {
    display: inline-block;
    font-size: 11px;
    letter-spacing: 1px;
    color: #a084ee;
    border: 1px solid #a084ee66;
    background: #1a1530;
    border-radius: 20px;
    padding: 2px 10px;
    margin-bottom: 8px;
}
.header-theme-text {
    font-size: 16px;
    font-weight: 700;
    color: #eaf0ff;
    line-height: 1.5;
}
.header-rules-list {
    font-size: 12px;
    color: #9aa4ad;
    line-height: 1.9;
}

/* ---- タイマーリング ---- */
.timer-ring-label {
    font-size: 10px;
    letter-spacing: 2px;
    color: #7d8aa0;
    text-align: center;
    margin-bottom: 2px;
}

/* ---- 発言の感情タグ・タイムスタンプ ---- */
.emotion-tag {
    font-size: 10px;
    letter-spacing: 1px;
    color: #6ea8fe;
    border: 1px solid #6ea8fe55;
    background: #0f1830;
    border-radius: 10px;
    padding: 1px 8px;
    margin-left: 2px;
}
.msg-timestamp {
    font-size: 10px;
    color: #5a6272;
    margin-left: 8px;
    font-weight: 400;
    letter-spacing: 0;
}

/* ---- プレイヤー一覧パネル ---- */
.player-panel-card {
    border: 1px solid #2a2f4a;
    background: linear-gradient(160deg, #121527, #0b0d16);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.player-panel-title {
    font-size: 12px;
    letter-spacing: 2px;
    color: #7fd3c7;
    font-weight: 700;
    margin-bottom: 10px;
}
.player-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 2px;
    border-bottom: 1px solid #1c2035;
}
.player-row:last-child { border-bottom: none; }
.player-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
    border: 1px solid #3a4a6e;
    background: #171b2c;
    color: #cfe0ff;
}
.player-row.self .player-avatar {
    border-color: #6ea8fe;
    box-shadow: 0 0 10px #6ea8fe66;
}
.player-row.dead { opacity: 0.4; }
.player-info { flex: 1; min-width: 0; }
.player-name-line {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 700;
    color: #eaf0ff;
}
.player-you-tag {
    font-size: 10px;
    color: #6ea8fe;
    border: 1px solid #6ea8fe66;
    border-radius: 8px;
    padding: 0 6px;
}
.player-badge {
    font-size: 11px;
    color: #9aa4ad;
    margin-top: 2px;
}
.status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4fd6a8;
    box-shadow: 0 0 6px #4fd6a8aa;
    flex-shrink: 0;
}
.status-indicator.dead { background: #4a4f55; box-shadow: none; }
.status-indicator.speaking {
    background: #f0c674;
    box-shadow: 0 0 8px #f0c674cc;
    animation: uai-blink 1s ease-in-out infinite;
}
@keyframes uai-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

/* ---- 音声再生キュー ---- */
.voice-queue-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #9aa4ad;
    padding: 6px 2px;
    border-bottom: 1px solid #1c2035;
}
.voice-queue-item:last-child { border-bottom: none; }
.voice-queue-item.current { color: #f0c674; }
.voice-queue-icon { width: 14px; text-align: center; flex-shrink: 0; }
.voice-queue-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.voice-queue-dur { font-size: 10px; color: #5a6272; flex-shrink: 0; }

/* ---- ゲームステータス ---- */
.status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.status-grid-item {
    border: 1px solid #1c2035;
    border-radius: 8px;
    padding: 8px 10px;
}
.status-grid-label {
    font-size: 10px;
    color: #7d8aa0;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.status-grid-value {
    font-size: 15px;
    font-weight: 700;
    color: #eaf0ff;
}
.influence-track {
    height: 6px;
    background: #1a1d2c;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 6px;
}
.influence-fill {
    height: 100%;
    background: linear-gradient(90deg, #6ea8fe, #e879c9);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ======================================================================
# ゲーム状態の初期化
# ======================================================================
def reset_day_state():
    """
    新しい昼フェーズ（自由チャット）用の状態にリセットする。
    chat_log はここではクリアしない（ゲーム全体を通した会話履歴を保持し、
    AIが日をまたいでも記憶を保ったまま会話・投票できるようにするため）。
    """
    st.session_state.phase = "day"
    st.session_state.day_phase_start = time.time()
    st.session_state.next_ai_speak_time = time.time() + random.uniform(3, 7)
    st.session_state.force_end_day = False
    st.session_state.pending_speakers = []
    st.session_state.human_vote = None
    st.session_state.votes = {}
    st.session_state.votes_done = False
    st.session_state.pending_elimination = None
    st.session_state.tie_result = False
    st.session_state.night_vote_draft = None


def initialize_game():
    seats = SEATS.copy()
    roles = [ROLE_HUMAN, ROLE_EMULATOR, ROLE_SEER_AI, ROLE_GENERAL_AI, ROLE_GENERAL_AI]
    random.shuffle(roles)
    seat_roles = dict(zip(seats, roles))
    human_seat = next(s for s, r in seat_roles.items() if r == ROLE_HUMAN)
    seer_seat = next(s for s, r in seat_roles.items() if r == ROLE_SEER_AI)

    st.session_state.seat_roles = seat_roles
    st.session_state.human_seat = human_seat
    st.session_state.seer_seat = seer_seat

    # 座席ごとに声(reference_id)を1つ固定で割り当てる。
    # プールが座席数より少ない場合は使い回すが、多い分にはそのぶん多彩になる。
    # プールが空（FISH_AUDIO_REFERENCE_IDS未設定）の場合は座席ごとの割り当ては
    # 行わず、get_seat_reference_id() が単一reference_id/デフォルト音声に
    # フォールバックする。
    ref_pool = get_fish_reference_ids()
    seat_voice_ids = {}
    if ref_pool:
        shuffled_pool = ref_pool.copy()
        random.shuffle(shuffled_pool)
        for i, seat in enumerate(seats):
            seat_voice_ids[seat] = shuffled_pool[i % len(shuffled_pool)]
    st.session_state.seat_voice_ids = seat_voice_ids

    # 各AI座席にランダムな「個性」を1つずつ割り当てる（役職とは無関係）。
    # プレイヤーには一切表示されないが、発言のトーン・話し方に反映される。
    personalities = random.sample(PERSONALITY_POOL, len(seats))
    st.session_state.seat_personalities = dict(zip(seats, personalities))

    # 画面表示用の「現在のテーマ」（アイスブレイク。会話は引き続き完全に自由）。
    st.session_state.discussion_topic = random.choice(DISCUSSION_TOPICS)

    # 占い師AIがこれまでに調査して判明した正体（座席名 → 役職）。
    # ゲーム全体を通して蓄積され、人間プレイヤーには一切表示されない。
    st.session_state.seer_investigations = {}
    st.session_state.alive = seats.copy()
    st.session_state.day = 1

    # ゲーム全体を通した会話履歴（日をまたいで保持される）。
    # 各エントリ: {"day": int, "seat": str, "text": str}
    # seat == "SYSTEM" のエントリは投票結果公開など、ゲーム進行上の公開情報。
    st.session_state.chat_log = []

    st.session_state.game_over = False
    st.session_state.game_result = None
    st.session_state.eliminated_last = None

    reset_day_state()


def run_seer_investigation():
    """
    占い師AIの夜の調査を1回実行し、対象の正体を記憶させる。
    調査対象はランダムではなく、占い師AI自身がこれまでの会話ログと
    既知の調査結果を踏まえてLLMで選ぶ。
    結果は seer_investigations に蓄積され、占い師AI自身の投票判断にのみ使われる。
    人間プレイヤーの画面には一切表示しない（占い師AIの能力はゲーム内で完全に非公開）。
    """
    seer_seat = st.session_state.get("seer_seat")
    if not seer_seat or seer_seat not in st.session_state.alive:
        return

    known = st.session_state.seer_investigations
    alive_others = [s for s in st.session_state.alive if s != seer_seat]
    if not alive_others:
        return

    # まだ調査していない相手を優先。全員調査済みなら生存者の中から選び直す。
    candidates = [s for s in alive_others if s not in known]
    if not candidates:
        candidates = alive_others

    known_facts = seer_known_facts_text()
    target = call_seer_choose_target(
        seer_seat, st.session_state.day, st.session_state.chat_log, candidates,
        known_facts=known_facts,
        personality=st.session_state.get("seat_personalities", {}).get(seer_seat),
    )
    known[target] = st.session_state.seat_roles[target]


def seer_known_facts_text():
    """占い師AIがこれまでに掴んだ情報を、投票プロンプトに渡す文字列のリストにして返す。"""
    known = st.session_state.get("seer_investigations", {})
    facts = []
    for seat, role in known.items():
        if seat in st.session_state.alive:
            facts.append(f"{seat}の正体は「{ROLE_LABEL_JP[role]}」であると判明している。")
    return facts


if "screen" not in st.session_state:
    st.session_state.screen = "title"  # "title" | "game"

if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True  # Fish Audioキーが無ければ実質無効化される
if "aizuchi_cache" not in st.session_state:
    st.session_state.aizuchi_cache = {}


def render_player_panel():
    """
    「プレイヤー一覧」「音声再生キュー」「ゲームステータス」パネル。

    デザイン参考画像では画面右側の専用カラムに配置されているが、それを
    実現するにはチャット描画・発言処理ロジック全体を1つの列(with句)の中に
    包み直す必要があり、かなり大掛かりで既存ロジックを壊すリスクが高い。
    st.sidebarはスクリプト内のどこからでも呼び出せて常に独立した領域に
    表示されるため、ここではサイドバー（左側の折りたたみ可能な領域）に
    同じ内容を配置することで、既存の発言処理フローに一切手を加えずに
    実現している。
    """
    if "seat_roles" not in st.session_state:
        return

    human_seat = st.session_state.human_seat
    alive = st.session_state.get("alive", [])
    personalities = st.session_state.get("seat_personalities", {})
    speaking_now = set(st.session_state.get("pending_speakers", []))

    # ---- プレイヤー一覧 ----
    st.markdown('<div class="player-panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="player-panel-title">◈ プレイヤー一覧</div>', unsafe_allow_html=True)
    rows_html = ""
    for seat in SEATS:
        is_self = seat == human_seat
        is_alive = seat in alive
        is_speaking = seat in speaking_now
        row_cls = "player-row" + (" self" if is_self else "") + ("" if is_alive else " dead")
        status_cls = "status-indicator" + (" dead" if not is_alive else (" speaking" if is_speaking else ""))
        name_label = f"YOU ({seat})" if is_self else seat
        you_tag = '<span class="player-you-tag">あなた</span>' if is_self else ""
        p = personalities.get(seat)
        badge = p.get("name", "") if isinstance(p, dict) else ""
        # 注意: このHTMLは必ず1行（内部に改行を含めない）で組み立てること。
        # 複数のdivブロックをループでf"""...""" (複数行) のまま連結すると、
        # 各断片の先頭/末尾に残る改行+インデントが「空行」として扱われ、
        # Markdown側がそこでHTMLブロックの継続を打ち切ってしまう。
        # その結果、2つ目以降の断片が「インデント付きコードブロック」と
        # 誤認識され、タグがそのまま文字として表示されてしまう
        # （実際に発生した不具合）。
        rows_html += (
            f'<div class="{row_cls}">'
            f'<div class="player-avatar">{seat.split("-")[-1]}</div>'
            f'<div class="player-info">'
            f'<div class="player-name-line">{name_label} {you_tag}</div>'
            f'<div class="player-badge">{badge}</div>'
            f'</div>'
            f'<div class="{status_cls}"></div>'
            f'</div>'
        )
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- 音声再生キュー（直近の再生履歴） ----
    queue_log = st.session_state.get("voice_queue_log", [])
    if queue_log:
        st.markdown('<div class="player-panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="player-panel-title">◈ 音声再生キュー</div>', unsafe_allow_html=True)
        items_html = ""
        for i, item in enumerate(reversed(queue_log[-6:])):
            cls = "voice-queue-item" + (" current" if i == 0 else "")
            icon = "🔊" if i == 0 else "✓"
            # 上と同じ理由で、1行のまま組み立てる。
            items_html += (
                f'<div class="{cls}">'
                f'<span class="voice-queue-icon">{icon}</span>'
                f'<span class="voice-queue-text">{item["seat"]}: {item["snippet"]}</span>'
                f'<span class="voice-queue-dur">{item["dur"]:.0f}s</span>'
                f'</div>'
            )
        st.markdown(items_html, unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px; color:#5a6272; margin-top:4px;">※ 直近の再生履歴です</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- ゲームステータス ----
    day = st.session_state.get("day", 1)
    today_msgs = [
        e for e in st.session_state.get("chat_log", [])
        if e.get("day") == day and e.get("seat") != "SYSTEM"
    ]
    total_msgs = len(today_msgs)
    your_msgs = len([e for e in today_msgs if e.get("seat") == human_seat])
    share_pct = round(your_msgs / total_msgs * 100) if total_msgs else 0
    elapsed_label = "--:--"
    if st.session_state.get("day_phase_start") and st.session_state.get("phase") == "day":
        elapsed = int(max(0, min(DAY_PHASE_SECONDS, time.time() - st.session_state.day_phase_start)))
        elapsed_label = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    st.markdown('<div class="player-panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="player-panel-title">◈ ゲームステータス</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-grid-item">
                <div class="status-grid-label">DAY</div>
                <div class="status-grid-value">{day}</div>
            </div>
            <div class="status-grid-item">
                <div class="status-grid-label">経過時間</div>
                <div class="status-grid-value">{elapsed_label}</div>
            </div>
            <div class="status-grid-item">
                <div class="status-grid-label">本日の発言数</div>
                <div class="status-grid-value">{total_msgs}</div>
            </div>
            <div class="status-grid-item">
                <div class="status-grid-label">あなたの発言シェア</div>
                <div class="status-grid-value">{share_pct}%</div>
                <div class="influence-track"><div class="influence-fill" style="width:{share_pct}%;"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_voice_sidebar():
    """サイドバーに、プレイヤー一覧パネルと音声読み上げのON/OFFトグルを表示する。"""
    with st.sidebar:
        render_player_panel()
        st.markdown("### 🔊 音声設定")
        fish_key_present = bool(get_fish_api_key())
        if not fish_key_present:
            st.caption(
                "`FISH_AUDIO_API_KEY` が未設定のため、音声読み上げは利用できません"
                "（テキストのみで進行します）。"
            )
        st.session_state.voice_enabled = st.checkbox(
            "AIの発言を読み上げる（Fish Audio）",
            value=st.session_state.get("voice_enabled", True),
            disabled=not fish_key_present,
        )
        render_manual_replay_button()
        st.caption(
            "※ ブラウザの自動再生ポリシーにより音が鳴らない場合は、"
            "上のボタンで手動再生してください（一度クリックすれば以降は自動再生されます）。"
        )


# ======================================================================
# タイトル画面
# ======================================================================
def render_title_screen():
    st.markdown(
        """
        <div class="title-wrap" style="padding-top:24px;">
            <div class="title-eyebrow">SOCIAL DEDUCTION SYSTEM</div>
            <div class="uai-glow-title">UAI</div>
            <div class="title-sub">U N I D E N T I F I E D &middot; A I</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align:center; color:#9aa4ad; font-size:14px;
                    max-width:520px; margin:18px auto 0; line-height:1.8;">
        5体のAI（AI-01〜AI-05）の中に、たった1人だけ紛れ込んだ「本物の人間」——それがあなたです。<br>
        AIに擬態して、最後まで見破られずに生き残ってください。
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5体のシルエット行。座席は「ゲームを開始する」を押した瞬間に
    # initialize_game() 内でランダムに割り当てられるため、この時点では
    # まだ誰が何番か（あなたがどこか）は決まっておらず、"????" のまま表示する。
    # 注意: 各スロットは必ず1行（内部に改行を含めない）で組み立てること。
    # 複数行のf"""...\"\"\"を"".join()で連結すると、各断片の先頭/末尾に
    # 残る改行+インデントが空行として扱われ、Markdown側がそこでHTML
    # ブロックの継続を打ち切ってしまう。その結果、2つ目以降の断片が
    # 「インデント付きコードブロック」と誤認識され、タグがそのまま
    # 文字として表示されてしまう（実際に発生した不具合）。
    slots_html = "".join(
        '<div class="silhouette-slot">'
        '<div class="silhouette-icon">?</div>'
        '<div class="silhouette-label">????</div>'
        '</div>'
        for _ in SEATS
    )
    st.markdown(f'<div class="silhouette-row">{slots_html}</div>', unsafe_allow_html=True)

    # ゲーム概要カード
    st.markdown(
        f"""
        <div class="overview-card-grid">
            <div class="overview-card">
                <div class="overview-card-label">プレイ人数</div>
                <div class="overview-card-value">{len(SEATS)}人（AI×{len(SEATS) - 1} + あなた）</div>
            </div>
            <div class="overview-card">
                <div class="overview-card-label">制限時間</div>
                <div class="overview-card-value">{DAY_PHASE_SECONDS // 60}分間 / 日</div>
            </div>
            <div class="overview-card">
                <div class="overview-card-label">目的</div>
                <div class="overview-card-value">AIに紛れて見破られず生き残る</div>
            </div>
            <div class="overview-card">
                <div class="overview-card-label">フェーズ構成</div>
                <div class="overview-card-value">自由議論 → 投票</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    with st.expander("📖 ルール / 役職説明", expanded=False):
        st.markdown('<div class="section-label">◈ 役職構成（5名・非公開）</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="role-card you">
                <div class="role-card-title">🧑 人間（あなた）× 1</div>
                <div class="role-card-desc">AIに擬態して生き残るのが目的。会話では「人間らしさ」を隠しきってください。</div>
            </div>
            <div class="role-card emulator">
                <div class="role-card-title">🤖 エミュレーター（特殊AI）× 1</div>
                <div class="role-card-desc">人間のふりをして疑いを集める撹乱役。あなたが生き残れば同時勝利になります。</div>
            </div>
            <div class="role-card seer">
                <div class="role-card-title">🔮 占い師AI（特殊AI）× 1</div>
                <div class="role-card-desc">夜の間、密かに1名の正体を調査できる一般AI側の隠し役職。その能力は誰にも公表されません。</div>
            </div>
            <div class="role-card ai">
                <div class="role-card-title">🤖 一般AI × 2</div>
                <div class="role-card-desc">会話の矛盾や「人間らしすぎる」発言から本物の人間を見つけ出し、追放するのが目的。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown('<div class="section-label">◈ 進行ルール</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="flow-step">
                <div class="flow-step-num">☀</div>
                <div>
                    <div class="flow-step-body-title">自由議論フェーズ</div>
                    <div class="flow-step-body-desc">決まった議題はありません。制限時間内、自由にチャットして誰が人間かを探ってください。</div>
                </div>
            </div>
            <div class="flow-step">
                <div class="flow-step-num">🌙</div>
                <div>
                    <div class="flow-step-body-title">投票フェーズ</div>
                    <div class="flow-step-body-desc">全員が「人間だと思う相手」に投票し、最多票の1名が追放されます。確信が持てなければ「{SKIP_LABEL_JP}」を選ぶこともできます（同数の場合や全員スキップの場合は誰も追放されず、次の日へ進みます）。</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="text-align:center; color:#8a939b; font-size:12px;
                        margin-top:8px; letter-spacing:1px;">
            人間が追放されれば一般AI側の勝利、生き残り続ければあなた（と、もしかしたらエミュレーター）の勝利です。
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🏆 実績 / ランキング", expanded=False):
        st.caption("この機能は準備中です。プレイ記録の保存・ランキング表示は今後のアップデートで追加予定です。")

    st.write("")
    render_mobile_audio_unlock_button()
    st.write("")
    if st.button("▶ ゲームを開始する", type="primary", use_container_width=True):
        # このクリックに直結させて無音を1回再生しておくことで、後でタイマー経由
        # （＝クリックを伴わずに）自動再生される最初のAIの発言がブラウザの
        # 自動再生制限でブロックされる事態を防ぐ（_silent_wav_bytesのコメント参照）。
        st.audio(_silent_wav_bytes(), format="audio/wav", autoplay=True)
        initialize_game()
        st.session_state.screen = "game"
        st.rerun()


# ======================================================================
# 共通表示ヘルパー
# ======================================================================
def render_header():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
            <span class="uai-glow-title" style="font-size:26px; letter-spacing:4px;">UAI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    phase = st.session_state.phase
    game_over = st.session_state.get("game_over", False)
    phase_cls = "day" if phase == "day" else "night"
    phase_label = "☀ 自由議論フェーズ" if phase == "day" else "🌙 投票フェーズ"

    if phase == "day" and not game_over:
        # サイバーパンク調のヘッダーカード行: テーマ / 残り時間（リング） / ルール＋終了ボタン
        col_theme, col_timer, col_rule = st.columns([2, 1, 1.3], gap="medium")
        with col_theme:
            st.markdown(
                f"""
                <div class="header-card">
                    <div class="header-card-label">DAY {st.session_state.day}　｜　現在のテーマ</div>
                    <div class="header-theme-badge">{phase_label}</div>
                    <div class="header-theme-text">{st.session_state.get("discussion_topic", "")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_timer:
            deadline = st.session_state.day_phase_start + DAY_PHASE_SECONDS
            render_client_side_timer(deadline, DAY_PHASE_SECONDS)
        with col_rule:
            st.markdown(
                f"""
                <div class="header-card">
                    <div class="header-card-label">◈ ルール</div>
                    <div class="header-rules-list">
                        ・{DAY_PHASE_SECONDS // 60}分間で自由に議論しよう<br>
                        ・誰が「本物の人間」か見破ろう
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("⏻ ゲームを終了する", key="header_exit_btn", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
    else:
        st.markdown(
            f"""
            <div class="phase-bar">
                <span class="phase-badge {phase_cls}">{phase_label}</span>
                <span class="phase-meta">DAY {st.session_state.day}　｜　あなたのID: {st.session_state.human_seat}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    chips = ""
    for s in SEATS:
        alive = s in st.session_state.alive
        is_self = s == st.session_state.human_seat
        cls = "status-chip"
        if not alive:
            cls += " dead"
        elif is_self:
            cls += " self"
        label = s + ("（あなた）" if is_self and alive else "")
        chips += f'<span class="{cls}"><span class="dot"></span><span class="chip-label">{label}</span></span>'
    st.markdown(f'<div class="status-bar">{chips}</div>', unsafe_allow_html=True)


_EMOTION_KEYWORDS = [
    ("excited", ("！", "!", "よし", "さあ", "始め")),
    ("smirking", ("笑", "ふふ", "なるほどね", "おっと")),
    ("thinking", ("？", "?", "かな", "だろうか", "気になる")),
    ("calm", ("確かに", "そうだね", "落ち着")),
    ("observing", ("見て", "観察", "様子")),
]


def guess_emotion_tag(text: str) -> str:
    """
    発言テキストから、表示用の感情タグを簡易的に推測する。
    LLMによる本格的な感情分析ではなく、記号や頻出語による軽い見た目上の
    演出（チャット画面に[calm]や[thinking]のようなラベルを添えるだけ）。
    ゲームの判定やAIの挙動には一切使わない。
    """
    if not text:
        return "calm"
    for tag, keywords in _EMOTION_KEYWORDS:
        if any(kw in text for kw in keywords):
            return tag
    return "calm"


def render_statement_card(seat, text, emotion=None, ts=None):
    if seat == "SYSTEM":
        st.markdown(
            f"""<div style="text-align:center; margin:16px 0;">
                    <div style="display:inline-block; max-width:90%; border:1px dashed #3a4149;
                                color:#9aa4ad; background-color:#0f1216; padding:10px 18px;
                                border-radius:14px; font-size:12px; letter-spacing:1px;
                                line-height:1.8; white-space:pre-line; text-align:left;">
                        {text}
                    </div>
                </div>""",
            unsafe_allow_html=True,
        )
        return
    is_self = seat == st.session_state.human_seat
    row_cls = "seat-row self" if is_self else "seat-row ai"
    card_cls = "seat-card self" if is_self else "seat-card ai"
    name_cls = "seat-name self" if is_self else "seat-name"
    suffix = "（あなた）" if is_self else ""
    icon = "🧑" if is_self else "🤖"
    emotion_html = f'<span class="emotion-tag">[{emotion}]</span>' if emotion else ""
    ts_html = f'<span class="msg-timestamp">{ts}</span>' if ts else ""
    st.markdown(
        f"""<div class="{row_cls}">
                <div class="{card_cls}">
                    <div class="{name_cls}"><span class="seat-avatar">{icon}</span>{seat} {suffix}{emotion_html}{ts_html}</div>
                    <div class="seat-text">{text}</div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


def get_aizuchi_clip_b64():
    """
    複数のAIが連続して話すときに間へ挟む、短い相槌の音声(base64)を1つ返す。
    相槌は種類が少なく使い回すので、一度合成した音声はセッション内でキャッシュし、
    毎回Fish Audioへ問い合わせずに済むようにする（レイテンシ・コスト削減）。
    生成に失敗した場合はNoneを返す（呼び出し側は相槌無しとして扱う）。
    """
    cache = st.session_state.setdefault("aizuchi_cache", {})
    line = random.choice(AIZUCHI_LINES)
    if line in cache:
        return cache[line]
    api_key = get_fish_api_key()
    if not api_key:
        return None
    audio = synthesize_sentence(line, api_key, FISH_MODEL_NAME, get_fish_reference_id())
    if not audio:
        return None
    b64 = base64.b64encode(audio).decode("ascii")
    cache[line] = b64
    return b64


def render_mobile_audio_unlock_button():
    """
    PCでは動くのにスマホ（特にiOS Safari）では自動再生されない問題への対策。

    デスクトップのブラウザ（Chrome等）は「そのページで過去に一度でも
    ユーザー操作があれば、以降は自動再生を許可する」という比較的緩い基準で
    判定する。一方iOS Safariは基本的に「ユーザー操作イベントの呼び出し
    スタックの"その場"でaudio.play()が呼ばれた場合だけ」再生を許可する、
    という厳しい基準で判定する。

    st.button()のクリックは、サーバーへ送信→サーバーが新しいUIを返す、
    という一往復（非同期）を挟んでしまうため、その後でst.audio(autoplay=True)
    を出しても、iOS Safariからは「ユーザー操作の"その場"ではない」と
    判定されて自動再生がブロックされることがある。これがPCでは動くのに
    スマホでは動かない主な原因と考えられる。

    対策として、ここではcomponents.v1.html()で"本物のHTMLボタン"を描画し、
    そのonclickハンドラの中で（Streamlitのサーバー往復を挟まずに）
    "その場"で無音を1回再生する。さらに、その<audio>要素はiframeの中では
    なく window.parent.document（アプリ本体のページ）に追加することで、
    音声再生機構の"アンロック"状態が正しくアプリ本体のページに対して
    記録されるようにしている。これにより、以降サーバー側から出される
    st.audio(autoplay=True)（AIの発言など）も自動再生されやすくなる。

    ※ iOS Safariの自動再生制限は非常に厳しく、機種・バージョン・設定に
    よってはこれでも100%保証はできない。最終的な保険として、
    render_manual_replay_button()（明確なタップなので確実に再生できる）
    を残してある。
    """
    silent_b64 = base64.b64encode(_silent_wav_bytes()).decode("ascii")
    html = f"""
    <button id="uai-unlock-btn" type="button" style="
        background:#1f6f5c; color:#fff; border:none; border-radius:8px;
        padding:10px 16px; font-size:14px; cursor:pointer; width:100%;
        font-family:inherit;
    ">🔊 音声を有効にする</button>
    <script>
    document.getElementById("uai-unlock-btn").addEventListener("click", function() {{
        var btn = this;
        try {{
            var doc = window.parent.document;
            var audio = doc.createElement("audio");
            audio.src = "data:audio/wav;base64,{silent_b64}";
            doc.body.appendChild(audio);
            var p = audio.play();
            if (p && p.catch) {{
                p.catch(function(e) {{ console.error("[UAI] unlock play failed:", e); }});
            }}
        }} catch (e) {{
            console.error("[UAI] unlock error:", e);
        }}
        btn.innerText = "✅ 音声が有効になりました";
        btn.disabled = true;
        btn.style.opacity = "0.6";
    }});
    </script>
    """
    components.html(html, height=50)


def _silent_wav_bytes(duration_sec=0.15, sample_rate=8000):
    """
    無音の極小WAVファイルをその場で生成する（ネットワーク不要・依存ライブラリ不要）。

    用途: 「ゲームを開始する」ボタンのクリック直後に、この無音音声を
    st.audio(..., autoplay=True) で1回再生しておく。ブラウザの自動再生制限
    （音声付きのメディアは、そのページ/オリジンでユーザー操作が一度も無いと
    自動再生できない）は、多くの場合いったん何か1つでも音声の自動再生に
    成功すると、それ以降のセッションでは自動再生が許可されるようになる。
    ここでボタンクリックという明確なユーザー操作に直結させて音声再生を
    "起動"しておくことで、その後タイマー経由で（＝ユーザー操作を伴わずに）
    再生される最初のAIの発言が自動再生をブロックされてしまう問題を防ぐ。
    """
    buf = io.BytesIO()
    n_frames = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def play_clip_and_wait(clip_bytes, dock=None, seat=None, snippet_text=None, extra_delay=AUDIO_SLEEP_BUFFER_SECONDS):
    """
    音声クリップを1つ再生し、その音声の実際の長さ分だけ処理を止めて待つ。

    重要: 以前は components.v1.html() のiframe内から window.parent.document へ
    <audio>要素を注入する方式（完全に見た目上の痕跡を残さない代わりに、
    毎回新しいiframeを生成する）を使っていたが、これが「音声がうまく流れる
    時と流れない時がある」の主因だったと判断し、st.audio()を使う元の方式に
    戻した。iframe方式には主に2つの弱点があった:
      1. componentsのiframeはStreamlit側のサンドボックス設定次第で
         window.parent.documentへのアクセスや自動再生の許可(ユーザー操作
         済み判定の伝播)が保証されない場合があり、ブラウザや
         タイミングによって自動再生が黙って失敗することがある。
      2. 発言のたびに新しいiframeを都度生成するため、前のiframeの
         読み込み・実行タイミングと重なると、同じid("uai-hidden-audio")を
         取り合う形になり、再生開始前に前の要素を消してしまう、
         といった競合が起こり得る。
    st.audio()はStreamlitのメイン画面と同じフレーム内のネイティブ要素なので、
    これらの問題が起こらない。表示上は、コントロールバーをできるだけ
    目立たなくするため、常に同じ場所（発言入力欄のすぐ上、dock引数で渡す
    プレースホルダー）に小さく表示する方式にしている。

    重要2（「途中で止まる」バグの根本原因と対策。こちらは既に解決済み）:
    以前は「テキストを全部chat_logに入れて音声だけキューに積み、次のrerunで
    まとめて再生する」という2段階の作りだった。しかしStreamlitの
    st_autorefresh（待機中の4秒ごとの自動更新）は、まさにその「音声再生中の
    次のrerun」を待たずに割り込んで発火することがあり、割り込まれた瞬間に
    画面全体が作り直されて再生中の<audio>要素ごと消えてしまっていた
    （＝ユーザーから見ると「途中で音声が止まる」）。

    この関数では、音声を鳴らし始めた"その場"でtime.sleep()により
    実際の再生時間ぶんだけ処理をブロックする。自動更新(st_autorefresh)が
    (再)登録されるのはこの関数を含む一連の処理が完全に終わったあと
    （render_day_phase側のstep 5）だけなので、再生中に自動更新が割り込んで
    要素を消してしまうことが構造的に起こらなくなる。

    副次効果として、「音声が鳴り始めた瞬間にその発言のテキストが表示される」
    という体感の同期も、このタイミング制御によって自然に実現される
    （呼び出し側でテキスト表示の直後にこの関数を呼ぶ設計になっている）。
    """
    if not clip_bytes:
        return
    target = dock if dock is not None else st
    target.audio(clip_bytes, format="audio/mp3", autoplay=True)
    duration = get_mp3_duration_seconds(clip_bytes)
    if seat is not None:
        # サイドバーの「音声再生キュー」パネル用に、直近の再生履歴を記録しておく
        # （表示用の演出。ゲーム進行やAIの判定には使わない）。
        queue_log = st.session_state.setdefault("voice_queue_log", [])
        snippet = (snippet_text or "")[:22] + ("…" if snippet_text and len(snippet_text) > 22 else "")
        queue_log.append({"seat": seat, "snippet": snippet, "dur": duration})
        st.session_state["voice_queue_log"] = queue_log[-12:]
    time.sleep(duration + extra_delay)


def render_manual_replay_button():
    """
    Fish Audioの音声合成そのものが失敗した場合や、何らかの事情で自動再生が
    ブロックされた場合のための保険。ボタンクリックは確実な「ユーザー操作」
    とみなされるため、ここからの再生はほぼ確実にブロックされない。
    """
    if st.session_state.get("last_audio_bytes"):
        if st.button("🔊 直前の音声を再生する", key="manual_replay_btn"):
            st.audio(st.session_state.last_audio_bytes, format="audio/mp3", autoplay=True)


def render_client_side_timer(deadline_epoch, total_seconds):
    """
    サーバーへの再実行(rerun)を発生させない、ブラウザ内だけで動くカウントダウン表示。
    タイマー更新のたびに画面全体がちらつく問題を避けるために使用する。
    円形のリング（SVGのstroke-dashoffsetをJSで更新）で残り時間を表現する。
    """
    deadline_ms = int(deadline_epoch * 1000)
    total_ms = int(total_seconds * 1000)
    radius = 46
    circumference = 2 * 3.14159265 * radius
    html = f"""
    <div style="font-family:'JetBrains Mono','Courier New',monospace;
                display:flex; flex-direction:column; align-items:center;">
      <div style="font-size:10px; letter-spacing:2px; color:#7d8aa0; margin-bottom:4px;">残り時間</div>
      <div style="position:relative; width:112px; height:112px;">
        <svg width="112" height="112" style="transform:rotate(-90deg);">
          <circle cx="56" cy="56" r="{radius}" fill="none" stroke="#1c2035" stroke-width="8"></circle>
          <circle id="rw_timer_ring" cx="56" cy="56" r="{radius}" fill="none"
                  stroke="url(#rw_timer_grad)" stroke-width="8" stroke-linecap="round"
                  stroke-dasharray="{circumference:.2f}" stroke-dashoffset="0"></circle>
          <defs>
            <linearGradient id="rw_timer_grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#6ea8fe"></stop>
              <stop offset="100%" stop-color="#e879c9"></stop>
            </linearGradient>
          </defs>
        </svg>
        <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center;">
          <span id="rw_timer_text" style="font-size:22px; font-weight:700; color:#eaf0ff;">--:--</span>
        </div>
      </div>
    </div>
    <script>
      (function() {{
        const deadline = {deadline_ms};
        const total = {total_ms};
        const circumference = {circumference:.2f};
        function tick() {{
          const now = Date.now();
          const remaining = Math.max(0, deadline - now);
          const s = Math.floor(remaining / 1000);
          const mm = String(Math.floor(s / 60)).padStart(2, '0');
          const ss = String(s % 60).padStart(2, '0');
          const textEl = document.getElementById('rw_timer_text');
          const ringEl = document.getElementById('rw_timer_ring');
          if (textEl) textEl.textContent = mm + ':' + ss;
          if (ringEl) {{
            const frac = Math.max(0, Math.min(1, remaining / total));
            ringEl.setAttribute('stroke-dashoffset', String(circumference * (1 - frac)));
          }}
          if (remaining <= 0) clearInterval(iv);
        }}
        tick();
        const iv = setInterval(tick, 500);
      }})();
    </script>
    """
    components.html(html, height=140)


def reset_game_button():
    debug_errors = st.session_state.get("debug_errors", [])
    if debug_errors:
        with st.expander("⚠ 通信の状態（トラブルが起きている場合はここを確認）", expanded=False):
            for line in debug_errors[-10:]:
                st.caption(line)

    if st.button("🔄 ゲームをリセット", key="reset_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ======================================================================
# 昼フェーズ（自由チャット・時間制限あり）
# ======================================================================
def render_day_phase():
    """
    昼フェーズ（自由チャット・時間制限あり）。

    ポイント:
    - 入力欄(st.text_area + 送信ボタン)は「必ず毎回」呼び出す。AIが発言を生成中でも
      入力欄自体は常に画面に存在し続けるため、プレイヤーはいつでも発言できる。
      プレイヤーが発言した場合は、AIの発言予定より常に優先して処理される。
    - AIの発言はまず「誰が話すか」だけ決めて即座に再描画し、次の再描画で
      実際の通信を行う（この回は自動更新を呼ばない）。これにより通信中に
      自動更新が割り込んでキャンセルされる事態を防ぐ。
    - 一定確率で複数のAIが同時に選ばれ、ThreadPoolExecutorで実際に並行して
      通信するため、複数のAIが同時に考えて発言する形になる。
    """
    alive = st.session_state.alive
    human_seat = st.session_state.human_seat

    # 自動更新（st_autorefresh）専用のプレースホルダー。
    # 「本当に何もしていない待機中（下の 5番）」の時だけ、ここに自動更新ウィジェットを描画する。
    # 毎回の実行で必ずこの行を通ることで、前回の実行で待機中に描画された
    # 自動更新タイマー（ブラウザ側で動き続けるJSタイマー）を、AIの発言生成などで
    # 処理がブロックされる「前」の時点で確実に消しておくことができる。
    #
    # これをしないと、次のようなバグが起きる:
    #   1. 待機中に自動更新タイマー(4秒間隔)がブラウザに仕込まれる
    #   2. AIが発言を考え始める（通信に数十秒かかることがある）
    #   3. しかしサーバーがまだ処理中の間、ブラウザの画面は「待機中だった時」の
    #      ままなので、古い自動更新タイマーは消えずに動き続けている
    #   4. 4秒後、そのタイマーが発火してサーバーに再実行を要求する
    #   5. Streamlitは「処理中だったスクリプト」を強制的に打ち切り、最初からやり直す
    #   6. pending_speakers（誰が話すか）はまだクリアされていないので、
    #      やり直した実行でも同じAIの発言生成が最初から再開される
    #   7. しかしまた数十秒かかるうちに次の自動更新が発火し、2に戻る……
    #   → 「AIが一生考え中のまま」になる（実際には内部で無限に再試行されている）
    autorefresh_slot = st.empty()

    deadline = st.session_state.day_phase_start + DAY_PHASE_SECONDS
    remaining = max(0.0, deadline - time.time())
    time_up = (remaining <= 0) or st.session_state.force_end_day

    st.caption("上のテーマはあくまで呼び水です。テーマに縛られず自由に会話して、誰が「本物の人間」か探ってください。")

    # 表示は「本日分」の発言のみに絞る（AIへ渡す文脈は日をまたいだ全履歴を使う）。
    for entry in st.session_state.chat_log:
        if entry.get("day") == st.session_state.day:
            render_statement_card(entry["seat"], entry["text"], entry.get("emotion"), entry.get("ts"))

    # --- AIの音声を再生するための固定枠 ---
    # 発言入力欄と同じ「画面下部に常に固定される」コンテナ(st.bottom)の中に、
    # 入力欄より先に空のプレースホルダーを置いておく。こうすると、実際に
    # 音声を再生するタイミング（このあとのstep 2、コード上はずっと後ろ）で
    # このプレースホルダーに書き込んでも、見た目の位置は「入力欄のすぐ上」に
    # 固定されたままになる（Streamlitのプレースホルダーは、生成した時点の
    # 位置を保持したまま、後から中身だけ差し替えられるため）。
    #
    # st.container(key="ai_voice_dock") で囲んでおくことで、CSS側からは
    # ".st-key-ai_voice_dock" というクラスで丸ごと非表示にできる。
    # data-testid="stAudio" 直指定のCSSだとStreamlitのバージョンによって
    # 内部のDOM構造（audio要素そのものにtestidが付くのか、それを包むdivに
    # 付くのか等）が変わり、非表示にできないことがあったため、
    # 「このコンテナの中身は中身が何であれ丸ごと消す」方式に変更した。
    with st.bottom:
        with st.container(key="ai_voice_dock"):
            audio_dock = st.empty()

    # --- 音声入力（任意）：マイクで録音した内容をWhisperで文字起こしし、
    #     下の発言欄に自動で入力する。送信は今まで通りボタンを押した時だけ
    #     行われるので、認識結果はここで一度確認・修正してから送れる。
    #
    # 発言欄(st.text_input)のkeyに対して事前に st.session_state[key]=... を
    # セットしておくと、その値がその回のウィジェット生成時の初期値になる
    # （ウィジェット生成"後"に同じ方法で値を変えることはできないので、
    #   このブロックは必ず下の st.text_input より前に置く必要がある）。
    text_input_key = f"chat_input_area_{st.session_state.day}"

    user_msg = None
    if not time_up:
        with st.bottom:
            # マイクボタンは発言欄と同じ行の左側に置きたいが、
            # st.form の中では通常のst.button()が使えない（st.form_submit_button
            # だと、押した瞬間にclear_on_submitで入力中の下書きまで消えてしまう）
            # ため、フォームの外に置いた列(col_mic)とフォーム自体(col_form)を
            # 横に並べる形にしている。
            # st.container(key=...) で囲むことで、CSS側から
            # ".st-key-mic_recorder_wrap" として高さを発言欄に揃えられるように
            # している（streamlit-mic-recorderは別iframeの独自UIのため、
            # 中身のボタン自体の見た目までは変えられないが、外枠のサイズは
            # 揃えられる）。
            col_mic, col_form = st.columns([1, 7], vertical_alignment="bottom")
            with col_mic:
                with st.container(key="mic_recorder_wrap"):
                    raw_bytes, rec_format = record_voice_input()

            if raw_bytes:
                with st.spinner("文字起こし中..."):
                    transcribed = transcribe_audio(raw_bytes, fmt=rec_format or "wav")
                if transcribed:
                    st.session_state[text_input_key] = transcribed[:150]
                else:
                    st.warning(
                        "うまく聞き取れませんでした。もう一度録音するか、直接入力してください。"
                    )

            with col_form:
                with st.form(
                    key="chat_form",
                    clear_on_submit=True,
                    enter_to_submit=False,
                    border=False,
                ):
                    col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
                    with col_input:
                        draft = st.text_input(
                            "発言を入力",
                            key=text_input_key,
                            max_chars=150,
                            label_visibility="collapsed",
                            placeholder="発言を入力（150文字以内）...",
                        )
                    with col_btn:
                        submitted = st.form_submit_button("送信", type="primary", use_container_width=True)
        if submitted and draft and draft.strip():
            user_msg = draft

    # --- 1) プレイヤーが発言した場合は最優先で処理する ---
    if user_msg:
        text = user_msg.strip()[:150]
        st.session_state.chat_log.append({
            "day": st.session_state.day, "seat": human_seat, "text": text,
            "emotion": guess_emotion_tag(text), "ts": time.strftime("%H:%M:%S"),
        })
        alive_ai_seats = [s for s in alive if s != human_seat]
        # 通信はまだ行わず「誰が反応するか」だけ確定し、即座に再描画する
        st.session_state.pending_speakers = decide_speakers(alive_ai_seats)
        st.rerun()
        return

    # --- 2) 発言予定が確定しているAIの通信を実行する（このrunでは自動更新を呼ばない） ---
    if st.session_state.pending_speakers:
        speakers = st.session_state.pending_speakers
        label = "、".join(speakers)
        seer_seat = st.session_state.get("seer_seat")
        voice_on = st.session_state.get("voice_enabled", True)
        fish_key = get_fish_api_key() if voice_on else ""
        tts_executor = get_tts_executor() if fish_key else None
        reply_stream = generate_replies_concurrently(
            speakers, st.session_state.day, st.session_state.chat_log, alive, st.session_state.seat_roles,
            seat_personalities=st.session_state.get("seat_personalities"),
            seer_seat=seer_seat,
            seer_known_facts=seer_known_facts_text() if seer_seat else None,
            tts_api_key=fish_key or None,
            seat_reference_ids={s: get_seat_reference_id(s) for s in speakers},
            tts_executor=tts_executor,
        )

        # 発言者1人ずつ「テキストを確定・表示 → その音声を再生 → 実際の長さぶん待つ」
        # の順で処理する。これにより、次の人の発言（と音声）に進むのは必ず
        # 前の人の音声が鳴り終わったあとになり、「音声が出た瞬間にその発言の
        # 文字が表示される」体感になる。
        # （このループはrerunを挟まず、この1回のスクリプト実行の中で完結する。
        #   自動更新(st_autorefresh)が再生の途中に割り込むことはない。詳細は
        #   play_clip_and_wait() のコメントを参照。）
        #
        # 重要: generate_replies_concurrently はジェネレータになっており、
        # ここで1人分をtime.sleep()で再生している間も、まだ順番が来ていない
        # 他のAIのLLM生成・TTS合成はThreadPoolExecutorの別スレッド上で
        # 裏で進み続ける（＝表示・再生だけが「前の音声が終わるまで」直列に
        # なり、思考・音声合成そのものは止めない）。最初の1人分が届くまでは
        # まだ何も表示できないため、そこだけスピナーで待っていることを示す。
        replay_clips = []
        any_played = False
        first_seat = first_text = first_clips = None
        try:
            with st.spinner(f"{label} が発言を考え中..."):
                first_seat, first_text, first_clips = next(reply_stream)
        except StopIteration:
            pass

        pending_results = itertools.chain(
            [(first_seat, first_text, first_clips)] if first_seat is not None else [],
            reply_stream,
        )
        for seat, text, audio_clips in pending_results:
            valid_clips = [c for c in audio_clips if c]
            if fish_key and not valid_clips:
                # 音声ONでテキストは生成できたのに音声が1つも得られなかった場合は、
                # 「無言で失敗」にせず、既存の通信状態パネルで確認できるようにする。
                _record_debug_error(seat, "音声合成に失敗しました（APIキー・残高・ネットワークをご確認ください）")

            if valid_clips and any_played and random.random() < AIZUCHI_INSERT_CHANCE:
                aizuchi_b64 = get_aizuchi_clip_b64()
                if aizuchi_b64:
                    aizuchi_bytes = base64.b64decode(aizuchi_b64)
                    play_clip_and_wait(aizuchi_bytes, dock=audio_dock)
                    replay_clips.append(aizuchi_bytes)

            # 発言のテキストは、その音声が再生され始める「今」確定・表示する
            # （音声が無い/失敗した場合も、ここで即座にテキストだけ表示する）。
            emotion = guess_emotion_tag(text)
            ts = time.strftime("%H:%M:%S")
            st.session_state.chat_log.append(
                {"day": st.session_state.day, "seat": seat, "text": text, "emotion": emotion, "ts": ts}
            )
            render_statement_card(seat, text, emotion, ts)

            for clip in valid_clips:
                play_clip_and_wait(clip, dock=audio_dock, seat=seat, snippet_text=text)
                replay_clips.append(clip)
                any_played = True

        if replay_clips:
            st.session_state.last_audio_bytes = concat_mp3_clips(replay_clips)  # 手動リプレイ用に保持

        st.session_state.pending_speakers = []
        st.session_state.next_ai_speak_time = time.time() + random.uniform(
            AI_SPEAK_MIN_INTERVAL, AI_SPEAK_MAX_INTERVAL
        )
        st.rerun()
        return

    # --- 3) 制限時間終了 ---
    if time_up:
        st.success("議論の制限時間が終了しました。")
        if st.button("🌙 投票フェーズへ進む", type="primary"):
            with st.spinner("夜の帳が下りています..."):
                run_seer_investigation()
            st.session_state.phase = "night"
            st.session_state.human_vote = None
            st.session_state.votes = {}
            st.session_state.votes_done = False
            st.rerun()
        return

    if st.button("⏭ 議論を早く終えて投票へ"):
        st.session_state.force_end_day = True
        st.rerun()
        return

    # --- 4) 自発的にAIが発言する時間かどうかチェック（ここも通信はせず予定を確定するだけ） ---
    alive_ai_seats = [s for s in alive if s != human_seat]
    if alive_ai_seats and time.time() >= st.session_state.next_ai_speak_time:
        last_seat = st.session_state.chat_log[-1]["seat"] if st.session_state.chat_log else None
        st.session_state.pending_speakers = decide_speakers(alive_ai_seats, exclude_last=last_seat)
        st.rerun()
        return

    # --- 5) 何も起きていない待機中のみ、軽量な自動更新で見張る ---
    # 必ず専用プレースホルダー(autorefresh_slot)の中に描画する。
    # こうすることで、次にこの関数が呼ばれた時に上でautorefresh_slotが
    # 再生成された瞬間、このウィジェット（＝ブラウザ側のJSタイマー）が
    # 確実に消去されるようになる。
    with autorefresh_slot:
        st_autorefresh(interval=IDLE_CHECK_INTERVAL_MS, key=f"day_watch_{st.session_state.day}")


# ======================================================================
# 夜フェーズ
# ======================================================================
def render_night_phase():
    alive = st.session_state.alive
    human_seat = st.session_state.human_seat

    with st.expander("📄 本日の会話を振り返る", expanded=False):
        for entry in st.session_state.chat_log:
            if entry.get("day") == st.session_state.day:
                render_statement_card(entry["seat"], entry["text"], entry.get("emotion"), entry.get("ts"))

    candidates = [s for s in alive if s != human_seat]

    # --- 1) プレイヤーの投票 ---
    if st.session_state.human_vote is None:
        st.markdown(
            '<div class="section-label">◈ 「本物の人間」だと思う相手に投票してください</div>',
            unsafe_allow_html=True,
        )
        st.caption("確信が持てない場合は、下の「スキップ」を選んで投票を見送ることもできます。")

        if st.session_state.get("night_vote_draft") is None:
            st.session_state.night_vote_draft = None

        cols = st.columns(3)
        for i, cand in enumerate(candidates):
            with cols[i % 3]:
                selected = st.session_state.night_vote_draft == cand
                if st.button(
                    ("✓ " if selected else "") + cand,
                    key=f"vote_btn_{cand}_{st.session_state.day}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.night_vote_draft = cand
                    st.rerun()

        st.write("")
        skip_selected = st.session_state.night_vote_draft == SKIP_VOTE
        if st.button(
            ("✓ " if skip_selected else "⏭ ") + SKIP_LABEL_JP,
            key=f"vote_skip_{st.session_state.day}",
            type="primary" if skip_selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.night_vote_draft = SKIP_VOTE
            st.rerun()

        st.write("")
        draft = st.session_state.night_vote_draft
        draft_label = SKIP_LABEL_JP if draft == SKIP_VOTE else draft
        if draft:
            st.caption(f"選択中: **{draft_label}**")
        if st.button("🗳 投票を確定する", type="primary", disabled=(draft is None)):
            st.session_state.human_vote = draft
            st.session_state.votes[human_seat] = draft
            st.rerun()
        return

    # --- 2) 他AIの自動投票 ---
    if not st.session_state.votes_done:
        remaining = [s for s in alive if s != human_seat and s not in st.session_state.votes]
        if remaining:
            seat = remaining[0]
            role = st.session_state.seat_roles[seat]
            seat_candidates = [s for s in alive if s != seat]
            known_facts = seer_known_facts_text() if seat == st.session_state.get("seer_seat") else None
            with st.spinner(f"{seat} が投票中..."):
                vote = call_ai_vote(
                    role, seat, st.session_state.day,
                    st.session_state.chat_log, seat_candidates,
                    known_facts=known_facts,
                    personality=st.session_state.get("seat_personalities", {}).get(seat),
                )
            st.session_state.votes[seat] = vote
            st.rerun()
        else:
            st.session_state.votes_done = True
            st.rerun()
        return

    # --- 3) 開票 ---
    st.markdown('<div class="section-label">◈ 開票結果</div>', unsafe_allow_html=True)

    tally = {}
    skip_count = 0
    for voter, target in st.session_state.votes.items():
        if target == SKIP_VOTE:
            skip_count += 1
        else:
            tally[target] = tally.get(target, 0) + 1

    total_votes = len(st.session_state.votes)
    max_votes = max(tally.values()) if tally else 0

    def render_tally_bar(label, count, is_max=False, is_skip=False):
        pct = int(round((count / total_votes) * 100)) if total_votes else 0
        fill_cls = "tally-fill"
        if is_skip:
            fill_cls += " skip"
        elif is_max and count > 0:
            fill_cls += " max"
        st.markdown(
            f"""<div class="tally-row">
                    <div class="tally-label"><span>{label}</span><span>{count}票</span></div>
                    <div class="tally-track"><div class="{fill_cls}" style="width:{pct}%;"></div></div>
                </div>""",
            unsafe_allow_html=True,
        )

    for seat in alive:
        count = tally.get(seat, 0)
        render_tally_bar(seat, count, is_max=(max_votes > 0 and count == max_votes))
    render_tally_bar(SKIP_LABEL_JP, skip_count, is_skip=True)

    with st.expander("🗳 投票内訳を見る（誰が誰に投票したか）", expanded=False):
        for voter in sorted(st.session_state.votes.keys()):
            target = st.session_state.votes[voter]
            target_label = SKIP_LABEL_JP if target == SKIP_VOTE else target
            voter_label = f"{voter}（あなた）" if voter == human_seat else voter
            st.caption(f"{voter_label} → {target_label}")

    top_seats = [s for s, c in tally.items() if c == max_votes] if tally else []

    if not top_seats:
        # 誰にも投票が集まらなかった（全員スキップ）場合
        st.session_state.tie_result = True
        st.session_state.pending_elimination = None
        st.info("有効な投票が誰にも集まらなかった（全員がスキップ）ため、今回は誰も追放されません。")
        if st.button("➡ 次の日へ進む", type="primary"):
            reveal_text = build_vote_reveal_text(st.session_state.day, st.session_state.votes)
            st.session_state.day += 1
            reset_day_state()
            st.session_state.chat_log.append({"day": st.session_state.day, "seat": "SYSTEM", "text": reveal_text})
            st.rerun()
    elif len(top_seats) >= 2:
        st.session_state.tie_result = True
        st.session_state.pending_elimination = None
        st.info(f"最多票が同数（{', '.join(top_seats)}）のため、今回は誰も追放されません。")
        if st.button("➡ 次の日へ進む", type="primary"):
            reveal_text = build_vote_reveal_text(st.session_state.day, st.session_state.votes)
            st.session_state.day += 1
            reset_day_state()
            st.session_state.chat_log.append({"day": st.session_state.day, "seat": "SYSTEM", "text": reveal_text})
            st.rerun()
    else:
        eliminated = top_seats[0]
        st.session_state.pending_elimination = eliminated
        st.warning(f"最多票により **{eliminated}** が追放されます。")
        if st.button("➡ 結果を確定する", type="primary"):
            reveal_text = build_vote_reveal_text(st.session_state.day, st.session_state.votes)
            process_elimination(eliminated)
            if not st.session_state.game_over:
                st.session_state.chat_log.append(
                    {"day": st.session_state.day, "seat": "SYSTEM", "text": reveal_text}
                )
            st.rerun()


# ======================================================================
# 追放処理・勝敗判定
# ======================================================================
def process_elimination(eliminated_seat):
    st.session_state.eliminated_last = eliminated_seat
    role = st.session_state.seat_roles[eliminated_seat]

    if role == ROLE_HUMAN:
        st.session_state.game_result = "general_ai_win"
        st.session_state.game_over = True
        return

    # AI（一般 or エミュレーター）が追放された場合、役職は非公開のまま人数だけ減らす
    st.session_state.alive.remove(eliminated_seat)

    if len(st.session_state.alive) <= 2:
        # 残数が「人間1名＋AI1名」になった → 人間＆エミュレーター側の勝利
        st.session_state.game_result = "human_side_win"
        st.session_state.game_over = True
        return

    # 次の日の昼フェーズへ
    st.session_state.day += 1
    reset_day_state()


# ======================================================================
# 結果画面
# ======================================================================
def render_result():
    result = st.session_state.game_result
    human_seat = st.session_state.human_seat

    if result == "general_ai_win":
        st.error("### ✕ 一般AI側の勝利\n人間が特定され、追放されました。")
    else:
        st.success("### ◎ 人間＆エミュレーター側の勝利\n人間が最後まで生き残りました。")

    st.markdown(f"最後に追放されたのは **{st.session_state.eliminated_last}** でした。")

    st.write("")
    st.markdown('<div class="section-label">◈ 全員の正体</div>', unsafe_allow_html=True)
    for seat in SEATS:
        role = st.session_state.seat_roles[seat]
        is_you = seat == human_seat
        card_cls = "reveal-card you" if is_you else "reveal-card"
        role_cls = f"reveal-role {role}"
        tag = "（あなた）" if is_you else ""
        st.markdown(
            f"""<div class="{card_cls}">
                    <span class="reveal-seat">{seat}{tag}</span>
                    <span class="{role_cls}">{ROLE_LABEL_JP[role]}</span>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()
    if st.button("🔄 もう一度プレイする", type="primary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ======================================================================
# メイン
# ======================================================================
def main():
    if st.session_state.screen == "title":
        render_title_screen()
        return

    render_voice_sidebar()
    render_header()

    if not get_api_key():
        st.error(
            "APIキーが設定されていません。ローカル実行時は `.env` に "
            "`OPENROUTER_API_KEY=...` を、Streamlit Cloudでは Secrets に "
            "`OPENROUTER_API_KEY` を設定してください。"
        )

    if st.session_state.game_over:
        render_result()
    else:
        if st.session_state.phase == "day":
            render_day_phase()
        else:
            render_night_phase()

    st.divider()
    reset_game_button()


if __name__ == "__main__":
    main()

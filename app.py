# -*- coding: utf-8 -*-
"""
app.py
UAI - Streamlit Webアプリ（自由チャット版・社会的推理ゲーム）

役職構成（合計5名 / AI-01〜AI-05）:
  - 人間（プレイヤー）    × 1
  - エミュレーター(特殊AI) × 1  … 人間に擬態。人間が生き残れば同時勝利。
  - 一般AI                × 3  … 人間を見つけ出すのが目的。

昼フェーズ: 固定の議題は無く、制限時間内は自由にチャットができる。
            人間はいつでも発言可能。AIは人間の発言に反応したり、
            一定間隔で自発的に発言に加わったりする。
夜フェーズ: 全員が「人間だと思う相手」に投票 → 最多票を追放
  - 追放者が「人間」        → 一般AI側の勝利でゲーム終了
  - 追放者が「AI」          → 役職非公開のまま人数減、残り2名(人間1+AI1)で
                              人間＆エミュレーター側の勝利としてゲーム終了
  - 最多票が同数(2名以上)   → 誰も追放されず、そのまま次の日の昼フェーズへ
"""

import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

from prompts import (
    ROLE_HUMAN,
    ROLE_EMULATOR,
    ROLE_GENERAL_AI,
    ROLE_LABEL_JP,
    build_chat_reply_messages,
    build_vote_messages,
    try_parse_vote,
)

# ======================================================================
# 基本設定
# ======================================================================
load_dotenv()  # ローカル実行時: .env を読み込む

MODEL_NAME = "openrouter/free"
BASE_URL = "https://openrouter.ai/api/v1"

SEATS = [f"AI-{i:02d}" for i in range(1, 6)]

DAY_PHASE_SECONDS = 180          # 昼フェーズ（自由チャット）の制限時間（秒）
AI_SPEAK_MIN_INTERVAL = 6        # AIが自発的に発言する最短間隔（秒）
AI_SPEAK_MAX_INTERVAL = 12       # AIが自発的に発言する最長間隔（秒）
IDLE_CHECK_INTERVAL_MS = 4000    # 待機中にサーバー側で状態確認する間隔（ミリ秒）
MULTI_SPEAK_CHANCE = 0.35        # 複数のAIが同時に発言を考え始める確率
MAX_SIMULTANEOUS_SPEAKERS = 2    # 同時に発言を考えるAIの最大数
LLM_TIMEOUT_SECONDS = 15         # OpenRouterへの通信タイムアウト（秒）
                                  # これが無いと、応答が返ってこない場合に「考え中」のまま
                                  # 永遠に待ち続けてしまう。

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


def call_ai_chat_reply(role, seat_name, day, chat_log, alive_seats):
    """
    AIの発言を生成する。応答が空、または日本語になっていない場合は
    最大2回まで「必ず日本語で」と念押しして再試行し、それでも駄目な場合は
    「(応答なし)」のような不自然な文言ではなく、キャラクター性を保った
    フォールバック発言を返す。
    """
    messages = build_chat_reply_messages(role, seat_name, day, chat_log, alive_seats)
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


def call_ai_vote(role, seat_name, day, chat_log, candidates):
    messages = build_vote_messages(role, seat_name, day, chat_log, candidates)
    raw = call_llm(messages, max_tokens=80, temperature=0.7)
    vote = try_parse_vote(raw, candidates)
    if vote is None:
        vote = random.choice(candidates)
    return vote


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


def generate_replies_concurrently(speakers, day, chat_log, alive_seats, seat_roles):
    """
    複数のAIの発言を、実際に並行して(同時に)通信・生成する。
    完了した順に (seat, text) のタプルとして返す。

    重要: ThreadPoolExecutorを `with` 文で使うと、ブロック終了時に
    「まだ終わっていないスレッドの完了を待つ」処理(shutdown(wait=True))が
    暗黙に走ってしまい、たとえ以下のタイムアウト処理が正しく働いていても、
    結局そこで固まってしまう（これが「永遠に考え中」の真因だった）。
    そのため、ここでは `with` を使わず、shutdown(wait=False) で
    「返事が来なくても待たずに関数を抜ける」ようにしている。
    取り残されたスレッドは、バックグラウンドで終わるか、やがて例外になるかする
    だけで、以後の画面表示をブロックすることはない。
    """
    results = []
    hard_timeout = LLM_TIMEOUT_SECONDS * 3 + 10  # リトライ3回分+余裕を持った絶対上限
    executor = ThreadPoolExecutor(max_workers=max(1, len(speakers)))
    try:
        future_to_seat = {
            executor.submit(
                call_ai_chat_reply, seat_roles[seat], seat, day, chat_log, alive_seats
            ): seat
            for seat in speakers
        }
        try:
            for future in as_completed(future_to_seat, timeout=hard_timeout + 5):
                seat = future_to_seat[future]
                try:
                    text = future.result(timeout=hard_timeout)
                except Exception as e:
                    text = random.choice(_FALLBACK_LINES)
                    _record_debug_error(seat, e)
                results.append((seat, text))
        except Exception as e:
            # as_completed自体が全体タイムアウトした場合もここで捕捉し、必ず先に進める
            _record_debug_error("全体", e)

        # 何らかの理由で結果が得られなかった座席は、必ずフォールバックで埋める
        done_seats = {seat for seat, _ in results}
        for future, seat in future_to_seat.items():
            if seat not in done_seats:
                results.append((seat, random.choice(_FALLBACK_LINES)))
                _record_debug_error(seat, "全体タイムアウトにより打ち切り")
    finally:
        # wait=False: 未完了のスレッドを待たずに、ここで即座に関数を抜ける
        executor.shutdown(wait=False, cancel_futures=True)
    return results


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
.seat-card {
    border: 1px solid #2a2f36;
    background-color: #101317;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.seat-name {
    font-size: 12px;
    letter-spacing: 3px;
    color: #7fd3c7;
    font-weight: 700;
    margin-bottom: 4px;
}
.seat-name.self {
    color: #f0c674;
}
.seat-text {
    font-size: 15px;
    color: #d7dbe0;
    line-height: 1.5;
}
.status-bar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.status-chip {
    border: 1px solid #2a2f36;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    letter-spacing: 1px;
    color: #9aa4ad;
}
.status-chip.dead {
    text-decoration: line-through;
    color: #4a4f55;
    border-color: #23272c;
}
div.stButton > button {
    background-color: #14181d;
    color: #e7ecf0;
    border: 1px solid #3a4149;
    border-radius: 4px;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
div.stButton > button:hover {
    border-color: #7fd3c7;
    color: #7fd3c7;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ======================================================================
# ゲーム状態の初期化
# ======================================================================
def reset_day_state():
    """新しい昼フェーズ（自由チャット）用の状態にリセットする。"""
    st.session_state.phase = "day"
    st.session_state.chat_log = []
    st.session_state.day_phase_start = time.time()
    st.session_state.next_ai_speak_time = time.time() + random.uniform(3, 7)
    st.session_state.force_end_day = False
    st.session_state.pending_speakers = []
    st.session_state.human_vote = None
    st.session_state.votes = {}
    st.session_state.votes_done = False
    st.session_state.pending_elimination = None
    st.session_state.tie_result = False


def initialize_game():
    seats = SEATS.copy()
    roles = [ROLE_HUMAN, ROLE_EMULATOR, ROLE_GENERAL_AI, ROLE_GENERAL_AI, ROLE_GENERAL_AI]
    random.shuffle(roles)
    seat_roles = dict(zip(seats, roles))
    human_seat = next(s for s, r in seat_roles.items() if r == ROLE_HUMAN)

    st.session_state.seat_roles = seat_roles
    st.session_state.human_seat = human_seat
    st.session_state.alive = seats.copy()
    st.session_state.day = 1

    st.session_state.game_over = False
    st.session_state.game_result = None
    st.session_state.eliminated_last = None

    reset_day_state()


if "screen" not in st.session_state:
    st.session_state.screen = "title"  # "title" | "game"


# ======================================================================
# タイトル画面
# ======================================================================
def render_title_screen():
    st.markdown(
        """
        <div style="text-align:center; padding: 48px 0 8px;">
            <div style="font-family:'JetBrains Mono','Courier New',monospace;
                        font-size:12px; letter-spacing:6px; color:#7fd3c7;
                        margin-bottom:10px;">SOCIAL DEDUCTION SYSTEM</div>
            <div style="font-family:'JetBrains Mono','Courier New',monospace;
                        font-size:56px; font-weight:700; letter-spacing:10px;
                        color:#e7ecf0;">UAI</div>
            <div style="font-family:'JetBrains Mono','Courier New',monospace;
                        font-size:12px; letter-spacing:3px; color:#8a939b;
                        margin-top:6px;">U N I D E N T I F I E D · A I</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
5体のAI（AI-01〜AI-05）の中に、たった1人だけ紛れ込んだ「本物の人間」——それがあなたです。
AIに擬態して、最後まで見破られずに生き残ってください。

**役職構成（5名・非公開）**
- 🧑 人間（あなた）× 1 —— AIに擬態して生き残るのが目的
- 🤖 エミュレーター × 1 —— 人間のふりをして疑いを集める特殊AI。人間が生き残れば同時勝利
- 🤖 一般AI × 3 —— 会話の矛盾から人間を見つけ出し、追放するのが目的

**進行ルール**
1. ☀ 自由議論フェーズ —— 制限時間内、決まったテーマはありません。自由にチャットしてください
2. 🌙 投票フェーズ —— 全員が「人間だと思う相手」に投票し、最多票の1名が追放されます
   （同数の場合は誰も追放されず、そのまま次の日へ進みます）

人間が追放されれば一般AI側の勝利、生き残り続ければあなた（と、もしかしたらエミュレーター）の勝利です。
        """
    )

    st.divider()
    if st.button("▶ ゲームを開始する", type="primary", use_container_width=True):
        initialize_game()
        st.session_state.screen = "game"
        st.rerun()


# ======================================================================
# 共通表示ヘルパー
# ======================================================================
def render_header():
    st.markdown("### ◈ UAI")
    phase_label = "☀ 自由議論" if st.session_state.phase == "day" else "🌙 投票"
    st.caption(
        f"あなたのID: **{st.session_state.human_seat}**　"
        f"｜ Day {st.session_state.day}　｜ フェーズ: {phase_label}"
    )
    chips = ""
    for s in SEATS:
        alive = s in st.session_state.alive
        cls = "status-chip" if alive else "status-chip dead"
        chips += f'<span class="{cls}">{s}</span>'
    st.markdown(f'<div class="status-bar">{chips}</div>', unsafe_allow_html=True)
    st.divider()


def render_statement_card(seat, text):
    is_self = seat == st.session_state.human_seat
    name_cls = "seat-name self" if is_self else "seat-name"
    suffix = "（あなた）" if is_self else ""
    st.markdown(
        f"""<div class="seat-card">
                <div class="{name_cls}">🤖 {seat} {suffix}</div>
                <div class="seat-text">{text}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_client_side_timer(deadline_epoch, total_seconds):
    """
    サーバーへの再実行(rerun)を発生させない、ブラウザ内だけで動くカウントダウン表示。
    タイマー更新のたびに画面全体がちらつく問題を避けるために使用する。
    """
    deadline_ms = int(deadline_epoch * 1000)
    total_ms = int(total_seconds * 1000)
    html = f"""
    <div style="font-family:'JetBrains Mono','Courier New',monospace;">
      <div style="border:1px solid #2a2f36;background-color:#12151a;padding:10px 16px;
                  margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;
                  color:#d7dbe0;border-radius:4px;">
        <span>⏱ 残り時間</span>
        <span id="rw_timer_text" style="font-size:20px;font-weight:700;">--:--</span>
      </div>
      <div style="height:6px;background:#20242a;border-radius:3px;overflow:hidden;margin-bottom:6px;">
        <div id="rw_timer_bar" style="height:100%;background:#7fd3c7;width:100%;"></div>
      </div>
    </div>
    <script>
      (function() {{
        const deadline = {deadline_ms};
        const total = {total_ms};
        function tick() {{
          const now = Date.now();
          const remaining = Math.max(0, deadline - now);
          const s = Math.floor(remaining / 1000);
          const mm = String(Math.floor(s / 60)).padStart(2, '0');
          const ss = String(s % 60).padStart(2, '0');
          const textEl = document.getElementById('rw_timer_text');
          const barEl = document.getElementById('rw_timer_bar');
          if (textEl) textEl.textContent = mm + ':' + ss;
          if (barEl) barEl.style.width = Math.max(0, Math.min(100, (remaining / total) * 100)) + '%';
          if (remaining <= 0) clearInterval(iv);
        }}
        tick();
        const iv = setInterval(tick, 500);
      }})();
    </script>
    """
    components.html(html, height=60)


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
    - 入力欄(st.chat_input)は「必ず毎回」呼び出す。AIが発言を生成中でも
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

    render_client_side_timer(deadline, DAY_PHASE_SECONDS)
    st.caption("議題は決まっていません。自由に会話して、誰が「本物の人間」か探ってください。")

    for entry in st.session_state.chat_log:
        render_statement_card(entry["seat"], entry["text"])

    # --- 0) 入力欄は必ず毎回呼び出す（AI生成中でも常に発言できるようにするため） ---
    user_msg = None
    if not time_up:
        user_msg = st.chat_input("発言を入力（150文字以内）...")

    # --- 1) プレイヤーが発言した場合は最優先で処理する ---
    if user_msg:
        text = user_msg.strip()[:150]
        st.session_state.chat_log.append({"seat": human_seat, "text": text})
        alive_ai_seats = [s for s in alive if s != human_seat]
        # 通信はまだ行わず「誰が反応するか」だけ確定し、即座に再描画する
        st.session_state.pending_speakers = decide_speakers(alive_ai_seats)
        st.rerun()
        return

    # --- 2) 発言予定が確定しているAIの通信を実行する（このrunでは自動更新を呼ばない） ---
    if st.session_state.pending_speakers:
        speakers = st.session_state.pending_speakers
        label = "、".join(speakers)
        with st.spinner(f"{label} が発言を考え中..."):
            results = generate_replies_concurrently(
                speakers, st.session_state.day, st.session_state.chat_log, alive, st.session_state.seat_roles
            )
        for seat, text in results:
            st.session_state.chat_log.append({"seat": seat, "text": text})
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
            render_statement_card(entry["seat"], entry["text"])

    candidates = [s for s in alive if s != human_seat]

    # --- 1) プレイヤーの投票 ---
    if st.session_state.human_vote is None:
        st.markdown("#### 「本物の人間」だと思う相手に投票してください")
        choice = st.radio(
            "投票先", candidates, label_visibility="collapsed", key=f"vote_radio_{st.session_state.day}"
        )
        if st.button("🗳 投票を確定する", type="primary"):
            st.session_state.human_vote = choice
            st.session_state.votes[human_seat] = choice
            st.rerun()
        return

    # --- 2) 他AIの自動投票 ---
    if not st.session_state.votes_done:
        remaining = [s for s in alive if s != human_seat and s not in st.session_state.votes]
        if remaining:
            seat = remaining[0]
            role = st.session_state.seat_roles[seat]
            seat_candidates = [s for s in alive if s != seat]
            with st.spinner(f"{seat} が投票中..."):
                vote = call_ai_vote(
                    role, seat, st.session_state.day,
                    st.session_state.chat_log, seat_candidates,
                )
            st.session_state.votes[seat] = vote
            st.rerun()
        else:
            st.session_state.votes_done = True
            st.rerun()
        return

    # --- 3) 開票 ---
    st.markdown("#### 🗳 開票結果")
    tally = {}
    for voter, target in st.session_state.votes.items():
        tally[target] = tally.get(target, 0) + 1

    for seat in alive:
        count = tally.get(seat, 0)
        st.write(f"　{seat} ： {'●' * count} {count}票")

    max_votes = max(tally.values()) if tally else 0
    top_seats = [s for s, c in tally.items() if c == max_votes]

    if len(top_seats) >= 2:
        st.session_state.tie_result = True
        st.session_state.pending_elimination = None
        st.info(f"最多票が同数（{', '.join(top_seats)}）のため、今回は誰も追放されません。")
        if st.button("➡ 次の日へ進む", type="primary"):
            st.session_state.day += 1
            reset_day_state()
            st.rerun()
    else:
        eliminated = top_seats[0]
        st.session_state.pending_elimination = eliminated
        st.warning(f"最多票により **{eliminated}** が追放されます。")
        if st.button("➡ 結果を確定する", type="primary"):
            process_elimination(eliminated)
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

    st.markdown("#### 🗂 全員の正体")
    for seat in SEATS:
        role = st.session_state.seat_roles[seat]
        tag = "（あなた）" if seat == human_seat else ""
        st.write(f"　{seat}{tag} ： {ROLE_LABEL_JP[role]}")

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
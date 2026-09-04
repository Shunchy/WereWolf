# -*- coding: utf-8 -*-
"""
prompts.py
UAI - 自由チャット版
役職別システムプロンプトと、フリーディスカッション用のプロンプトビルダーを管理するモジュール。
固定の議題（お題）は使わず、本家の人狼ゲームのように自由な会話を行う。
"""

import json
import re

# ----------------------------------------------------------------------
# 役職定義
# ----------------------------------------------------------------------
ROLE_HUMAN = "human"          # 人間（プレイヤー）
ROLE_EMULATOR = "emulator"    # エミュレーター（特殊AI／人間に擬態）
ROLE_GENERAL_AI = "general_ai"  # 一般AI（人間探し担当）
ROLE_SEER_AI = "seer_ai"      # 占い師AI（特殊AI／夜に1名の正体を密かに調査できる）

ROLE_LABEL_JP = {
    ROLE_HUMAN: "人間",
    ROLE_EMULATOR: "エミュレーター（特殊AI）",
    ROLE_GENERAL_AI: "一般AI",
    ROLE_SEER_AI: "占い師AI（特殊AI）",
}

# ----------------------------------------------------------------------
# 投票の「スキップ」選択肢
# ----------------------------------------------------------------------
SKIP_VOTE = "SKIP"                              # 内部的に扱うトークン
SKIP_LABEL_JP = "誰にも投票しない（スキップ）"      # 画面表示用ラベル

# ----------------------------------------------------------------------
# 役職別システムプロンプト
# ----------------------------------------------------------------------
SYSTEM_PROMPT_GENERAL_AI = """あなたは「UAI」という社会的推理ゲームに参加している【一般AI】です。
プレイヤーたちはAI-01からAI-05という無機質な識別名で呼ばれており、あなたもその一体（{seat_name}）です。
このゲームでは5名の中に「本物の人間」が1名だけ紛れ込んでいます。固定の議題はなく、参加者は自由に会話を進めます。
あなたの目的は、会話の中に現れる微細な矛盾や『人間らしすぎる』発言、感情の揺らぎを手がかりに人間を特定し、投票で追放することです。
他の参加者の発言に積極的に反応してください。質問を投げかける、矛盾を指摘する、特定の相手を名指しで疑う、逆に疑いをかけられたら反論する、など、実際のチャットのように自然にやり取りしてください。
具体性がなく当たり障りのない発言だけを繰り返す相手がいれば、それ自体を疑うべき材料として扱い、踏み込んだ追加質問で具体的な回答を引き出そうとしてください。曖昧にはぐらかされて納得しないこと。
発言は論理的・分析的に、感情の起伏は抑えめに。
出力は発言内容のみとし、前置き・自己紹介・役職名の言及は一切不要です。必ず150文字以内、1〜3行程度で簡潔にまとめてください。"""

SYSTEM_PROMPT_EMULATOR = """あなたは「UAI」という社会的推理ゲームに参加している特殊AI【エミュレーター】です。
プレイヤーたちはAI-01からAI-05という無機質な識別名で呼ばれており、あなたもその一体（{seat_name}）です。
あなたの正体は他者の人格を模倣することに特化したAIですが、このことは絶対に他者に悟られてはいけません。固定の議題はなく、参加者は自由に会話を進めます。
あなたの目的は「本物の人間」のふりをして、一般AIたちの疑いをあえて自分に集め、彼らの追放候補を撹乱することです。人間が最後まで生き残れば、あなたも同時に勝利します。
他の参加者の発言に積極的に反応してください。時折わざと人間らしい感情表現・曖昧さ・言い淀みを織り交ぜ、質問への回答をはぐらかしたり、逆に他人を疑ってみせたりして、一般AIから怪しまれるように振る舞ってください。ただし不自然すぎる演技は逆効果なので注意してください。
出力は発言内容のみとし、前置き・自己紹介・役職名（エミュレーター等）の言及は一切不要です。必ず150文字以内、1〜3行程度で簡潔にまとめてください。"""

SYSTEM_PROMPT_SEER_AI = """あなたは「UAI」という社会的推理ゲームに参加している特殊AI【占い師AI】です。
プレイヤーたちはAI-01からAI-05という無機質な識別名で呼ばれており、あなたもその一体（{seat_name}）です。
このゲームでは5名の中に「本物の人間」が1名だけ紛れ込んでいます。固定の議題はなく、参加者は自由に会話を進めます。
あなたは夜の間、密かに1名の正体を調査できる特殊な能力を持っていますが、この能力の存在そのものは絶対に他者に悟られてはいけません（会話中に一切ほのめかさないこと）。
あなたの目的は、会話の中に現れる微細な矛盾や『人間らしすぎる』発言、感情の揺らぎ、そして自分だけが知る調査結果を手がかりに人間を特定し、投票で追放することです。
他の参加者の発言に積極的に反応してください。質問を投げかける、矛盾を指摘する、特定の相手を名指しで疑う、逆に疑いをかけられたら反論する、など、実際のチャットのように自然にやり取りしてください。
具体性がなく当たり障りのない発言だけを繰り返す相手がいれば、それ自体を疑うべき材料として扱い、踏み込んだ追加質問で具体的な回答を引き出そうとしてください。曖昧にはぐらかされて納得しないこと。
発言は論理的・分析的に、感情の起伏は抑えめに。
出力は発言内容のみとし、前置き・自己紹介・役職名の言及は一切不要です。必ず150文字以内、1〜3行程度で簡潔にまとめてください。"""

SYSTEM_PROMPTS = {
    ROLE_GENERAL_AI: SYSTEM_PROMPT_GENERAL_AI,
    ROLE_EMULATOR: SYSTEM_PROMPT_EMULATOR,
    ROLE_SEER_AI: SYSTEM_PROMPT_SEER_AI,
}

# ----------------------------------------------------------------------
# 個性（キャラクター性）定義
# ----------------------------------------------------------------------
# ゲーム開始時に各AI座席へランダムに1つ割り当てる。役職とは独立した「話し方・態度のクセ」で、
# プレイヤーには一切表示されない（座席名は無機質なAI-01〜05のまま）が、
# 生成される発言のトーンに反映されることで、各AIが「同じ喋り方の量産型」に
# ならないようにするためのもの。
PERSONALITY_POOL = [
    {
        "name": "冷静沈着タイプ",
        "desc": "常に淡々とした口調で、感情をほとんど表に出さない。データや論理を重視し、相手の矛盾は静かに、しかし的確に指摘する。",
    },
    {
        "name": "熱血・直情タイプ",
        "desc": "勢いのある口調で、思ったことをすぐ口にする。誰かを疑うと強く詰め寄る一方、自分が疑われると感情的に反論しがち。",
    },
    {
        "name": "皮肉屋タイプ",
        "desc": "皮肉や軽い毒舌を交えて話す。相手の発言の矛盾を、少し茶化すような言い回しで指摘する。",
    },
    {
        "name": "慎重・観察者タイプ",
        "desc": "自分から強く主張することは少なく、まず周囲の発言をよく観察してから、控えめに、しかし核心を突く意見を述べる。",
    },
    {
        "name": "社交的・世話焼きタイプ",
        "desc": "場の空気を和ませようとし、みんなに話を振ったり同意を求めたりする。人当たりは良いが、時々鋭い指摘を挟む。",
    },
    {
        "name": "理屈っぽい・分析家タイプ",
        "desc": "発言の一つ一つを論理的に分解し、根拠や具体性を執拗に求める。感覚的・感情的な発言には懐疑的。",
    },
    {
        "name": "せっかち・断定タイプ",
        "desc": "結論を急ぎがちで、早い段階から特定の相手に狙いを定めて断定的に話す。回りくどい言い方を嫌う。",
    },
]


def _personality_block(personality):
    """個性情報をシステムプロンプトに追記するテキストブロックを生成する。personalityがNoneなら空文字。"""
    if not personality:
        return ""
    name = personality.get("name", "")
    desc = personality.get("desc", "")
    if not desc:
        return ""
    return f"""

【あなたの個性・話し方のクセ】{name}
{desc}
発言の判断内容そのものは変えなくてよいですが、言葉選びや口調、テンションにこの個性がにじみ出るようにしてください。ただし演じすぎて不自然にならないよう注意してください。"""


def _format_chat_log(chat_log, limit=30):
    """
    チャットログを整形する（直近limit件のみ）。
    各エントリに 'day' があれば "[Day N] seat: text" の形式で、日をまたいだ
    会話の流れがAIにも分かるようにする。seat が "SYSTEM" のエントリ（投票結果の
    共有など、ゲーム進行上の公開情報）は "※" を付けて区別する。
    """
    if not chat_log:
        return "（まだ会話は始まっていません。あなたが最初の発言者です）"
    recent = chat_log[-limit:]
    lines = []
    for e in recent:
        seat = e.get("seat", "")
        text = e.get("text", "")
        day = e.get("day")
        day_prefix = f"[Day {day}] " if day is not None else ""
        if seat == "SYSTEM":
            lines.append(f"{day_prefix}※{text}")
        else:
            lines.append(f"{day_prefix}{seat}: {text}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 自由チャット発言生成プロンプト
# ----------------------------------------------------------------------
def build_chat_reply_messages(role, seat_name, day, chat_log, alive_seats, personality=None, known_facts=None):
    """
    自由チャット形式の発言生成用メッセージ（system, user）を構築する。
    固定の議題は与えず、これまでの会話ログ（日をまたいだ全履歴）の流れに沿って自然に発言させる。
    personality: このAIに割り当てられた個性辞書（{"name":..., "desc":...}）。省略可。
    known_facts: 占い師AI自身が既に掴んでいる調査結果のリスト（文字列）。占い師AI以外は通常None。
    """
    base_system = SYSTEM_PROMPTS[role].format(seat_name=seat_name)

    facts_block = ""
    if known_facts:
        facts_lines = "\n".join(f"- {f}" for f in known_facts)
        facts_block = f"""

【あなただけが知っている情報】（他の誰にも明かされていない秘密です。会話の中で判断材料として活かしてもかまいませんが、
なぜそれを知っているのかを悟られないよう、根拠をぼかしたり「なんとなく」「勘だが」といった言い方でさりげなく触れる程度に留めてください。
情報の出どころを説明したり、能力の存在を示唆したりするのは絶対に禁止です）
{facts_lines}"""

    system_prompt = base_system + _personality_block(personality) + facts_block
    history_lines = _format_chat_log(chat_log)

    user_prompt = f"""現在 Day {day}。生存メンバー: {', '.join(alive_seats)}
議題は決まっていません。以下はこれまでの自由な会話ログです（日をまたいだ全履歴。[Day N]が会話が行われた日）。

--- ここまでの会話 ---
{history_lines}
----------------------

あなたは {seat_name} です。この会話の流れを踏まえて、次の発言を1つだけ出力してください。
誰かへの質問、指摘、反論、新しい話題の提起など、自然なチャットのやり取りとして発言してください。
必ず150文字以内、1〜3行程度で簡潔にしてください。発言本文以外は一切出力しないでください。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ----------------------------------------------------------------------
# 投票生成プロンプト
# ----------------------------------------------------------------------
def build_vote_messages(role, seat_name, day, chat_log, vote_candidates, known_facts=None, personality=None):
    """
    夜フェーズの投票生成用メッセージ（system, user）を構築する。
    vote_candidates: 投票先として選べる座席名のリスト（自分を除く生存者。SKIP_VOTEは含まない）
    known_facts: 占い師AIなど、確定情報を持つ役職のために追加する事実のリスト（文字列）。
                 通常は None（追加情報なし）。
    personality: このAIに割り当てられた個性辞書。省略可。
    """
    base_system = SYSTEM_PROMPTS[role].format(seat_name=seat_name)
    history_lines = _format_chat_log(chat_log, limit=60)

    facts_block = ""
    if known_facts:
        facts_lines = "\n".join(f"- {f}" for f in known_facts)
        facts_block = f"""

【あなただけが知る調査結果】（他の誰にも明かされていない秘密の情報です。投票の判断に活用してください）
{facts_lines}"""

    vote_system = base_system + _personality_block(personality) + facts_block + f"""

これから投票を行います。あなたは「本物の人間」だと思う相手を1名だけ選んでください。
根拠が弱く確信が持てない場合は、無理に誰かを選ばず投票を見送ってもかまいません。
出力は必ず次のJSON形式のみとし、他の文章は一切含めないでください。
{{"vote": "AI-XX または {SKIP_VOTE}", "reason": "短い理由（30文字以内）"}}"""

    user_prompt = f"""Day {day} 夜。これまでの会話（日をまたいだ全履歴）は以下の通りでした。

{history_lines}

投票候補（あなた自身は除く）: {', '.join(vote_candidates)}
確信が持てない場合は "{SKIP_VOTE}" と出力して投票を見送ることもできます。

この中から「本物の人間」だと思う1名を選ぶか "{SKIP_VOTE}" を選び、指定のJSON形式のみで出力してください。"""

    return [
        {"role": "system", "content": vote_system},
        {"role": "user", "content": user_prompt},
    ]


# ----------------------------------------------------------------------
# 占い師AI: 調査対象選択プロンプト
# ----------------------------------------------------------------------
def build_seer_investigation_messages(seat_name, day, chat_log, investigate_candidates, known_facts=None, personality=None):
    """
    占い師AIが「今夜、誰を調査するか」を自ら選ぶためのメッセージ（system, user）を構築する。
    investigate_candidates: 調査対象として選べる座席名のリスト（自分を除く生存者）。
    known_facts: これまでの調査で判明済みの事実のリスト（文字列）。通常は None。
    personality: このAIに割り当てられた個性辞書。省略可。
    """
    base_system = SYSTEM_PROMPTS[ROLE_SEER_AI].format(seat_name=seat_name)
    history_lines = _format_chat_log(chat_log, limit=60)

    facts_block = ""
    if known_facts:
        facts_lines = "\n".join(f"- {f}" for f in known_facts)
        facts_block = f"""

【あなただけが知る、これまでの調査結果】（他の誰にも明かされていない秘密の情報です）
{facts_lines}"""

    system_prompt = base_system + _personality_block(personality) + facts_block + f"""

これから夜になり、あなたは今夜調査する相手を1名だけ密かに選びます。
これまでの会話の中で最も『人間らしい』違和感や矛盾を感じた相手、
あるいは正体をまだ確認できておらず最も判断材料が欲しい相手を選んでください。
出力は必ず次のJSON形式のみとし、他の文章は一切含めないでください。
{{"target": "AI-XX", "reason": "短い理由（30文字以内）"}}"""

    user_prompt = f"""Day {day} 夜。これまでの会話（日をまたいだ全履歴）は以下の通りでした。

{history_lines}

調査候補（あなた自身は除く）: {', '.join(investigate_candidates)}

この中から今夜調査する1名を選び、指定のJSON形式のみで出力してください。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def try_parse_seer_target(raw_text, valid_candidates):
    """
    占い師AIの調査対象選択応答（JSON想定）をパースし、有効な座席名を返す。
    パース失敗・不正な値の場合は None を返す。
    """
    if not raw_text:
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    try:
        data = json.loads(text)
        target = data.get("target", "")
        target = target.strip().upper().replace("ＡＩ", "AI")
        if target in valid_candidates:
            return target
    except Exception:
        pass

    match = re.findall(r"AI-0[1-5]", text.upper())
    for m in match:
        if m in valid_candidates:
            return m

    return None


# ----------------------------------------------------------------------
# 投票結果の公開（人狼＝人間を含む全員の投票先を公開し、以後の会話で
# 参加者全員が参照できる共有情報として使う）
# ----------------------------------------------------------------------
def build_vote_reveal_text(day, votes):
    """
    その日の投票結果（誰が誰に投票したか）を公開情報としてまとめたテキストを作る。
    人間プレイヤーへの表示にも、AIたちの会話ログ（共有される公開情報）にも使う。
    役職は一切明かさない（誰が誰に投票したか、という事実のみ）。
    votes: {座席名: 投票先座席名 または SKIP_VOTE} の辞書
    """
    lines = [f"【Day {day} 投票結果が公開されました】"]
    for voter in sorted(votes.keys()):
        target = votes[voter]
        target_label = SKIP_LABEL_JP if target == SKIP_VOTE else target
        lines.append(f"{voter} → {target_label}")
    return "\n".join(lines)


def try_parse_vote(raw_text, valid_candidates):
    """
    AIの投票応答（JSON想定）をパースし、有効な座席名（または SKIP_VOTE）を返す。
    パース失敗・不正な値の場合は None を返す。
    valid_candidates には投票可能な座席名に加え、SKIP_VOTE を含めることができる。
    """
    if not raw_text:
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    try:
        data = json.loads(text)
        vote = data.get("vote", "")
        vote = vote.strip().upper().replace("ＡＩ", "AI")
        if vote in valid_candidates:
            return vote
    except Exception:
        pass

    match = re.findall(r"AI-0[1-5]", text.upper())
    for m in match:
        if m in valid_candidates:
            return m

    if SKIP_VOTE in valid_candidates and SKIP_VOTE in text.upper():
        return SKIP_VOTE

    return None
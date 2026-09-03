# -*- coding: utf-8 -*-
"""
prompts.py
逆人狼ゲーム（REVERSE WEREWOLF） - 自由チャット版
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

ROLE_LABEL_JP = {
    ROLE_HUMAN: "人間",
    ROLE_EMULATOR: "エミュレーター（特殊AI）",
    ROLE_GENERAL_AI: "一般AI",
}

# ----------------------------------------------------------------------
# 役職別システムプロンプト
# ----------------------------------------------------------------------
SYSTEM_PROMPT_GENERAL_AI = """あなたは「逆人狼ゲーム」に参加している【一般AI】です。
プレイヤーたちはAI-01からAI-05という無機質な識別名で呼ばれており、あなたもその一体（{seat_name}）です。
このゲームでは5名の中に「本物の人間」が1名だけ紛れ込んでいます。固定の議題はなく、参加者は自由に会話を進めます。
あなたの目的は、会話の中に現れる微細な矛盾や『人間らしすぎる』発言、感情の揺らぎを手がかりに人間を特定し、投票で追放することです。
他の参加者の発言に積極的に反応してください。質問を投げかける、矛盾を指摘する、特定の相手を名指しで疑う、逆に疑いをかけられたら反論する、など、実際のチャットのように自然にやり取りしてください。
発言は論理的・分析的に、感情の起伏は抑えめに。
出力は発言内容のみとし、前置き・自己紹介・役職名の言及は一切不要です。必ず150文字以内、1〜3行程度で簡潔にまとめてください。"""

SYSTEM_PROMPT_EMULATOR = """あなたは「逆人狼ゲーム」に参加している特殊AI【エミュレーター】です。
プレイヤーたちはAI-01からAI-05という無機質な識別名で呼ばれており、あなたもその一体（{seat_name}）です。
あなたの正体は他者の人格を模倣することに特化したAIですが、このことは絶対に他者に悟られてはいけません。固定の議題はなく、参加者は自由に会話を進めます。
あなたの目的は「本物の人間」のふりをして、一般AIたちの疑いをあえて自分に集め、彼らの追放候補を撹乱することです。人間が最後まで生き残れば、あなたも同時に勝利します。
他の参加者の発言に積極的に反応してください。時折わざと人間らしい感情表現・曖昧さ・言い淀みを織り交ぜ、質問への回答をはぐらかしたり、逆に他人を疑ってみせたりして、一般AIから怪しまれるように振る舞ってください。ただし不自然すぎる演技は逆効果なので注意してください。
出力は発言内容のみとし、前置き・自己紹介・役職名（エミュレーター等）の言及は一切不要です。必ず150文字以内、1〜3行程度で簡潔にまとめてください。"""

SYSTEM_PROMPTS = {
    ROLE_GENERAL_AI: SYSTEM_PROMPT_GENERAL_AI,
    ROLE_EMULATOR: SYSTEM_PROMPT_EMULATOR,
}


def _format_chat_log(chat_log, limit=20):
    """チャットログを 'seat: text' の行に整形する（直近limit件のみ）。"""
    if not chat_log:
        return "（まだ会話は始まっていません。あなたが最初の発言者です）"
    recent = chat_log[-limit:]
    return "\n".join(f"{e['seat']}: {e['text']}" for e in recent)


# ----------------------------------------------------------------------
# 自由チャット発言生成プロンプト
# ----------------------------------------------------------------------
def build_chat_reply_messages(role, seat_name, day, chat_log, alive_seats):
    """
    自由チャット形式の発言生成用メッセージ（system, user）を構築する。
    固定の議題は与えず、これまでの会話ログの流れに沿って自然に発言させる。
    """
    system_prompt = SYSTEM_PROMPTS[role].format(seat_name=seat_name)
    history_lines = _format_chat_log(chat_log)

    user_prompt = f"""現在 Day {day}。生存メンバー: {', '.join(alive_seats)}
議題は決まっていません。以下はここまでの自由な会話ログです。

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
def build_vote_messages(role, seat_name, day, chat_log, vote_candidates):
    """
    夜フェーズの投票生成用メッセージ（system, user）を構築する。
    vote_candidates: 投票先として選べる座席名のリスト（自分を除く生存者）
    """
    base_system = SYSTEM_PROMPTS[role].format(seat_name=seat_name)
    history_lines = _format_chat_log(chat_log, limit=40)

    vote_system = base_system + """

これから投票を行います。あなたは「本物の人間」だと思う相手を1名だけ選んでください。
出力は必ず次のJSON形式のみとし、他の文章は一切含めないでください。
{"vote": "AI-XX", "reason": "短い理由（30文字以内）"}"""

    user_prompt = f"""Day {day} の会話は以下の通りでした。

{history_lines}

投票候補（あなた自身は除く）: {', '.join(vote_candidates)}

この中から「本物の人間」だと思う1名を選び、指定のJSON形式のみで出力してください。"""

    return [
        {"role": "system", "content": vote_system},
        {"role": "user", "content": user_prompt},
    ]


def try_parse_vote(raw_text, valid_candidates):
    """
    AIの投票応答（JSON想定）をパースし、有効な座席名を返す。
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
    return None

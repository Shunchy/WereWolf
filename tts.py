# -*- coding: utf-8 -*-
"""
tts.py
Fish Audio (https://docs.fish.audio) を使ったテキスト読み上げ(TTS)クライアント。

Streamlitには一切依存しない独立モジュール。app.py側から
「文字列→音声バイト列(mp3)」の変換と、「LLMのストリーミング出力を
文単位でリアルタイムに音声化していくパイプライン」を提供する。

設計のポイント（レイテンシの隠蔽）:
  LLM(OpenRouter)からテキストが少しずつ届くたびに `SentenceTTSPipeline.feed()`
  へ流し込むと、「。！？」などで文が確定した瞬間に、その文だけを
  即座に ThreadPoolExecutor 経由で Fish Audio へ非同期リクエストする。
  つまり1文目の音声を生成している間にも、LLMは2文目・3文目のテキストを
  生成し続けており、両者は並行して進む。
  ストリーミングが終わった時点では、最後の1〜2文分のTTSレスポンス待ち
  だけが残っている状態になるため、「全文が出そろってから音声化」する
  場合に比べて、体感の待ち時間を大きく圧縮できる。
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

FISH_TTS_URL = "https://api.fish.audio/v1/tts"
DEFAULT_MODEL = "s2.1-pro-free"
REQUEST_TIMEOUT = 30  # 1文あたりのTTSリクエストのタイムアウト（秒）

# 日本語の文区切り「。！？」または改行までを1文として拾う正規表現。
# 区切り記号自体も文に含める（読み上げの間の取り方が自然になるため）。
_SENTENCE_END_RE = re.compile(r"([^。！？\n]*[。！？\n]+)")


def split_into_sentences(text):
    """
    テキストを「。！？」及び改行で1文ずつに分割する。
    末尾に区切り記号の無い残り部分がある場合は、それも最後の1文として含める。
    """
    if not text:
        return []
    sentences = []
    pos = 0
    for m in _SENTENCE_END_RE.finditer(text):
        s = m.group(1).strip()
        if s:
            sentences.append(s)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def synthesize_sentence(text, api_key, model=DEFAULT_MODEL, reference_id=None, timeout=REQUEST_TIMEOUT):
    """
    1文をFish Audio API (POST /v1/tts) に送信し、mp3の音声バイト列を返す。
    通信に失敗した場合は例外を投げず、Noneを返す
    （呼び出し側はその文の音声だけを諦めて、テキスト表示は止めない）。

    バックグラウンドスレッドから呼ばれるためStreamlitのAPIは使えない。
    失敗の詳細は（アプリのログに残るよう）標準エラー出力に書き出しておく。
    """
    if not text:
        return None
    if not api_key:
        print("[tts] FISH_AUDIO_API_KEY が設定されていないため、音声合成をスキップしました。", file=sys.stderr)
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Fish AudioはHTTPヘッダーでモデルを切り替える方式
        # （例: "s2.1-pro-free"）。ボディではなくヘッダーに載せる点に注意。
        "model": model,
    }
    payload = {"text": text, "format": "mp3"}
    if reference_id:
        payload["reference_id"] = reference_id
    try:
        resp = requests.post(FISH_TTS_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        if not resp.content:
            print("[tts] Fish Audioから空の音声データが返されました。", file=sys.stderr)
            return None
        return resp.content
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        print(f"[tts] Fish Audio APIエラー: {e} / response: {body}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[tts] Fish Audio 通信エラー: {e}", file=sys.stderr)
        return None


def concat_mp3_clips(clips):
    """
    複数のmp3バイト列を単純連結し、1本のmp3として返す。
    MP3はフレーム単位のフォーマットのため、素朴な連結でもほとんどのプレイヤー
    （ブラウザのHTMLAudioElementを含む）で問題なく連続再生できる。
    Noneや空の要素は無視する。全て空ならNoneを返す。
    """
    valid = [c for c in (clips or []) if c]
    if not valid:
        return None
    return b"".join(valid)


def synthesize_text_batch(text, api_key, reference_id, executor, model=DEFAULT_MODEL, timeout=REQUEST_TIMEOUT + 10):
    """
    ストリーミングを使わない場面（フォールバック発言・相槌の事前生成など）向けの簡易版。
    1つのテキストを文単位に分割し、全文をまとめて並行してTTSに投げる。
    戻り値は文の順番通りの [bytes|None, ...]。
    """
    if not (text and api_key and executor):
        return []
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    futures = [
        executor.submit(synthesize_sentence, s, api_key, model, reference_id)
        for s in sentences
    ]
    results = []
    for f in futures:
        try:
            results.append(f.result(timeout=timeout))
        except Exception:
            results.append(None)
    return results


class SentenceTTSPipeline:
    """
    LLMのストリーミング出力を受け取りながら、文が1つ確定するたびに
    「即座に」非同期タスクとしてFish AudioへTTSリクエストを投げるパイプライン。

    使い方:
        executor = ThreadPoolExecutor(max_workers=4)
        pipeline = SentenceTTSPipeline(api_key, executor)
        for delta in llm_stream:            # LLMからの断片テキスト
            pipeline.feed(delta)
        pipeline.finish()                   # 末尾に残った断片を最後の1文として処理
        audio_clips = pipeline.collect_audio()  # 文の順番通りの [bytes|None, ...]
        full_text = pipeline.full_text
    """

    def __init__(self, api_key, executor, model=DEFAULT_MODEL, reference_id=None):
        self.api_key = api_key
        self.executor = executor
        self.model = model
        self.reference_id = reference_id
        self.full_text = ""
        self._buffer = ""
        self._futures = []  # 文の順番を保ったまま Future を積んでいく

    def _dispatch(self, sentence):
        sentence = sentence.strip()
        if not sentence:
            return
        future = self.executor.submit(
            synthesize_sentence, sentence, self.api_key, self.model, self.reference_id
        )
        self._futures.append(future)

    def feed(self, chunk_text):
        """LLMから届いた断片テキストを取り込み、文が完成した分だけ即座にTTSを投げる。"""
        if not chunk_text:
            return
        self.full_text += chunk_text
        self._buffer += chunk_text

        sentences = []
        pos = 0
        for m in _SENTENCE_END_RE.finditer(self._buffer):
            sentences.append(m.group(1))
            pos = m.end()
        if sentences:
            for s in sentences:
                self._dispatch(s)
            self._buffer = self._buffer[pos:]

    def finish(self):
        """ストリーム終了後、区切り記号の無いまま残ったバッファを最後の1文として処理する。"""
        if self._buffer.strip():
            self._dispatch(self._buffer)
            self._buffer = ""

    def collect_audio(self, timeout=REQUEST_TIMEOUT + 15):
        """
        文の順番通りに音声バイト列(または失敗時はNone)のリストを返す。
        ストリーミング中にバックグラウンドで既にかなり進行しているため、
        ここで実際に待つのは基本的に最後の1〜2文分だけで済む。
        """
        results = []
        for f in self._futures:
            try:
                results.append(f.result(timeout=timeout))
            except Exception:
                results.append(None)
        return results

    def cancel(self):
        """バリデーション失敗などで、この試行分のTTS結果を丸ごと破棄したい場合に呼ぶ。"""
        for f in self._futures:
            f.cancel()
        self._futures = []

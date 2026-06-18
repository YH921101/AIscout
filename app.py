import json
import logging
import os
import re
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from openai import OpenAI
from pdfminer.high_level import extract_text
from zoneinfo import ZoneInfo


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TDNET_BASE_URL = "https://www.release.tdnet.info/inbs/"
DEFAULT_KEYWORDS = [
    "月次",
    "受注",
    "受注残",
    "大型受注",
    "契約締結",
    "業績予想",
    "上方修正",
    "下方修正",
    "中期経営計画",
    "補助金",
    "採択",
    "新工場",
    "設備投資",
    "増産",
    "提携",
    "資本業務提携",
    "AI",
    "半導体",
    "防衛",
    "宇宙",
    "データセンター",
    "電力",
    "蓄電池",
    "造船",
    "インバウンド",
]

SCOUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "judgement": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "watch_points": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "one_line_summary": {"type": "string"},
    },
    "required": [
        "score",
        "judgement",
        "reasons",
        "watch_points",
        "risks",
        "one_line_summary",
    ],
}


@dataclass(frozen=True)
class Settings:
    timezone: str
    tdnet_base_url: str
    tdnet_lookback_days: int
    tdnet_max_pages: int
    request_timeout_sec: int
    pdf_max_bytes: int
    pdf_text_max_chars: int
    max_ai_candidates_per_run: int
    notify_threshold: int
    log_threshold: int
    run_token: str
    openai_api_key: str
    openai_model: str
    openai_reasoning_effort: str
    discord_webhook_url: str
    data_dir: Path
    processed_path: Path
    log_path: Path
    dry_run: bool

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(__file__).resolve().parent
        data_dir = Path(os.getenv("DATA_DIR", base_dir / "data")).resolve()
        return cls(
            timezone=os.getenv("TIMEZONE", "Asia/Tokyo"),
            tdnet_base_url=os.getenv("TDNET_BASE_URL", TDNET_BASE_URL),
            tdnet_lookback_days=max(1, env_int("TDNET_LOOKBACK_DAYS", 1)),
            tdnet_max_pages=max(1, env_int("TDNET_MAX_PAGES", 5)),
            request_timeout_sec=max(5, env_int("REQUEST_TIMEOUT_SEC", 20)),
            pdf_max_bytes=max(100_000, env_int("PDF_MAX_BYTES", 8_000_000)),
            pdf_text_max_chars=max(1_000, env_int("PDF_TEXT_MAX_CHARS", 16_000)),
            max_ai_candidates_per_run=max(1, env_int("MAX_AI_CANDIDATES_PER_RUN", 12)),
            notify_threshold=env_int("NOTIFY_THRESHOLD", 95),
            log_threshold=env_int("LOG_THRESHOLD", 85),
            run_token=os.getenv("RUN_TOKEN", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
            data_dir=data_dir,
            processed_path=Path(os.getenv("PROCESSED_PATH", data_dir / "processed.json")).resolve(),
            log_path=Path(os.getenv("LOG_PATH", data_dir / "scout_log.jsonl")).resolve(),
            dry_run=env_bool("DRY_RUN", False),
        )


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


settings = Settings.from_env()
app = Flask(__name__)
session = requests.Session()
session.headers.update(
    {
        "User-Agent": "tdnet-ai-scout/0.1 (+https://render.com)",
        "Accept-Language": "ja,en;q=0.8",
    }
)
run_lock = threading.Lock()
state_lock = threading.Lock()
job_status_lock = threading.Lock()
job_status: dict[str, Any] = {
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_http_status": None,
    "last_result": None,
    "last_error": None,
}


def now_jst() -> datetime:
    return datetime.now(ZoneInfo(settings.timezone))


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").lower()


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip()


def clean_pdf_text(value: str) -> str:
    cleaned = clean_text(value)
    return re.sub(
        r"(?<=[ぁ-んァ-ヶー一-龯々])\s+(?=[ぁ-んァ-ヶー一-龯々])",
        "",
        cleaned,
    )


def normalize_code(raw_code: str) -> str:
    code = clean_text(raw_code)
    if len(code) == 5 and code.endswith("0"):
        return code[:4]
    return code


def disclosure_id_from_url(url: str) -> str:
    filename = Path(urlparse(url).path).name
    return Path(filename).stem or url


def title_matches(title: str) -> list[str]:
    normalized_title = normalize_text(title)
    return [kw for kw in DEFAULT_KEYWORDS if normalize_text(kw) in normalized_title]


def tdnet_dates() -> list[str]:
    today = now_jst().date()
    return [
        (today - timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(settings.tdnet_lookback_days)
    ]


def load_state() -> dict[str, Any]:
    if not settings.processed_path.exists():
        return {"version": 1, "seen": {}}
    try:
        with settings.processed_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load state: %s", exc)
        return {"version": 1, "seen": {}}
    if not isinstance(data, dict):
        return {"version": 1, "seen": {}}
    data.setdefault("version", 1)
    data.setdefault("seen", {})
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_log(record: dict[str, Any]) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def fetch_text(url: str) -> str:
    response = session.get(url, timeout=settings.request_timeout_sec)
    response.raise_for_status()
    return response.content.decode("utf-8", errors="replace")


def fetch_tdnet_disclosures() -> list[dict[str, Any]]:
    disclosures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for ymd in tdnet_dates():
        disclosures.extend(fetch_tdnet_date(ymd, seen_ids))
    disclosures.sort(key=lambda item: item.get("published_at", ""), reverse=True)
    return disclosures


def fetch_tdnet_date(ymd: str, seen_ids: set[str]) -> list[dict[str, Any]]:
    first_page = f"I_list_001_{ymd}.html"
    queue = [first_page]
    seen_pages: set[str] = set()
    rows: list[dict[str, Any]] = []

    while queue and len(seen_pages) < settings.tdnet_max_pages:
        page_name = queue.pop(0)
        if page_name in seen_pages:
            continue
        seen_pages.add(page_name)
        page_url = urljoin(settings.tdnet_base_url, page_name)

        try:
            html = fetch_text(page_url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.info("TDnet page not found: %s", page_url)
                continue
            raise

        rows.extend(parse_tdnet_page(html, page_url, ymd, seen_ids))
        page_links = sorted(
            set(re.findall(rf"I_list_\d{{3}}_{re.escape(ymd)}\.html", html))
        )
        for link in page_links:
            if link not in seen_pages and link not in queue:
                queue.append(link)

    return rows


def parse_tdnet_page(
    html: str, page_url: str, ymd: str, seen_ids: set[str]
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []

    for tr in soup.select("table#main-list-table tr"):
        time_cell = tr.select_one("td.kjTime")
        code_cell = tr.select_one("td.kjCode")
        name_cell = tr.select_one("td.kjName")
        title_cell = tr.select_one("td.kjTitle")
        place_cell = tr.select_one("td.kjPlace")
        link = title_cell.find("a", href=True) if title_cell else None
        if not all([time_cell, code_cell, name_cell, title_cell, link]):
            continue

        pdf_url = urljoin(page_url, link["href"])
        disclosure_id = disclosure_id_from_url(pdf_url)
        if disclosure_id in seen_ids:
            continue
        seen_ids.add(disclosure_id)

        disclosed_time = clean_text(time_cell.get_text(" ", strip=True))
        published_at = parse_published_at(ymd, disclosed_time)
        title = clean_text(link.get_text(" ", strip=True))
        raw_code = clean_text(code_cell.get_text(" ", strip=True))

        rows.append(
            {
                "id": disclosure_id,
                "date": ymd,
                "time": disclosed_time,
                "published_at": published_at,
                "code": normalize_code(raw_code),
                "tdnet_code": raw_code,
                "company": clean_text(name_cell.get_text(" ", strip=True)),
                "title": title,
                "exchange": clean_text(place_cell.get_text(" ", strip=True))
                if place_cell
                else "",
                "url": pdf_url,
                "source_page": page_url,
            }
        )

    return rows


def parse_published_at(ymd: str, disclosed_time: str) -> str:
    try:
        dt = datetime.strptime(f"{ymd} {disclosed_time}", "%Y%m%d %H:%M")
        return dt.replace(tzinfo=ZoneInfo(settings.timezone)).isoformat()
    except ValueError:
        return f"{ymd}T{disclosed_time}"


def fetch_pdf_text(pdf_url: str) -> str:
    response = session.get(pdf_url, timeout=settings.request_timeout_sec)
    response.raise_for_status()
    content = response.content
    if len(content) > settings.pdf_max_bytes:
        logger.warning("PDF too large, skipping text extraction: %s", pdf_url)
        return ""

    try:
        text = extract_text(BytesIO(content)) or ""
    except Exception as exc:
        logger.warning("PDF text extraction failed for %s: %s", pdf_url, exc)
        return ""

    lines = [clean_pdf_text(line) for line in text.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return compact[: settings.pdf_text_max_chars]


def score_disclosure(disclosure: dict[str, Any], pdf_text: str) -> dict[str, Any]:
    if settings.dry_run:
        return heuristic_score(disclosure, pdf_text)
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=settings.openai_api_key)
    matched_keywords = disclosure.get("matched_keywords", [])
    prompt = build_scout_prompt(disclosure, pdf_text, matched_keywords)

    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたは日本株の短中期スイング投資向けに、TDnet適時開示から"
                    "初動検知に値する材料だけを厳選するアナリストです。"
                    "投資助言ではなく、材料の大きさ・新規性・短期反応可能性・リスクを"
                    "保守的に採点してください。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": settings.openai_reasoning_effort},
        text={
            "format": {
                "type": "json_schema",
                "name": "tdnet_ai_scout_score",
                "schema": SCOUT_SCHEMA,
                "strict": True,
            },
            "verbosity": "low",
        },
        temperature=0.2,
        store=False,
    )
    return json.loads(response.output_text)


def build_scout_prompt(
    disclosure: dict[str, Any], pdf_text: str, matched_keywords: list[str]
) -> str:
    body = pdf_text.strip() or "PDF本文の抽出に失敗、または本文なし。タイトルとメタ情報だけで慎重に判定。"
    return f"""
以下のTDnet開示を0〜100点で採点してください。

通知基準:
- 95点以上: Discord通知に値する。数値インパクト・新規性・短期反応可能性がかなり強い。
- 85〜94点: 有望だが、規模・織り込み・リスクに不確実性がある。ログ保存のみ。
- 84点以下: 通常の開示、既出感が強い、短期材料性が弱い、または失望リスクが大きい。

評価観点:
- 売上高比で大きい案件か。
- 受注残・受注高・月次KPIが前年同期比で大きく増えているか。
- 会社予想にまだ織り込まれていない可能性があるか。
- 時価総額に対して材料が大きいか。
- 国策・テーマ性があるか。
- 短期で株価反応しやすい材料か。
- 出尽くし売りや失望リスクがあるか。
- 既に株価が大きく上がりすぎていないか。

減点:
- 役員人事、株式報酬、定款、自己株取得の進捗、形式的な訂正など、短期材料性が弱い開示。
- 数値規模が不明、既に業績予想へ反映済み、または一過性の可能性が高い開示。
- 下方修正は原則低評価。ただし悪材料出尽くしや構造改善が明確なら理由を明記。

開示:
- 銘柄: {disclosure.get("company", "")}
- 証券コード: {disclosure.get("code", "")}
- 開示日時: {disclosure.get("published_at", "")}
- 開示タイトル: {disclosure.get("title", "")}
- 一次フィルタ一致: {", ".join(matched_keywords)}
- URL: {disclosure.get("url", "")}

PDF本文抜粋:
{body}
""".strip()


def heuristic_score(disclosure: dict[str, Any], pdf_text: str) -> dict[str, Any]:
    text = normalize_text(disclosure.get("title", "") + "\n" + pdf_text)
    score = 60
    boosts = {
        "大型受注": 25,
        "受注": 15,
        "上方修正": 20,
        "業績予想": 12,
        "資本業務提携": 18,
        "契約締結": 15,
        "補助金": 10,
        "採択": 10,
        "半導体": 8,
        "防衛": 8,
        "宇宙": 8,
        "データセンター": 8,
        "蓄電池": 8,
        "新工場": 8,
        "増産": 8,
    }
    for keyword, points in boosts.items():
        if normalize_text(keyword) in text:
            score += points
    if "下方修正" in text:
        score -= 30
    score = max(0, min(score, 100))
    return {
        "score": score,
        "judgement": "DRY_RUNの簡易判定です。本番ではOpenAI APIの判定を使用してください。",
        "reasons": ["タイトルと本文キーワードから機械的に仮採点しました。"],
        "watch_points": ["本番運用前にOPENAI_API_KEYとDRY_RUN=falseを設定してください。"],
        "risks": ["簡易判定は数値規模や織り込み度を十分に評価できません。"],
        "one_line_summary": "簡易スコアリング",
    }


def format_discord_message(disclosure: dict[str, Any], score: dict[str, Any]) -> str:
    reasons = bullet_lines(score.get("reasons", []), limit=4)
    watch_points = bullet_lines(score.get("watch_points", []), limit=3)
    risks = bullet_lines(score.get("risks", []), limit=3)
    message = f"""🔥 全市場AIスカウト

銘柄：{disclosure.get("company", "")}
証券コード：{disclosure.get("code", "")}
開示タイトル：{disclosure.get("title", "")}
スコア：{score.get("score", "")}

判定：
{score.get("judgement", "")}

理由：
{reasons}

見るべきポイント：
{watch_points}

リスク：
{risks}

URL：
{disclosure.get("url", "")}"""
    return truncate_discord_message(message)


def bullet_lines(items: list[Any], limit: int) -> str:
    cleaned = [clean_text(str(item)) for item in items if clean_text(str(item))]
    if not cleaned:
        cleaned = ["特記事項なし"]
    return "\n".join(f"- {item}" for item in cleaned[:limit])


def truncate_discord_message(message: str) -> str:
    if len(message) <= 1900:
        return message
    return message[:1870].rstrip() + "\n...(省略)"


def send_discord(message: str) -> None:
    if settings.dry_run:
        logger.info("DRY_RUN Discord message:\n%s", message)
        return
    if not settings.discord_webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL is not set")
    response = session.post(
        settings.discord_webhook_url,
        json={"content": message, "allowed_mentions": {"parse": []}},
        timeout=settings.request_timeout_sec,
    )
    response.raise_for_status()


def mark_seen(
    state: dict[str, Any],
    disclosure: dict[str, Any],
    status: str,
    score: int | None = None,
) -> None:
    state["seen"][disclosure["id"]] = {
        "seen_at": now_jst().isoformat(),
        "date": disclosure.get("date"),
        "time": disclosure.get("time"),
        "code": disclosure.get("code"),
        "company": disclosure.get("company"),
        "title": disclosure.get("title"),
        "url": disclosure.get("url"),
        "status": status,
        "score": score,
    }


def run_once() -> tuple[dict[str, Any], int]:
    if not run_lock.acquire(blocking=False):
        return {"ok": False, "error": "previous run is still in progress"}, 409
    try:
        return _run_once_locked()
    finally:
        run_lock.release()


def _run_once_locked() -> tuple[dict[str, Any], int]:
    with state_lock:
        state = load_state()

    disclosures = fetch_tdnet_disclosures()
    seen = state.get("seen", {})
    new_disclosures = [item for item in disclosures if item["id"] not in seen]

    result = {
        "ok": True,
        "fetched": len(disclosures),
        "new": len(new_disclosures),
        "title_matched": 0,
        "scored": 0,
        "logged": 0,
        "notified": 0,
        "ignored": 0,
        "deferred": 0,
        "deferred_items": [],
        "errors": [],
        "auth_warning": not bool(settings.run_token),
    }

    ai_candidates_used = 0
    for disclosure in new_disclosures:
        matched = title_matches(disclosure["title"])
        disclosure["matched_keywords"] = matched
        if not matched:
            mark_seen(state, disclosure, "ignored_title")
            result["ignored"] += 1
            continue

        result["title_matched"] += 1
        if ai_candidates_used >= settings.max_ai_candidates_per_run:
            result["deferred"] += 1
            if len(result["deferred_items"]) < 10:
                result["deferred_items"].append(
                    {
                        "id": disclosure["id"],
                        "title": disclosure["title"],
                        "reason": "MAX_AI_CANDIDATES_PER_RUN reached; will retry next run",
                    }
                )
            continue

        ai_candidates_used += 1
        try:
            pdf_text = fetch_pdf_text(disclosure["url"])
            score = score_disclosure(disclosure, pdf_text)
            score_value = int(score.get("score", 0))
            result["scored"] += 1
        except Exception as exc:
            logger.exception("Failed to score disclosure %s", disclosure.get("id"))
            result["errors"].append(
                {"id": disclosure["id"], "title": disclosure["title"], "error": str(exc)}
            )
            continue

        log_record = {
            "logged_at": now_jst().isoformat(),
            "disclosure": disclosure,
            "score": score,
        }

        if score_value >= settings.notify_threshold:
            try:
                send_discord(format_discord_message(disclosure, score))
            except Exception as exc:
                logger.exception("Failed to send Discord notification %s", disclosure.get("id"))
                result["errors"].append(
                    {
                        "id": disclosure["id"],
                        "title": disclosure["title"],
                        "score": score_value,
                        "error": str(exc),
                    }
                )
                append_log({**log_record, "status": "discord_error"})
                continue
            append_log({**log_record, "status": "notified"})
            mark_seen(state, disclosure, "notified", score_value)
            result["notified"] += 1
        elif score_value >= settings.log_threshold:
            append_log({**log_record, "status": "logged_candidate"})
            mark_seen(state, disclosure, "logged_candidate", score_value)
            result["logged"] += 1
        else:
            mark_seen(state, disclosure, "scored_low", score_value)
            result["ignored"] += 1

    with state_lock:
        write_json_atomic(settings.processed_path, state)

    status = 207 if result["errors"] else 200
    return result, status


def run_once_background() -> bool:
    with job_status_lock:
        if job_status["running"]:
            return False
        job_status.update(
            {
                "running": True,
                "last_started_at": now_jst().isoformat(),
                "last_finished_at": None,
                "last_http_status": None,
                "last_error": None,
            }
        )

    thread = threading.Thread(target=_background_worker, daemon=True)
    thread.start()
    return True


def _background_worker() -> None:
    try:
        result, status = run_once()
        with job_status_lock:
            job_status.update(
                {
                    "running": False,
                    "last_finished_at": now_jst().isoformat(),
                    "last_http_status": status,
                    "last_result": result,
                    "last_error": None,
                }
            )
    except Exception as exc:
        logger.exception("Background run failed")
        with job_status_lock:
            job_status.update(
                {
                    "running": False,
                    "last_finished_at": now_jst().isoformat(),
                    "last_http_status": 500,
                    "last_error": str(exc),
                }
            )


def request_authorized() -> bool:
    if not settings.run_token:
        return True
    supplied = request.args.get("token") or request.headers.get("X-Run-Token", "")
    return supplied == settings.run_token


@app.get("/")
def index():
    return jsonify(
        {
            "name": "全市場AIスカウトBOT",
            "health": "/health",
            "run": "/run?token=YOUR_RUN_TOKEN",
            "status": "/status?token=YOUR_RUN_TOKEN",
            "dry_run": settings.dry_run,
            "model": settings.openai_model,
            "notify_threshold": settings.notify_threshold,
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "time": now_jst().isoformat(),
            "state_exists": settings.processed_path.exists(),
        }
    )


@app.get("/status")
def status_endpoint():
    if not request_authorized():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    with job_status_lock:
        return jsonify({"ok": True, **job_status})


@app.get("/run")
def run_endpoint():
    if not request_authorized():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    if request.args.get("sync") == "1":
        result, status = run_once()
        return jsonify(result), status

    started = run_once_background()
    if not started:
        return jsonify({"ok": True, "started": False, "message": "previous run is still in progress"}), 200
    return jsonify({"ok": True, "started": True, "status": "/status"}), 202


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

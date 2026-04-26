"""subtask_results envelope 공용 유틸.

여러 노드(refiner, var_binder 등)에서 동일하게 쓰이는
- envelope payload 평면화
- "subtask_id별 최신 verdict=True attempt" 선택
- 문자열 자르기
- reference_features 포맷
"""

from typing import Any, Dict, Iterable, List, Optional


def result_payload(entry: Dict) -> Dict:
    """envelope `{id, attempt, verdict, result: {...}}`에서 결과 payload 추출.

    구 형태(평면 키)도 backward-compat fallback.
    """
    payload = entry.get("result") or {}
    return {
        "subtask_answer": payload.get("subtask_answer") or entry.get("subtask_answer", ""),
        "refined_text": payload.get("refined_text") or entry.get("refined_text", ""),
        "reference_features": payload.get("reference_features") or entry.get("reference_features", []),
    }


def pick_latest_successful(
    subtask_results: Iterable[Dict],
    *,
    exclude_id: Any = None,
    key_as_str: bool = False,
) -> Dict[Any, Dict]:
    """verdict=True인 entry 중 subtask_id별 최신 attempt만 남긴 dict 반환.

    Args:
        exclude_id: 결과에서 제외할 subtask id (현재 처리 중인 id 등)
        key_as_str: True면 dict 키를 str로 강제 (LLM JSON 직렬화용)
    """
    latest: Dict[Any, Dict] = {}
    for r in subtask_results:
        if r.get("verdict") is not True:
            continue
        sid = r.get("id")
        if sid is None or sid == exclude_id:
            continue
        key = str(sid) if key_as_str else sid
        prev = latest.get(key)
        if prev is None or r.get("attempt", 0) > prev.get("attempt", 0):
            latest[key] = r
    return latest


def truncate(s: str, n: int, suffix: str = "...") -> str:
    """길이 n 초과 시 잘라서 suffix 부착."""
    if not s:
        return s
    return s if len(s) <= n else s[:n] + suffix


def format_features(
    features: List[Dict],
    *,
    sep: Optional[str] = None,
    line_prefix: str = "",
    empty: str = "(없음)",
) -> str:
    """reference_features 리스트를 'fid: fname' 문자열로 포맷.

    sep=None → 한 줄에 하나(`{line_prefix}{fid}: {fname}`)
    sep=", " → 한 줄에 모두(`fid:fname, fid:fname`)
    """
    parts: List[str] = []
    for f in features or []:
        if not isinstance(f, dict):
            continue
        fid = f.get("feature_id", "?")
        fname = f.get("feature_name", "")
        if sep is None:
            parts.append(f"{line_prefix}{fid}: {fname}" if fname else f"{line_prefix}{fid}")
        else:
            parts.append(f"{fid}:{fname}" if fname else f"{fid}")
    if not parts:
        return empty
    return (sep if sep is not None else "\n").join(parts)

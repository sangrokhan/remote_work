"""aipt.gateway.profiles -- named netem parameter presets (DESIGN.md 4.7,
"구성" table, 웹 UI dropdown values: clean/wired/wireless/custom).

Each :class:`Profile` is the full set of ``tc netem`` knobs this package
controls: ``delay_ms``/``jitter_ms`` (variable one-way delay),
``loss_pct`` (packet loss %), ``reorder_pct`` (packet reordering %). Values
are intentionally simple (no distribution/correlation parameters) --
DESIGN.md's own "미해결 세부사항" leaves those for later phases; this is
the first, concrete cut.

``custom`` is not itself a fixed preset -- it is the escape hatch that lets
a caller (env vars, the POST /gateway/profile body) supply arbitrary
values, still expressed as the same :class:`Profile` shape so
``netem_control`` never has to special-case it.

2026-09 재설계 (3-profile, 근거 명시): 원래 5개 프리셋(broadband/3g/satellite/
lossy)은 illustrative 숫자였고 근거 문서가 없었다. 세 가지 문제로 재설계했다:

1. netem 원산지: ``tc netem``에는 "profile" 개념 자체가 없다 -- delay/loss/
   reorder는 순수 raw 파라미터고, 이름 붙은 프리셋은 이 모듈이 만든 추상화다.
2. 무선(LTE/NR) 구간은 MAC 계층 HARQ + RLC AM ARQ로 로컬 재전송을 하기 때문에,
   IP/TCP 계층에 실제로 보이는 것은 "패킷 손실"이 아니라 "재전송으로 인한
   지연/지터 증가"인 경우가 대부분이다. 즉 netem의 ``loss``(균등 확률로 그냥
   드롭)로 무선 구간을 모사하면, 실제로는 상위 계층까지 새지 않는 손실을
   인위적으로 만들어 TCP 재전송/cwnd 감소를 과대 유발한다 -- 3GPP 5QI PER
   목표치가 유선보다도 낮게 잡히는 이유가 이것이다.
3. 그래서 프리셋을 clean/wired/wireless 3개 + custom으로 줄이고, wireless는
   loss를 낮게 유지하는 대신 jitter로 "재전송에 의해 늦게 도착하지만 결국은
   전달됨"을 근사한다. 여전히 근사 모델이다 -- HARQ/RLC의 최대 재전송 횟수를
   다 쓰고도 실패하는 드문 진짜 IP-loss 케이스는 표현하지 못한다(그건
   ``custom``으로 별도 설정).

값 근거:
  - wired loss_pct=0.1: ITU-T Rec. Y.1541 (Network performance objectives
    for IP-based services) Table 1, QoS Class 0-4의 IP Packet Loss Ratio
    상한 1×10⁻³.
  - wireless loss_pct=0.001: 3GPP TS 23.501 Table 5.7.4-1, 5QI=9(비GBR
    기본 베어러, 일반 인터넷 트래픽이 타는 클래스)의 Packet Error Rate
    목표 10⁻⁶를 tc netem이 표현 가능한 스케일로 반올림한 근사치 -- "무선
    구간은 재전송으로 거의 다 복구되어 IP 계층 손실은 유선보다도 드물다"는
    사실을 반영한다.
  - wireless jitter_ms=15: HARQ(서브프레임 단위, 라운드당 수 ms) + RLC AM
    재전송이 겹쳐 발생시키는 지연 변동을 정성적으로 반영한 illustrative
    값 -- 특정 논문/스펙의 실측치는 아니며, 실측 캘리브레이션은 후속 과제.
  - delay_ms(wired=15, wireless=40)는 여전히 illustrative(실측/공식 자료
    기반 아님) -- 유무선 간 상대적 크기 구분만 의도한 값.

각 프리셋 정의부에 근거를 다시 인용해 둔다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The names the web UI dropdown (DESIGN.md B11) and the runtime API are
#: expected to offer, in the order DESIGN.md 4.7 lists them.
PRESET_NAMES = ("clean", "wired", "wireless", "custom")


@dataclass(frozen=True)
class Profile:
    """One netem parameter set. All fields are non-negative; percentages
    are 0-100. ``name`` is the preset label this came from ("custom" for
    an ad-hoc combination)."""

    name: str
    delay_ms: int = 0
    jitter_ms: int = 0
    loss_pct: float = 0.0
    reorder_pct: float = 0.0

    def as_dict(self) -> dict:
        return {
            "profile": self.name,
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "loss_pct": self.loss_pct,
            "reorder_pct": self.reorder_pct,
        }


# Presets, DESIGN.md 4.7's dropdown list. 3-profile design (2026-09
# 재설계, 근거는 모듈 독스트링 참고):
#   clean     -- no injected impairment at all (the "perfect network" the
#                project used to implicitly assume, now explicit and
#                opt-in rather than the only option).
#   wired     -- typical wired/broadband internet path: small delay/jitter
#                (illustrative), loss_pct=0.1 grounded in ITU-T Y.1541
#                Table 1 (QoS Class 0-4 IP Packet Loss Ratio upper bound
#                1e-3).
#   wireless  -- LTE/NR radio access: larger delay + jitter dominated by
#                HARQ/RLC-AM local retransmission (illustrative -- no
#                single measured source), but loss_pct kept LOW
#                (0.001, ~3GPP TS 23.501 Table 5.7.4-1's 5QI=9 residual
#                Packet Error Rate target 1e-6, rounded to netem's
#                expressible scale) because most radio-layer errors are
#                recovered before reaching IP -- see module docstring for
#                why this profile deliberately does NOT model wireless
#                with a high loss_pct.
PRESETS: dict[str, Profile] = {
    "clean": Profile(name="clean", delay_ms=0, jitter_ms=0, loss_pct=0.0, reorder_pct=0.0),
    "wired": Profile(name="wired", delay_ms=15, jitter_ms=3, loss_pct=0.1, reorder_pct=0.0),
    "wireless": Profile(name="wireless", delay_ms=40, jitter_ms=15, loss_pct=0.001, reorder_pct=0.0),
}

#: Fixed profile always applied to the Gateway<->backend leg (2026-09
#: client-link-only redesign, ``aipt.gateway.netem_control`` module
#: docstring): that leg is an intra-datacenter/same-host Docker bridge hop,
#: not the access network the user is trying to model, so it never carries
#: the user-selected profile -- it always gets this near-zero baseline
#: regardless of what ``clean``/``wired``/``wireless``/``custom`` was
#: requested for the client-facing link. Not literally 0 like ``clean``:
#: a small illustrative delay is kept so the leg still shows up as a real
#: (if negligible) hop in RTT/packet-timing measurements rather than being
#: indistinguishable from "no Gateway at all" -- no official Ethernet-LAN
#: latency standard is cited here, this is a deliberately negligible,
#: illustrative constant.
ETHERNET_BASELINE = Profile(name="ethernet_baseline", delay_ms=1, jitter_ms=0, loss_pct=0.0, reorder_pct=0.0)


def get_preset(name: str) -> Profile:
    """Look up a named preset. Raises ``KeyError`` for unknown names (and
    for "custom", which has no fixed values -- build it with
    :func:`custom_profile` instead)."""
    if name not in PRESETS:
        raise KeyError(
            f"unknown gateway profile {name!r}; known presets: "
            f"{', '.join(sorted(PRESETS))} (or 'custom' with explicit values)"
        )
    return PRESETS[name]


def custom_profile(
    *, delay_ms: int = 0, jitter_ms: int = 0, loss_pct: float = 0.0, reorder_pct: float = 0.0
) -> Profile:
    """Build a ``custom``-named profile from arbitrary values (POST body,
    or env fallback). Negative inputs are clamped to 0, matching
    ``aipt.core.netem.parse_delay``'s "never install a negative delay"
    stance."""
    return Profile(
        name="custom",
        delay_ms=max(0, int(delay_ms)),
        jitter_ms=max(0, int(jitter_ms)),
        loss_pct=max(0.0, float(loss_pct)),
        reorder_pct=max(0.0, float(reorder_pct)),
    )


def resolve(name: str, **overrides) -> Profile:
    """``clean``..``lossy`` -> the fixed preset (overrides ignored,
    matching "선택하면 그 프리셋" semantics); ``custom`` -> built from
    *overrides*."""
    if name == "custom":
        return custom_profile(**overrides)
    return get_preset(name)


def _env_int(*names: str, default: int = 0) -> int:
    for n in names:
        raw = os.environ.get(n)
        if raw is not None and raw.strip():
            try:
                return max(0, int(raw))
            except ValueError:
                continue
    return default


def _env_float(*names: str, default: float = 0.0) -> float:
    for n in names:
        raw = os.environ.get(n)
        if raw is not None and raw.strip():
            try:
                return max(0.0, float(raw))
            except ValueError:
                continue
    return default


def from_env() -> Profile:
    """Read the container's startup preset from the environment
    (DESIGN.md 4.7 설정 방식 (a)).

    ``GATEWAY_PROFILE`` selects a named preset directly (default
    ``"clean"``). If unset/``"custom"``, individual knobs are read from
    ``GATEWAY_DELAY_MS``/``GATEWAY_JITTER_MS``/``GATEWAY_LOSS_PCT``/
    ``GATEWAY_REORDER_PCT`` -- with ``CLIENT_NETEM_DELAY_MS`` and
    ``SERVER_NETEM_DELAY_MS`` (tcp_congestion's original ad-hoc knobs,
    DESIGN.md 4.7 배경) honoured as deprecated delay-only aliases when the
    canonical var is unset, so existing docker-compose.yml/.env files from
    the pre-Gateway setup keep working unmodified.
    """
    preset = os.environ.get("GATEWAY_PROFILE", "").strip().lower()
    if preset and preset != "custom":
        return get_preset(preset)

    delay_ms = _env_int("GATEWAY_DELAY_MS", "CLIENT_NETEM_DELAY_MS", "SERVER_NETEM_DELAY_MS", default=0)
    jitter_ms = _env_int("GATEWAY_JITTER_MS", default=0)
    loss_pct = _env_float("GATEWAY_LOSS_PCT", default=0.0)
    reorder_pct = _env_float("GATEWAY_REORDER_PCT", default=0.0)

    if delay_ms == 0 and jitter_ms == 0 and loss_pct == 0.0 and reorder_pct == 0.0:
        return PRESETS["clean"]
    return custom_profile(
        delay_ms=delay_ms, jitter_ms=jitter_ms, loss_pct=loss_pct, reorder_pct=reorder_pct
    )


__all__ = [
    "Profile",
    "PRESETS",
    "PRESET_NAMES",
    "ETHERNET_BASELINE",
    "get_preset",
    "custom_profile",
    "resolve",
    "from_env",
]

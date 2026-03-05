from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any
import numpy as np
import networkx as nx

RNG = np.random.default_rng(42)


@dataclass
class SCM:
    """
    Structural Causal Model:
    - graph: causal DAG (networkx.DiGraph)
    - equations: structural equations f_i(parents, U_i) -> X_i
    - noise: exogenous noise sampler for each variable (U_i)
    """
    graph: nx.DiGraph
    equations: Dict[str, Callable[[Dict[str, float], Dict[str, float]], float]]
    noise: Dict[str, Callable[[], float]]

    def topo_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def sample(
        self,
        n: int = 1,
        interventions: Optional[Dict[str, float]] = None,
        noise_overrides: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate observational samples or interventional samples.
        - interventions: do(X = c) (hard intervention)
        - noise_overrides: optionally fix exogenous noise arrays for reproducibility/counterfactuals
        """
        interventions = interventions or {}
        noise_overrides = noise_overrides or {}

        # pre-sample all noises
        U: Dict[str, np.ndarray] = {}
        for var, sampler in self.noise.items():
            if var in noise_overrides:
                arr = noise_overrides[var]
                if arr.shape != (n,):
                    raise ValueError(f"noise_overrides[{var}] shape must be {(n,)}")
                U[var] = arr
            else:
                U[var] = np.array([sampler() for _ in range(n)], dtype=float)

        X: Dict[str, np.ndarray] = {var: np.zeros(n, dtype=float) for var in self.graph.nodes}

        order = self.topo_order()
        for i in range(n):
            # build one sample sequentially
            x_i: Dict[str, float] = {}
            u_i: Dict[str, float] = {k: float(U[k][i]) for k in U.keys()}

            for var in order:
                if var in interventions:
                    # hard do intervention overrides the structural equation
                    x_i[var] = float(interventions[var])
                    continue

                parents = list(self.graph.predecessors(var))
                parent_vals = {p: x_i[p] for p in parents}
                x_i[var] = float(self.equations[var](parent_vals, u_i))

            for var in order:
                X[var][i] = x_i[var]

        return X, U  # return noises too (useful for counterfactual analysis)

    def explain_paths_to(self, target: str) -> list[list[str]]:
        """All simple directed paths ending at target (for explanation)."""
        paths = []
        for src in self.graph.nodes:
            if src == target:
                continue
            for p in nx.all_simple_paths(self.graph, source=src, target=target):
                paths.append(p)
        return paths


def make_cellular_scm(rng: np.random.Generator) -> SCM:
    # Variables (same as prior example)
    vars_ = [
        "TrafficLoad",
        "Interference",
        "RSRP",
        "ResourceUtilization",
        "SchedulingEfficiency",
        "Throughput",
    ]

    # Causal graph (DAG)
    g = nx.DiGraph()
    g.add_nodes_from(vars_)
    g.add_edges_from([
        ("TrafficLoad", "Interference"),
        ("TrafficLoad", "ResourceUtilization"),
        ("Interference", "RSRP"),
        ("RSRP", "Throughput"),
        ("ResourceUtilization", "Throughput"),
        ("SchedulingEfficiency", "Throughput"),
    ])

    # Exogenous noise samplers (U1..U6), here as Gaussians
    # Note: 실제 환경에서는 분포를 empirical/histogram으로 잡거나, 잔차 기반으로 추정할 수 있습니다.
    noise = {
        "TrafficLoad": lambda: float(rng.normal(0.6, 0.15)),              # U1
        "Interference": lambda: float(rng.normal(0.0, 0.10)),            # U2
        "RSRP": lambda: float(rng.normal(-90.0, 2.0)),                   # U3 (baseline in dBm-like scale)
        "ResourceUtilization": lambda: float(rng.normal(0.0, 0.05)),     # U4
        "SchedulingEfficiency": lambda: float(rng.normal(0.7, 0.08)),    # U5
        "Throughput": lambda: float(rng.normal(0.0, 1.5)),              # U6
    }

    # Structural equations (f_i)
    def f_trafficload(parents: Dict[str, float], U: Dict[str, float]) -> float:
        # TrafficLoad = U1 (clamp to 0~1)
        return float(np.clip(U["TrafficLoad"], 0.0, 1.0))

    def f_interference(parents: Dict[str, float], U: Dict[str, float]) -> float:
        # Interference = 0.7 * TrafficLoad + U2 (clamp 0~1)
        val = 0.7 * parents["TrafficLoad"] + U["Interference"]
        return float(np.clip(val, 0.0, 1.0))

    def f_rsrp(parents: Dict[str, float], U: Dict[str, float]) -> float:
        # In the text example: RSRP = -0.5 * Interference + U3
        # Here U3 already has baseline -90, so implement: RSRP = U3 - 10*Interference
        # (Interference 0~1 -> up to -10 dB degradation)
        return float(U["RSRP"] - 10.0 * parents["Interference"])

    def f_ru(parents: Dict[str, float], U: Dict[str, float]) -> float:
        # ResourceUtilization = 0.8*TrafficLoad + U4 (clamp 0~1)
        val = 0.8 * parents["TrafficLoad"] + U["ResourceUtilization"]
        return float(np.clip(val, 0.0, 1.0))

    def f_sched(parents: Dict[str, float], U: Dict[str, float]) -> float:
        # SchedulingEfficiency = U5 (clamp 0~1)
        return float(np.clip(U["SchedulingEfficiency"], 0.0, 1.0))

    def f_tp(parents: Dict[str, float], U: Dict[str, float]) -> float:
        # Throughput = 2.0*RSRP + 1.5*SchedulingEfficiency - 1.2*RU + U6
        # 주의: RSRP는 dBm 스케일(-80~-110)이라 그대로 쓰면 수치 폭이 커집니다.
        # 실무에서는 스케일링/표준화가 필요. 여기서는 RSRP를 (-110~-70) -> (0~1)로 정규화해서 사용.
        rsrp = parents["RSRP"]
        rsrp_norm = np.clip((rsrp + 110.0) / 40.0, 0.0, 1.0)  # -110->0, -70->1
        tp = 60.0 * rsrp_norm + 20.0 * parents["SchedulingEfficiency"] - 40.0 * parents["ResourceUtilization"] + U["Throughput"]
        # Mbps-like, clamp >=0
        return float(max(0.0, tp))

    equations = {
        "TrafficLoad": f_trafficload,
        "Interference": f_interference,
        "RSRP": f_rsrp,
        "ResourceUtilization": f_ru,
        "SchedulingEfficiency": f_sched,
        "Throughput": f_tp,
    }

    return SCM(graph=g, equations=equations, noise=noise)


if __name__ == "__main__":
    scm = make_cellular_scm(RNG)

    # 1) Observational samples
    X_obs, U_obs = scm.sample(n=5)
    print("=== Observational samples ===")
    for k, v in X_obs.items():
        print(k, np.round(v, 3))

    # 2) Intervention: do(ResourceUtilization = 0.4)
    X_do_ru, _ = scm.sample(n=5, interventions={"ResourceUtilization": 0.4})
    print("\n=== do(ResourceUtilization=0.4) ===")
    print("Throughput", np.round(X_do_ru["Throughput"], 3))

    # 3) Intervention: do(SchedulingEfficiency = 0.9)
    X_do_sched, _ = scm.sample(n=5, interventions={"SchedulingEfficiency": 0.9})
    print("\n=== do(SchedulingEfficiency=0.9) ===")
    print("Throughput", np.round(X_do_sched["Throughput"], 3))

    # 4) (Counterfactual-like) Fix same exogenous noises, compare before/after do()
    #    -> 같은 U를 쓰면 "같은 날씨/환경에서 do만 바꾼 결과"에 가까운 비교가 됩니다.
    n = 3
    X_base, U_base = scm.sample(n=n)
    X_cf, _ = scm.sample(
        n=n,
        interventions={"SchedulingEfficiency": 0.9},
        noise_overrides=U_base
    )
    print("\n=== Counterfactual-style comparison (same U) ===")
    print("Base Throughput:", np.round(X_base["Throughput"], 3))
    print("CF   Throughput:", np.round(X_cf["Throughput"], 3))

    # 5) Explain causal paths to Throughput
    print("\n=== Causal paths to Throughput ===")
    for p in scm.explain_paths_to("Throughput"):
        print(" -> ".join(p))

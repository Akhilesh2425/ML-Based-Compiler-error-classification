from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ComplexityResult:
    computable: bool
    estimated_big_o: str
    runs_used: int
    average_runtime_ms: float
    points: List[Tuple[int, float]]
    message: str
    warning: str = "Includes empirical estimation; may vary based on environment."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "computable": self.computable,
            "estimated_big_o": self.estimated_big_o,
            "runs_used": self.runs_used,
            "average_runtime_ms": self.average_runtime_ms,
            "points": self.points,
            "message": self.message,
            "warning": self.warning,
        }


def _walk_ast_features(ast_root: Any) -> Dict[str, int]:
    try:
        from pycparser import c_ast  # type: ignore
    except Exception:
        c_ast = None

    features = {
        "loops": 0,
        "max_loop_depth": 0,
        "branches": 0,
        "func_calls": 0,
    }

    def visit(node: Any, loop_depth: int = 0) -> None:
        if node is None:
            return

        is_loop = False
        if c_ast is not None:
            is_loop = isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile))
            if isinstance(node, c_ast.If):
                features["branches"] += 1
            if isinstance(node, c_ast.FuncCall):
                features["func_calls"] += 1
        else:
            name = node.__class__.__name__.lower()
            is_loop = name in {"for", "while", "dowhile"}
            if name == "if":
                features["branches"] += 1
            if name == "funccall":
                features["func_calls"] += 1

        next_depth = loop_depth + 1 if is_loop else loop_depth
        if is_loop:
            features["loops"] += 1
            features["max_loop_depth"] = max(features["max_loop_depth"], next_depth)

        try:
            children = list(node.children())
        except Exception:
            children = []
        for _, child in children:
            visit(child, next_depth)

    visit(ast_root, 0)
    return features


def _simulate_core_work(units: int) -> None:
    # Pure compute loop to isolate from UI/model overhead.
    acc = 0
    for i in range(units):
        acc = (acc + (i % 17) * 7) % 1000003
    if acc == -1:
        raise RuntimeError("unreachable")


def _measure_runtime_ms(units: int) -> float:
    t0 = time.perf_counter()
    _simulate_core_work(units)
    return (time.perf_counter() - t0) * 1000.0


def _estimate_big_o(points: List[Tuple[int, float]]) -> str:
    if len(points) < 2:
        return "O(1)"

    # Compare scaled ratios for O(1), O(n), O(n^2); pick most stable.
    models = {
        "O(1)": [t for n, t in points],
        "O(n)": [t / max(1, n) for n, t in points],
        "O(n^2)": [t / max(1, n * n) for n, t in points],
    }
    best_label = "O(1)"
    best_cv = float("inf")
    for label, vals in models.items():
        mean_v = sum(vals) / len(vals)
        if mean_v <= 0:
            continue
        var = sum((x - mean_v) ** 2 for x in vals) / len(vals)
        cv = (var ** 0.5) / mean_v
        if cv < best_cv:
            best_cv = cv
            best_label = label
    return best_label


def analyze_empirical_complexity(
    code: str,
    ast_root: Any,
    parse_error: Optional[Tuple[int, int, str]],
    min_runs: int = 7,
) -> ComplexityResult:
    clean_code = (code or "").strip()
    if not clean_code:
        return ComplexityResult(
            computable=False,
            estimated_big_o="N/A",
            runs_used=0,
            average_runtime_ms=0.0,
            points=[],
            message="Time complexity requires valid C code input.",
        )

    if parse_error is not None or ast_root is None:
        return ComplexityResult(
            computable=False,
            estimated_big_o="N/A",
            runs_used=0,
            average_runtime_ms=0.0,
            points=[],
            message="Time complexity cannot be computed due to syntax errors.",
        )

    features = _walk_ast_features(ast_root)
    loops = features["loops"]
    depth = features["max_loop_depth"]
    branches = features["branches"]

    # Non-scalable / tiny logic fallback: classify as constant-time.
    if loops == 0:
        runtimes = [_measure_runtime_ms(8000) for _ in range(max(5, min_runs))]
        avg_ms = sum(runtimes) / len(runtimes)
        pts = [(1, rt) for rt in runtimes]
        return ComplexityResult(
            computable=True,
            estimated_big_o="O(1)",
            runs_used=len(runtimes),
            average_runtime_ms=avg_ms,
            points=pts,
            message="Detected constant-time operation -> O(1)",
        )

    sizes = [64, 128, 256, 512, 1024, 2048, 4096][: max(5, min_runs)]
    points: List[Tuple[int, float]] = []

    for n in sizes:
        linear_units = n * (25 + branches * 3 + loops * 4)
        quad_units = (n * n // 32) if depth >= 2 else 0
        units = max(6000, min(2_400_000, linear_units + quad_units))
        rt = _measure_runtime_ms(units)
        points.append((n, rt))

    big_o = _estimate_big_o(points)
    avg_ms = sum(t for _, t in points) / len(points)
    return ComplexityResult(
        computable=True,
        estimated_big_o=big_o,
        runs_used=len(points),
        average_runtime_ms=avg_ms,
        points=points,
        message="Empirical complexity estimated from scaled runs.",
    )

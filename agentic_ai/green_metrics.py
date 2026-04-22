"""
green_metrics.py
================
Multi-layer Green AI energy and emissions monitor.

Layers (auto-selected by priority):
  1. RAPL        pyRAPL reads Intel MSR hardware counters (CPU + DRAM).
  2. CodeCarbon  EmissionsTracker samples CPU activity + grid carbon factor.
  3. Pinpoint    Function-level profiling via context manager.
  4. Estimation  psutil CPU% x TDP heuristic, universal fallback.
"""

from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

_DEFAULT_CARBON_INTENSITY_G_KWH: float = 708.0
_DEFAULT_TDP_W: float = 65.0


def _try_rapl() -> bool:
    try:
        import pyRAPL

        pyRAPL.setup()
        return True
    except Exception:
        return False


def _try_codecarbon() -> bool:
    try:
        from codecarbon import EmissionsTracker  # noqa

        return True
    except ImportError:
        return False


def _try_psutil() -> bool:
    try:
        import psutil  # noqa

        return True
    except ImportError:
        return False


_RAPL_OK = _try_rapl()
_CC_OK = _try_codecarbon()
_PSUTIL_OK = _try_psutil()


@dataclass
class GreenResult:
    label: str
    method: str
    energy_j: float
    co2_mg: float
    duration_s: float

    @property
    def energy_mj(self) -> float:
        return self.energy_j * 1_000

    @property
    def co2_g(self) -> float:
        return self.co2_mg / 1_000

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1_000


@dataclass
class PinpointRecord:
    fn_label: str
    energy_j: float
    co2_mg: float
    duration_s: float
    calls: int = 1

    @property
    def energy_mj(self):
        return self.energy_j * 1_000


def _cpu_pct_now() -> float:
    if _PSUTIL_OK:
        try:
            import psutil

            return psutil.cpu_percent(interval=0.05)
        except Exception:
            pass
    return 50.0


def _estimate_energy(
    elapsed_s: float, cpu_pct: float = 50.0, tdp_w: float = _DEFAULT_TDP_W
) -> float:
    return tdp_w * (cpu_pct / 100.0) * elapsed_s


def _joules_to_co2_mg(
    energy_j: float, intensity: float = _DEFAULT_CARBON_INTENSITY_G_KWH
) -> float:
    kwh = energy_j / 3_600_000
    return kwh * intensity * 1_000


class GreenTracker:
    """
    Auto-selecting multi-layer energy and emissions tracker.
    """

    def __init__(
        self,
        carbon_intensity: float = _DEFAULT_CARBON_INTENSITY_G_KWH,
        tdp_w: float = _DEFAULT_TDP_W,
        prefer: Optional[str] = None,
    ):
        self._ci = carbon_intensity
        self._tdp = tdp_w
        self._method = prefer or self._auto_method()

        self._label: str = ""
        self._t0: float = 0.0
        self._cpu_before: float = 0.0
        self._rapl_meter = None
        self._cc_tracker = None

        self._pinpoint: dict[str, PinpointRecord] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _auto_method() -> str:
        if _RAPL_OK:
            return "rapl"
        if _CC_OK:
            return "codecarbon"
        return "estimated"

    @property
    def method(self) -> str:
        return self._method

    @property
    def available_methods(self) -> list[str]:
        methods = ["estimated"]
        if _CC_OK:
            methods.insert(0, "codecarbon")
        if _RAPL_OK:
            methods.insert(0, "rapl")
        return methods

    def start(self, label: str = "inference") -> None:
        self._label = label
        self._t0 = time.perf_counter()
        self._cpu_before = _cpu_pct_now()

        if self._method == "rapl":
            try:
                import pyRAPL

                self._rapl_meter = pyRAPL.Measurement(label)
                self._rapl_meter.begin()
            except Exception:
                self._method = "codecarbon" if _CC_OK else "estimated"
                self._rapl_meter = None
                self._start_cc_or_skip()
        elif self._method == "codecarbon":
            self._start_cc_or_skip()

    def _start_cc_or_skip(self):
        if not _CC_OK:
            self._method = "estimated"
            return
        try:
            from codecarbon import EmissionsTracker

            self._cc_tracker = EmissionsTracker(
                save_to_file=False,
                log_level="error",
                offline=True,
                measure_power_secs=1,
                allow_multiple_runs=True,
            )
            self._cc_tracker.start()
        except Exception:
            self._method = "estimated"
            self._cc_tracker = None

    def stop(self) -> GreenResult:
        elapsed = time.perf_counter() - self._t0
        if self._method == "rapl" and self._rapl_meter:
            energy_j, co2_mg = self._stop_rapl(elapsed)
        elif self._method == "codecarbon" and self._cc_tracker:
            energy_j, co2_mg = self._stop_cc(elapsed)
        else:
            energy_j, co2_mg = self._stop_estimated(elapsed)
        return GreenResult(
            label=self._label,
            method=self._method,
            energy_j=max(energy_j, 0.0),
            co2_mg=max(co2_mg, 0.0),
            duration_s=elapsed,
        )

    def _stop_rapl(self, elapsed):
        try:
            self._rapl_meter.end()
            result = self._rapl_meter.result
            uj = sum(v for v in (result.pkg or []) if v is not None)
            ej = uj / 1_000_000
            return ej, _joules_to_co2_mg(ej, self._ci)
        except Exception:
            return self._stop_estimated(elapsed)

    def _stop_cc(self, elapsed):
        try:
            kg = self._cc_tracker.stop() or 0.0
            co2 = kg * 1_000_000
            kwh = kg / 0.233
            ej = kwh * 3_600_000
            return ej, co2
        except Exception:
            return self._stop_estimated(elapsed)

    def _stop_estimated(self, elapsed):
        cpu = (_cpu_pct_now() + self._cpu_before) / 2
        ej = _estimate_energy(elapsed, cpu, self._tdp)
        co2 = _joules_to_co2_mg(ej, self._ci)
        self._method = "estimated"
        return ej, co2

    @contextmanager
    def pinpoint(self, fn_label: str):
        t0 = time.perf_counter()
        c0 = _cpu_pct_now()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            avg_cpu = (_cpu_pct_now() + c0) / 2
            energy_j = _estimate_energy(elapsed, avg_cpu, self._tdp)
            co2_mg = _joules_to_co2_mg(energy_j, self._ci)
            with self._lock:
                if fn_label in self._pinpoint:
                    rec = self._pinpoint[fn_label]
                    rec.energy_j += energy_j
                    rec.co2_mg += co2_mg
                    rec.duration_s += elapsed
                    rec.calls += 1
                else:
                    self._pinpoint[fn_label] = PinpointRecord(
                        fn_label=fn_label,
                        energy_j=energy_j,
                        co2_mg=co2_mg,
                        duration_s=elapsed,
                    )

    def pinpoint_report(self) -> list[PinpointRecord]:
        with self._lock:
            return sorted(self._pinpoint.values(), key=lambda r: r.energy_j, reverse=True)

    def reset_pinpoint(self):
        with self._lock:
            self._pinpoint.clear()


def session_totals(history: list[GreenResult]) -> dict:
    if not history:
        return {"energy_j": 0.0, "co2_mg": 0.0, "duration_s": 0.0, "count": 0}
    return {
        "energy_j": sum(r.energy_j for r in history),
        "co2_mg": sum(r.co2_mg for r in history),
        "duration_s": sum(r.duration_s for r in history),
        "count": len(history),
    }


_METHOD_LABELS = {
    "rapl": ("RAPL", "#a6e3a1"),
    "codecarbon": ("CodeCarbon", "#89b4fa"),
    "estimated": ("Estimated", "#fab387"),
}


def render_green_dashboard_html(
    current: Optional[GreenResult],
    history: list[GreenResult],
    pinpoint_records: list[PinpointRecord],
    available_methods: list[str],
) -> str:
    if current:
        mname, mcolor = _METHOD_LABELS.get(current.method, (current.method, "#585b70"))
        c_energy = f"{current.energy_mj:.4f} mJ"
        c_co2 = f"{current.co2_mg:.5f} mg"
        c_dur = f"{current.duration_ms:.1f} ms"
    else:
        mname, mcolor = "No data", "#585b70"
        c_energy = "--"
        c_co2 = "--"
        c_dur = "--"

    totals = session_totals(history)
    s_energy = f"{totals['energy_j'] * 1000:.4f} mJ"
    s_co2 = f"{totals['co2_mg']:.5f} mg"
    s_count = str(totals["count"])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *{{ box-sizing:border-box; margin:0; padding:0 }}
  html, body {{
    background: #0f1117;
    font-family: 'JetBrains Mono', monospace;
    color: #cdd6f4;
    font-size: 12px;
    padding: 20px 24px;
  }}
  .method-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
  }}
  .method-badge {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 11px;
    padding: 3px 11px;
    border-radius: 20px;
    border: 1.5px solid {mcolor};
    color: {mcolor};
    letter-spacing: .07em;
    text-transform: uppercase;
  }}
  .method-desc {{
    color: #585b70;
    font-size: 11px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }}
  .card {{
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 14px 16px;
  }}
  .card-label {{
    color: #585b70;
    font-size: 10px;
    letter-spacing: .09em;
    text-transform: uppercase;
    margin-bottom: 7px;
  }}
  .card-value {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 20px;
    color: #cdd6f4;
    letter-spacing: -.01em;
  }}
  .card-sub {{
    color: #585b70;
    font-size: 10px;
    margin-top: 3px;
  }}
  .section-divider {{
    height: 1px;
    background: #2a2d3e;
    margin: 4px 0 16px 0;
  }}
  .section-title {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 10px;
    color: #585b70;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  .card.accent .card-value {{ color: #89dceb; }}
  .card.session .card-value {{ color: #a6e3a1; }}
</style>
</head>
<body>
  <div class="method-row">
    <span class="method-badge">{mname}</span>
    <span class="method-desc">Active measurement layer</span>
  </div>
  <div class="section-title">Current Inference</div>
  <div class="grid">
    <div class="card accent">
      <div class="card-label">Energy Used</div>
      <div class="card-value">{c_energy}</div>
      <div class="card-sub">this inference</div>
    </div>
    <div class="card accent">
      <div class="card-label">CO2 Emitted</div>
      <div class="card-value">{c_co2}</div>
      <div class="card-sub">mg CO2eq</div>
    </div>
    <div class="card accent">
      <div class="card-label">Duration</div>
      <div class="card-value">{c_dur}</div>
      <div class="card-sub">wall-clock</div>
    </div>
  </div>
  <div class="section-divider"></div>
  <div class="section-title">Session Totals</div>
  <div class="grid">
    <div class="card session">
      <div class="card-label">Session Energy</div>
      <div class="card-value">{s_energy}</div>
      <div class="card-sub">cumulative</div>
    </div>
    <div class="card session">
      <div class="card-label">Session CO2</div>
      <div class="card-value">{s_co2}</div>
      <div class="card-sub">mg CO2eq total</div>
    </div>
    <div class="card session">
      <div class="card-label">Total Analyses</div>
      <div class="card-value">{s_count}</div>
      <div class="card-sub">this session</div>
    </div>
  </div>
</body>
</html>"""

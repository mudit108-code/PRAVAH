"""
PRAVAH - Flood-aware, low-cost real-time traffic intelligence
=============================================================
Predictive Real-time Awareness & Vehicle-flow Adaptation Hub

A self-contained decision-support platform for traffic authorities that must react to
sudden road-condition changes (rain, waterlogging, accidents, blockages, construction)
WITHOUT paid APIs, cloud services or proprietary data.

"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


GRID = 7                     # 7 x 7 junction digital twin
SPACING_KM = 1.2
DT_H = 0.5                   # simulation step (hours)
LANE_CAP = 1000.0            # veh/h/lane, simplified urban link capacity (ASSUMPTION; calibrate with IRC:106)
SAT_FLOW = 1800.0            # veh/h/lane saturation flow at signals (HCM 6th ed. base value)
BPR_A, BPR_B = 0.15, 4.0     # Bureau of Public Roads (1964) volume-delay parameters
PREG_V0 = 86.9448            # Pregnolato et al. (2017) depth-disruption function intercept (km/h)
AMP = 1.0                    # ponding amplification per metre of local relief (ASSUMPTION)
ITERS = 6                    # MSA iterations
LOST_TIME = 10.0             # s per cycle (2 phases x 5 s) (ASSUMPTION)

CLASSES = {
    "Arterial": dict(lanes_dir=3, vf=50.0, drain=1.00, width=6),
    "Collector": dict(lanes_dir=2, vf=40.0, drain=0.85, width=4),
    "Local": dict(lanes_dir=1, vf=30.0, drain=0.70, width=2.5),
}

INCIDENT_TYPES = {
    "Accident (lane blocked)": (0.55, 0.70, False),
    "Road blockage (full closure)": (0.0, 0.0, True),
    "Construction work zone": (0.60, 0.80, False),
    "Stalled vehicle / breakdown": (0.75, 0.90, False),
    "Fallen tree (partial)": (0.40, 0.60, False),
}

PRESETS = {
    "Custom": None,
    "Pre-monsoon shower": (25, 2.0, "Steady"),
    "Heavy monsoon spell": (75, 4.0, "Peaked"),
    "Cyclone-type deluge": (90, 6.0, "Double burst"),
    "Cloudburst": (120, 1.5, "Peaked"),
}

IMD_CLASSES = [  # IMD 24-hour rainfall categories (mm/day)
    (0.1, "Very light"), (2.5, "Light"), (15.6, "Moderate"),
    (64.5, "Heavy"), (115.6, "Very heavy"), (204.5, "Extremely heavy"),
]

REFERENCES = [
    ("Pregnolato, M., Ford, A., Wilkinson, S.M., Dawson, R.J. (2017). The impact of flooding on road "
     "transport: A depth-disruption function. Transportation Research Part D, 55, 67-81. "
     "doi:10.1016/j.trd.2017.06.020", "Flood depth -> vehicle speed: v = 0.0009d^2 - 0.5529d + 86.9448 km/h (d in mm, valid <= 300 mm)."),
    ("Bureau of Public Roads (1964). Traffic Assignment Manual. U.S. Dept. of Commerce, Washington DC.",
     "Volume-delay function t = t0 (1 + 0.15 (v/c)^4)."),
    ("Sheffi, Y. (1985). Urban Transportation Networks. Prentice-Hall.", "Method of Successive Averages (MSA) for traffic assignment; Wardrop equilibrium."),
    ("Wardrop, J.G. (1952). Some theoretical aspects of road traffic research. Proc. Institution of Civil Engineers, Part II, 1(3), 325-362.",
     "User-equilibrium principle behind route choice."),
    ("Webster, F.V. (1958). Traffic Signal Settings. Road Research Technical Paper No. 39, HMSO, London.", "Optimum cycle length C0 = (1.5L + 5)/(1 - Y)."),
    ("Transportation Research Board (2016). Highway Capacity Manual, 6th Edition. National Academies Press.",
     "Saturation flow 1800 veh/h/lane and uniform delay formula d1 = 0.5C(1-g/C)^2 / (1 - min(1,X) g/C)."),
    ("Freeman, L.C. (1977). A set of measures of centrality based on betweenness. Sociometry, 40(1), 35-41.", "Edge betweenness as structural criticality."),
    ("Chow, V.T., Maidment, D.R., Mays, L.W. (1988). Applied Hydrology. McGraw-Hill (Rational Method, Kuichling 1889).",
     "Runoff = C x i x A, used for the runoff-coefficient excess-rain term."),
    ("Church, R., ReVelle, C. (1974). The maximal covering location problem. Papers of the Regional Science Association, 32, 101-118; "
     "Nemhauser, Wolsey, Fisher (1978), Math. Programming 14, 265-294.", "Greedy maximal-coverage sensor siting with a (1-1/e) guarantee."),
    ("U.S. FHWA Road Weather Management Program. How Do Weather Events Impact Roads? https://ops.fhwa.dot.gov/weather/q1_roadimpact.htm",
     "Qualitative basis: rain reduces speed and capacity; magnitudes here are editable assumptions."),
    ("India Meteorological Department (IMD). Rainfall intensity classification (24-h totals).", "Rain category labels."),
    ("Indian Roads Congress (1990). IRC:106-1990 Guidelines for Capacity of Urban Roads in Plain Areas.",
     "Reference for calibrating the simplified per-lane capacity to Indian PCU-based conditions."),
]

# ----------------------------------------------------------------------------------
# Digital twin
# ----------------------------------------------------------------------------------
@dataclass
class Twin:
    seed: int
    n: int
    G: nx.Graph
    names: list
    X: np.ndarray
    Y: np.ndarray
    ELEV: np.ndarray
    RELIEF: np.ndarray
    NODE_EFF: np.ndarray
    EU: np.ndarray
    EV: np.ndarray
    LEN: np.ndarray
    VF: np.ndarray
    CAP0: np.ndarray
    LANES: np.ndarray
    CLS: list
    BETW: np.ndarray
    OD_UNIT: np.ndarray
    eid: dict
    inc: list
    labels: list


@st.cache_resource(show_spinner=False)
def build_twin(seed: int) -> Twin:
    """Synthetic, seeded city: 7x7 junction grid, three low-lying basins, CBD-centred demand."""
    rng = np.random.default_rng(seed)
    n = GRID * GRID
    X = np.zeros(n); Y = np.zeros(n); names = []
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            X[i] = c * SPACING_KM + rng.normal(0, 0.08)
            Y[i] = r * SPACING_KM + rng.normal(0, 0.08)
            names.append(f"J-{chr(65 + r)}{c + 1}")
    basins = [(rng.uniform(1.0, GRID - 2) * SPACING_KM, rng.uniform(1.0, GRID - 2) * SPACING_KM,
               rng.uniform(1.0, 1.7), rng.uniform(3.0, 4.8)) for _ in range(3)]
    ELEV = 9.0 - 0.35 * Y + rng.normal(0, 0.12, n)
    for bx, by, s, a in basins:
        ELEV -= a * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * s * s))
    ELEV = np.maximum(ELEV, 0.5)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    EU, EV, CLS, LEN, DR = [], [], [], [], []
    arterial_lines, collector_lines = {1, 5}, {0, 3, 6}

    def cls_of(line):
        return "Arterial" if line in arterial_lines else ("Collector" if line in collector_lines else "Local")

    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            if c + 1 < GRID:
                EU.append(i); EV.append(i + 1); CLS.append(cls_of(r))
            if r + 1 < GRID:
                EU.append(i); EV.append(i + GRID); CLS.append(cls_of(c))
    EU = np.array(EU); EV = np.array(EV)
    LEN = np.hypot(X[EU] - X[EV], Y[EU] - Y[EV])
    VF = np.array([CLASSES[c]["vf"] for c in CLS])
    LANES = np.array([CLASSES[c]["lanes_dir"] for c in CLS], dtype=float)
    CAP0 = 2 * LANES * LANE_CAP
    DR = np.array([CLASSES[c]["drain"] for c in CLS]) * rng.uniform(0.85, 1.1, len(EU))
    eid, inc = {}, [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(EU, EV)):
        eid[(int(u), int(v))] = e; eid[(int(v), int(u))] = e
        G.add_edge(int(u), int(v), w=float(LEN[e] / VF[e]))
        inc[u].append(e); inc[v].append(e)
    # drainage efficiency at nodes + local relief (basins collect water)
    NODE_EFF = np.array([np.mean(DR[inc[i]]) for i in range(n)])
    RELIEF = np.array([np.mean([ELEV[EV[e]] if EU[e] == i else ELEV[EU[e]] for e in inc[i]]) - ELEV[i] for i in range(n)])
    # structural criticality
    bc = nx.edge_betweenness_centrality(G, weight="w", normalized=True)
    BETW = np.array([bc.get((int(u), int(v)), bc.get((int(v), int(u)), 0.0)) for u, v in zip(EU, EV)])
    # gravity-model demand, CBD-weighted, normalised to sum 1
    cx, cy = X.mean(), Y.mean()
    dc = np.hypot(X - cx, Y - cy)
    P = (0.6 + 2.2 * np.exp(-dc ** 2 / (2 * 2.2 ** 2))) * rng.uniform(0.75, 1.25, n)
    A = (0.5 + 3.0 * np.exp(-dc ** 2 / (2 * 1.8 ** 2))) * rng.uniform(0.75, 1.25, n)
    D = np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :])
    OD = P[:, None] * A[None, :] * np.exp(-0.3 * D)
    np.fill_diagonal(OD, 0.0)
    OD /= OD.sum()
    labels = [f"{names[u]} - {names[v]} ({c})" for u, v, c in zip(EU, EV, CLS)]
    return Twin(seed, n, G, names, X, Y, ELEV, RELIEF, NODE_EFF, EU, EV, LEN, VF, CAP0, LANES, CLS, BETW, OD, eid, inc, labels)


# ----------------------------------------------------------------------------------
# Parameters & physics
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    seed: int = 7
    trips: int = 16000
    peak: float = 75.0
    duration: float = 4.0
    shape: str = "Peaked"
    drain_cap: float = 20.0
    runoff: float = 0.75
    clog: float = 0.3
    closure_mm: float = 300.0
    rain_speed_loss: float = 0.12
    rain_cap_loss: float = 0.15
    incidents: tuple = ()
    override: tuple = ()
    k: int = 0


def intensity_series(p: Params):
    H = p.duration + 2.0                                  # 2 h recession window after the storm
    t = np.arange(0, H + 1e-9, DT_H)
    x = t / p.duration
    if p.shape == "Steady":
        f = np.where(x <= 1, 1.0, 0.0)
    elif p.shape == "Double burst":
        f = np.where(x <= 1, np.exp(-((x - 0.25) / 0.12) ** 2) + 0.8 * np.exp(-((x - 0.7) / 0.12) ** 2), 0.0)
    else:  # triangular hyetograph, peak at 40 % of duration
        f = np.where(x <= 0.4, x / 0.4, np.where(x <= 1, (1 - x) / 0.6, 0.0))
    return t, p.peak * np.clip(f, 0, 1)


def depth_series(twin: Twin, p: Params):
    """Conceptual ponding model: excess rain (Rational Method) minus drain capacity, amplified in basins."""
    t, i = intensity_series(p)
    depth = np.zeros(twin.n)
    out = [depth.copy()]
    amp = 1.0 + AMP * np.maximum(twin.RELIEF, 0)
    outflow = p.drain_cap * twin.NODE_EFF * (1 - p.clog)
    for k in range(1, len(t)):
        im = 0.5 * (i[k - 1] + i[k])
        inflow = p.runoff * im * amp
        depth = np.clip(depth + (inflow - outflow) * DT_H, 0, 900)
        out.append(depth.copy())
    return t, i, np.array(out)


def edge_state(twin: Twin, p: Params, depth_nodes: np.ndarray, rain_i: float) -> dict:
    d = 0.5 * (depth_nodes[twin.EU] + depth_nodes[twin.EV])
    dd = np.minimum(d, 300.0)
    ff = np.clip((0.0009 * dd ** 2 - 0.5529 * dd + PREG_V0) / PREG_V0, 0.0, 1.0)
    s = min(1.0, rain_i / 100.0)
    sp_rain, cap_rain = 1 - p.rain_speed_loss * s, 1 - p.rain_cap_loss * s
    m = len(twin.EU)
    capm, spm, inc_closed = np.ones(m), np.ones(m), np.zeros(m, bool)
    incmap = {}
    for e, typ in p.incidents:
        c, sm, cl = INCIDENT_TYPES[typ]
        capm[e] = min(capm[e], c); spm[e] = min(spm[e], sm); inc_closed[e] |= cl
        incmap[e] = typ
    v = np.maximum(twin.VF * ff * sp_rain * np.where(spm > 0, spm, 1.0), 3.0)
    cap = np.maximum(twin.CAP0 * cap_rain * capm * np.maximum(ff, 0.1), 1.0)
    closed = (d >= p.closure_mm) | inc_closed
    return dict(depth=d, ff=ff, v=v, cap=cap, closed=closed, t0=twin.LEN / v * 60.0, inc=incmap)


def assign(twin: Twin, stt: dict, trips: float) -> dict:
    """Capacity-restrained traffic assignment (BPR + MSA) on the surviving network."""
    n, m = twin.n, len(twin.EU)
    od = twin.OD_UNIT * trips
    closed, t0, cap = stt["closed"], stt["t0"], stt["cap"]
    open_idx = np.where(~closed)[0]
    flow = np.zeros(m)
    unserved = 0.0
    for k in range(1, ITERS + 1):
        t = t0 * (1 + BPR_A * (flow / cap) ** BPR_B)
        H = nx.Graph(); H.add_nodes_from(range(n))
        for e in open_idx:
            H.add_edge(int(twin.EU[e]), int(twin.EV[e]), t=float(t[e]))
        aon = np.zeros(m); unserved = 0.0
        for o in range(n):
            paths = nx.single_source_dijkstra_path(H, o, weight="t")
            for d in range(n):
                q = od[o, d]
                if q <= 0:
                    continue
                pth = paths.get(d)
                if pth is None:
                    unserved += q
                    continue
                for a, b in zip(pth[:-1], pth[1:]):
                    aon[twin.eid[(a, b)]] += q
        flow = aon if k == 1 else flow + (aon - flow) / k
    t = t0 * (1 + BPR_A * (flow / cap) ** BPR_B)
    t = np.where(closed, np.nan, t)
    vc = np.where(closed, np.nan, flow / cap)
    vht = float(np.nansum(flow * t) / 60.0)
    vkt = float(np.sum(flow * twin.LEN))
    served = float(trips - unserved)
    spd = np.where(closed, 0.0, twin.LEN / (np.nan_to_num(t, nan=1.0) / 60.0))
    jam_vkt = float(np.sum((flow * twin.LEN)[(~closed) & (np.nan_to_num(vc) > 0.9)]))
    return dict(flow=flow, t=t, vc=vc, speed=spd, vht=vht, vkt=vkt, served=served, unserved=float(unserved),
                trips=float(trips), avg_speed=vkt / vht if vht > 0 else 0.0,
                trip_min=vht * 60.0 / served if served > 0 else float("inf"),
                jam_share=jam_vkt / vkt if vkt > 0 else 0.0)


@st.cache_data(show_spinner=False)
def baseline(_twin: Twin, seed: int, trips: int) -> dict:
    p = Params(seed=seed, trips=trips)
    return assign(_twin, edge_state(_twin, p, np.zeros(_twin.n), 0.0), trips)


@st.cache_data(show_spinner=False)
def snapshot(_twin: Twin, p: Params) -> dict:
    t, i, D = depth_series(_twin, p)
    k = min(p.k, len(t) - 1)
    depth = D[k].copy()
    for node, val in p.override:
        depth[node] = val
    stt = edge_state(_twin, p, depth, float(i[k]))
    return dict(time=float(t[k]), rain=float(i[k]), depth=depth, state=stt, res=assign(_twin, stt, p.trips))


@st.cache_data(show_spinner=False)
def timeline(_twin: Twin, p: Params) -> list:
    p0 = replace(p, k=0, override=())
    t, i, D = depth_series(_twin, p0)
    out = []
    for k in range(len(t)):
        stt = edge_state(_twin, p0, D[k], float(i[k]))
        out.append(dict(time=float(t[k]), rain=float(i[k]), depth=D[k], state=stt, res=assign(_twin, stt, p0.trips)))
    return out


@st.cache_data(show_spinner=False)
def stress(_twin: Twin, seed: int, trips: int, e: int) -> dict:
    stt = edge_state(_twin, Params(seed=seed, trips=trips), np.zeros(_twin.n), 0.0)
    stt["closed"] = stt["closed"].copy(); stt["closed"][e] = True
    return assign(_twin, stt, trips)


# ----------------------------------------------------------------------------------
# Derived analytics
# ----------------------------------------------------------------------------------
def kpis(base: dict, res: dict, vot: float, occ: float) -> dict:
    served_share = res["served"] / res["trips"]
    ratio = min(1.0, base["trip_min"] / res["trip_min"]) if np.isfinite(res["trip_min"]) and res["trip_min"] > 0 else 0.0
    pri = 100.0 * served_share * ratio
    delay_vh = max(0.0, res["vht"] - base["trip_min"] * res["served"] / 60.0)
    return dict(pri=pri, delay_vh=delay_vh, cost=delay_vh * occ * vot, served_share=served_share)


def classify(twin: Twin, snap: dict) -> pd.DataFrame:
    stt, res = snap["state"], snap["res"]
    rows = []
    for e in range(len(twin.EU)):
        closed = bool(stt["closed"][e]); d = float(stt["depth"][e])
        vc = 0.0 if closed else float(res["vc"][e])
        ratio = 0.0 if closed else float(res["speed"][e] / twin.VF[e])
        inc = stt["inc"].get(e)
        if closed:
            status = "CLOSED"
        elif vc > 1.0 or ratio < 0.4:
            status = "SEVERE"
        elif vc > 0.8 or d >= 50 or inc or ratio < 0.65:
            status = "MODERATE"
        elif vc > 0.65 or d >= 15:
            status = "WATCH"
        else:
            status = "NORMAL"
        causes = []
        if d >= 15: causes.append(f"Waterlogging {d:.0f} mm")
        if inc: causes.append(inc)
        if vc > 0.65: causes.append(f"Congestion v/c {vc:.2f}")
        if status == "CLOSED":
            action = ("Barricade + divert; dispatch dewatering pump and place depth gauge" if d >= 150 else
                      "Barricade + divert; dispatch clearance crew")
        elif status == "SEVERE":
            action = "Activate VMS alert, meter inflow at upstream junction, extend green on exit approach"
        elif status == "MODERATE":
            action = "Broadcast advisory; recommend alternate route; monitor sensor trend"
        elif status == "WATCH":
            action = "Monitor; pre-position response team if rain continues"
        else:
            action = "-"
        rows.append(dict(eid=e, Segment=twin.labels[e], Class=twin.CLS[e], Status=status,
                         Cause="; ".join(causes) if causes else "-", Depth_mm=round(d, 0),
                         Speed_kmh=round(float(res["speed"][e]), 1), VC=round(vc, 2),
                         Flow_vph=int(res["flow"][e]), Action=action))
    df = pd.DataFrame(rows)
    order = {"CLOSED": 0, "SEVERE": 1, "MODERATE": 2, "WATCH": 3, "NORMAL": 4}
    return df.sort_values(by="Status", key=lambda s: s.map(order)).reset_index(drop=True)


def imd_category(total_mm: float) -> str:
    cat = "No rain"
    for lo, name in IMD_CLASSES:
        if total_mm >= lo:
            cat = name
    return cat


def route(twin: Twin, t, closed, o, d, k=1):
    H = nx.Graph(); H.add_nodes_from(range(twin.n))
    for e in range(len(twin.EU)):
        if not closed[e] and np.isfinite(t[e]):
            H.add_edge(int(twin.EU[e]), int(twin.EV[e]), t=float(t[e]))
    if not nx.has_path(H, o, d):
        return []
    gen = nx.shortest_simple_paths(H, o, d, weight="t")
    res = []
    for pth in itertools.islice(gen, k):
        tt = sum(H[a][b]["t"] for a, b in zip(pth[:-1], pth[1:]))
        res.append((tt, pth))
    return res


def node_risk(twin: Twin, p: Params):
    _, _, D = depth_series(twin, replace(p, override=()))
    peak = D.max(axis=0)
    crit = np.array([np.mean(twin.BETW[twin.inc[i]]) for i in range(twin.n)])
    crit = crit / crit.max()
    risk = (peak / max(peak.max(), 1.0)) * (0.4 + 0.6 * crit)
    return peak, crit, risk


def greedy_sensors(twin: Twin, risk: np.ndarray, radius: float, kmax: int):
    Dm = np.hypot(twin.X[:, None] - twin.X[None, :], twin.Y[:, None] - twin.Y[None, :])
    cover = Dm <= radius
    covered = np.zeros(twin.n, bool); chosen, curve = [], []
    total = max(risk.sum(), 1e-9)
    for _ in range(kmax):
        gains = [(risk[cover[j] & ~covered].sum(), j) for j in range(twin.n) if j not in chosen]
        g, j = max(gains)
        if g <= 0:
            break
        chosen.append(j); covered |= cover[j]; curve.append(risk[covered].sum() / total)
    return chosen, curve


def webster(twin: Twin, flow: np.ndarray, topn: int = 10) -> pd.DataFrame:
    rows = []
    for nd in range(twin.n):
        inc = twin.inc[nd]
        if len(inc) < 3:
            continue
        ns, ew = [], []
        for e in inc:
            o = twin.EV[e] if twin.EU[e] == nd else twin.EU[e]
            (ew if abs(twin.X[o] - twin.X[nd]) > abs(twin.Y[o] - twin.Y[nd]) else ns).append(e)
        if not ns or not ew:
            continue
        yph = lambda es: max((flow[e] / 2) / (twin.LANES[e] * SAT_FLOW) for e in es)
        yns, yew = max(yph(ns), 0.02), max(yph(ew), 0.02)
        Y = yns + yew
        C = 120.0 if Y >= 0.85 else float(np.clip((1.5 * LOST_TIME + 5) / (1 - Y), 40, 120))
        g_tot = C - LOST_TIME
        g_ns = float(np.clip(g_tot * yns / Y, 8, g_tot - 8)); g_ew = g_tot - g_ns
        q_ns, q_ew = sum(flow[e] for e in ns) / 2, sum(flow[e] for e in ew) / 2

        def d1(g, c, y):
            x = y / (g / c)
            return 0.5 * c * (1 - g / c) ** 2 / (1 - min(1.0, x) * g / c), x

        def avg_delay(c, gn, ge):
            dn, xn = d1(gn, c, yns); de, xe = d1(ge, c, yew)
            return (dn * q_ns + de * q_ew) / max(q_ns + q_ew, 1e-9), max(xn, xe)

        fixed, xf = avg_delay(90.0, (90 - LOST_TIME) / 2, (90 - LOST_TIME) / 2)
        web, xw = avg_delay(C, g_ns, g_ew)
        rows.append({"Junction": twin.names[nd], "Y (flow ratio)": round(Y, 2), "Fixed 90s delay (s/veh)": round(fixed, 1),
                     "Webster cycle (s)": round(C), "NS green (s)": round(g_ns), "EW green (s)": round(g_ew),
                     "Webster delay (s/veh)": round(web, 1), "Delay cut %": round(100 * (fixed - web) / fixed, 1) if fixed > 0 else 0.0,
                     "Max X (v/c)": round(xw, 2), "Total approach flow": int(q_ns + q_ew), "State": "Oversaturated" if xw > 1 else "OK"})
    return pd.DataFrame(rows).sort_values("Total approach flow", ascending=False).head(topn).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Visual layer
# ----------------------------------------------------------------------------------
LAYERS = {
    "Congestion (v/c)": dict(cuts=[0.5, 0.75, 0.9, 1.0], labels=["< 0.5 free", "0.5-0.75", "0.75-0.9", "0.9-1.0", "> 1.0 jammed"],
                             colors=["#22c55e", "#a3e635", "#facc15", "#f97316", "#ef4444"]),
    "Waterlogging depth": dict(cuts=[15, 50, 150, 300], labels=["dry (<15 mm)", "15-50 mm", "50-150 mm hazardous", "150-300 mm severe", ">= 300 mm impassable"],
                               colors=["#475569", "#38bdf8", "#3b82f6", "#6366f1", "#d946ef"]),
    "Speed retention": dict(cuts=[0.25, 0.5, 0.75, 0.9], labels=["< 25 %", "25-50 %", "50-75 %", "75-90 %", "> 90 % of free-flow"],
                            colors=["#ef4444", "#f97316", "#facc15", "#a3e635", "#22c55e"]),
    "Structural criticality": dict(cuts=None, labels=["lowest 40 %", "40-70 %", "70-90 %", "90-97 %", "top 3 %"],
                                   colors=["#334155", "#64748b", "#38bdf8", "#f59e0b", "#ef4444"]),
}


def layer_values(twin: Twin, layer: str, snap: dict):
    stt, res = snap["state"], snap["res"]
    if layer == "Congestion (v/c)":
        v = np.nan_to_num(res["vc"], nan=0.0)
    elif layer == "Waterlogging depth":
        v = stt["depth"]
    elif layer == "Speed retention":
        v = np.where(stt["closed"], 0.0, res["speed"] / twin.VF)
    else:
        v = twin.BETW
    cuts = LAYERS[layer]["cuts"]
    if cuts is None:
        cuts = list(np.quantile(v, [0.4, 0.7, 0.9, 0.97]))
    return v, cuts


def make_map(twin, bucket, layer, closed, depth_nodes, hover, overlays=(), sensors=(), height=650, risk_nodes=None):
    cfg = LAYERS[layer]
    fig = go.Figure()
    shown = set()
    for b, col in enumerate(cfg["colors"]):
        for cls, spec in CLASSES.items():
            idx = [e for e in range(len(twin.EU)) if bucket[e] == b and not closed[e] and twin.CLS[e] == cls]
            if not idx:
                continue
            xs, ys = [], []
            for e in idx:
                xs += [twin.X[twin.EU[e]], twin.X[twin.EV[e]], None]; ys += [twin.Y[twin.EU[e]], twin.Y[twin.EV[e]], None]
            lab = cfg["labels"][b]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=spec["width"]), name=lab,
                                     legendgroup=lab, showlegend=lab not in shown, hoverinfo="skip"))
            shown.add(lab)
    cl = [e for e in range(len(twin.EU)) if closed[e]]
    if cl:
        xs, ys = [], []
        for e in cl:
            xs += [twin.X[twin.EU[e]], twin.X[twin.EV[e]], None]; ys += [twin.Y[twin.EU[e]], twin.Y[twin.EV[e]], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#fb7185", width=5, dash="dot"), name="CLOSED", hoverinfo="skip"))
    for ov in overlays:
        pth = ov["path"]
        fig.add_trace(go.Scatter(x=[twin.X[i] for i in pth], y=[twin.Y[i] for i in pth], mode="lines",
                                 line=dict(color=ov["color"], width=ov.get("width", 9), dash=ov.get("dash", "solid")),
                                 name=ov["name"], opacity=0.85, hoverinfo="skip"))
    mx = [(twin.X[u] + twin.X[v]) / 2 for u, v in zip(twin.EU, twin.EV)]
    my = [(twin.Y[u] + twin.Y[v]) / 2 for u, v in zip(twin.EU, twin.EV)]
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=16, opacity=0), hovertext=hover, hoverinfo="text", showlegend=False))
    col = depth_nodes if risk_nodes is None else risk_nodes
    cbar = "Waterlogging (mm)" if risk_nodes is None else "Risk score"
    cmax = 300 if risk_nodes is None else max(float(np.max(risk_nodes)), 1e-6)
    fig.add_trace(go.Scatter(
        x=twin.X, y=twin.Y, mode="markers+text", text=twin.names, textposition="top center", textfont=dict(size=9, color="#94a3b8"),
        marker=dict(size=9 + np.minimum(col / cmax, 1) * 10, color=col, cmin=0, cmax=cmax,
                    colorscale=[[0, "#1e293b"], [0.15, "#38bdf8"], [0.55, "#6366f1"], [1, "#d946ef"]],
                    colorbar=dict(title=cbar, thickness=12, len=0.55, x=1.02), line=dict(color="#0b1220", width=1)),
        hovertext=[f"<b>{twin.names[i]}</b><br>elevation {twin.ELEV[i]:.1f} m<br>{cbar.lower()} {col[i]:.2f}" if risk_nodes is not None else
                   f"<b>{twin.names[i]}</b><br>elevation {twin.ELEV[i]:.1f} m<br>water {col[i]:.0f} mm" for i in range(twin.n)],
        hoverinfo="text", showlegend=False))
    if len(sensors):
        fig.add_trace(go.Scatter(x=[twin.X[i] for i in sensors], y=[twin.Y[i] for i in sensors], mode="markers",
                                 marker=dict(symbol="star", size=20, color="#fde047", line=dict(color="#0b1220", width=1.5)),
                                 name="Low-cost sensor node", hoverinfo="skip"))
    fig.update_layout(height=height, paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(font=dict(color="#cbd5e1", size=11), bgcolor="rgba(15,23,42,.7)", orientation="h", y=-0.02))
    return fig


def dark(fig, h=330):
    fig.update_layout(height=h, paper_bgcolor="#0b1220", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.15))
    fig.update_xaxes(gridcolor="#1e293b"); fig.update_yaxes(gridcolor="#1e293b")
    return fig


# ----------------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="PRAVAH | Flood-aware Traffic Intelligence", page_icon="🌊", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;}
.hero{background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#0e7490 100%);padding:18px 24px;border-radius:14px;margin-bottom:10px}
.hero h1{margin:0;color:#f8fafc;letter-spacing:3px;font-size:2.0rem}
.hero p{margin:2px 0 0 0;color:#bae6fd;font-size:.95rem}
div[data-testid="stMetric"]{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:10px 14px}
.brief{background:#0f172a;border-left:4px solid #38bdf8;padding:10px 14px;border-radius:8px;color:#e2e8f0;font-size:.92rem}
</style>
<div class="hero"><h1>PRAVAH <span style="font-size:.9rem;letter-spacing:1px;color:#7dd3fc">प्रवाह</span></h1>
<p>Predictive Real-time Awareness &amp; Vehicle-flow Adaptation Hub for sudden road-condition changes.</p></div>
""", unsafe_allow_html=True)

# ---- session defaults
for k_, v_ in dict(peak=75, dur=4.0, shape="Peaked", snap=1.5, incidents=[], preset="Heavy monsoon spell").items():
    st.session_state.setdefault(k_, v_)


def apply_preset():
    v = PRESETS.get(st.session_state.preset)
    if v:
        st.session_state.peak, st.session_state.dur, st.session_state.shape = v


with st.sidebar:
    st.header("🎛️ Scenario controls")
    seed = st.number_input("Digital-twin seed", 1, 999, 7, help="Regenerates the synthetic city (terrain, demand, drains).")
    twin = build_twin(int(seed))
    st.selectbox("Storm preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.slider("Peak rainfall (mm/h)", 5, 150, key="peak")
    st.slider("Storm duration (h)", 1.0, 8.0, step=0.5, key="dur")
    st.selectbox("Hyetograph shape", ["Peaked", "Steady", "Double burst"], key="shape")
    with st.expander("Drainage, terrain & demand", expanded=False):
        drain_cap = st.slider("Design drain capacity (mm/h)", 5, 60, 20, help="ASSUMPTION - set to your ward's design standard.")
        clog = st.slider("Drain blockage (%)", 0, 80, 30) / 100
        runoff = st.slider("Runoff coefficient C", 0.3, 0.95, 0.75, 0.05, help="Urban impervious ~0.7-0.9 (Rational Method).")
        closure = st.slider("Closure depth (mm)", 150, 300, 300, 10, help="Pregnolato et al. 2017 treat >300 mm as impassable.")
        trips = st.slider("Peak-hour demand (veh/h)", 6000, 40000, 16000, 1000)
        rsl = st.slider("Rain speed loss at 100 mm/h (%)", 0, 30, 12) / 100
        rcl = st.slider("Rain capacity loss at 100 mm/h (%)", 0, 40, 15) / 100
        vot = st.number_input("Value of time (INR / person-hour)", 10, 1000, 100, help="ASSUMPTION for cost estimates.")
        occ = st.number_input("Average occupancy (persons/veh)", 1.0, 4.0, 1.6, 0.1)
    H_total = st.session_state.dur + 2.0
    st.session_state.snap = float(min(st.session_state.snap, H_total))
    st.slider("Snapshot time (h since rain start)", 0.0, float(H_total), step=DT_H, key="snap")

    st.subheader("🚧 Incidents")
    itype = st.selectbox("Type", list(INCIDENT_TYPES))
    seg = st.selectbox("Segment", range(len(twin.EU)), format_func=lambda e: twin.labels[e])
    c1, c2 = st.columns(2)
    if c1.button("➕ Add", use_container_width=True):
        st.session_state.incidents.append((int(seg), itype))
    if c2.button("🎲 Random", use_container_width=True):
        rng_ = np.random.default_rng()
        st.session_state.incidents.append((int(rng_.integers(len(twin.EU))), str(rng_.choice(list(INCIDENT_TYPES)))))
    for n_, (e_, t_) in enumerate(list(st.session_state.incidents)):
        st.caption(f"{n_ + 1}. {t_} @ {twin.labels[e_]}")
    if st.session_state.incidents and st.button("🧹 Clear incidents", use_container_width=True):
        st.session_state.incidents = []
        st.rerun()

    with st.expander("📥 Ingest real sensor readings (CSV)"):
        st.caption("Columns: `junction`, `depth_mm`. No API required.")
        tmpl = pd.DataFrame({"junction": [twin.names[10], twin.names[24]], "depth_mm": [120, 40]})
        st.download_button("Download template", tmpl.to_csv(index=False), "sensor_template.csv", use_container_width=True)
        up = st.file_uploader("Upload readings", type="csv")
    override = ()
    if up is not None:
        try:
            df_up = pd.read_csv(up)
            lut = {nm: i for i, nm in enumerate(twin.names)}
            override = tuple((lut[r.junction], float(r.depth_mm)) for r in df_up.itertuples() if r.junction in lut)
            st.success(f"{len(override)} sensor readings applied.")
        except Exception as ex:  # noqa: BLE001
            st.error(f"Could not parse CSV: {ex}")

p = Params(seed=int(seed), trips=int(trips), peak=float(st.session_state.peak), duration=float(st.session_state.dur),
           shape=st.session_state.shape, drain_cap=float(drain_cap), runoff=float(runoff), clog=float(clog),
           closure_mm=float(closure), rain_speed_loss=float(rsl), rain_cap_loss=float(rcl),
           incidents=tuple(st.session_state.incidents), override=override, k=int(round(st.session_state.snap / DT_H)))

base = baseline(twin, p.seed, p.trips)
with st.spinner("Solving traffic equilibrium..."):
    snap = snapshot(twin, p)
res, stt = snap["res"], snap["state"]
K = kpis(base, res, vot, occ)
adv = classify(twin, snap)

tabs = st.tabs(["🛰️ Command Center", "📢 Advisories & Routing", "⏱️ Storm Timeline", "🧪 Resilience Lab",
                "📡 Low-Cost Sensing", "🚦 Signal Assist", "🔍 Problem Lens", "📚 Method & Citations"])

# ================= TAB 1: COMMAND CENTER =================
with tabs[0]:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Resilience index (PRI)", f"{K['pri']:.0f} / 100", help="100 x served-trip share x (baseline trip time / current trip time)")
    m2.metric("Network speed", f"{res['avg_speed']:.1f} km/h", f"{res['avg_speed'] - base['avg_speed']:+.1f} vs normal")
    m3.metric("Closed segments", int(stt["closed"].sum()), f"of {len(twin.EU)}", delta_color="off")
    m4.metric("Trips stranded", f"{res['unserved']:,.0f}/h", help="Demand with no surviving path.")
    m5.metric("Delay cost", f"₹{K['cost']:,.0f}/h", f"{K['delay_vh']:,.0f} veh-h lost", delta_color="off")

    left, right = st.columns([3, 1.15])
    with right:
        layer = st.radio("Map layer", list(LAYERS), index=0)
        n_status = adv["Status"].value_counts()
        total_rain = float(np.sum(0.5 * (intensity_series(p)[1][:-1] + intensity_series(p)[1][1:]) * DT_H))
        worst = adv.iloc[0]
        st.markdown(f"""<div class="brief"><b>Situation brief · t = {snap['time']:.1f} h</b><br>
Rain now <b>{snap['rain']:.0f} mm/h</b>; storm total <b>{total_rain:.0f} mm</b> (IMD class: {imd_category(total_rain)}).<br>
🟣 Closed: <b>{n_status.get('CLOSED', 0)}</b> &nbsp; 🔴 Severe: <b>{n_status.get('SEVERE', 0)}</b><br>
🟠 Moderate: <b>{n_status.get('MODERATE', 0)}</b> &nbsp; 🟡 Watch: <b>{n_status.get('WATCH', 0)}</b><br>
Most affected: <b>{worst['Segment']}</b> – {worst['Status']} ({worst['Cause']}).<br>
Max water: <b>{snap['depth'].max():.0f} mm</b> at <b>{twin.names[int(snap['depth'].argmax())]}</b>.</div>""",
                    unsafe_allow_html=True)
        if len(override):
            st.info("Sensor CSV overrides active at this snapshot.")
    v, cuts = layer_values(twin, layer, snap)
    bucket = np.digitize(v, cuts)
    hov = [f"<b>{twin.labels[e]}</b><br>{adv.set_index('eid').loc[e, 'Status']} · depth {stt['depth'][e]:.0f} mm<br>"
           f"speed {res['speed'][e]:.0f} km/h · v/c {np.nan_to_num(res['vc'][e]):.2f} · flow {res['flow'][e]:.0f} veh/h<br>"
           f"criticality {twin.BETW[e]:.3f}" for e in range(len(twin.EU))]
    with left:
        st.plotly_chart(make_map(twin, bucket, layer, stt["closed"], snap["depth"], hov), use_container_width=True)
    st.caption("Line width = road class (arterial / collector / local); dotted pink = closed; node colour = water depth.")

# ================= TAB 2: ADVISORIES & ROUTING =================
with tabs[1]:
    st.subheader("Live advisory feed")
    f1, f2 = st.columns([2, 1])
    sel = f1.multiselect("Status filter", ["CLOSED", "SEVERE", "MODERATE", "WATCH"], default=["CLOSED", "SEVERE", "MODERATE"])
    feed = adv[adv["Status"].isin(sel)].drop(columns="eid")
    st.dataframe(feed, use_container_width=True, hide_index=True, height=260)
    b1, b2 = f2.columns(2)
    b1.download_button("⬇ CSV", feed.to_csv(index=False), "pravah_advisories.csv", use_container_width=True)
    b2.download_button("⬇ JSON", json.dumps(feed.to_dict("records"), indent=1), "pravah_advisories.json", use_container_width=True)

    st.subheader("Diversion planner")
    r1, r2, r3 = st.columns(3)
    o = r1.selectbox("Origin", range(twin.n), index=0, format_func=lambda i: twin.names[i])
    d = r2.selectbox("Destination", range(twin.n), index=twin.n - 1, format_func=lambda i: twin.names[i])
    kalt = r3.slider("Alternatives", 1, 3, 3)
    if o != d:
        normal = route(twin, base["t"], np.zeros(len(twin.EU), bool), o, d, 1)
        now = route(twin, res["t"], stt["closed"], o, d, kalt)
        if not now:
            st.error("No surviving path between these junctions — escalate to emergency diversion / boat-pump protocol.")
        else:
            palette = ["#22d3ee", "#a78bfa", "#fb923c"]
            ovs = [dict(path=normal[0][1], color="#ffffff", name=f"Normal route ({normal[0][0]:.0f} min)", width=14, dash="dot")]
            for i_, (tt, pth) in enumerate(now):
                ovs.append(dict(path=pth, color=palette[i_], name=f"Option {i_ + 1}: {tt:.0f} min"))
            bucket2 = np.digitize(np.nan_to_num(res["vc"]), LAYERS["Congestion (v/c)"]["cuts"])
            st.plotly_chart(make_map(twin, bucket2, "Congestion (v/c)", stt["closed"], snap["depth"], hov, overlays=ovs, height=520),
                            use_container_width=True)
            best_t, best_p = now[0]
            via = " → ".join(twin.names[i] for i in best_p)
            blocked = [twin.labels[e] for e in np.where(stt["closed"])[0] if e in {twin.eid[(a, b)] for a, b in zip(normal[0][1][:-1], normal[0][1][1:])}]
            vms = (f"⚠ {'ROUTE AFFECTED: ' + ', '.join(blocked[:2]) + '. ' if blocked else ''}"
                   f"{twin.names[o]}→{twin.names[d]}: ETA {best_t:.0f} min (normally {normal[0][0]:.0f}). "
                   f"VIA {twin.names[best_p[1]]}…{twin.names[best_p[-2]]}.")
            st.code(vms, language="text")
            st.caption(f"Recommended path: {via}. Variable-message-sign text above is auto-generated.")

# ================= TAB 3: TIMELINE =================
with tabs[2]:
    st.subheader("Storm evolution — impact over time")
    with st.spinner("Simulating every 30-minute step..."):
        TL = timeline(twin, p)
    tt = [x["time"] for x in TL]
    pri = [kpis(base, x["res"], vot, occ)["pri"] for x in TL]
    cost = [kpis(base, x["res"], vot, occ)["cost"] for x in TL]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=tt, y=[x["rain"] for x in TL], name="Rain (mm/h)", marker_color="#38bdf8", opacity=0.55), secondary_y=False)
    fig.add_trace(go.Scatter(x=tt, y=[x["depth"].max() for x in TL], name="Max water depth (mm)", line=dict(color="#d946ef", width=3)), secondary_y=True)
    fig.add_vline(x=snap["time"], line=dict(color="#fde047", dash="dash"))
    fig.update_layout(title="Rainfall input and peak waterlogging")
    fig.update_yaxes(title_text="mm/h", secondary_y=False); fig.update_yaxes(title_text="mm", secondary_y=True)
    st.plotly_chart(dark(fig), use_container_width=True)
    c1, c2 = st.columns(2)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=tt, y=pri, name="Resilience index", fill="tozeroy", line=dict(color="#22d3ee", width=3)))
    f2.add_vline(x=snap["time"], line=dict(color="#fde047", dash="dash"))
    f2.update_layout(title="Network resilience index (PRI)", yaxis=dict(range=[0, 105]))
    c1.plotly_chart(dark(f2), use_container_width=True)
    f3 = make_subplots(specs=[[{"secondary_y": True}]])
    f3.add_trace(go.Bar(x=tt, y=[int(x["state"]["closed"].sum()) for x in TL], name="Closed segments", marker_color="#fb7185"), secondary_y=False)
    f3.add_trace(go.Scatter(x=tt, y=cost, name="Delay cost (INR/h)", line=dict(color="#facc15", width=3)), secondary_y=True)
    f3.update_layout(title="Closures and economic cost")
    c2.plotly_chart(dark(f3), use_container_width=True)

    st.subheader("Smart departure-time advisor")
    a1, a2 = st.columns(2)
    o2 = a1.selectbox("From", range(twin.n), index=0, format_func=lambda i: twin.names[i], key="o2")
    d2 = a2.selectbox("To", range(twin.n), index=twin.n - 1, format_func=lambda i: twin.names[i], key="d2")
    if o2 != d2:
        times = []
        for x in TL:
            r_ = route(twin, x["res"]["t"], x["state"]["closed"], o2, d2, 1)
            times.append(r_[0][0] if r_ else np.nan)
        nominal = route(twin, base["t"], np.zeros(len(twin.EU), bool), o2, d2, 1)[0][0]
        fig4 = go.Figure(go.Bar(x=tt, y=times, marker_color=["#22c55e" if (not np.isnan(v_) and v_ == np.nanmin(times)) else "#6366f1" for v_ in times]))
        fig4.add_hline(y=nominal, line=dict(color="#fff", dash="dot"), annotation_text=f"normal {nominal:.0f} min")
        fig4.update_layout(title="Best-route travel time by departure hour", xaxis_title="hours since rain start", yaxis_title="minutes")
        st.plotly_chart(dark(fig4), use_container_width=True)
        if np.all(np.isnan(times)):
            st.error("Destination unreachable throughout the storm.")
        else:
            bt = int(np.nanargmin(times))
            wt = int(np.nanargmax(times))
            st.success(f"Best departure ≈ t+{tt[bt]:.1f} h ({times[bt]:.0f} min). Worst ≈ t+{tt[wt]:.1f} h ({np.nan_to_num(times[wt], nan=0):.0f} min"
                       f"{', route cut' if np.isnan(times[wt]) else ''}).")

# ================= TAB 4: RESILIENCE LAB =================
with tabs[3]:
    st.subheader("Where is the network fragile?")
    rl, rr = st.columns([2, 1])
    top = np.argsort(-twin.BETW)[:10]
    vcrit = twin.BETW
    cuts_c = list(np.quantile(vcrit, [0.4, 0.7, 0.9, 0.97]))
    with rl:
        st.plotly_chart(make_map(twin, np.digitize(vcrit, cuts_c), "Structural criticality", np.zeros(len(twin.EU), bool),
                                 np.zeros(twin.n), [f"<b>{twin.labels[e]}</b><br>criticality {twin.BETW[e]:.3f}" for e in range(len(twin.EU))],
                                 height=520), use_container_width=True)
    with rr:
        st.markdown("**Top-10 structurally critical links**")
        st.dataframe(pd.DataFrame({"Segment": [twin.labels[e] for e in top], "Betweenness": np.round(twin.BETW[top], 3)}),
                     hide_index=True, use_container_width=True, height=380)
    st.markdown("#### Intervention priority = criticality × flood exposure")
    _, _, Dall = depth_series(twin, replace(p, override=()))
    pk = Dall.max(axis=0)
    e_peak = 0.5 * (pk[twin.EU] + pk[twin.EV])
    score = twin.BETW / twin.BETW.max() * np.minimum(e_peak, 300) / 300
    pr = np.argsort(-score)[:8]
    st.dataframe(pd.DataFrame({"Segment": [twin.labels[e] for e in pr], "Peak depth (mm)": e_peak[pr].round(0), "Criticality": twin.BETW[pr].round(3),
                               "Priority score": score[pr].round(3), "Suggested pre-monsoon action": ["Silt-clear drains, fix inlet grating, stage pump"] * len(pr)}),
                 hide_index=True, use_container_width=True)
    st.markdown("#### Stress-test: what if one critical link fails on a normal day?")
    kst = st.slider("Links to test", 3, 10, 6)
    if st.button("Run stress test"):
        rows = []
        prog = st.progress(0.0)
        for i_, e in enumerate(top[:kst]):
            r_ = stress(twin, p.seed, p.trips, int(e))
            kk = kpis(base, r_, vot, occ)
            rows.append({"Link removed": twin.labels[e], "PRI": round(kk["pri"], 1), "Extra delay (veh-h/h)": round(kk["delay_vh"], 0),
                         "Stranded trips/h": round(r_["unserved"]), "Avg speed (km/h)": round(r_["avg_speed"], 1)})
            prog.progress((i_ + 1) / kst)
        sdf = pd.DataFrame(rows).sort_values("PRI")
        st.dataframe(sdf, hide_index=True, use_container_width=True)
        st.plotly_chart(dark(go.Figure(go.Bar(x=sdf["Link removed"], y=100 - sdf["PRI"], marker_color="#f97316")).update_layout(
            title="Resilience loss when link fails (100 − PRI)")), use_container_width=True)

# ================= TAB 5: SENSING =================
with tabs[4]:
    st.subheader("Low-cost detection layer — where to put the next sensor rupee")
    peak_d, crit_n, risk = node_risk(twin, p)
    s1, s2, s3 = st.columns(3)
    radius = s1.slider("Representative radius of one sensor (km)", 0.6, 4.0, 1.8, 0.2,
                       help="ASSUMPTION: nodes within this radius share similar hydrology/traffic response.")
    budget = s2.number_input("Budget (INR)", 10000, 5000000, 150000, 10000)
    bench = s3.number_input("Benchmark ITS node cost (INR)", 20000, 2000000, 150000, 10000,
                            help="PLACEHOLDER for a conventional camera/radar+controller node. Replace with your tender rate.")
    with st.expander("Bill of materials for one DIY node (indicative, editable)"):
        bcols = st.columns(5)
        parts = {"ESP32 board": bcols[0].number_input("ESP32 MCU", 100, 5000, 600),
                 "Waterproof ultrasonic (JSN-SR04T class)": bcols[1].number_input("Ultrasonic", 100, 5000, 900),
                 "Solar + Li-ion + charger": bcols[2].number_input("Power", 100, 8000, 1800),
                 "Pole, enclosure, mounting": bcols[3].number_input("Mounting", 100, 8000, 1200),
                 "Connectivity (LoRa/GSM, 1 yr)": bcols[4].number_input("Comms", 100, 8000, 800)}
        unit = sum(parts.values())
        st.caption(f"Unit cost ≈ ₹{unit:,.0f}. Prices are indicative hobby-market estimates, not quotations — verify locally.")
    kmax = int(min(budget // unit, 25, twin.n))
    chosen, curve = greedy_sensors(twin, risk, radius, max(kmax, 1))
    cov = curve[-1] if curve else 0
    q1, q2, q3 = st.columns(3)
    q1.metric("Nodes affordable", f"{len(chosen)}", f"₹{len(chosen) * unit:,.0f} spend")
    q2.metric("Flood-risk coverage", f"{cov * 100:.0f} %")
    q3.metric("Nodes for same budget (conventional)", f"{max(int(budget // bench), 0)}", f"{(budget // unit) / max(budget // bench, 1):.0f}× more with DIY", delta_color="off")
    ml, mr = st.columns([3, 2])
    with ml:
        bucket3 = np.digitize(twin.BETW, cuts_c)
        st.plotly_chart(make_map(twin, bucket3, "Structural criticality", np.zeros(len(twin.EU), bool), np.zeros(twin.n),
                                 [twin.labels[e] for e in range(len(twin.EU))], sensors=chosen, risk_nodes=risk, height=500),
                        use_container_width=True)
    with mr:
        fc = go.Figure(go.Scatter(x=list(range(1, len(curve) + 1)), y=[c * 100 for c in curve], mode="lines+markers", line=dict(color="#22d3ee", width=3)))
        fc.update_layout(title="Diminishing returns of sensor count", xaxis_title="# sensors", yaxis_title="% risk covered")
        st.plotly_chart(dark(fc, 280), use_container_width=True)
        st.dataframe(pd.DataFrame({"Order": range(1, len(chosen) + 1), "Junction": [twin.names[i] for i in chosen],
                                   "Peak depth (mm)": peak_d[chosen].round(0), "Risk": risk[chosen].round(2)}), hide_index=True,
                     use_container_width=True, height=200)
    st.markdown("""**Detection logic for the node:** take the median of 5 ultrasonic pings, convert
`depth = mounting_height − distance`, then raise levels at **15 / 50 / 150 / 300 mm** (the same bands PRAVAH uses). Transmit only on level change
(LoRa/GSM) to keep battery and data cost minimal; upload the log as the CSV in the sidebar to re-calibrate the model.""")

# ================= TAB 6: SIGNAL ASSIST =================
with tabs[5]:
    st.subheader("Adaptive signal retiming from live flows")
    wt = webster(twin, res["flow"])
    st.dataframe(wt, hide_index=True, use_container_width=True)
    fw = go.Figure()
    fw.add_trace(go.Bar(x=wt["Junction"], y=wt["Fixed 90s delay (s/veh)"], name="Fixed 90 s, 50/50 plan", marker_color="#64748b"))
    fw.add_trace(go.Bar(x=wt["Junction"], y=wt["Webster delay (s/veh)"], name="Webster-adapted plan", marker_color="#22d3ee"))
    fw.update_layout(barmode="group", title="Average uniform delay per vehicle (HCM d1 term)", yaxis_title="s/veh")
    st.plotly_chart(dark(fw, 360), use_container_width=True)
    st.caption("Two-phase approximation (NS/EW), lost time 10 s/cycle, approach flow = half of link flow. Oversaturated junctions "
               "(X>1) need metering/diversion rather than retiming alone — see advisory feed.")

# ================= TAB 7: PROBLEM LENS =================
with tabs[6]:
    st.subheader("Why sudden road-condition changes break traffic management")
    st.markdown("""
**Problem statement:** unexpected changes — heavy rain, waterlogging, accidents, blockages, construction — cause congestion, delay and poor route
management, and affordable real-time monitoring is missing.

**Deep analysis: the disruption is not one problem but a chain of four gaps**

| # | Gap | What actually happens on the ground | PRAVAH response |
|---|-----|--------------------------------------|-----------------|
| 1 | **Detection gap** | Water depth/incidents are known only after commuters are stuck; cameras and loop detectors are too costly to blanket a city. | Cost model + greedy siting of ₹-level ultrasonic nodes (Sensing tab); CSV ingest of real readings. |
| 2 | **Impact-understanding gap** | A flooded 200 mm underpass is not simply “slower”; it can cut a *structurally critical* link and re-route flow onto weaker roads (network effect). | Depth-disruption speed model + equilibrium re-assignment shows true network-wide delay, stranded trips, cost. |
| 3 | **Decision gap** | Controllers lack ranked, actionable guidance: what to close, divert, pump, retime? | Auto-ranked advisory feed, VMS text, diversion planner, Webster signal retiming. |
| 4 | **Anticipation gap** | Response is reactive; the storm profile is predictable minute-to-minute and *departure time* is a lever. | Storm timeline, departure-time advisor, pre-monsoon intervention priority (criticality × exposure), stress tests. |

**Cascade logic:** rain ↓ speed & capacity everywhere → low-lying nodes pond → links beyond ~300 mm close → flow shifts to remaining links →
v/c exceeds 1 on alternates (BPR delay grows with the 4th power) → queues spill back to junctions → fixed signal plans mis-allocate green.
Breaking the chain early, cheaply, and *where it matters* is the design goal.

""")

# ================= TAB 8: METHOD & CITATIONS =================
with tabs[7]:
    st.subheader("Method pipeline")
    st.markdown("""
1. **Digital twin** – seeded 7×7 junction grid, three basins (elevation), class-based lanes/speeds/drains, gravity-model OD demand (CBD-weighted).
2. **Rain → water** – Rational-Method excess rain (`C·i`) amplified by local relief, minus drain capacity × (1 − blockage); integrated every 30 min.
3. **Water → speed** – Pregnolato et al. (2017) depth-disruption function; ≥ closure depth ⇒ link closed. Rain additionally trims speed/capacity.
4. **Incidents** – capacity/speed multipliers or full closure per incident type (editable defaults).
5. **Traffic assignment** – BPR volume-delay function with Method of Successive Averages toward user equilibrium (Wardrop).
6. **Analytics** – PRI, delay cost, advisories, routing (Dijkstra / Yen-type k-shortest), betweenness criticality, Webster timings, maximal-coverage sensor siting.
""")
    st.markdown("**PRI (Pravah Resilience Index)** = 100 × (served trips ÷ demand) × min(1, normal avg trip time ÷ current avg trip time). "
                "It is a transparent composite for comparison, not a standard.")
    st.markdown("**IMD rainfall categories (24-h totals, mm):** " + ", ".join(f"≥{lo:g} {nm}" for lo, nm in IMD_CLASSES))
    st.subheader("References")
    for i_, (ref, use) in enumerate(REFERENCES, 1):
        st.markdown(f"**[{i_}]** {ref}  \n<span style='color:#7dd3fc'>Used for:</span> {use}", unsafe_allow_html=True)
    

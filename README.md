# 🌊 PRAVAH (प्रवाह)
### Predictive Real-time Awareness & Vehicle-flow Adaptation Hub
**Low-cost, API-free, real-time traffic management for sudden road-condition changes** – heavy rain, waterlogging, accidents, blockages and construction.

> *Pravah* = "flow". The platform keeps traffic flowing when the road suddenly stops being what the map says it is.

---

## 1. Problem analysis

Unexpected disruptions create congestion not because roads become "a bit slower", but because **four gaps** open up at once:

| # | Gap | Reality on the ground | PRAVAH module |
|---|-----|-----------------------|---------------|
| 1 | **Detection** | Depth/incidents are known only after commuters are stuck; cameras and loop detectors are too expensive to blanket a city. | *Low-Cost Sensing*: DIY ultrasonic node BOM, greedy maximal-coverage siting, CSV ingest of real readings |
| 2 | **Impact understanding** | A 200 mm underpass doesn't just slow traffic – it can remove a *structurally critical* link and push flow onto weaker roads. | Depth-disruption model + user-equilibrium re-assignment → network-wide delay, stranded trips, ₹ cost |
| 3 | **Decision** | Controllers lack ranked, actionable guidance (close? divert? pump? retime?). | *Advisories & Routing*, VMS text, *Signal Assist* |
| 4 | **Anticipation** | Response is reactive although storm profiles and departure time are controllable levers. | *Storm Timeline*, departure-time advisor, *Resilience Lab* (criticality × flood exposure, stress tests) |

**Cascade modelled:** rain ↓ speed & capacity → low-lying nodes pond → links beyond closure depth shut → flow shifts → v/c > 1 on alternates (BPR delay ∝ (v/c)⁴) → queues spill to junctions → fixed signal plans mis-allocate green.

## 2. Feature overview

| Tab | What you get |
|-----|--------------|
| 🛰️ **Command Center** | KPIs (Pravah Resilience Index, network speed, closures, stranded trips, delay cost), auto-written situation brief, switchable map layers (congestion, water depth, speed retention, criticality) |
| 📢 **Advisories & Routing** | Ranked live advisory feed (CLOSED / SEVERE / MODERATE / WATCH) with actions, CSV/JSON export, diversion planner with 3 alternatives and auto-generated VMS text |
| ⏱️ **Storm Timeline** | 30-minute evolution of rain, water depth, resilience, closures, ₹ cost; **smart departure-time advisor** |
| 🧪 **Resilience Lab** | Betweenness criticality map, pre-monsoon **intervention priority** list, single-link **stress tests** |
| 📡 **Low-Cost Sensing** | Budget-driven sensor siting, coverage-vs-count curve, DIY node BOM vs. conventional ITS node |
| 🚦 **Signal Assist** | Webster optimum cycle & splits for the busiest junctions, delay saving vs. fixed 90 s plan |
| 🔍 **Problem Lens** | Deep problem decomposition built into the app |
| 📚 **Method & Citations** | Pipeline, formulas, embedded references, limitations |

## 3. Quick start (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. No API keys, no internet access required at runtime.

## 4. Models & data (no external APIs)

All city data is a **seeded digital twin** (7×7 junctions, three low-lying basins, CBD-weighted gravity demand). Scientific relationships are cited in-app:

| Component | Method | Source |
|-----------|--------|--------|
| Flood depth → speed | `v = 0.0009d² − 0.5529d + 86.9448` km/h (d ≤ 300 mm; closed beyond) | Pregnolato et al. (2017), *Transp. Res. D* 55:67-81 |
| Rain → ponding | Rational Method excess `C·i`, relief amplification, drain capacity & blockage | Chow, Maidment & Mays (1988) |
| Volume-delay | `t = t0 (1 + 0.15 (v/c)⁴)` | Bureau of Public Roads (1964) |
| Assignment | Method of Successive Averages → user equilibrium | Sheffi (1985); Wardrop (1952) |
| Criticality | Edge betweenness centrality | Freeman (1977) |
| Signals | `C0 = (1.5L+5)/(1−Y)`; HCM uniform delay | Webster (1958); HCM 6th ed. (2016) |
| Sensor siting | Greedy maximal coverage | Church & ReVelle (1974); Nemhauser et al. (1978) |
| Rain classes | IMD 24-h categories | India Meteorological Department |
| Weather impact | Qualitative basis for rain speed/capacity loss | U.S. FHWA Road Weather Management |
| Capacity calibration | PCU-based urban capacity | IRC:106-1990 |


## 5. Using your own data
* **Sensor readings:** sidebar → *Ingest real sensor readings* → CSV with columns `junction,depth_mm` (template provided). Overrides simulated depth at the snapshot instant.
* **Real network:** replace `build_twin()` with a loader for your GIS export (nodes, edges, length, class, elevation, drainage efficiency, OD matrix). The rest of the pipeline is unchanged.




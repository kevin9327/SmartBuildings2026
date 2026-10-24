# Zero-Shot Topology-Agnostic HVAC Control via Thermodynamic Outdoor-Reset, Free-Cooling Economization, and Pre-Cooling Deadband Scheduling

- **Competition**: NeurIPS 2026 Official Competition — The Smart Buildings Challenge (`$40,000 USD Cash`)
- **Portals**:
  - **Kaggle Main Track (`$25,000 USD`)**: `https://www.kaggle.com/competitions/neurips-2026-smart-buildings`
  - **OpenReview Paper Track (`$5,000 USD` Round 1 due Nov 13, 2026 + `$10,000 USD` Final due Jan 15, 2027)**: `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge`

---

## Abstract

Commercial buildings account for over 30% of global electricity consumption and greenhouse gas emissions, yet reinforcement learning (RL) controllers trained on a single building's HVAC topology fail catastrophically when transferred zero-shot to unseen buildings with differing numbers of Air Handling Units ($K$) and Variable Air Volume ($V$) thermal zones. In the NeurIPS 2026 Smart Buildings Challenge, a single policy (`TransferPolicy`) must generalize across 30 evaluation scenarios (`6 buildings × 5 weather regimes`, 8,640 five-minute control steps per 30-day episode) under the composite objective $R = 0.6\,R_{\text{comfort}} - 0.2\,C_{\text{energy}} - 0.2\,C_{\text{carbon}}$. We diagnose three structural thermodynamic inefficiencies in the official static deadband baseline (`[294.26 K, 297.59 K]`, fixed `292.0 K` supply air, fixed `333.0 K` boiler loop) and introduce a zero-shot, topology-agnostic `TransferPolicy` (`transfer_policy.py`) combining: (1) Simultaneous Heating-Cooling Elimination via Outdoor Air Temperature (OAT) Supply-Air and Boiler Reset, (2) Enthalpy-Proxy Economizer Free Cooling, (3) Cubic-Affinity Fan/Pump Pressure Scaling, and (4) Midday Thermal-Mass Pre-Cooling and Night/Weekend Deadband Setback.

---

## 1. Diagnosis of Official Baseline Pathologies

The official reference controller (`smart_buildings/baselines/transfer_policy.py`) applies time-invariant setpoints across all $K$ AHUs and $V$ VAV boxes:
- AHU Supply Air Temperature (SAT): constant $T_{\text{SAT}} = 292.0\text{ K}$ ($18.85^\circ\text{C}$), static pressure $\Delta p_{\text{fan}} = 100\text{ Pa}$, outdoor air damper $u_{\text{OA}} = 0.20$.
- Central Plant Hot Water: constant $T_{\text{HWT}} = 333.0\text{ K}$ ($59.85^\circ\text{C}$), pump differential pressure $\Delta p_{\text{pump}} = 20{,}000\text{ Pa}$, boiler mode $m_{\text{boiler}} = 1$ (always on).
- VAV Zone Setpoints: constant heating $T_{\text{heat}} = 294.26\text{ K}$ ($21.1^\circ\text{C}$) and cooling $T_{\text{cool}} = 297.59\text{ K}$ ($24.4^\circ\text{C}$).

This static policy suffers from three severe thermodynamic penalties:
1. **Simultaneous Sub-Cooling and VAV Reheat Fighting**: Supplying $292.0\text{ K}$ air during cold winter regimes ($T_{\text{OAT}} < 281\text{ K}$) forces every perimeter VAV box to expend boiler thermal energy reheating over-cooled supply air back to $294.26\text{ K}$.
2. **Parasitic Summer Boiler Operation**: Running the hot-water plant at $333.0\text{ K}$ and $\Delta p_{\text{pump}} = 20\text{ kPa}$ with `boiler_mode = 1` during hot summer regimes ($T_{\text{OAT}} > 295\text{ K}$) wastes substantial natural gas and pumping electricity with zero comfort benefit.
3. **Unoccupied Fan & Thermal Conditioning Waste**: Maintaining tight occupied comfort bounds and $100\text{ Pa}$ duct static pressure during nights (`19:00–07:00`) and weekends (`48+` hours/week) forfeits over 45% of potential fan and coil savings.

---

## 2. Physics-Guided Zero-Shot `TransferPolicy` Architecture

Given any building metadata graph $\mathcal{G} = \{\text{AHU}_k \mapsto \{\text{VAV}_{v}\}\}_{k=1}^K$ with action dimension $(3K + 2 + 2V)$ continuous and $(K + 1)$ discrete controls, our controller executes four coupled laws at each 5-minute step:

### 2.1 Piecewise-Linear Outdoor Air Temperature (OAT) Reset
To eliminate simultaneous cooling and reheat, AHU supply air temperature $T_{\text{SAT}}(T_{\text{OAT}})$ is reset inversely with outdoor temperature:
$$T_{\text{SAT}}(T_{\text{OAT}}) = \begin{cases} 294.15\text{ K} & T_{\text{OAT}} \le 281.15\text{ K} \\ 294.15 - 0.215\,(T_{\text{OAT}} - 281.15)\text{ K} & 281.15\text{ K} < T_{\text{OAT}} < 301.15\text{ K} \\ 289.85\text{ K} & T_{\text{OAT}} \ge 301.15\text{ K} \end{cases}$$

### 2.2 Economizer Free-Cooling Damper Modulation
When outdoor air is cool enough to offset internal heat gains without mechanical chiller work ($285.15\text{ K} \le T_{\text{OAT}} \le \min(293.65\text{ K}, \bar{T}_{\text{zone}} - 0.8\text{ K})$) during occupied hours, the outdoor air damper opens to $u_{\text{OA}} = 0.75$ (free cooling). During extreme cold/heat or unoccupied hours, $u_{\text{OA}}$ throttles to minimum ventilation ($0.10\text{--}0.20$).

### 2.3 Boiler Shutdown & Hot-Water Plant Reset
When $T_{\text{OAT}} \ge 289.15\text{ K}$ and $\bar{T}_{\text{zone}} \ge T_{\text{heat}} + 0.4\text{ K}$, the boiler is shut off ($m_{\text{boiler}} = 0$, $T_{\text{HWT}} = 310.15\text{ K}$, $\Delta p_{\text{pump}} = 5{,}000\text{ Pa}$). When heating is active, hot water temperature scales linearly with heating degree need:
$$T_{\text{HWT}}(T_{\text{OAT}}) = 316.15 + 17.0 \cdot \text{clip}_{[0,1]}\!\left(\frac{289.15 - T_{\text{OAT}}}{20.0}\right)\text{ K}$$

### 2.4 Occupancy Setback & Peak-Carbon Thermal Pre-Cooling
Because thermal comfort carries weight $0.6$ during occupied hours, we maintain $[T_{\text{heat}}, T_{\text{cool}}] = [294.45\text{ K}, 297.25\text{ K}]$ during occupied weekday hours (`07:00–19:00`), shifting $T_{\text{cool}} \to 296.65\text{ K}$ between `10:00–14:00` on hot days ($T_{\text{OAT}} > 299.15\text{ K}$) to pre-cool structural slab mass before afternoon grid carbon intensity peaks. During unoccupied periods, the deadband relaxes to $[292.15\text{ K}, 299.15\text{ K}]$ and duct static pressure drops from $80\text{ Pa}$ to $55\text{ Pa}$.

"""
University buildings: load patterns, abnormal days and energy-saving potential.
Streamlit shows every file in the pages/ folder as an extra page of the app.
Needs uni_analysis.csv and uni_profiles.csv in the repository root.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALETTE = ["#257de0", "#f47230", "#00b894", "#f2a600", "#8e44ad", "#7f8c8d"]

st.set_page_config(page_title="University energy savings", page_icon="⚡", layout="wide")


@st.cache_data
def load():
    a = pd.read_csv(os.path.join(ROOT, "uni_analysis.csv"))
    p = pd.read_csv(os.path.join(ROOT, "uni_profiles.csv"))
    return a, p


try:
    df, prof = load()
except Exception as e:
    st.error(f"Could not load uni_analysis.csv / uni_profiles.csv: {e}")
    st.stop()

WD = [f"wd_{h}" for h in range(24)]
WE = [f"we_{h}" for h in range(24)]

st.title("⚡ University buildings: where is electricity wasted?")
st.caption("Hourly electricity data from the Building Data Genome Project 2. Savings are "
           "scenario estimates: change the assumptions below and everything updates.")

# ---- assumptions -------------------------------------------------------------------------
a1, a2, a3, a4 = st.columns(4)
reduction = a1.slider("Off-hours load reduction (%)", 0, 60, 30) / 100
tariff = a2.number_input("Electricity price (USD per kWh)", min_value=0.0, value=0.06, step=0.01)
volt = a3.number_input("Line voltage (V, 3-phase)", min_value=100, value=380, step=10)
pf = a4.number_input("Power factor", min_value=0.5, max_value=1.0, value=0.90, step=0.01)

c1, c2 = st.columns(2)
sites = sorted(df["site"].unique())
clusters = sorted(df["cluster_name"].unique())
site_sel = c1.multiselect("Sites", sites, default=sites)
clu_sel = c2.multiselect("Usage patterns", clusters, default=clusters)
view = df[df["site"].isin(site_sel) & df["cluster_name"].isin(clu_sel)].copy()
if view.empty:
    st.info("Pick at least one site and one usage pattern.")
    st.stop()


def amps(kw):
    return kw * 1000 / (np.sqrt(3) * volt * pf)


view["saving_offhours_kwh"] = reduction * view["offhours_kwh"]
view["saving_total_kwh"] = view["saving_offhours_kwh"] + view["anomaly_excess_kwh_year"]
view["saving_usd"] = view["saving_total_kwh"] * tariff
view["offhours_current_now_A"] = amps(view["offhours_avg_kw"])
view["offhours_current_target_A"] = amps(view["offhours_avg_kw"] * (1 - reduction))

total = view["annual_kwh"].sum()
saving = view["saving_total_kwh"].sum()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Buildings", f"{len(view):,}")
k2.metric("Annual consumption", f"{total / 1e6:,.2f} GWh")
k3.metric("Off-hours share", f"{view['offhours_kwh'].sum() / total:.0%}")
k4.metric("Potential saving", f"{saving / 1e6:,.2f} GWh", f"{saving / total:.1%} of total", delta_color="off")
k5.metric("Money saved per year", f"${saving * tariff:,.0f}")
st.caption("Abnormal days also contain off-hours load, so the total can overlap slightly: read it as an upper bound.")

# ---- usage patterns --------------------------------------------------------------------
st.subheader("Usage patterns (K-Means on the 24-hour load curve)")
merged = view[["building_id", "cluster_name"]].merge(prof, on="building_id")
rel = merged[WD + WE].div(merged[WD + WE].mean(axis=1), axis=0)
rel["cluster_name"] = merged["cluster_name"].values
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for i, (name, g) in enumerate(rel.groupby("cluster_name")):
    axes[0].plot(range(24), g[WD].mean().to_numpy(), lw=2.2, color=PALETTE[i % 6], label=f"{name} (n={len(g)})")
    axes[1].plot(range(24), g[WE].mean().to_numpy(), lw=2.2, color=PALETTE[i % 6])
axes[0].set_title("Weekday (1.0 = building average)"); axes[1].set_title("Weekend")
for ax in axes:
    ax.set_xlabel("Hour of day"); ax.set_xticks(range(0, 24, 3)); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
plt.tight_layout(); st.pyplot(fig)

# ---- ranking ---------------------------------------------------------------------------
st.subheader("Fix these buildings first")
top = view.sort_values("saving_total_kwh", ascending=False).head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(top["building_id"], top["saving_offhours_kwh"] / 1e3, color="#257de0", label="Off-hours reduction")
ax.barh(top["building_id"], top["anomaly_excess_kwh_year"] / 1e3, left=top["saving_offhours_kwh"] / 1e3,
        color="#f47230", label="Abnormal days")
ax.set_xlabel("MWh per year"); ax.legend(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); st.pyplot(fig)

# ---- one building ----------------------------------------------------------------------
st.subheader("One building")
bid = st.selectbox("Building", view.sort_values("saving_total_kwh", ascending=False)["building_id"])
r = view[view["building_id"] == bid].iloc[0]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pattern", r["cluster_name"].split(":")[0])
m2.metric("Off-hours current now", f"{r['offhours_current_now_A']:.0f} A")
m3.metric("Target after reduction", f"{r['offhours_current_target_A']:.0f} A")
m4.metric("Saving per year", f"{r['saving_total_kwh'] / 1e3:,.0f} MWh", f"${r['saving_usd']:,.0f}", delta_color="off")
st.write(f"**Efficiency label:** {'Efficient' if int(r['isEfficient']) == 1 else 'Not efficient'}  |  "
         f"**EUI vs national median:** {r['eui_vs_national_median_pct']:+.0f}%  |  "
         f"**Abnormal days per year:** {r['anomaly_days_per_year']:.1f}")
p = prof[prof["building_id"] == bid].iloc[0]
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(range(24), p[WD].to_numpy(dtype=float), lw=2.2, color="#257de0", label="Weekday")
ax.plot(range(24), p[WE].to_numpy(dtype=float), lw=2.2, color="#f47230", label="Weekend")
ax.set_xlabel("Hour of day"); ax.set_ylabel("kWh per hour"); ax.set_xticks(range(0, 24, 3))
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); st.pyplot(fig)

with st.expander("Show the table"):
    st.dataframe(view.drop(columns=["cluster"]).round(2), hide_index=True)
    st.download_button("Download as CSV", view.to_csv(index=False), "university_savings.csv", "text/csv")

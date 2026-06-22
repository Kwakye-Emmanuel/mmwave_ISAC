# replot.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isac.config import SystemConfig
from isac.simulate import plot_rsec_vs_snr, plot_outage_vs_snr, _topology_tag

cfg  = SystemConfig()
data = np.load('outputs/eval_results_Eve20dB_dbe20m_CU40-60m.npz')
tag  = _topology_tag(cfg)

# Add floor point at 7.5dB for Genie-Aided
snr_db      = data['snr_db']
pout_oracle = data['pout_oracle'].copy()
idx         = np.argmin(np.abs(snr_db - 7.5))
pout_oracle[idx] = 1e-4
print(f"  Added floor point at SNR={snr_db[idx]}dB "
      f"(index={idx}) → pout_oracle={1e-4:.0e}")

def nan_zeros(arr):
    out = arr.copy().astype(float)
    out[out == 0.0] = np.nan
    return out

for ext in ["png", "pdf"]:
    fig = plot_rsec_vs_snr(
        data['snr_db'], data['rsec_B1'],
        data['rsec_B2'], data['rsec_oracle'],
        rsec_proposed = data['rsec_proposed'],
        rsec_conv_dl  = data['rsec_conv_dl'],
        cfg           = cfg,
        save_path     = f"outputs/replot_rsec_vs_snr{tag}.{ext}",
    )
    plt.close(fig)

for ext in ["png", "pdf"]:
    fig = plot_outage_vs_snr(
        data['snr_db'],
        nan_zeros(data['pout_B1']),
        nan_zeros(data['pout_B2']),
        nan_zeros(pout_oracle),          
        pout_proposed = nan_zeros(data['pout_proposed']),
        pout_conv_dl  = nan_zeros(data['pout_conv_dl']),
        cfg           = cfg,
        save_path     = f"outputs/replot_outage_vs_snr{tag}.{ext}",
    )
    plt.close(fig)

print("Done!")

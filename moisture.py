import numpy as np
from data_loader import UnifiedERA5Dataset
from utils.YParams import YParams

params = YParams('config.yaml', 'base')
dataset = UnifiedERA5Dataset(
    years=[2020],
    input_channels=['u10', 'v10', 'q_1000', 'q_800', 'q_600'],
    output_channels=['avg_tprate'],
    region='us_midwest',
)

sample = dataset[500]
u10 = sample['input'][0].numpy()
v10 = sample['input'][1].numpy()
q_1000 = sample['input'][2].numpy()
q_800 = sample['input'][3].numpy()
q_600 = sample['input'][4].numpy()

lat = dataset.lat
lon = dataset.lon

def compute_moisture_flux_convergence(u10, v10, q, lat, lon):
    flux_u = u10 * q
    flux_v = v10 * q
    lat_spacing_m = np.abs(lat[1] - lat[0]) * 111000
    lon_spacing_m = np.abs(lon[1] - lon[0]) * 111000 * np.cos(np.radians(lat.mean()))
    dflux_u_dx = np.gradient(flux_u, axis=1) / lon_spacing_m
    dflux_v_dy = np.gradient(flux_v, axis=0) / lat_spacing_m
    return -(dflux_u_dx + dflux_v_dy)

def compute_moisture_proxy(q_1000, q_800, q_600):
    return (q_1000 + q_800 + q_600) / 3.0

convergence = compute_moisture_flux_convergence(u10, v10, q_800, lat, lon)
moisture_proxy = compute_moisture_proxy(q_1000, q_800, q_600)

print("Moisture flux convergence  min/max/mean:", convergence.min(), convergence.max(), convergence.mean())
print("Moisture proxy  min/max/mean:", moisture_proxy.min(), moisture_proxy.max(), moisture_proxy.mean())

# Sanity check: correlate with actual precipitation at this timestamp
actual_precip = sample['output'][0].numpy()
corr_convergence = np.corrcoef(convergence.flatten(), actual_precip.flatten())[0, 1]
corr_moisture = np.corrcoef(moisture_proxy.flatten(), actual_precip.flatten())[0, 1]
print(f"Correlation of convergence with actual precip: {corr_convergence:.4f}")
print(f"Correlation of moisture proxy with actual precip: {corr_moisture:.4f}")

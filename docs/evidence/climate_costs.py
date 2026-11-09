"""Derived costs for the climate build report: per TB and whole catalogue, from measured rates (sources in comments)."""
from datetime import date

def hours(a, b): return (b - a).days * 24
USD_EECU = 0.40          # Earth Engine pay-as-you-go, first 10,000 EECU-h/month (cloud.google.com/earth-engine/pricing)
USD_VCPU = 0.0446        # AWS c7i on-demand us-east-1 per vCPU-hour (holori aggregator)
CONTRIB = 1000           # EECU-h per month, Contributor tier
TB = 1e12

print("== ERA5 hourly, ARCO-ERA5 (anonymous; free egress)")
f_era5 = 1440 * 721 * 4                                   # float32 field, CLIMATE:239 '~4 MB'
rate = 882144 / (10.75 * 3600)                            # R1024:9
print(f"field {f_era5/1e6:.2f} MB; {rate:.1f} fields/s; {rate*f_era5/1e6:.0f} MB/s field-equivalent; {TB/f_era5/rate/3600:.1f} h per TB")
h = hours(date(1940, 1, 1), date(2026, 7, 1)); n = 262 * h # 262 hourly single-level variables (EV/era5_variables.py:3), final to 2026-06-30 (CLIMATE:26)
print(f"whole catalogue: 262 vars x {h:,} h = {n:,} fields = {n*f_era5/TB:,.0f} TB; {n/rate/86400:,.0f} days at {rate:.0f} fields/s; one variable {h*f_era5/TB:.2f} TB, {h/rate/3600:.0f} h")

print("== ERA5-Land hourly, Earth Engine (stored images)")
s_band_hour = 186 / 11                                     # CLIMATE:244, 11 bands
f_land = 3600 * 1801 * 4
eecu_tb = TB / f_land * s_band_hour / 3600
print(f"{s_band_hour:.1f} EECU-s per band-hour; {eecu_tb:.0f} EECU-h per TB (float32) = ${eecu_tb*USD_EECU:.0f}")
hl = hours(date(1950, 1, 1), date(2026, 10, 1)); nl = 49 * hl
e = nl * s_band_hour / 3600
print(f"whole catalogue 49 vars x {hl:,} h = {nl:,} band-hours, {nl*f_land/TB:,.0f} TB float32: {e:,.0f} EECU-h = {e/CONTRIB:,.0f} Contributor-months, {e/100000:.1f} Partner-months, ~${e*USD_EECU:,.0f} at $0.40 (list tiers lower above 10k)")

print("== ERA5-Land hourly, Earth Data Hub (500k requests/month free per key)")
rps = 330000 / (22.7 * 3600)                               # job 1808, group 1 ~330k requests (CLIMATE:221)
mb_req = 2.0                                               # 1-3 MB per chunk (CLIMATE:193), midpoint
print(f"group 1: {rps:.1f} requests/s, ~{rps*mb_req:.0f} MB/s at {mb_req} MB/request; {TB/(mb_req*1e6)/rps/3600:.0f} h per TB; {TB/(mb_req*1e6):,.0f} requests per TB = {TB/(mb_req*1e6)/5e5:.1f} free key-months")
time_chunks = -(-hl // 24 // 60); space_chunks = -(-1472 // 50) * (3600 // 100)   # 60 days x 5 deg x 10 deg; 1,472 rows (CLIMATE:210)
req = 50 * time_chunks * space_chunks
print(f"whole store: 50 vars x {time_chunks} x {space_chunks} chunks = {req:,} requests (before skipping sea) = {req/5e5:.0f} free key-months; ~{req*mb_req*1e6/TB:.0f} TB compressed; {req/rps/86400:.0f} days at {rps:.1f}/s")

print("== ERA5-Land hourly, CDS (south fill)")
south_tb, south_fields = 1.68, 2404920                       # CLIMATE:212
gd = 2045 * 6
mb_per_gd = south_tb * 1e6 / gd
pace = 520                                                   # group-days per day across 12 jobs, measured 2026-11-06..09
print(f"{mb_per_gd:.0f} MB per group-day; at {pace} group-days/day: {pace*mb_per_gd/1e3:.0f} GB/day, {1e6/(pace*mb_per_gd):.1f} days per TB")
full_field = 0.70 * 1801 / 336                               # MB, scale the south cut to the globe
print(f"global field ~{full_field:.2f} MB; whole catalogue 50 vars x {hl:,} h = {50*hl*full_field/1e6:,.0f} TB = {50*hl*full_field/(pace*mb_per_gd):,.0f} days ({50*hl*full_field/(pace*mb_per_gd)/365:.1f} years) at that pace; {50*hl/1848:,.0f} requests of 1,848 fields")

print("== IMERG half-hourly, Earth Engine")
mb_hh = 1.24e3 / 48; s_hh = 4.85                            # R1024:33; calibration CLIMATE:262
print(f"{mb_hh:.1f} MB per half-hour; per TB: {TB/(mb_hh*1e6)*s_hh/3600:.0f} EECU-h (${TB/(mb_hh*1e6)*s_hh/3600*USD_EECU:.0f}), {TB/12.9e6/3600:.1f} h at 12.9 MB/s")
hh = hours(date(1998, 6, 1), date(2026, 1, 1)) * 2
print(f"whole catalogue {hh:,} half-hours = {hh*mb_hh/1e6:.1f} TB, {hh*s_hh/3600:,.0f} EECU-h ({hh*s_hh/3600/CONTRIB:.1f} Contributor-months, ${hh*s_hh/3600*USD_EECU:,.0f}), {hh*mb_hh/12.9/86400:.1f} days")
print(f"GES DISC instead: {hh*mb_hh/0.09/86400:,.0f} days at 0.09 MB/s")

print("== CAMS EAC4, ADS")
print(f"{TB/0.62e6/86400:.1f} days per TB at 0.62 MB/s (3 in flight); our 19-variable selection 2003-2025: 276 months x 1.25 GB = {276*1.25:.0f} GB, {276*1.25e9/0.62e6/86400:.1f} days")

print("== Sentinel-5P L3, Earth Engine")
d = (date(2026, 1, 1) - date(2018, 6, 28)).days
print(f"{d} days x 7 products x ~50 EECU-s = {d*7*50/3600:.0f} EECU-h (${d*7*50/3600*USD_EECU:.0f})")

print("== Uploads to source.coop")
for name, gb, h in (("Core-Weather", 395.2, 4 + 1/60), ("ERA5-Monthly", 2800, 18.8), ("ERA5 by_cell", 2141, 7.0 + 9.2 + 4.5)):
    print(f"{name}: {gb} GB in {h:.1f} h = {gb*1e3/h/3600:.0f} MB/s, {h/(gb/1000):.1f} h per TB")

print("== Compute on cirrus (Slurm accounting)")
print(f"allocated 11,458 CPU-h = ${11458*USD_VCPU:,.0f} at ${USD_VCPU}/vCPU-h; of which running south fill 3,842")

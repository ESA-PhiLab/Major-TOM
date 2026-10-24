"""All ARCO-ERA5 single-level variables, grouped by theme with a recommendation, as a table for review.

    ARCO-ERA5 metadata (262 hourly single-level variables: name, long name, units)
      + THEMES below (theme -> recommendation -> variables, with a reason per group or variable)
      -> docs/ERA5_VARIABLES.md          fails if any variable is unclassified or classified twice

    python docs/evidence/era5_variables.py
Recommendations: Keep (clearly useful for Major TOM users), Optional (useful for some), Skip (duplicate or model
internal), Static (constant in time: a one-off table per cell, not a time series).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CURRENT = {"2m_temperature", "2m_dewpoint_temperature", "total_precipitation", "10m_u_component_of_wind",
           "10m_v_component_of_wind", "surface_pressure", "surface_solar_radiation_downwards", "snowfall", "snow_depth",
           "volumetric_soil_water_layer_1", "sea_surface_temperature", "sea_ice_cover"}

# theme -> list of (recommendation, reason, [variables])
THEMES: dict[str, list[tuple[str, str, list[str]]]] = {
    "Near-surface temperature and humidity": [
        ("Keep", "basic weather state", ["2m_temperature", "2m_dewpoint_temperature"]),
        ("Keep", "radiative surface temperature, comparable with satellite land surface temperature",
         ["skin_temperature"]),
        ("Keep", "true extremes within each hour (better than min/max of hourly snapshots)",
         ["maximum_2m_temperature_since_previous_post_processing",
          "minimum_2m_temperature_since_previous_post_processing"]),
    ],
    "Wind": [
        ("Keep", "basic wind state", ["10m_u_component_of_wind", "10m_v_component_of_wind"]),
        ("Keep", "hub-height wind (energy, vegetation stress)", ["100m_u_component_of_wind", "100m_v_component_of_wind"]),
        ("Keep", "strongest gust in the hour (storm damage)", ["10m_wind_gust_since_previous_post_processing"]),
        ("Optional", "instantaneous gust at the stamp; the hourly maximum above usually serves better",
         ["instantaneous_10m_wind_gust"]),
        ("Optional", "neutral-stability wind, matches scatterometer and SAR wind retrievals over sea",
         ["10m_u_component_of_neutral_wind", "10m_v_component_of_neutral_wind"]),
        ("Optional", "wave-model wind (ocean only)", ["ocean_surface_stress_equivalent_10m_neutral_wind_speed",
                                                     "ocean_surface_stress_equivalent_10m_neutral_wind_direction"]),
    ],
    "Pressure and atmospheric column": [
        ("Keep", "synoptic situation", ["surface_pressure", "mean_sea_level_pressure"]),
        ("Keep", "atmospheric correction of optical data; radar path delay", ["total_column_water_vapour"]),
        ("Keep", "atmospheric correction (ozone absorption)", ["total_column_ozone"]),
        ("Keep", "mixing height (air quality, haze)", ["boundary_layer_height"]),
        ("Keep", "freezing level (rain vs snow in mountains)", ["zero_degree_level"]),
        ("Keep", "convective potential (storms)", ["convective_available_potential_energy"]),
        ("Optional", "all water in the column (vapour, cloud, precipitation)", ["total_column_water"]),
        ("Optional", "storm indices, overlapping with CAPE", ["convective_inhibition", "k_index", "total_totals_index"]),
        ("Optional", "column moisture convergence (heavy rain setups)", ["vertically_integrated_moisture_divergence"]),
        ("Skip", "ocean-only diagnostics of the wave model", ["air_density_over_the_oceans",
                                                             "free_convective_velocity_over_the_oceans"]),
    ],
    "Clouds": [
        ("Keep", "cloudiness at the time of an optical image", ["total_cloud_cover", "low_cloud_cover",
                                                                "medium_cloud_cover", "high_cloud_cover"]),
        ("Keep", "height of the cloud base", ["cloud_base_height"]),
        ("Keep", "cloud water and ice (cloud thickness, radar and microwave effects)",
         ["total_column_cloud_liquid_water", "total_column_cloud_ice_water"]),
        ("Keep", "supercooled liquid water: marks mixed-phase clouds, which drive precipitation formation and the "
                 "radiation budget, and change optical, thermal and microwave signatures",
         ["total_column_supercooled_liquid_water"]),
        ("Optional", "falling rain and snow in the column", ["total_column_rain_water", "total_column_snow_water"]),
    ],
    "Precipitation": [
        ("Keep", "amount in the hour ending at the stamp", ["total_precipitation"]),
        ("Keep", "rain, snow, freezing rain, ... at the stamp (code table)", ["precipitation_type"]),
        ("Keep", "peak intensity within the hour (flash floods)",
         ["maximum_total_precipitation_rate_since_previous_post_processing"]),
        ("Optional", "convective vs large-scale split of precipitation", ["convective_precipitation",
                                                                         "large_scale_precipitation"]),
        ("Skip", "instantaneous rates and fractions duplicating the accumulations",
         ["convective_rain_rate", "large_scale_rain_rate", "large_scale_precipitation_fraction",
          "instantaneous_large_scale_surface_precipitation_fraction",
          "minimum_total_precipitation_rate_since_previous_post_processing"]),
    ],
    "Snow": [
        ("Keep", "snowfall and snowpack", ["snowfall", "snow_depth", "snow_density", "snow_albedo", "snowmelt"]),
        ("Optional", "sublimation and snowpack temperature", ["snow_evaporation", "temperature_of_snow_layer"]),
        ("Skip", "convective / large-scale split of snowfall and instantaneous rates",
         ["convective_snowfall", "large_scale_snowfall", "convective_snowfall_rate_water_equivalent",
          "large_scale_snowfall_rate_water_equivalent"]),
    ],
    "Radiation at the surface": [
        ("Keep", "incoming sunlight and sky radiation", ["surface_solar_radiation_downwards",
                                                         "surface_thermal_radiation_downwards"]),
        ("Keep", "net radiation (energy balance)", ["surface_net_solar_radiation", "surface_net_thermal_radiation"]),
        ("Keep", "direct beam (shadows, solar energy)", ["total_sky_direct_solar_radiation_at_surface"]),
        ("Keep", "surface albedo used by the model", ["forecast_albedo"]),
        ("Optional", "clear-sky counterparts: the cloud effect is the difference",
         ["surface_solar_radiation_downward_clear_sky", "surface_thermal_radiation_downward_clear_sky",
          "surface_net_solar_radiation_clear_sky", "surface_net_thermal_radiation_clear_sky",
          "clear_sky_direct_solar_radiation_at_surface"]),
        ("Optional", "UV dose", ["downward_uv_radiation_at_the_surface"]),
        ("Optional", "spectral albedos (model background, not observed)",
         ["near_ir_albedo_for_diffuse_radiation", "near_ir_albedo_for_direct_radiation",
          "uv_visible_albedo_for_diffuse_radiation", "uv_visible_albedo_for_direct_radiation"]),
    ],
    "Radiation at the top of the atmosphere": [
        ("Optional", "incoming sunlight (day length, season)", ["toa_incident_solar_radiation"]),
        ("Optional", "outgoing longwave: a proxy for deep convective cloud", ["top_net_thermal_radiation"]),
        ("Skip", "other top-of-atmosphere budget terms", ["top_net_solar_radiation", "top_net_solar_radiation_clear_sky",
                                                          "top_net_thermal_radiation_clear_sky"]),
    ],
    "Surface heat, water and momentum fluxes": [
        ("Keep", "latent and sensible heat (evaporation, heating)", ["surface_latent_heat_flux",
                                                                     "surface_sensible_heat_flux"]),
        ("Keep", "actual and potential evaporation (drought)", ["evaporation", "potential_evaporation"]),
        ("Optional", "wind stress on the surface", ["eastward_turbulent_surface_stress",
                                                    "northward_turbulent_surface_stress", "friction_velocity"]),
        ("Optional", "aerodynamic roughness", ["forecast_surface_roughness"]),
        ("Skip", "instantaneous duplicates of the accumulated fluxes",
         ["instantaneous_surface_sensible_heat_flux", "instantaneous_moisture_flux",
          "instantaneous_eastward_turbulent_surface_stress", "instantaneous_northward_turbulent_surface_stress"]),
        ("Skip", "model momentum and dissipation terms",
         ["eastward_gravity_wave_surface_stress", "northward_gravity_wave_surface_stress", "gravity_wave_dissipation",
          "boundary_layer_dissipation", "forecast_logarithm_of_surface_roughness_for_heat"]),
    ],
    "Soil and runoff": [
        ("Keep", "soil water at 0-7, 7-28, 28-100, 100-289 cm", ["volumetric_soil_water_layer_1",
                                                                "volumetric_soil_water_layer_2",
                                                                "volumetric_soil_water_layer_3",
                                                                "volumetric_soil_water_layer_4"]),
        ("Keep", "soil temperature at the same depths", ["soil_temperature_level_1", "soil_temperature_level_2",
                                                         "soil_temperature_level_3", "soil_temperature_level_4"]),
        ("Keep", "runoff, total and its surface / sub-surface parts", ["runoff", "surface_runoff", "sub_surface_runoff"]),
        ("Optional", "water held on leaves and the surface (interception)", ["skin_reservoir_content"]),
    ],
    "Vegetation": [
        ("Optional", "leaf area index: a fixed seasonal cycle in ERA5, not observed year to year",
         ["leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"]),
    ],
    "Lakes": [
        ("Keep", "lake surface temperature and ice", ["lake_mix_layer_temperature", "lake_ice_depth"]),
        ("Optional", "further lake model state", ["lake_ice_temperature", "lake_total_layer_temperature",
                                                  "lake_mix_layer_depth"]),
        ("Skip", "lake bottom temperature (model internal)", ["lake_bottom_temperature"]),
    ],
    "Sea surface and sea ice": [
        ("Keep", "sea surface temperature and sea ice", ["sea_surface_temperature", "sea_ice_cover"]),
        ("Optional", "sea ice temperature profile", ["ice_temperature_layer_1", "ice_temperature_layer_2",
                                                     "ice_temperature_layer_3", "ice_temperature_layer_4"]),
    ],
    "Ocean waves": [
        ("Keep", "wave height, period and direction (SAR sea clutter, coasts)",
         ["significant_height_of_combined_wind_waves_and_swell", "significant_height_of_wind_waves",
          "significant_height_of_total_swell", "mean_wave_direction", "mean_wave_period"]),
        ("Keep", "sea-surface slope: drives radar backscatter and sun glint in optical images",
         ["mean_square_slope_of_waves"]),
        ("Optional", "peak period, extremes and the wind-wave / swell split",
         ["peak_wave_period", "maximum_individual_wave_height", "mean_direction_of_wind_waves",
          "mean_direction_of_total_swell", "mean_period_of_wind_waves", "mean_period_of_total_swell"]),
        ("Skip", "swell partitions, spectral moments and shape, wave-model coupling terms",
         ["significant_wave_height_of_first_swell_partition", "significant_wave_height_of_second_swell_partition",
          "significant_wave_height_of_third_swell_partition", "mean_wave_direction_of_first_swell_partition",
          "mean_wave_direction_of_second_swell_partition", "mean_wave_direction_of_third_swell_partition",
          "mean_wave_period_of_first_swell_partition", "mean_wave_period_of_second_swell_partition",
          "mean_wave_period_of_third_swell_partition", "mean_wave_period_based_on_first_moment",
          "mean_wave_period_based_on_first_moment_for_swell", "mean_wave_period_based_on_first_moment_for_wind_waves",
          "mean_wave_period_based_on_second_moment_for_swell", "mean_wave_period_based_on_second_moment_for_wind_waves",
          "mean_zero_crossing_wave_period", "period_corresponding_to_maximum_individual_wave_height",
          "wave_spectral_directional_width", "wave_spectral_directional_width_for_swell",
          "wave_spectral_directional_width_for_wind_waves", "wave_spectral_kurtosis", "wave_spectral_peakedness",
          "wave_spectral_skewness", "benjamin_feir_index", "charnock", "coefficient_of_drag_with_waves",
          "normalized_energy_flux_into_ocean", "normalized_energy_flux_into_waves", "normalized_stress_into_ocean",
          "u_component_stokes_drift", "v_component_stokes_drift"]),
    ],
    "Radio propagation (ducting)": [
        ("Skip", "radio refractivity layers", ["duct_base_height", "trapping_layer_base_height",
                                               "trapping_layer_top_height",
                                               "mean_vertical_gradient_of_refractivity_inside_trapping_layer",
                                               "minimum_vertical_gradient_of_refractivity_inside_trapping_layer"]),
    ],
    "Mean rates (duplicates)": [
        ("Skip", "the same quantities as the accumulations above, as hourly mean rates in other units", []),
    ],
    "Vertical integrals": [
        ("Optional", "water vapour transport (atmospheric rivers)", ["vertical_integral_of_eastward_water_vapour_flux",
                                                                    "vertical_integral_of_northward_water_vapour_flux"]),
        ("Skip", "other column budgets of mass, energy, ozone and cloud water", []),
    ],
    "Static fields (one-off table per cell)": [
        ("Static", "terrain and land surface description", ["geopotential_at_surface", "land_sea_mask", "soil_type",
                                                            "type_of_high_vegetation", "type_of_low_vegetation",
                                                            "high_vegetation_cover", "low_vegetation_cover"]),
        ("Static", "sub-grid terrain", ["standard_deviation_of_orography",
                                        "standard_deviation_of_filtered_subgrid_orography",
                                        "slope_of_sub_gridscale_orography", "angle_of_sub_gridscale_orography",
                                        "anisotropy_of_sub_gridscale_orography"]),
        ("Static", "lakes and sea floor", ["lake_cover", "lake_depth", "lake_shape_factor", "model_bathymetry"]),
    ],
}


def classify(names: list[str]) -> dict[str, tuple[str, str, str]]:
    """variable -> (theme, recommendation, reason); the two catch-all groups take mean_* and vertical_integral_*."""
    out = {}
    for theme, groups in THEMES.items():
        for rec, why, variables in groups:
            for v in variables:
                assert v not in out, f"{v} classified twice"
                out[v] = (theme, rec, why)
    for v in names:
        if v not in out and v.startswith("mean_"):
            out[v] = ("Mean rates (duplicates)", "Skip", THEMES["Mean rates (duplicates)"][0][1])
        elif v not in out and v.startswith("vertical_integral_"):
            out[v] = ("Vertical integrals", "Skip", THEMES["Vertical integrals"][1][1])
    missing = [v for v in names if v not in out]
    extra = [v for v in out if v not in names]
    assert not missing and not extra, f"unclassified {missing}, unknown {extra}"
    return out


def main() -> None:
    import gcsfs

    meta = json.loads(gcsfs.GCSFileSystem(token="anon").cat(f"{STORE}/.zmetadata"))["metadata"]
    rows = {}
    for k in meta:
        if k.endswith("/.zarray") and len(meta[k]["shape"]) == 3:
            v = k.split("/")[0]
            attrs = meta.get(f"{v}/.zattrs", {})
            rows[v] = (attrs.get("long_name", ""), attrs.get("units", ""))
    kinds = classify(sorted(rows))
    for v, (theme, rec, why) in kinds.items():                    # principle A (Miko, 2026-10-24): keep all but
        decision = "Static" if rec == "Static" else ("Skip" if theme == "Mean rates (duplicates)" else "Keep")
        kinds[v] = (theme, decision, why if decision != "Skip" else why)
    counts = {r: sum(1 for t in kinds.values() if t[1] == r) for r in ("Keep", "Skip", "Static")}
    lines = ["> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.", "",
             "> **Review status:** decision by Miko, 2026-10-24 (principle A, complete record). Generated by "
             "`docs/evidence/era5_variables.py`; edit there, not here.", "",
             "# ERA5 variables for Core-Weather and ERA5-Monthly", "",
             f"All {len(rows)} hourly single-level variables in ARCO-ERA5, by theme. **Principle A, complete:** keep "
             f"every variable except exact duplicates, because any of them may matter to someone's research. "
             f"Keep {counts['Keep']}, Skip {counts['Skip']} (mean rates: the hourly totals divided by 3,600 s), "
             f"Static {counts['Static']} (fixed in time: one table per cell). ✓ marks the 12 built first.", "",
             "Instantaneous and accumulated versions of a flux are both kept: the first is the value at the time stamp, "
             "the second the total over the hour ending at it. The note column says what each group is for.", ""]
    for theme in THEMES:
        members = sorted((v for v, t in kinds.items() if t[0] == theme),
                         key=lambda v: (["Keep", "Static", "Skip"].index(kinds[v][1]), v))
        lines += [f"## {theme} ({len(members)})", "", "| | Variable | Long name | Units | Decision | Note |",
                  "|---|---|---|---|---|---|"]
        for v in members:
            long_name, units = rows[v]
            lines.append(f"| {'✓' if v in CURRENT else ''} | `{v}` | {long_name} | {units} | {kinds[v][1]} | {kinds[v][2]} |")
        lines.append("")
    (ROOT / "docs" / "ERA5_VARIABLES.md").write_text("\n".join(lines))
    print(counts, "->", ROOT / "docs" / "ERA5_VARIABLES.md")


if __name__ == "__main__":
    main()

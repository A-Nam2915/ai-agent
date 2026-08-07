"""
Generates a synthetic Urban Heat Equity dataset at the census-tract level.
Built so income/impervious-surface/tree-canopy are correlated with land
surface temperature in a realistic (not perfectly linear) way, so the
agent actually has something non-trivial to discover.
"""
import numpy as np
import pandas as pd

def generate(n_tracts=250, seed=42, out_path="urban_heat_sample.csv"):
    rng = np.random.default_rng(seed)

    median_income = rng.lognormal(mean=10.9, sigma=0.45, size=n_tracts).clip(18000, 220000)
    pct_minority = rng.beta(2, 2, size=n_tracts) * 100
    # lower income tracts tend to have less tree canopy, more impervious surface
    income_z = (median_income - median_income.mean()) / median_income.std()

    tree_canopy_pct = (28 + 9 * income_z + rng.normal(0, 6, n_tracts)).clip(2, 65)
    pct_impervious = (48 - 6 * income_z + rng.normal(0, 8, n_tracts)).clip(10, 92)
    population_density = rng.lognormal(mean=8.2, sigma=0.6, size=n_tracts).clip(500, 40000)
    pct_poverty = (22 - 0.00012 * (median_income - median_income.mean()) + rng.normal(0, 5, n_tracts)).clip(1, 55)
    distance_to_park_miles = (1.6 - 0.15 * income_z + rng.exponential(0.4, n_tracts)).clip(0.05, 6)

    # land surface temperature: driven by impervious surface + inverse tree canopy + density, plus noise
    land_surface_temp_f = (
        88
        + 0.14 * pct_impervious
        - 0.22 * tree_canopy_pct
        + 0.00008 * population_density
        + rng.normal(0, 2.2, n_tracts)
    ).clip(78, 112)

    lat = 30.25 + rng.normal(0, 0.08, n_tracts)   # roughly Austin, TX area
    lon = -97.75 + rng.normal(0, 0.08, n_tracts)

    df = pd.DataFrame({
        "tract_id": [f"T{i:04d}" for i in range(n_tracts)],
        "latitude": lat.round(5),
        "longitude": lon.round(5),
        "land_surface_temp_f": land_surface_temp_f.round(1),
        "tree_canopy_pct": tree_canopy_pct.round(1),
        "pct_impervious_surface": pct_impervious.round(1),
        "median_household_income": median_income.round(0).astype(int),
        "pct_minority": pct_minority.round(1),
        "pct_poverty": pct_poverty.round(1),
        "population_density": population_density.round(0).astype(int),
        "distance_to_park_miles": distance_to_park_miles.round(2),
    })
    df.to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    path = generate()
    print(f"Wrote {path}")
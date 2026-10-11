*******************************************************
* run_fua_resilience_regressions.do
* Baseline regressions for urban resilience analysis
* Author: <your name>, Date: <YYYY-MM-DD>
*******************************************************

version 17.0
clear all
set more off


* -------------------- Data import ------------------ *
* Expecting a wide panel with eFUA-level covariates
import delimited using "`infile'", clear varn(1) stringcols(_all)

* -------------------- Basic cleaning --------------- *
* Ensure expected variable names are lowercase
rename *, lower

* Drop extreme outcomes (consistent with prior logic)
drop if resistance >  3 | resistance < -3
drop if recoverability >  3 | recoverability < -3

* Encode country code for clustering (cntry_iso is typically string)
capture confirm variable cntry_iso
if _rc != 0 {
    di as error "Variable cntry_iso not found. Please ensure country ISO3 is present."
    exit 198
}
encode cntry_iso, gen(country_id)

* -------------------- Standardized variables ------- *
* Create z-scores for outcomes and key predictors
egen resistance_std      = std(resistance)
egen recoverability_std  = std(recoverability)
egen resilience_std      = std(resilience)
egen instability_std     = std(instability_count)

egen highways_std        = std(mean_road_density_highways)
egen local_roads_std     = std(mean_road_density_local_roads)
egen total_std           = std(mean_road_density_total)
egen nightlight_std      = std(nightlight_mean)
egen pop_mean_std        = std(pop_mean)
egen old_pop_std         = std(old_pop)
egen distance_std        = std(nearest_coastline_distance)
egen stringency_std      = std(stringency_index)

* Optional controls if present; silently skip if missing
capture confirm variable rd_mean
if _rc==0 egen rd_mean_std = std(rd_mean)
capture confirm variable trading
if _rc==0 egen trading_std = std(trading)
capture confirm variable idv
if _rc==0 egen idv_std = std(idv)

* Interaction example used in prior drafts
capture gen night_stringency = nightlight_mean * stringency_index

* -------------------- Output helper ---------------- *
* Install outreg2 if not available (silently)
cap which outreg2
if _rc ssc install outreg2, replace

* Remove old output to avoid unintended appends
cap erase "`outfile'"

* -------------------- Regressions ------------------ *
* Sanity checks (simple bivariate)
regress resistance_std  distance_std, vce(cluster country_id)
regress resilience_std  nightlight_std, vce(cluster country_id)

* Model 1: Resistance
regress resistance_std  ///
    highways_std local_roads_std nightlight_std pop_mean_std old_pop_std ///
    distance_std stringency_std, vce(cluster country_id)
outreg2 using "`outfile'", replace ctitle("Model 1: Resistance") se p

* Model 2: Recoverability
regress recoverability_std ///
    highways_std local_roads_std nightlight_std pop_mean_std old_pop_std ///
    distance_std stringency_std, vce(cluster country_id)
outreg2 using "`outfile'", append ctitle("Model 2: Recoverability") se p

* Model 3: Resilience
regress resilience_std ///
    highways_std local_roads_std nightlight_std pop_mean_std old_pop_std ///
    distance_std stringency_std, vce(cluster country_id)
outreg2 using "`outfile'", append ctitle("Model 3: Resilience") se p

* Model 4: Instability
regress instability_std ///
    highways_std local_roads_std nightlight_std pop_mean_std old_pop_std ///
    distance_std stringency_std, vce(cluster country_id)
outreg2 using "`outfile'", append ctitle("Model 4: Instability") se p

di as result "All models estimated. Results saved to: `outfile'"

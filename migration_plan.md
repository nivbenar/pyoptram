# rOPTRAM → Python migration plan

## Goal
Convert the rOPTRAM R package into a Python module with similar functionality and workflow.

---

## Public functions (R)

- optram()
- optram_options()
- optram_acquire_s2()
- optram_calculate_str()
- optram_ndvi_str()
- optram_wetdry_coefficients()
- optram_calculate_soil_moisture()
- optram_landsat()
- optram_safe()

---

## Workflow

optram()  # main wrapper
 -> acquire Sentinel-2
 -> calculate VI + STR
 -> build dataframe
 -> fit trapezoid
 -> calculate soil moisture / save coefficients

---

## Notes

- `optram()` is the main wrapper function.
- It connects the full workflow but does not implement the scientific calculations itself.
- Start from the scientific core, not from the wrapper or API.

---

## Initial R → Python Mapping

- optram() -> OPTRAMModel.fit() / run_optram()
- optram_options() -> OPTRAMConfig
- optram_acquire_s2() -> acquire_sentinel2()
- optram_calculate_str() -> calculate_str()
- optram_ndvi_str() -> build_vi_str_dataframe()
- optram_wetdry_coefficients() -> fit_trapezoid()
- optram_calculate_soil_moisture() -> calculate_soil_moisture()
- optram_landsat() -> run_landsat_optram()
- optram_safe() -> run_safe_optram()

---

## My work plan:
### Phase 1 - Understanding 
- [ ] Analyze `optram()`
- [ ] Understand the full workflow
- [ ] Identify that `optram()` is a wrapper

### Phase 2 - Core Implementation (CURRENT)
- [ ] calculate_str()
- [ ] calculate_vi()
- [ ] build_vi_str_dataframe()

### Phase 3 - Modeling
- [ ] fit_trapezoid()
- [ ] calculate_soil_moisture()

### Phase 4 - Integration
- [ ] OPTRAMModel / wrapper
- [ ] package structure

### Phase 5 - Data Acquisition
- [ ] Sentinel-2 API (CDSE)
- [ ] credentials handling
- [ ] local SAFE workflow

from pyoptram.acquire import load_evalscript


def test_load_evalscript_includes_scl():
    script = load_evalscript("SCL")
    assert "bands: [\"SCL\"]" in script
    assert "sampleType: \"UINT8\"" in script


def test_load_evalscript_can_mask_ndvi_with_scl():
    script = load_evalscript("NDVI", scm_mask=True)
    assert "bands: [\"B04\", \"B08\", \"SCL\"]" in script
    assert "[2, 4, 5, 10].includes(sample.SCL)" in script
    assert "return [NaN]" in script

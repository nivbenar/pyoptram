from pathlib import Path
import json

import requests


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def get_cdse_token(client_id, client_secret):
    """
    Get an access token from Copernicus Data Space.
    """
    response = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def load_aoi(aoi):
    """
    Convert AOI input into a GeoJSON geometry.

    Supported inputs:
    - GeoJSON geometry dict
    - path to a .geojson file
    - bbox tuple: (minx, miny, maxx, maxy)
    """
    if isinstance(aoi, dict):
        if "type" not in aoi:
            raise ValueError("AOI dictionary must contain a GeoJSON 'type'.")
        if aoi["type"] == "Feature":
            return aoi["geometry"]
        if aoi["type"] == "FeatureCollection":
            return aoi["features"][0]["geometry"]
        return aoi

    if isinstance(aoi, (str, Path)):
        path = Path(aoi)
        if not path.exists():
            raise FileNotFoundError(f"AOI file not found: {path}")

        if path.suffix.lower() != ".geojson":
            raise ValueError("Only .geojson files are supported for AOI paths.")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data["type"] == "FeatureCollection":
            return data["features"][0]["geometry"]
        if data["type"] == "Feature":
            return data["geometry"]
        return data

    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        minx, miny, maxx, maxy = aoi
        return {
            "type": "Polygon",
            "coordinates": [[
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]],
        }

    raise ValueError(
        "AOI must be a GeoJSON dict, a .geojson file path, or a bbox tuple "
        "(minx, miny, maxx, maxy)."
    )


def prepare_output_folders(output_dir, veg_index="NDVI", only_vi_str=False):
    """
    Create output folders for BOA, vegetation index, and STR.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = {
        "vi": output_dir / veg_index,
        "str": output_dir / "STR",
    }

    if not only_vi_str:
        folders["boa"] = output_dir / "BOA"

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def load_evalscript(script_name, swir_band=11):
    """
    Return the evalscript text for the requested product.
    """
    if script_name == "NDVI":
        return """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B04", "B08"] }],
        output: { bands: 1, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    return [ndvi];
}
""".strip()

    if script_name == "STR":
        band = f"B{swir_band}"
        return f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{ bands: ["{band}"], units: "DN" }}],
        output: {{ bands: 1, sampleType: "FLOAT32" }}
    }};
}}
function evaluatePixel(sample) {{
    let value = sample.{band};
    if (value !== 0) {{
        let v = value / 10000.0;
        let str_value = ((1 - v) ** 2) / (2 * v);
        return [str_value];
    }}
    return [0];
}}
""".strip()

    if script_name == "BOA":
        return """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B01","B02","B03","B04","B05","B06",
                    "B07","B08","B8A","B09","B11","B12"],
            units: "DN"
        }],
        output: { bands: 12, sampleType: "UINT16" }
    };
}
function evaluatePixel(sample) {
    return [
        sample.B01, sample.B02, sample.B03, sample.B04,
        sample.B05, sample.B06, sample.B07, sample.B08,
        sample.B8A, sample.B09, sample.B11, sample.B12
    ];
}
""".strip()

    raise ValueError(f"Unknown script_name: {script_name}")


def search_catalog(aoi_geometry, from_date, to_date, token, max_cloud=20, limit=20):
    """
    Search Sentinel-2 L2A scenes that intersect the AOI.
    """
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "intersects": aoi_geometry,
        "datetime": f"{from_date}T00:00:00Z/{to_date}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": limit,
        "filter": f"eo:cloud_cover <= {max_cloud}",
    }

    response = requests.post(CATALOG_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json().get("features", [])


def download_index(
    aoi_geometry,
    scene_datetime,
    script_name,
    output_path,
    token,
    swir_band=11,
):
    """
    Request one raster product from the Process API and save it to disk.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff",
    }

    payload = {
        "input": {
            "bounds": {"geometry": aoi_geometry},
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": scene_datetime,
                            "to": scene_datetime,
                        }
                    },
                }
            ],
        },
        "output": {
            "width": 2500,
            "height": 2500,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": load_evalscript(script_name, swir_band=swir_band),
    }

    response = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=180)
    response.raise_for_status()

    output_path = Path(output_path)
    output_path.write_bytes(response.content)
    return str(output_path)


def acquire_optram_inputs(
    aoi,
    from_date,
    to_date,
    output_dir,
    client_id,
    client_secret,
    veg_index="NDVI",
    swir_band=11,
    max_cloud=20,
    only_vi_str=True,
):
    """
    Download OPTRAM input rasters from Copernicus Data Space.
    """
    if swir_band not in (11, 12):
        raise ValueError("swir_band must be 11 or 12")

    aoi_geometry = load_aoi(aoi)

    token = get_cdse_token(client_id, client_secret)
    folders = prepare_output_folders(
        output_dir,
        veg_index=veg_index,
        only_vi_str=only_vi_str,
    )

    scenes = search_catalog(
        aoi_geometry=aoi_geometry,
        from_date=from_date,
        to_date=to_date,
        token=token,
        max_cloud=max_cloud,
    )

    if not scenes:
        return {"NDVI": [], "STR": [], "BOA": []}

    results = {"NDVI": [], "STR": [], "BOA": []}

    for scene in scenes:
        scene_id = scene["id"]
        scene_datetime = scene["properties"]["datetime"]
        safe_time = scene_datetime.replace(":", "-")

        ndvi_path = folders["vi"] / f"{veg_index}_{safe_time}_{scene_id}.tif"
        str_path = folders["str"] / f"STR_{safe_time}_{scene_id}.tif"

        results["NDVI"].append(
            download_index(
                aoi_geometry=aoi_geometry,
                scene_datetime=scene_datetime,
                script_name="NDVI",
                output_path=ndvi_path,
                token=token,
                swir_band=swir_band,
            )
        )

        results["STR"].append(
            download_index(
                aoi_geometry=aoi_geometry,
                scene_datetime=scene_datetime,
                script_name="STR",
                output_path=str_path,
                token=token,
                swir_band=swir_band,
            )
        )

        if not only_vi_str and "boa" in folders:
            boa_path = folders["boa"] / f"BOA_{safe_time}_{scene_id}.tif"
            results["BOA"].append(
                download_index(
                    aoi_geometry=aoi_geometry,
                    scene_datetime=scene_datetime,
                    script_name="BOA",
                    output_path=boa_path,
                    token=token,
                    swir_band=swir_band,
                )
            )

    return results

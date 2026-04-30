from datetime import datetime
from pathlib import Path
import json

import requests


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
DEFAULT_MAX_SIZE = 2500


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


def validate_date(date_text, name):
    """
    Validate an ISO date string and return it unchanged.
    """
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be formatted as YYYY-MM-DD") from exc

    return date_text


def _geometry_from_vector_file(path):
    """
    Read the first geometry from a vector file using geopandas, when available.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "Reading this AOI file type requires geopandas. Install geopandas "
            "or pass a GeoJSON file, GeoJSON dictionary, or bbox tuple."
        ) from exc

    data = gpd.read_file(path)

    if data.empty:
        raise ValueError(f"AOI file has no features: {path}")

    if data.crs is not None and data.crs.to_epsg() != 4326:
        data = data.to_crs(4326)

    geometry = data.geometry.unary_union
    return json.loads(gpd.GeoSeries([geometry], crs="EPSG:4326").to_json())[
        "features"
    ][0]["geometry"]


def load_aoi(aoi):
    """
    Convert AOI input into a GeoJSON geometry.

    Supported inputs:
    - GeoJSON geometry, Feature, or FeatureCollection dict
    - path to a .geojson/.json file
    - path to another vector file readable by geopandas, for example .gpkg
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

        if path.suffix.lower() in (".geojson", ".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data["type"] == "FeatureCollection":
                return data["features"][0]["geometry"]
            if data["type"] == "Feature":
                return data["geometry"]
            return data

        return _geometry_from_vector_file(path)

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
        "AOI must be a GeoJSON dict, a vector file path, or a bbox tuple "
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


def _scene_tile(scene):
    """
    Return a Sentinel-2 MGRS tile id from common STAC property names.
    """
    properties = scene.get("properties", {})

    for key in ("s2:mgrs_tile", "grid:code"):
        value = properties.get(key)
        if value:
            return str(value).replace("MGRS-", "").upper()

    scene_id = scene.get("id", "")
    parts = scene_id.split("_")
    for part in parts:
        if part.startswith("T") and len(part) == 6:
            return part[1:].upper()

    return None


def _filter_scenes_by_tile(scenes, tile):
    if tile is None:
        return scenes

    wanted = str(tile).upper().removeprefix("T")
    return [scene for scene in scenes if _scene_tile(scene) == wanted]


def search_catalog(
    aoi_geometry,
    from_date,
    to_date,
    token,
    max_cloud=20,
    limit=20,
    tile=None,
):
    """
    Search Sentinel-2 L2A scenes that intersect the AOI.
    """
    headers = {"Authorization": f"Bearer {token}"}
    request_limit = max(limit, 100) if tile is not None else limit

    payload = {
        "intersects": aoi_geometry,
        "datetime": f"{from_date}T00:00:00Z/{to_date}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": request_limit,
        "query": {"eo:cloud_cover": {"lte": max_cloud}},
    }

    response = requests.post(CATALOG_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    scenes = response.json().get("features", [])
    return _filter_scenes_by_tile(scenes, tile)[:limit]


def download_index(
    aoi_geometry,
    scene_datetime,
    script_name,
    output_path,
    token,
    swir_band=11,
    width=DEFAULT_MAX_SIZE,
    height=DEFAULT_MAX_SIZE,
    overwrite=False,
):
    """
    Request one raster product from the Process API and save it to disk.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": load_evalscript(script_name, swir_band=swir_band),
    }

    response = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=180)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    return str(output_path)


def _scene_record(scene, ndvi_path, str_path, boa_path=None):
    properties = scene.get("properties", {})
    return {
        "id": scene.get("id"),
        "datetime": properties.get("datetime"),
        "cloud_cover": properties.get("eo:cloud_cover"),
        "tile": _scene_tile(scene),
        "NDVI": ndvi_path,
        "STR": str_path,
        "BOA": boa_path,
    }


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
    tile=None,
    limit=20,
    width=DEFAULT_MAX_SIZE,
    height=DEFAULT_MAX_SIZE,
    overwrite=False,
):
    """
    Download OPTRAM input rasters from Copernicus Data Space.
    """
    if swir_band not in (11, 12):
        raise ValueError("swir_band must be 11 or 12")

    if veg_index != "NDVI":
        raise NotImplementedError("Only NDVI is implemented at the moment")

    if width > DEFAULT_MAX_SIZE or height > DEFAULT_MAX_SIZE:
        raise ValueError("width and height cannot exceed 2500 pixels")

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive integers")

    from_date = validate_date(from_date, "from_date")
    to_date = validate_date(to_date, "to_date")

    if from_date > to_date:
        raise ValueError("from_date must be earlier than or equal to to_date")

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
        limit=limit,
        tile=tile,
    )

    if not scenes:
        return {"NDVI": [], "STR": [], "BOA": [], "scenes": []}

    results = {"NDVI": [], "STR": [], "BOA": [], "scenes": []}

    for scene in scenes:
        scene_id = scene["id"]
        scene_datetime = scene["properties"]["datetime"]
        safe_time = scene_datetime.replace(":", "-").replace("/", "-")

        ndvi_path = folders["vi"] / f"{veg_index}_{safe_time}_{scene_id}.tif"
        str_path = folders["str"] / f"STR_{safe_time}_{scene_id}.tif"

        ndvi_file = download_index(
            aoi_geometry=aoi_geometry,
            scene_datetime=scene_datetime,
            script_name="NDVI",
            output_path=ndvi_path,
            token=token,
            swir_band=swir_band,
            width=width,
            height=height,
            overwrite=overwrite,
        )

        str_file = download_index(
            aoi_geometry=aoi_geometry,
            scene_datetime=scene_datetime,
            script_name="STR",
            output_path=str_path,
            token=token,
            swir_band=swir_band,
            width=width,
            height=height,
            overwrite=overwrite,
        )

        boa_file = None

        if not only_vi_str and "boa" in folders:
            boa_path = folders["boa"] / f"BOA_{safe_time}_{scene_id}.tif"
            boa_file = download_index(
                aoi_geometry=aoi_geometry,
                scene_datetime=scene_datetime,
                script_name="BOA",
                output_path=boa_path,
                token=token,
                swir_band=swir_band,
                width=width,
                height=height,
                overwrite=overwrite,
            )

        results["NDVI"].append(ndvi_file)
        results["STR"].append(str_file)
        if boa_file is not None:
            results["BOA"].append(boa_file)

        results["scenes"].append(_scene_record(scene, ndvi_file, str_file, boa_file))

    return results

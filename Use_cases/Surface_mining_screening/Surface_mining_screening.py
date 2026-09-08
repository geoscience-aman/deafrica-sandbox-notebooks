"""
Surface mining screening — STAC-only build.

Runs outside the DE Africa Sandbox. All data is loaded through STAC endpoints
via odc-stac, so no Open Data Cube index or database connection is required.

Data sources
------------
Sentinel-1 RTC   Microsoft Planetary Computer  (sentinel-1-rtc)
Sentinel-2 GeoMAD  DE Africa Explorer          (gm_s2_annual, gm_s2_semiannual)
WOfS annual summary  DE Africa Explorer        (wofs_ls_summary_annual)

deafrica_tools is used for band indices, rasterisation and RGB plotting. Those
functions do not touch the ODC index.
"""

import calendar
import gc
import os
import tempfile
import time
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, Point, Polygon, box

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import ipywidgets as widgets
from IPython.display import Markdown, clear_output, display

import odc.stac
import planetary_computer
import pystac_client
from odc.geo.geom import Geometry
from odc.stac import configure_rio

from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk

from deafrica_tools.bandindices import calculate_indices, dualpol_indices
from deafrica_tools.plotting import rgb
from deafrica_tools.spatial import xr_rasterize


# =============================================================================
# Endpoints and STAC configuration
# =============================================================================

DEAFRICA_STAC = "https://explorer.digitalearth.africa/stac"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

STAC_TIMEOUT = 600          # seconds; without this a blocked host hangs forever
STAC_PAGE_LIMIT = 1000     # ODC-backed STAC APIs truncate to 20 without this

S2_MINING_PRODUCTS = {"s2", "s2_semiannual", "s2_imagery"}

# STAC items do not always carry dtype/nodata, so supply them explicitly.
WOFS_CFG = {
    "wofs_ls_summary_annual": {
        "assets": {
            "frequency": {"data_type": "float32", "nodata": float("nan")},
            "count_wet": {"data_type": "int16", "nodata": -999},
            "count_clear": {"data_type": "int16", "nodata": -999},
        }
    }
}

GEOMAD_CFG = {
    "*": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0},
        }
    }
}


def _stamp(msg):
    """Flushed print. Buffered stdout inside widgets.Output looks like a hang."""
    print(msg, flush=True)


def _configure_deafrica_rio():
    """Point rasterio at DE Africa's public bucket in af-south-1."""
    configure_rio(
        cloud_defaults=True,
        aws={"aws_unsigned": True},
        AWS_S3_ENDPOINT="s3.af-south-1.amazonaws.com",
    )


def _open_catalog(url, sign=False):
    """Open a STAC catalogue with a timeout so unreachable hosts raise."""
    kwargs = {"timeout": STAC_TIMEOUT}
    if sign:
        kwargs["modifier"] = planetary_computer.sign_inplace
    return pystac_client.Client.open(url, **kwargs)


def _search_items(catalog, collection, bbox, start_date, end_date, label):
    items = list(
        catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            limit=STAC_PAGE_LIMIT,
        ).item_collection()
    )
    _stamp(f"{label}: {len(items)} STAC items")
    if not items:
        raise ValueError(
            f"No {collection} items found for this AOI and date range. "
            f"Check the AOI location and that the product covers these years."
        )
    return items


# =============================================================================
# STAC loaders
# =============================================================================

def load_s1_stac(
    bbox,
    start_date,
    end_date,
    resolution=20,
    output_crs="EPSG:6933",
    dask_chunks=None,
):
    """
    Load Sentinel-1 RTC gamma-0 VV/VH from Planetary Computer.

    odc.stac.load reprojects and clips lazily onto one common grid, so scenes
    in different UTM zones combine correctly without a full-scene read.

    Note: the RTC assets live on sentinel1euwestrtc.blob.core.windows.net.
    That host is not reachable from inside the DE Africa Sandbox.
    """
    if dask_chunks is None:
        dask_chunks = {"time": 1, "x": 512, "y": 512}

    catalog = _open_catalog(PC_STAC, sign=True)
    items = _search_items(
        catalog, "sentinel-1-rtc", bbox, start_date, end_date, "Sentinel-1 RTC"
    )

    res = abs(resolution[1]) if isinstance(resolution, (tuple, list)) else abs(resolution)

    return odc.stac.load(
        items,
        bands=["vv", "vh"],
        bbox=bbox,
        crs=output_crs,
        resolution=res,
        groupby="solar_day",
        chunks=dask_chunks,
        dtype="float32",
    )


def load_geomad_stac(
    collection,
    bbox,
    start_date,
    end_date,
    resolution=30,
    output_crs="EPSG:6933",
    bands=("red", "green", "blue", "nir"),
    dask_chunks=None,
):
    """
    Load a DE Africa Sentinel-2 GeoMAD composite via STAC.

    collection is "gm_s2_annual" or "gm_s2_semiannual".
    """
    if dask_chunks is None:
        dask_chunks = {"time": 1, "x": 512, "y": 512}

    _configure_deafrica_rio()

    catalog = _open_catalog(DEAFRICA_STAC)
    items = _search_items(catalog, collection, bbox, start_date, end_date, collection)

    res = abs(resolution[1]) if isinstance(resolution, (tuple, list)) else abs(resolution)

    return odc.stac.load(
        items,
        bands=list(bands),
        bbox=bbox,
        crs=output_crs,
        resolution=res,
        groupby="solar_day",
        chunks=dask_chunks,
        stac_cfg=GEOMAD_CFG,
    )


def load_wofs_stac(geobox, start_date, end_date, bands=("frequency",), dask_chunks=None):
    """
    Load the DE Africa WOfS annual summary onto an existing geobox.

    Passing geobox= aligns the output exactly to the imagery grid, so no
    post-hoc reprojection is needed and nothing is materialised at full
    resolution first.
    """
    if dask_chunks is None:
        dask_chunks = {"time": 1, "x": 512, "y": 512}

    _configure_deafrica_rio()

    catalog = _open_catalog(DEAFRICA_STAC)
    bbox = list(geobox.geographic_extent.boundingbox)
    items = _search_items(
        catalog, "wofs_ls_summary_annual", bbox, start_date, end_date, "WOfS annual"
    )

    return odc.stac.load(
        items,
        bands=list(bands),
        geobox=geobox,
        resampling="nearest",
        groupby="solar_day",
        chunks=dask_chunks,
        stac_cfg=WOFS_CFG,
    )


# =============================================================================
# AOI helpers
# =============================================================================

def create_aoi_from_latlon(lat, lon, buffer_km=5, output_crs="EPSG:6933"):
    """Circular AOI around a point, returned in EPSG:4326."""
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    if buffer_km <= 0:
        raise ValueError("Buffer distance must be greater than 0 km.")

    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    buffered = point_gdf.to_crs(output_crs).buffer(buffer_km * 1000)
    return gpd.GeoDataFrame(geometry=buffered, crs=output_crs).to_crs("EPSG:4326")


def create_rectangular_aoi_from_latlon(
    lat, lon, width_km, height_km, output_crs="EPSG:6933"
):
    """Rectangular (bounding-box style) AOI around a point, in EPSG:4326."""
    if width_km <= 0 or height_km <= 0:
        raise ValueError("Rectangle width and height must be greater than 0 km.")

    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    projected = point_gdf.to_crs(output_crs)
    cx = float(projected.geometry.iloc[0].x)
    cy = float(projected.geometry.iloc[0].y)
    half_w = float(width_km) * 1000.0 / 2.0
    half_h = float(height_km) * 1000.0 / 2.0
    rect = box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    return gpd.GeoDataFrame(geometry=[rect], crs=output_crs).to_crs("EPSG:4326")


def convert_3D_polygon_to_2D(poly_3D):
    exterior = [(x, y) for x, y, *_ in poly_3D.exterior.coords]
    interiors = [[(x, y) for x, y, *_ in ring.coords] for ring in poly_3D.interiors]
    return Polygon(exterior, interiors)


def convert_3D_geometry_to_2D(geom_3D):
    if geom_3D.geom_type == "Polygon":
        return convert_3D_polygon_to_2D(geom_3D)
    if geom_3D.geom_type == "MultiPolygon":
        return MultiPolygon([convert_3D_polygon_to_2D(p) for p in geom_3D.geoms])
    return geom_3D


def load_vector_file(vector_file):
    """Read an AOI vector file and return (GeoDataFrame, odc Geometry)."""
    extension = str(vector_file).split(".")[-1].lower()
    if extension == "kml":
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(vector_file, driver="KML")
    else:
        gdf = gpd.read_file(vector_file)

    if gdf.empty:
        raise ValueError("AOI vector is empty.")
    if gdf.crs is None:
        raise ValueError("AOI vector has no CRS. Define the projection first.")

    gdf["geometry"] = gdf["geometry"].apply(convert_3D_geometry_to_2D)
    geom = Geometry(gdf.union_all(), gdf.crs)
    return gdf, geom


def _save_aoi_gdf_to_temp_file(aoi_gdf, prefix="interactive_aoi"):
    temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
    out_path = Path(temp_dir) / "aoi.geojson"
    aoi_gdf.to_file(out_path, driver="GeoJSON")
    return str(out_path)


def _estimate_aoi_pixels(aoi_gdf, resolution_m=30):
    """Return (area_km2, pixels_per_layer) for the AOI at a given resolution."""
    if aoi_gdf.crs is None:
        raise ValueError("AOI has no CRS.")
    gdf_proj = aoi_gdf.to_crs("EPSG:6933")
    area_m2 = float(gdf_proj.geometry.area.sum())
    return area_m2 / 1_000_000.0, int(area_m2 / float(resolution_m**2))


def _count_years_from_dates(start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    return int(end_dt.year - start_dt.year + 1)


# =============================================================================
# Shapefile upload helpers
# =============================================================================

def _get_uploaded_file(upload_widget):
    """Extract (name, content) from an ipywidgets FileUpload, v7 or v8 format."""
    if not upload_widget.value:
        raise ValueError("No shapefile uploaded.")

    value = upload_widget.value

    if isinstance(value, dict):
        uploaded = list(value.values())[0]
        file_name = uploaded.get("metadata", {}).get("name")
        content = uploaded.get("content")
    elif isinstance(value, (tuple, list)):
        uploaded = value[0]
        file_name = uploaded.get("name") or uploaded.get("metadata", {}).get("name")
        content = uploaded.get("content")
    else:
        raise ValueError("Unsupported upload format from FileUpload widget.")

    if file_name is None or content is None:
        raise ValueError("Could not read uploaded file name or content.")

    return file_name, content


def read_uploaded_shapefile(upload_widget):
    """Read a zipped shapefile from a FileUpload widget into EPSG:4326."""
    file_name, content = _get_uploaded_file(upload_widget)

    if not file_name.lower().endswith(".zip"):
        raise ValueError("Please upload the shapefile as a zipped .zip file.")

    temp_dir = tempfile.mkdtemp(prefix="uploaded_aoi_")
    zip_path = os.path.join(temp_dir, file_name)

    with open(zip_path, "wb") as f:
        f.write(content)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    shp_files = list(Path(temp_dir).rglob("*.shp"))
    if not shp_files:
        raise ValueError("No .shp file found inside the uploaded zip.")

    gdf = gpd.read_file(shp_files[0])
    if gdf.empty:
        raise ValueError("The uploaded shapefile is empty.")
    if gdf.crs is None:
        raise ValueError("The shapefile has no CRS. Define its projection first.")

    return gdf.to_crs("EPSG:4326")


# =============================================================================
# Compositing
# =============================================================================

def get_sentinel_stat_options():
    return [
        "Median",
        "Mean",
        "Minimum",
        "Maximum",
        "Standard deviation",
        "Geomedian",
    ]


def _reduce_time_by_statistic(ds, statistic):
    """
    Reduce a time-series Dataset with the selected statistic.

    Rechunks time into a single block first. Dask's nanmedian and friends need
    the whole reduction axis in one chunk, and with chunks={"time": 1} the
    implicit rechunk builds an enormous graph.
    """
    statistic = str(statistic or "Median").strip()

    if "time" in ds.dims:
        ds = ds.chunk({"time": -1})

    if statistic == "Median":
        return ds.median(dim="time", skipna=True)
    if statistic == "Mean":
        return ds.mean(dim="time", skipna=True)
    if statistic == "Minimum":
        return ds.min(dim="time", skipna=True)
    if statistic == "Maximum":
        return ds.max(dim="time", skipna=True)
    if statistic == "Standard deviation":
        return ds.std(dim="time", skipna=True)
    if statistic == "Geomedian":
        try:
            from odc.algo import xr_geomedian
        except ImportError as exc:
            raise ImportError("Geomedian requires odc.algo.xr_geomedian.") from exc
        return xr_geomedian(ds)

    raise ValueError(f"Unsupported statistic: {statistic}")


def _annual_composite_by_statistic(ds, statistic):
    """One composite per calendar year, preserving a yearly time dimension."""
    composites = []
    for year_value, yearly_ds in ds.groupby("time.year"):
        comp = _reduce_time_by_statistic(yearly_ds, statistic)
        comp = comp.expand_dims(time=[pd.Timestamp(f"{int(year_value)}-12-31")])
        composites.append(comp)

    if not composites:
        raise ValueError("No imagery was loaded for the selected date range.")

    return xr.concat(composites, dim="time")


# =============================================================================
# Data loading
# =============================================================================

def process_data(
    gdf,
    geom,
    start_date,
    end_date,
    product="s2_semiannual",
    output_crs="EPSG:6933",
    sentinel_statistic="Median",
    resolution_m=30,
    dask_chunks=None,
    keep_rgb_bands=True,
):
    """
    Load imagery and the WOfS water summary for the AOI, entirely via STAC.

    Products
    --------
    s2_semiannual  Sentinel-2 semi-annual GeoMAD
    s2             Sentinel-2 annual GeoMAD
    s1             Sentinel-1 RTC, composited annually after computing RVI

    Set keep_rgb_bands=False to drop red/green/blue and keep only NDVI. That
    cuts the cube by roughly 80% when no RGB plots are needed.
    """
    if dask_chunks is None:
        dask_chunks = {"time": 1, "x": 512, "y": 512}

    bbox = list(geom.to_crs("EPSG:4326").boundingbox)

    if product == "s2_semiannual":
        ds = load_geomad_stac(
            "gm_s2_semiannual", bbox, start_date, end_date,
            resolution=resolution_m, output_crs=output_crs, dask_chunks=dask_chunks,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")

    elif product == "s2":
        ds = load_geomad_stac(
            "gm_s2_annual", bbox, start_date, end_date,
            resolution=resolution_m, output_crs=output_crs, dask_chunks=dask_chunks,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")

    elif product == "s1":
        ds = load_s1_stac(
            bbox, start_date, end_date,
            resolution=resolution_m, output_crs=output_crs, dask_chunks=dask_chunks,
        )
        ds["vh/vv"] = ds.vh / ds.vv
        ds = dualpol_indices(ds, index="RVI")
        ds = _annual_composite_by_statistic(ds, sentinel_statistic)

    else:
        raise ValueError("product must be 's2_semiannual', 's2', or 's1'")

    _stamp(f"Imagery: {ds.sizes.get('time', 0)} timesteps, "
           f"{ds.sizes.get('y', 0)} x {ds.sizes.get('x', 0)} pixels")

    if not keep_rgb_bands and product in S2_MINING_PRODUCTS:
        ds = ds[["NDVI"]]

    ds_wofs = load_wofs_stac(
        geobox=ds.odc.geobox,
        start_date=start_date,
        end_date=end_date,
        dask_chunks=dask_chunks,
    ).frequency
    _stamp(f"WOfS: {ds_wofs.sizes.get('time', 0)} annual summaries")

    # Rasterise the AOI onto the analysis grid.
    mask = xr_rasterize(gdf, ds).astype(bool)

    # Cast to float32 before masking. .where() promotes integers to float64,
    # which quadruples the size of the largest object in the pipeline.
    ds = ds.astype("float32").where(mask)
    ds_wofs = ds_wofs.where(mask)

    # Water present in more than 10% of clear observations in a given year.
    water_frequency_sum = (ds_wofs > 0.1).sum("time", dtype=np.uint16).where(mask)

    return ds, water_frequency_sum, mask


# =============================================================================
# Analysis
# =============================================================================

def calculate_vegetation_loss(ds, product="s2", threshold=-0.15):
    """
    Year-on-year index change, and the pixels where it drops below threshold.

    Returns (loss_bool, loss_sum, change, threshold_used).
    """
    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
    if index not in ds:
        raise KeyError(f"{index} not found. Available: {list(ds.data_vars)}")

    change = ds[index] - ds[index].shift(time=1)

    if threshold == "otsu":
        # Subsample rather than materialising the whole cube.
        sample = change.isel(
            x=slice(None, None, 10), y=slice(None, None, 10)
        ).values
        thr = float(threshold_otsu(np.nan_to_num(sample, nan=0.0)))
    else:
        thr = float(threshold)

    # Boolean AND rather than .where(), which would promote bool to float64.
    loss_bool = (change < thr) & np.isfinite(ds[index])
    loss_sum = loss_bool.sum("time", dtype=np.uint16)

    return loss_bool, loss_sum, change, thr


def possible_mining_masks(vegetation_loss_sum, water_frequency_sum, ds, buffer_m=90.0):
    """
    Screen for possible mining: vegetation loss coincident with nearby water.

    The buffer is a raster dilation rather than a vector buffer, which is far
    cheaper and accurate enough at these resolutions.
    """
    base_mining = ((vegetation_loss_sum > 0) & (water_frequency_sum > 0)).fillna(False)

    res_m = float(abs(ds.x[1] - ds.x[0]))
    radius_px = max(1, int(np.ceil(buffer_m / res_m)))

    buffered = binary_dilation(
        base_mining.values.astype(bool), footprint=disk(radius_px)
    )
    buffered_mining = xr.DataArray(
        buffered.astype(np.uint8),
        coords=base_mining.coords,
        dims=base_mining.dims,
        name="buffered_mining",
    )

    veg_loss_in_buffer = (
        ((vegetation_loss_sum > 0) & (buffered_mining == 1))
        .fillna(False)
        .astype(np.uint8)
    )

    return base_mining.astype(np.uint8), buffered_mining, veg_loss_in_buffer


def pixel_area_km2_from_coords(ds):
    """Pixel area in km2 from coordinate spacing; handles negative resolution."""
    dx = float(abs(ds.x[1] - ds.x[0]))
    dy = float(abs(ds.y[1] - ds.y[0]))
    return (dx * dy) / 1_000_000.0


def build_summary_table(ds, vegetation_loss_bool, veg_loss_in_buffer_mask, product="s2"):
    """Per-year vegetation loss areas, total and within the mining buffer."""
    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
    background = ds[index].isel(time=0)

    pix_area = pixel_area_km2_from_coords(ds)
    total_area = int(np.count_nonzero(np.isfinite(background.values))) * pix_area

    years = pd.to_datetime(vegetation_loss_bool.time.values).year

    loss_any = vegetation_loss_bool.fillna(False)
    loss_any_area = loss_any.sum(dim=["y", "x"]).values * pix_area

    buf = veg_loss_in_buffer_mask
    if buf.dtype != bool:
        buf = buf == 1
    loss_in_buffer_area = (loss_any & buf).sum(dim=["y", "x"]).values * pix_area

    df = pd.DataFrame(
        {
            "year": years,
            "any_veg_loss_km2": loss_any_area,
            "any_veg_loss_%": (loss_any_area / total_area) * 100.0,
            "veg_loss_in_from_mining_km2": loss_in_buffer_area,
            "veg_loss_in_from_mining_%": (loss_in_buffer_area / total_area) * 100.0,
        }
    )
    meta = pd.DataFrame({"metric": ["total_aoi_area_km2"], "value": [total_area]})
    return df, meta


def _convert_area_outputs(df_yearly, df_meta, area_unit="km2"):
    """Convert the summary tables between km2 and hectares."""
    unit = str(area_unit).strip().lower()
    if unit in ("km2", "km²", "square kilometres", "square kilometers"):
        return df_yearly.copy(), df_meta.copy()
    if unit not in ("hectares", "ha", "hectare"):
        raise ValueError("area_unit must be either 'km2' or 'hectares'.")

    yearly = df_yearly.copy()
    meta = df_meta.copy()

    for src, dst in {
        "any_veg_loss_km2": "any_veg_loss_ha",
        "veg_loss_in_from_mining_km2": "veg_loss_in_from_mining_ha",
    }.items():
        if src in yearly.columns:
            yearly[dst] = yearly[src] * 100.0
            yearly = yearly.drop(columns=[src])

    if "metric" in meta.columns:
        sel = meta["metric"] == "total_aoi_area_km2"
        meta.loc[sel, "value"] = meta.loc[sel, "value"] * 100.0
        meta.loc[sel, "metric"] = "total_aoi_area_ha"

    return yearly, meta


# =============================================================================
# Output helpers
# =============================================================================

def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataarray_geotiff(da, out_path, nodata=None, dtype=None, compress="deflate"):
    """Write a 2D DataArray to a tiled, compressed GeoTIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if da.ndim != 2:
        raise ValueError(f"Expected 2D DataArray, got dims {da.dims}")

    try:
        gb = da.odc.geobox
        transform, crs = gb.transform, gb.crs
    except AttributeError:
        xmin, xmax = float(da.x.min()), float(da.x.max())
        ymin, ymax = float(da.y.min()), float(da.y.max())
        transform = from_bounds(xmin, ymin, xmax, ymax, da.sizes["x"], da.sizes["y"])
        crs = da.attrs.get("crs")

    arr = da.values
    dtype = dtype or str(arr.dtype)

    if nodata is None:
        nodata = np.nan if np.issubdtype(arr.dtype, np.floating) else -9999

    write_nodata = nodata
    if np.issubdtype(np.dtype(dtype), np.floating):
        if np.isnan(nodata):
            write_nodata = -9999.0
        write_arr = np.where(np.isfinite(arr), arr, write_nodata).astype(dtype)
    elif np.issubdtype(arr.dtype, np.floating):
        write_arr = np.where(np.isfinite(arr), arr, write_nodata).astype(dtype)
    else:
        write_arr = arr.astype(dtype)

    profile = dict(
        driver="GTiff",
        height=write_arr.shape[0],
        width=write_arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=write_nodata,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(write_arr, 1)


def save_time_stack_geotiffs(da_time, out_dir, prefix, nodata=None, dtype=None):
    """Write one GeoTIFF per timestep of a (time, y, x) DataArray."""
    out_dir = _ensure_dir(out_dir)
    years = pd.to_datetime(da_time.time.values).year
    for i, y in enumerate(years):
        save_dataarray_geotiff(
            da_time.isel(time=i),
            out_dir / f"{prefix}_{int(y)}.tif",
            nodata=nodata,
            dtype=dtype,
        )


# =============================================================================
# Plotting
# =============================================================================

def plot_possible_mining_map(ds, veg_loss_in_buffer_mask, product="s2",
                             out_png=None, dpi=150):
    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
    bg = ds[index].isel(time=0)

    fig, ax = plt.subplots(figsize=(12, 12))
    bg.plot.imshow(ax=ax, cmap="Greys", add_colorbar=False)
    veg_loss_in_buffer_mask.where(veg_loss_in_buffer_mask == 1).plot.imshow(
        ax=ax, cmap=ListedColormap(["Gold"]), add_colorbar=False
    )
    ax.legend([Patch(facecolor="Gold")], ["Possible mining site"], loc="upper left")
    ax.set_title("Possible mining areas")
    ax.set_axis_off()

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_vegetation_loss_timeseries(ds, vegetation_loss_bool, veg_loss_in_buffer_mask,
                                    out_png=None, dpi=150):
    pix_area = pixel_area_km2_from_coords(ds)
    years = pd.to_datetime(vegetation_loss_bool.time.values).year

    loss_any = vegetation_loss_bool.fillna(False)
    loss_any_area = loss_any.sum(dim=["y", "x"]).values * pix_area

    buf = veg_loss_in_buffer_mask
    if buf.dtype != bool:
        buf = buf == 1
    loss_in_buffer_area = (loss_any & buf).sum(dim=["y", "x"]).values * pix_area

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(years, loss_any_area, marker="o", label="Any vegetation loss (km²)")
    ax.plot(years, loss_in_buffer_area, marker="^",
            label="Vegetation loss in mining buffer (km²)")
    ax.grid(True)
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (km²)")
    ax.set_title("Annual vegetation loss")
    ax.legend()

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_rgb_and_mining_veg_loss_by_year(ds, vegetation_loss_bool,
                                         veg_loss_in_buffer_mask, product="s2",
                                         out_png=None, dpi=150,
                                         max_years_in_legend=12):
    """Two panels: most recent RGB composite, and yearly loss within the buffer."""
    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"

    years = pd.to_datetime(vegetation_loss_bool.time.values).year
    if len(years) < 2:
        raise ValueError("Need at least 2 timesteps to plot loss by year.")

    background = ds[index].isel(time=0)
    last_i = len(years) - 1

    loss_any = vegetation_loss_bool.fillna(False).astype(bool)
    buf = veg_loss_in_buffer_mask
    if buf.dtype != bool:
        buf = buf == 1
    loss_in_mining = loss_any & buf.fillna(False).astype(bool)

    tableau = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    colors = [tableau[i % len(tableau)] for i in range(len(years))]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax0 = axes[0]
    if product in S2_MINING_PRODUCTS:
        rgb(ds, index=[last_i], ax=ax0)
    else:
        med_s1 = ds[["vv", "vh", "vh/vv"]].median()
        rgb(ds[["vv", "vh", "vh/vv"]] / med_s1,
            bands=["vv", "vh", "vh/vv"], index=[last_i], ax=ax0)
    ax0.set_title("RGB from most recent composite")
    ax0.set_axis_off()

    ax1 = axes[1]
    background.plot.imshow(ax=ax1, cmap="Greys", add_colorbar=False)
    ax1.set_axis_off()

    patches, labels = [], []
    year_indices = list(range(1, len(years)))  # first year has no change
    if max_years_in_legend and len(year_indices) > max_years_in_legend:
        year_indices = year_indices[-max_years_in_legend:]

    for i in year_indices:
        da = loss_in_mining.isel(time=i)
        if int(da.sum().values) == 0:
            continue
        da.where(da).plot.imshow(
            ax=ax1, add_colorbar=False, cmap=ListedColormap([colors[i]])
        )
        patches.append(Patch(facecolor=colors[i]))
        labels.append(str(int(years[i])))

    ax1.legend(patches, labels, loc="upper left")
    ax1.set_title(
        f"Vegetation loss from possible mining, {int(years[0])} to {int(years[-1])}"
    )

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def _normalise_plot_outputs(plot_outputs=None, make_plots=False):
    if plot_outputs is None:
        return ["Possible mining map"] if make_plots else []
    if isinstance(plot_outputs, str):
        return [plot_outputs]
    return list(plot_outputs)


# =============================================================================
# Runner
# =============================================================================

def run_surface_mining_screening(
    vector_file,
    start_date,
    end_date,
    product="s2_semiannual",
    sentinel_statistic=None,
    threshold=-0.15,
    buffer_m=90.0,
    out_dir="results",
    resolution_m=30,
    max_pixels=5_000_000,
    area_unit="km2",
    export_geotiffs=False,
    export_yearly_loss_geotiffs=False,
    make_plots=False,
    plot_outputs=None,
    dpi=150,
):
    """
    End-to-end screening run.

    Guards on AOI size before loading anything, then persists the intermediate
    results so the lazy graph is evaluated once rather than on every downstream
    .values call.
    """
    t0 = time.time()
    out_dir = _ensure_dir(out_dir)

    gdf, geom = load_vector_file(vector_file)
    area_km2, estimated_pixels = _estimate_aoi_pixels(gdf, resolution_m=resolution_m)
    n_years = _count_years_from_dates(start_date, end_date)

    if estimated_pixels > int(max_pixels):
        raise MemoryError(
            f"AOI is too large at {resolution_m} m. Estimated pixels: "
            f"{estimated_pixels:,}; limit: {int(max_pixels):,}. Reduce the AOI, "
            f"coarsen the resolution, or raise max_pixels if RAM allows."
        )

    if str(area_unit).strip().lower() in ("hectares", "ha", "hectare"):
        _stamp(f"AOI area: {area_km2 * 100.0:,.2f} ha")
    else:
        _stamp(f"AOI area: {area_km2:,.2f} km²")
    _stamp(f"Estimated pixels per layer: {estimated_pixels:,}")
    _stamp(f"Date range: {start_date} to {end_date} ({n_years} years)")
    _stamp(f"Resolution: {resolution_m} m")

    selected_plots = _normalise_plot_outputs(plot_outputs, make_plots)
    needs_rgb = any("RGB" in p for p in selected_plots)

    ds, water_frequency_sum, mask = process_data(
        gdf=gdf,
        geom=geom,
        start_date=start_date,
        end_date=end_date,
        product=product,
        sentinel_statistic=sentinel_statistic or "Median",
        resolution_m=resolution_m,
        keep_rgb_bands=needs_rgb,
    )

    # Evaluate once. Without this, every downstream .values call re-reads
    # every COG from source, which roughly triples the runtime.
    _stamp("Reading data...")
    ds = _materialise(ds)
    water_frequency_sum = _materialise(water_frequency_sum)
    _stamp(f"Data read in {time.time() - t0:.1f}s")

    veg_loss_bool, veg_loss_sum, change, thr = calculate_vegetation_loss(
        ds, product=product, threshold=threshold
    )
    veg_loss_bool = _materialise(veg_loss_bool)
    veg_loss_sum = _materialise(veg_loss_sum)
    _stamp(f"Vegetation-loss threshold used: {thr}")

    base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask = (
        possible_mining_masks(veg_loss_sum, water_frequency_sum, ds, buffer_m=buffer_m)
    )
    _stamp("Computed possible mining masks.")

    df_yearly, df_meta = build_summary_table(
        ds, veg_loss_bool, veg_loss_in_buffer_mask, product=product
    )
    df_yearly, df_meta = _convert_area_outputs(df_yearly, df_meta, area_unit=area_unit)

    df_yearly.to_csv(out_dir / "surface_mining_summary_by_year.csv", index=False)
    df_meta.to_csv(out_dir / "surface_mining_summary_meta.csv", index=False)
    _stamp(f"Saved CSV outputs to: {out_dir}")

    if export_geotiffs:
        save_dataarray_geotiff(
            water_frequency_sum.astype(np.float32),
            out_dir / "water_frequency_sum.tif", nodata=-9999.0, dtype="float32")
        save_dataarray_geotiff(
            veg_loss_sum.astype(np.uint16),
            out_dir / "vegetation_loss_sum.tif", nodata=0, dtype="uint16")
        save_dataarray_geotiff(
            base_mining_mask.astype(np.uint8),
            out_dir / "possible_mining_base_mask.tif", nodata=0, dtype="uint8")
        save_dataarray_geotiff(
            buffered_mining_mask.astype(np.uint8),
            out_dir / "possible_mining_buffer_mask.tif", nodata=0, dtype="uint8")
        save_dataarray_geotiff(
            veg_loss_in_buffer_mask.astype(np.uint8),
            out_dir / "veg_loss_in_mining_buffer_mask.tif", nodata=0, dtype="uint8")
        _stamp("Saved core GeoTIFF outputs.")

    if export_yearly_loss_geotiffs:
        save_time_stack_geotiffs(
            veg_loss_bool.fillna(False).astype(np.uint8),
            out_dir / "yearly_veg_loss_masks",
            prefix="veg_loss", nodata=0, dtype="uint8",
        )
        _stamp("Saved yearly vegetation-loss GeoTIFFs.")

    if selected_plots:
        _stamp("Creating plots...")

        if "Possible mining map" in selected_plots:
            plot_possible_mining_map(
                ds, veg_loss_in_buffer_mask, product=product,
                out_png=out_dir / "Possible_Mining.png", dpi=dpi)
            plt.close("all")
            gc.collect()

        if "Vegetation loss time series" in selected_plots:
            plot_vegetation_loss_timeseries(
                ds, veg_loss_bool, veg_loss_in_buffer_mask,
                out_png=out_dir / "veg_loss_timeseries.png", dpi=dpi)
            plt.close("all")
            gc.collect()

        if "Two-panel RGB + vegetation loss" in selected_plots:
            plot_rgb_and_mining_veg_loss_by_year(
                ds, veg_loss_bool, veg_loss_in_buffer_mask, product=product,
                out_png=out_dir / "RGB_and_VegLoss_From_Mining.png",
                dpi=dpi, max_years_in_legend=12)
            plt.close("all")
            gc.collect()

    _stamp(f"Done in {time.time() - t0:.1f}s. Outputs in {out_dir}")

    result = {
        "out_dir": str(out_dir),
        "summary_by_year": df_yearly,
        "summary_meta": df_meta,
        "threshold_used": thr,
        "estimated_aoi_area_km2": area_km2,
        "estimated_aoi_area_ha": area_km2 * 100.0,
        "area_unit": area_unit,
        "estimated_pixels_per_layer": estimated_pixels,
    }

    del ds, water_frequency_sum, mask, veg_loss_bool, veg_loss_sum, change
    del base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask
    gc.collect()

    return result


def _is_dask(obj):
    """True if the xarray object is dask-backed."""
    if isinstance(obj, xr.Dataset):
        return any(v.chunks is not None for v in obj.data_vars.values())
    return getattr(obj, "chunks", None) is not None


def _materialise(obj):
    """Compute a dask-backed object once, so it is not recomputed downstream."""
    return obj.compute() if _is_dask(obj) else obj


# =============================================================================
# Notebook UI
# =============================================================================

def display_surface_mining_screening_ui():
    """Interactive panel for the screening workflow."""

    help_text = widgets.HTML("""
<h4>Surface Mining Screening (STAC build)</h4>
<p>Screens for possible surface mining by detecting year-on-year vegetation
loss and testing whether it coincides with nearby surface water. All data is
loaded over STAC, so no Open Data Cube index is required.</p>
<p><b>Sources.</b> Sentinel-1 RTC comes from Microsoft Planetary Computer;
Sentinel-2 GeoMAD and WOfS come from Digital Earth Africa. Coverage is limited
to Africa for the GeoMAD and WOfS products.</p>
<p><b>Products.</b> Semi-annual GeoMAD is the best first choice: cloud-reduced
and light. Annual GeoMAD gives fewer timesteps over a longer span. Sentinel-1
is radar, useful where cloud is persistent, and uses RVI instead of NDVI.</p>
<p><b>Workflow.</b> Set the AOI, click Preview Size to check the job fits in
memory, then Run. Start small: a 2 km buffer, 30 m resolution, two or three
years, plots limited to the possible mining map.</p>
""")

    # --- AOI -----------------------------------------------------------
    aoi_mode = widgets.ToggleButtons(
        options=["Upload shapefile", "Lat/Lon buffer"],
        description="AOI:", layout=widgets.Layout(width="360px"))

    shapefile_upload = widgets.FileUpload(
        accept=".zip", multiple=False, description="Upload .zip",
        layout=widgets.Layout(width="220px"))

    lat_input = widgets.FloatText(value=5.6, description="Latitude:",
                                  layout=widgets.Layout(width="240px"))
    lon_input = widgets.FloatText(value=-0.2, description="Longitude:",
                                  layout=widgets.Layout(width="240px"))
    buffer_km_input = widgets.FloatText(value=2, description="Buffer km:",
                                        layout=widgets.Layout(width="240px"))

    aoi_shape = widgets.ToggleButtons(
        options=["Circle buffer", "Square/rectangle"], value="Circle buffer",
        description="Shape:", layout=widgets.Layout(width="360px"))

    rect_width_km = widgets.FloatText(value=4, description="Width km:",
                                      layout=widgets.Layout(width="240px"))
    rect_height_km = widgets.FloatText(value=4, description="Height km:",
                                       layout=widgets.Layout(width="240px"))

    circle_box = widgets.VBox([buffer_km_input])
    rect_box = widgets.VBox([rect_width_km, rect_height_km])
    dynamic_shape_box = widgets.VBox([circle_box])

    aoi_map_output = widgets.Output(
        layout=widgets.Layout(height="280px", overflow="auto"))
    preview_aoi_button = widgets.Button(
        description="Preview AOI Map", button_style="info", icon="map",
        layout=widgets.Layout(width="170px"))

    shapefile_box = widgets.VBox([
        widgets.HTML("<b>Upload zipped shapefile</b>"),
        shapefile_upload,
        widgets.HTML("<small>Zip must contain .shp, .shx, .dbf and .prj.</small>"),
    ])
    latlon_box = widgets.VBox([
        widgets.HTML("<b>Point and buffer</b>"),
        lat_input, lon_input, aoi_shape, dynamic_shape_box,
    ])
    dynamic_aoi_box = widgets.VBox([shapefile_box])

    # --- Product and dates ---------------------------------------------
    mining_product = widgets.Dropdown(
        options={
            "Semi-annual GeoMAD": "s2_semiannual",
            "Annual GeoMAD": "s2",
            "Sentinel-1": "s1",
        },
        value="s2_semiannual", description="Product:",
        layout=widgets.Layout(width="300px"))

    product_note = widgets.HTML(
        "<small>Cloud-reduced Sentinel-2 composite; good first option.</small>")

    sentinel_stat_select = widgets.Dropdown(
        options=get_sentinel_stat_options(), value="Median",
        description="Statistic:", layout=widgets.Layout(width="300px"))

    stat_box = widgets.VBox([
        widgets.HTML("<b>Compositing statistic</b>"),
        sentinel_stat_select,
        widgets.HTML("<small>Median is safest. Geomedian is heavier.</small>"),
    ])

    current_year = date.today().year
    years = list(range(2015, current_year + 1))
    months = list(range(1, 13))

    start_year = widgets.Dropdown(options=years, value=max(2018, current_year - 3),
                                  description="Start year:",
                                  layout=widgets.Layout(width="240px"))
    end_year = widgets.Dropdown(options=years, value=current_year - 1,
                                description="End year:",
                                layout=widgets.Layout(width="240px"))
    start_month = widgets.Dropdown(options=months, value=1,
                                   description="Start month:",
                                   layout=widgets.Layout(width="240px"))
    end_month = widgets.Dropdown(options=months, value=12,
                                 description="End month:",
                                 layout=widgets.Layout(width="240px"))

    month_box = widgets.VBox([
        widgets.HTML("<b>Month range</b>"), start_month, end_month,
        widgets.HTML("<small>Applies to Sentinel-1 only.</small>"),
    ])

    # --- Processing -----------------------------------------------------
    resolution_m = widgets.Dropdown(options=[10, 20, 30, 60, 100], value=30,
                                    description="Resolution m:",
                                    layout=widgets.Layout(width="260px"))
    max_pixels = widgets.IntText(value=5_000_000, description="Max pixels:",
                                 layout=widgets.Layout(width="260px"))
    area_unit = widgets.Dropdown(
        options={"Square kilometres (km²)": "km2", "Hectares (ha)": "hectares"},
        value="km2", description="Area unit:", layout=widgets.Layout(width="300px"))
    threshold_input = widgets.Text(value="-0.15", description="Threshold:",
                                   layout=widgets.Layout(width="260px"))
    mining_buffer_m = widgets.FloatText(value=90.0, description="Mining buffer m:",
                                        layout=widgets.Layout(width="260px"))
    out_dir_input = widgets.Text(value="results", description="Output folder:",
                                 layout=widgets.Layout(width="320px"))

    export_core_tifs = widgets.Checkbox(value=False, description="Export core GeoTIFFs")
    export_yearly_tifs = widgets.Checkbox(
        value=False, description="Export yearly loss GeoTIFFs")

    plot_outputs = widgets.SelectMultiple(
        options=["Possible mining map", "Vegetation loss time series",
                 "Two-panel RGB + vegetation loss"],
        value=("Possible mining map",), description="Plots:",
        layout=widgets.Layout(width="360px", height="95px"))

    preview_button = widgets.Button(description="Preview Size", button_style="info",
                                    icon="search", layout=widgets.Layout(width="150px"))
    run_button = widgets.Button(description="Run", button_style="success",
                                icon="play", layout=widgets.Layout(width="140px"))
    clear_button = widgets.Button(description="Clear", button_style="warning",
                                  icon="trash", layout=widgets.Layout(width="100px"))

    output = widgets.Output()
    last_preview = {"ok": False}

    # --- Behaviour ------------------------------------------------------
    def update_aoi_box(change=None):
        dynamic_aoi_box.children = (
            [shapefile_box] if aoi_mode.value == "Upload shapefile" else [latlon_box])
        last_preview["ok"] = False

    def update_shape_box(change=None):
        dynamic_shape_box.children = (
            [circle_box] if aoi_shape.value == "Circle buffer" else [rect_box])
        last_preview["ok"] = False

    def update_product_controls(change=None):
        is_s1 = mining_product.value == "s1"
        month_box.layout.display = "block" if is_s1 else "none"
        stat_box.layout.display = "block" if is_s1 else "none"

        notes = {
            "s2_semiannual": "Cloud-reduced Sentinel-2 composite, two per year.",
            "s2": "Yearly Sentinel-2 composite; fewer timesteps over a longer span.",
            "s1": "Radar, works through cloud. Uses VV/VH and RVI rather than NDVI.",
        }
        product_note.value = f"<small>{notes[mining_product.value]}</small>"
        last_preview["ok"] = False

    def invalidate(change=None):
        last_preview["ok"] = False

    def parse_threshold(value):
        value = str(value).strip().lower()
        return "otsu" if value == "otsu" else float(value)

    def build_dates():
        sy, ey = int(start_year.value), int(end_year.value)
        if mining_product.value == "s1":
            sm, em = int(start_month.value), int(end_month.value)
            start_dt = date(sy, sm, 1)
            end_dt = date(ey, em, calendar.monthrange(ey, em)[1])
        else:
            start_dt, end_dt = date(sy, 1, 1), date(ey, 12, 31)

        if start_dt > end_dt:
            raise ValueError("Start date must be before or equal to end date.")
        return start_dt.isoformat(), end_dt.isoformat()

    def prepare_aoi():
        if aoi_mode.value == "Upload shapefile":
            aoi_gdf = read_uploaded_shapefile(shapefile_upload)
        elif aoi_shape.value == "Circle buffer":
            aoi_gdf = create_aoi_from_latlon(
                lat=lat_input.value, lon=lon_input.value,
                buffer_km=buffer_km_input.value)
        else:
            aoi_gdf = create_rectangular_aoi_from_latlon(
                lat=lat_input.value, lon=lon_input.value,
                width_km=rect_width_km.value, height_km=rect_height_km.value)
        return aoi_gdf, _save_aoi_gdf_to_temp_file(aoi_gdf)

    def show_aoi_map(aoi_gdf):
        with aoi_map_output:
            clear_output()
            try:
                gdf4326 = aoi_gdf.to_crs("EPSG:4326")
                minx, miny, maxx, maxy = gdf4326.total_bounds
                center = ((miny + maxy) / 2.0, (minx + maxx) / 2.0)
                try:
                    from ipyleaflet import GeoData, LayersControl, Map, basemaps
                    m = Map(center=center, zoom=12,
                            basemap=basemaps.OpenStreetMap.Mapnik,
                            layout=widgets.Layout(width="100%", height="260px"))
                    m.add_layer(GeoData(
                        geo_dataframe=gdf4326, name="AOI",
                        style={"color": "red", "fillColor": "red", "opacity": 1,
                               "weight": 2, "fillOpacity": 0.15}))
                    m.add_control(LayersControl())
                    display(m)
                except Exception:
                    display(gdf4326.explore())
            except Exception as exc:
                print("Could not display AOI map:", exc)

    def on_preview_aoi(button):
        with output:
            clear_output()
            try:
                aoi_gdf, vector_file = prepare_aoi()
                show_aoi_map(aoi_gdf)
                print(f"AOI written to {vector_file}")
            except Exception as e:
                print("Error:", e)

    def on_preview(button):
        with output:
            clear_output()
            try:
                aoi_gdf, vector_file = prepare_aoi()
                show_aoi_map(aoi_gdf)
                start_dt, end_dt = build_dates()
                area_km2, pixels = _estimate_aoi_pixels(
                    aoi_gdf, resolution_m=resolution_m.value)
                n_years = _count_years_from_dates(start_dt, end_dt)
                layers = n_years * (3 if mining_product.value == "s1" else 5)

                print("Preview estimate")
                print("----------------")
                print(f"Product: {mining_product.label}")
                print(f"Date range: {start_dt} to {end_dt}")
                if area_unit.value == "hectares":
                    print(f"AOI area: {area_km2 * 100.0:,.2f} ha")
                else:
                    print(f"AOI area: {area_km2:,.2f} km²")
                print(f"Pixels per layer: {pixels:,}")
                print(f"Approximate analysis cells: {pixels * layers:,}")
                print(f"Estimated peak memory: "
                      f"{pixels * layers * 4 / 1e9:,.2f} GB (float32)")
                print(f"Resolution: {resolution_m.value} m")

                if pixels > max_pixels.value:
                    print("\nStatus: NOT SAFE — shrink the AOI or coarsen resolution.")
                    last_preview["ok"] = False
                else:
                    print("\nStatus: OK to run.")
                    last_preview["ok"] = True

                last_preview.update({
                    "vector_file": vector_file,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                })
            except Exception as e:
                last_preview["ok"] = False
                print("Error:", e)

    def on_run(button):
        with output:
            try:
                if not last_preview.get("ok"):
                    print("Click 'Preview Size' first and check the status is OK.")
                    return

                print("\nRunning...")
                stat = (sentinel_stat_select.value
                        if mining_product.value == "s1" else None)

                result = run_surface_mining_screening(
                    vector_file=last_preview["vector_file"],
                    start_date=last_preview["start_dt"],
                    end_date=last_preview["end_dt"],
                    product=mining_product.value,
                    sentinel_statistic=stat,
                    threshold=parse_threshold(threshold_input.value),
                    buffer_m=float(mining_buffer_m.value),
                    out_dir=out_dir_input.value,
                    resolution_m=float(resolution_m.value),
                    max_pixels=int(max_pixels.value),
                    area_unit=area_unit.value,
                    export_geotiffs=bool(export_core_tifs.value),
                    export_yearly_loss_geotiffs=bool(export_yearly_tifs.value),
                    plot_outputs=list(plot_outputs.value),
                )

                display(result["summary_meta"])
                display(result["summary_by_year"])
            except Exception as e:
                print("Error:", e)
            finally:
                plt.close("all")
                gc.collect()

    def on_clear(button):
        with output:
            clear_output()
        plt.close("all")
        gc.collect()

    for w in [start_year, start_month, end_year, end_month, resolution_m,
              max_pixels, area_unit, threshold_input, mining_buffer_m,
              out_dir_input, sentinel_stat_select, buffer_km_input,
              rect_width_km, rect_height_km, lat_input, lon_input, plot_outputs]:
        w.observe(invalidate, names="value")

    aoi_mode.observe(update_aoi_box, names="value")
    aoi_shape.observe(update_shape_box, names="value")
    mining_product.observe(update_product_controls, names="value")
    preview_aoi_button.on_click(on_preview_aoi)
    preview_button.on_click(on_preview)
    run_button.on_click(on_run)
    clear_button.on_click(on_clear)

    # --- Layout ---------------------------------------------------------
    panel_style = dict(border="1px solid lightgray", padding="12px",
                       margin="0 8px 0 0")

    aoi_panel = widgets.VBox([
        widgets.HTML("<h3>AOI</h3>"), aoi_mode, dynamic_aoi_box,
        preview_aoi_button, aoi_map_output,
    ], layout=widgets.Layout(width="390px", **panel_style))

    product_panel = widgets.VBox([
        widgets.HTML("<h3>Product &amp; date</h3>"), mining_product, product_note,
        start_year, end_year, month_box, stat_box,
    ], layout=widgets.Layout(width="360px", **panel_style))

    processing_panel = widgets.VBox([
        widgets.HTML("<h3>Processing</h3>"), resolution_m, max_pixels, area_unit,
        threshold_input,
        widgets.HTML("<small>A number like -0.15, or type otsu.</small>"),
        mining_buffer_m, out_dir_input, export_core_tifs, export_yearly_tifs,
        widgets.HTML("<b>Plots</b>"), plot_outputs,
        widgets.HTML("<br>"),
        widgets.HBox([preview_button, run_button, clear_button]),
    ], layout=widgets.Layout(width="390px", **panel_style))

    controls = widgets.HBox([aoi_panel, product_panel, processing_panel],
                            layout=widgets.Layout(width="100%",
                                                  align_items="stretch"))

    output_panel = widgets.VBox([
        widgets.HTML("<h3>Output</h3>"), output,
    ], layout=widgets.Layout(width="100%", border="1px solid lightgray",
                             padding="12px", margin="10px 0 0 0"))

    intro = widgets.VBox([help_text], layout=widgets.Layout(
        width="100%", border="1px solid lightgray", padding="12px",
        margin="0 0 10px 0"))

    update_aoi_box()
    update_shape_box()
    update_product_controls()

    display(widgets.VBox([intro, controls, output_panel]))

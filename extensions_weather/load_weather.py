import xarray as xr
import gcsfs
import matplotlib.pyplot as plt
import os

fs = gcsfs.GCSFileSystem(token="anon")  # public bucket
mapper = fs.get_mapper(
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

ds = xr.open_zarr(mapper, consolidated=True)
print(ds)

da = ds["10m_wind_speed"]
print(da)
print(da.dims)
print(da.shape)
print(da.coords)
slice0 = da.isel(time=0)
print(slice0)

t_0 = 21413*4
slice_t = da.isel(time=t_0)

slice_t_ds = slice_t.isel(latitude=slice(None, None, 5), longitude=slice(None, None, 5))
print("Downsampled slice dims:", slice_t_ds.dims)
print("Downsampled slice shape:", slice_t_ds.shape)
lon_ds = slice_t_ds.longitude
use_0_360_ds = bool(lon_ds.min() >= 0.0 and lon_ds.max() > 180.0)
slice_t_ds_plot = slice_t_ds
if use_0_360_ds:
    slice_t_ds_plot = slice_t_ds_plot.assign_coords(longitude=(((slice_t_ds_plot.longitude + 180.0) % 360.0) - 180.0))
slice_t_ds_plot = slice_t_ds_plot.sortby("longitude")
slice_t_ds_plot = slice_t_ds_plot.sortby("latitude")
print("Downsampled slice (sorted) dims:", slice_t_ds_plot.dims)
print("Downsampled slice (sorted) shape:", slice_t_ds_plot.shape)

plt.figure(figsize=(14, 5))
slice_t_ds_plot.plot(x="longitude", y="latitude", cmap="viridis", robust=True)
plt.title(f"10m_wind_speed (downsample x5) at time={str(slice_t['time'].values)}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
slice_t.plot(x="longitude", y="latitude", cmap="viridis", robust=True)
plt.title(f"10m_wind_speed at time={str(slice_t['time'].values)}")
plt.tight_layout()
plt.show()

na_lat_min = 20.0
na_lat_max = 70.0
na_lon_min = -85.0
na_lon_max = 20.0

lon = slice_t.longitude
use_0_360 = bool(lon.min() >= 0.0 and lon.max() > 180.0)
slice_t_na = slice_t
if use_0_360:
    slice_t_na = slice_t_na.assign_coords(longitude=(((slice_t_na.longitude + 180.0) % 360.0) - 180.0))

slice_t_na = slice_t_na.sortby("longitude")
slice_t_na = slice_t_na.sortby("latitude")

north_atlantic = slice_t_na.sel(latitude=slice(na_lat_min, na_lat_max), longitude=slice(na_lon_min, na_lon_max))

plt.figure(figsize=(10, 7))
north_atlantic.plot(x="longitude", y="latitude", cmap="viridis", robust=True)
plt.title(
    f"North Atlantic (full-res) lat {na_lat_min}..{na_lat_max}, lon {na_lon_min}..{na_lon_max} at time={str(slice_t['time'].values)}"
)
plt.tight_layout()
plt.show()

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "frames_global_downsampled"))
os.makedirs(out_dir, exist_ok=True)

out_dir_na = os.path.abspath(os.path.join(os.path.dirname(__file__), "frames_north_atlantic_fullres"))
os.makedirs(out_dir_na, exist_ok=True)

for t in range(t_0,t_0+80):
    slice_t = da.isel(time=t)
    slice_t_ds = slice_t.isel(latitude=slice(None, None, 5), longitude=slice(None, None, 5))
    lon_ds = slice_t_ds.longitude
    use_0_360_ds = bool(lon_ds.min() >= 0.0 and lon_ds.max() > 180.0)
    slice_t_ds_plot = slice_t_ds
    if use_0_360_ds:
        slice_t_ds_plot = slice_t_ds_plot.assign_coords(longitude=(((slice_t_ds_plot.longitude + 180.0) % 360.0) - 180.0))
    slice_t_ds_plot = slice_t_ds_plot.sortby("longitude")
    slice_t_ds_plot = slice_t_ds_plot.sortby("latitude")

    fig = plt.figure(figsize=(14, 5))
    slice_t_ds_plot.plot(x="longitude", y="latitude", cmap="viridis", robust=True)
    plt.title(f"10m_wind_speed (downsample x5) at time={str(slice_t['time'].values)}")
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"global_downsampled_t{t:03d}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    lon_full = slice_t.longitude
    use_0_360_full = bool(lon_full.min() >= 0.0 and lon_full.max() > 180.0)
    slice_t_na = slice_t
    if use_0_360_full:
        slice_t_na = slice_t_na.assign_coords(longitude=(((slice_t_na.longitude + 180.0) % 360.0) - 180.0))
    slice_t_na = slice_t_na.sortby("longitude")
    slice_t_na = slice_t_na.sortby("latitude")

    north_atlantic = slice_t_na.sel(latitude=slice(na_lat_min, na_lat_max), longitude=slice(na_lon_min, na_lon_max))

    fig = plt.figure(figsize=(10, 7))
    north_atlantic.plot(x="longitude", y="latitude", cmap="viridis", robust=True)
    plt.title(
        f"North Atlantic (full-res) lat {na_lat_min}..{na_lat_max}, lon {na_lon_min}..{na_lon_max} at time={str(slice_t['time'].values)}"
    )
    plt.tight_layout()

    out_png_na = os.path.join(out_dir_na, f"north_atlantic_fullres_t{t:03d}.png")
    fig.savefig(out_png_na, dpi=200)
    plt.close(fig)

t_uk = 0
slice_t = da.isel(time=t_uk)

uk_lat_max = 61.0
uk_lat_min = 49.0
uk_lon_min = -8.0
uk_lon_max = 2.0

lon = slice_t.longitude
use_0_360 = bool(lon.min() >= 0.0 and lon.max() > 180.0)

slice_t_plot = slice_t
if use_0_360:
    slice_t_plot = slice_t_plot.assign_coords(longitude=(((slice_t_plot.longitude + 180.0) % 360.0) - 180.0))

slice_t_plot = slice_t_plot.sortby("longitude")
slice_t_plot = slice_t_plot.sortby("latitude")

uk = slice_t_plot.sel(latitude=slice(uk_lat_min, uk_lat_max), longitude=slice(uk_lon_min, uk_lon_max))

plt.figure(figsize=(8, 8))
uk.plot(x="longitude", y="latitude", cmap="viridis", robust=True)
plt.title(f"UK subset (lat {uk_lat_min}..{uk_lat_max}, lon {uk_lon_min}..{uk_lon_max}) at time={str(slice_t['time'].values)}")
plt.tight_layout()
plt.show()

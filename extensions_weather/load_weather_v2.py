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

atlas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "atlas2.png"))
atlas_img = plt.imread(atlas_path)

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.imshow(atlas_img, extent=[-180, 180, -90, 90], origin="upper", aspect="auto")

lon_plot = slice_t_ds_plot.longitude.values
lat_plot = slice_t_ds_plot.latitude.values
data_plot = slice_t_ds_plot.values

pcm = ax.pcolormesh(lon_plot, lat_plot, data_plot, cmap="viridis", shading="auto", alpha=0.6)
fig.colorbar(pcm, ax=ax, label=str(da.name))

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title(f"{da.name} overlay on atlas (downsample x5) at time={str(slice_t['time'].values)}")
fig.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
slice_t.plot(x="longitude", y="latitude", cmap="viridis", robust=True)
plt.title(f"10m_wind_speed at time={str(slice_t['time'].values)}")
plt.tight_layout()
plt.show()

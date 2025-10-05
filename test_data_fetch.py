import pytest
import os
from src.data_fetch import download_tropomi, download_openaq_stations

def test_download_tropomi(tmp_path):
    ds = download_tropomi([77.5,12.5,78.5,13.5], '2024-01-01', '2024-01-03', tmp_path)
    assert "no2" in ds.data_vars
    assert "qa_value" in ds.data_vars
    assert ds.dims["time"] == 3
    assert ds.no2.shape == (3,10,10)

def test_download_tropomi_local(tmp_path):
    ds1 = download_tropomi([77.5,12.5,78.5,13.5], '2024-01-01', '2024-01-01', tmp_path)
    local_path = os.path.join(tmp_path, "synthetic_no2.nc")
    ds2 = download_tropomi([0,0,1,1], '2024-01-01', '2024-01-01', tmp_path, load_local=local_path)
    assert (ds2.no2 == ds1.no2).all()

def test_download_openaq_stations():
    ds = download_openaq_stations([77.5,12.5,78.5,13.5], '2024-01-01')
    assert "pm25" in ds.data_vars
    assert ds.no2.shape[0] == 2
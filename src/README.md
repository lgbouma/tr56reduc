# usage

first:
```
make_master_bias_and_master_flat.py
get_timestamp_in_BJD_TDB.py
 ```

then with the output of `get_timestamp_in_BJD_TDB.py` go to Jason Eastman's
applet discussed in docstring of `get_timestamp_in_BJD_TDB`, get timestamps in
BJD in barycentric dynamical time. This requires entering the TR 56 sky coords,
and the observing site (Las Campanas).

Then:
```
do_simple_aperture_phot.py
do_centroid_clustering.py
do_differential_phot.py
```

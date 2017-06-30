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

Then (after defining parameters in `define_arbitrary_parameters.py` which
unfortunately are not so arbitrary):

```
do_simple_aperture_phot.py
do_centroid_clustering.py
do_differential_phot.py
```

Selecting particular bands is left as something to be done in the code. Most of
my reduction focused on 'r' and 'i', although I looked briefly at 'g' and 'z'
as well.

Python can save rich hierarchical datasets in hdf5 format. Matlab can
read hdf5, but the api is so heavy it is almost unusable. Here are some
matlab scripts (written by Gaël Varoquaux) to load and save data in hdf5
format under Matlab with the same signature as the standard matlab
load/save function.

.. image:: hdf5_in_Matlab_attachments/hdf5matlab.zip

These Matlab scripts cannot load every type allowed in hdf5. Feel free
to provide python scripts to use pytables to implement simple load/save
functions compatible with this hdf5 subset.

One notice: these script use the "Workspace" namespace to store some
variables, they will pollute your workspace when saving data from
Matlab. Nothing that I find unacceptable.

Another loader script
---------------------

Here is a second HDF5 loader script, which loads (optionally partial)
data from a HDF5 file to a Matlab structure

``   ``\ ```.. image:: hdf5_in_Matlab_attachments/h5load.m`` <.. image:: hdf5_in_Matlab_attachments/h5load.m>`__

It can deal with more varied HDF5 datasets than the Matlab high-level
functions (at least R2008a hdf5info fails with chunked compressed
datasets), via using only the low-level HDF5 API.

The script also recognizes complex numbers in the Pytables format, and
permutes array dimensions to match the logical order in the file (ie. to
match Python. The builtin Matlab functions by default return data in the
opposite order, so the first dimension in Python would be the last in
Matlab).

--------------

CategoryCookbook


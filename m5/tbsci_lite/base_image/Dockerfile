# tasksynth-sci: the shared base of every synthesized TB-Sci-lite-style task (#353). M1's lean base
# (local/tasksynth-base:2: Ubuntu 24.04, CLI tools, Python 3.12 + numpy/pandas/scipy/requests/pyyaml/pytest) plus the
# scientific stack the lite tasks build on, across all fields. Anything else a task needs goes in its thin layer
# (apt/pip through the package cache at build time).
FROM local/tasksynth-base:2

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        gfortran libopenblas-dev liblapack-dev libhdf5-dev libnetcdf-dev \
        r-base-core ngspice gnuplot-nox graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        matplotlib scikit-learn statsmodels sympy networkx mpmath numba \
        h5py netCDF4 xarray pyarrow openpyxl tifffile \
        astropy biopython scikit-image pillow PyWavelets \
        emcee lmfit uncertainties pint \
        cvxpy pulp ortools python-sat \
        shapely pyproj \
        rdkit ase \
        control joblib tqdm

WORKDIR /app

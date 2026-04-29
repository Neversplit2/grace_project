# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Collect all hidden imports from heavy data science libraries
hidden_imports = [
    'uvicorn',
    'fastapi',
    'matplotlib',
    'pandas',
    'numpy',
    'xgboost',
    'sklearn',
    'cartopy',
    'scipy',
    'joblib',
    'shapely',
    'xarray',
    'netCDF4',
    'cdsapi',
    'colorama',
    'plotly',
    'requests',
    'seaborn',
    'geopandas',
    'pyproj',
    'fiona',
    'dask',
]

# Add specific submodules that PyInstaller often misses during static analysis
hidden_imports += collect_submodules('uvicorn')
hidden_imports += collect_submodules('sklearn')
hidden_imports += collect_submodules('scipy')
hidden_imports += collect_submodules('cartopy')
hidden_imports += collect_submodules('xarray')
hidden_imports += collect_submodules('plotly')

# Collect necessary data files (including the static React UI and Python source code)
datas = [
    ('frontend_react/dist', 'frontend_react/dist'),
    ('code', 'code'),
    ('backend_api', 'backend_api')
]

# Collect hidden package data files (like cartopy shapefiles and plotly templates)
datas += collect_data_files('cartopy')
datas += collect_data_files('plotly')
datas += collect_data_files('xarray')
datas += collect_data_files('xgboost')
try:
    datas += collect_data_files('netCDF4')
except:
    pass

binaries = collect_dynamic_libs('xgboost')

a = Analysis(
    ['main_exe.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GraceDownscalingEngine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keeps the console open to see any runtime errors
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GraceDownscalingEngine',
)

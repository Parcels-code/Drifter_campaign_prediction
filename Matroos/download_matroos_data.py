import os
import subprocess
from getpass import getpass
import numpy as np
from glob import glob

PRODUCTS = {
    "flow": ("dcsm_fm100m_harmonie", 1),
    "waves": ("swan_kuststrook_harmonie", 1),
    "wind": ("knmi_harmonie43", 0),
}

DIR = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmonie"

# Ask these credentials with @erikvansebille
username = os.getenv('MATROOS_USERNAME')
password = os.getenv('MATROOS_PASSWORD')
if not username:
    username = input('MATROOS username: ').strip()
if not password:
    password = getpass('MATROOS password: ')

times = np.arange(np.datetime64('2025-11-01T00:00'), np.datetime64('2025-12-01T00:00'), np.timedelta64(3, 'h'))

for t in times:
    for product, (product_name, is_hindcast) in PRODUCTS.items():
        dirname = os.path.join(DIR, product)
        if product == "flow":
            dirname = os.path.join(dirname, "raw")
        filename = os.path.join(dirname, f"{product_name}_{t.astype('datetime64[m]').astype(str).replace('T', '').replace(':', '').replace('-', '')}.nc")
        print(filename)
        if not os.path.exists(filename):
            os.makedirs(dirname, exist_ok=True)
            subprocess.run(
                [
                    'wget',
                    '--user', username,
                    '--password', password,
                    '--content-disposition',
                    '--output-document', filename,
                    (
                        f'https://matroos.rws.nl/direct/get_netcdf.php'
                        f'?database=maps2d&hindcast={is_hindcast}'
                        f'&zip=0&source={product_name}&analysis={t}'
                    ),
                ],
                check=True,
            )

import requests
import io
from libs.foxutils.utils.core_utils import mkdir_if_not_exist


def fetch_h5_file_from_drive(url, savename='dataset.hd5'):
    with requests.Session() as session:
        r = session.get(url, stream=True)
        r.raise_for_status()
        mkdir_if_not_exist(savename)
        with open(savename, 'wb') as hd5:
            for chunk in r.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE):
                hd5.write(chunk)

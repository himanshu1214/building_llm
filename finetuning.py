aleimport urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

url_path = "https://archive.ics.uci.edu/static/public/228/sms+spam+collections.zip"

zip_filename = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "sms_spam_collection.tsv"

def download_url_contents(url_path, zip_filename):
    
    get_url_contents = urllib.request.urlopen(url_path)
    cwd = os.getcwd()
    zip_filepath = os.path.join(cwd, zip_filename)

    with (zip_filepath, 'wb') as f:
        f.write(get_url_contents)


    with zipfile.ZipFile(zip_filepath, 'r') as zp:
        zp.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "sms_spam_collection"
    os.rename(original_file_path, data_file_path)

    print(f"file downloaded and saved to : {data_file_path}")


download_url_contents(url_path, zip_filename, extracted_path, data_file_path)





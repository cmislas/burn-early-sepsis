import gzip
import shutil

with open('sepsis_model.joblib', 'rb') as f_in:
    with gzip.open('sepsis_model.joblib.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)



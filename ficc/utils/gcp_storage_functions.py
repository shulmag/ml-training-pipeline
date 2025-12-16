'''
Description: Convenience functions to upload and download data from Google cloud buckets.
'''
import pickle

from ficc.utils.auxiliary_functions import run_ten_times_before_raising_gcp_bucket_access_error


def upload_data(storage_client, bucket_name, file_name, file_path: str = None):
    '''Upload data to the cloud bucket `bucket_name` with filename `file_name` from a local `file_path`.'''
    if file_path is None: file_path = file_name
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_path)
    print(f'File from {file_path} uploaded to {file_name} in Google cloud bucket: {bucket_name}')


@run_ten_times_before_raising_gcp_bucket_access_error
def download_data(storage_client, bucket_name, file_name, deserialize_from_pickle: bool = True):
    '''Download file `file_name` from the cloud bucket `bucket_name`. Assumes 
    that `file_name` is a pickle file.'''
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    if not blob.exists():    # `file_name` in `bucket_name` does not exist
        print(f'File {file_name} does not exist in {bucket_name}.')
        return None
    data = blob.download_as_bytes()
    if deserialize_from_pickle: data = pickle.loads(data)
    print(f'File {file_name} downloaded from Google cloud bucket: {bucket_name}')
    return data

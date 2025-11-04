import boto3
import joblib
import io
import sys
import os
import shutil
import mimetypes
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
dotenv_path = os.path.abspath(os.path.normpath(dotenv_path))
load_dotenv(dotenv_path)

USE_S3 = os.getenv("USE_S3", "0") == "1"

S3_BUCKET = os.getenv("S3_BUCKET", "telco-customer-churn-analysis")
s3 = None
if USE_S3:
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )

    except Exception as e:
        print(f"Warning: Could not initialize S3 client: {e}", file=sys.stderr)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "../static")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def upload_to_s3(obj, filename: str):
    """
    Uploads a Python object (e.g., trained model) to S3 or saves it locally.

    Parameters
    ----------
    obj : Any
        Python object to serialize and save (e.g., scikit-learn Pipeline).
    filename : str
        Filename for saving the object.

    Side Effects
    ------------
    - Saves the object to S3 if `USE_S3` is True and S3 client is initialized.
    - Otherwise, saves the object to the local models directory (`MODELS_DIR`).
    """

    if USE_S3 and s3:
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        s3_key = f"models/{filename}"
        s3.upload_fileobj(buffer, S3_BUCKET, s3_key)
        print(f"Uploaded {s3_key} to S3.", file=sys.stderr)

    else:
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(obj, path)
        print(f"Saved {path} locally", file=sys.stderr)


def download_from_s3(filename: str):
    """
    Downloads a Python object (e.g., trained model) from S3 or loads it locally.

    Parameters
    ----------
    filename : str
        Filename of the object to retrieve.

    Returns
    -------
    Any
        The deserialized Python object.

    Side Effects
    ------------
    - Downloads from S3 if `USE_S3` is True and S3 client is initialized.
    - Otherwise, loads the object from the local models directory (`MODELS_DIR`).
    """

    if USE_S3 and s3:
        buffer = io.BytesIO()
        s3_key = f"models/{filename}"
        s3.download_fileobj(S3_BUCKET, s3_key, buffer)
        buffer.seek(0)
        obj = joblib.load(buffer)
        print(f"Downloaded {s3_key} from S3", file=sys.stderr)

    else:
        path = os.path.join(MODELS_DIR, filename)
        obj = joblib.load(path)
        print(f"Loaded {path} locally", file=sys.stderr)
    
    return obj


def upload_image_to_s3(file_path: str, s3_key: str):
    """
    Uploads an image file to S3 or copies it to a local static directory.

    Parameters
    ----------
    file_path : str
        Path to the image file to upload.
    s3_key : str
        Target S3 key or filename.

    Side Effects
    ------------
    - Uploads the image to S3 with the correct MIME type if `USE_S3` is True.
    - Otherwise, copies the file to the local static directory (`STATIC_DIR`).
    """

    if USE_S3 and s3:
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"
        with open(file_path, 'rb') as f:
            s3.upload_fileobj(f, S3_BUCKET, s3_key, ExtraArgs={'ContentType': content_type})
        print(f"Uploaded image {s3_key} to S3", file=sys.stderr)
        print('Object Type: ', content_type)

    else:
        target_path = os.path.join(STATIC_DIR, os.path.basename(file_path))
        shutil.copy(file_path, target_path)
        print(f"Saved image locally at {target_path}", file=sys.stderr)



def download_image_from_s3(file_path: str, s3_key: str):
    """
    Downloads an image from S3 or uses a local copy.

    Parameters
    ----------
    file_path : str
        Local filename to save the image to.
    s3_key : str
        S3 key of the image to download.

    Returns
    -------
    str
        Relative path to the image for use in the project (e.g., for visualization).

    Side Effects
    ------------
    - Downloads the image from S3 if `USE_S3` is True and S3 client is initialized.
    - Otherwise, uses the local static file at `STATIC_DIR`.
    """
    
    path = os.path.join(STATIC_DIR, file_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if USE_S3 and s3:
        with open(path, "wb") as f:
            s3.download_fileobj(S3_BUCKET, s3_key, f)
        print(f"Downloaded image {s3_key} from S3", file=sys.stderr)
    
    else:
        print(f"Using local image {path}", file=sys.stderr)

    return os.path.join('eda', os.path.basename(file_path))
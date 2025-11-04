import io
import os
import sys
import shutil
import joblib
import pytest
import mimetypes
from unittest import mock

from src.telco_customer_churn_analysis.s3_utils import (download_from_s3, upload_to_s3,
                                                                                      download_image_from_s3, upload_image_to_s3)

## upload_to_s3 tests

def test_upload_to_s3_local(monkeypatch, tmp_path, capsys):
    """Test uploading object locally when USE_S3 is False."""
    dummy_obj = {"data": 123}
    dummy_file = tmp_path / "model.pkl"

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", False)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.MODELS_DIR", str(tmp_path))
    mock_joblib = mock.Mock()
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.joblib.dump", mock_joblib)

    upload_to_s3(dummy_obj, "model.pkl")
    captured = capsys.readouterr()

    mock_joblib.assert_called_once_with(dummy_obj, os.path.join(str(tmp_path), "model.pkl"))
    assert "Saved" in captured.err


def test_upload_to_s3_s3(monkeypatch, capsys):
    """Test uploading to S3 when USE_S3 is True."""
    dummy_obj = {"model": "ok"}
    mock_s3 = mock.Mock()

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", True)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.s3", mock_s3)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.S3_BUCKET", "test-bucket")

    with mock.patch("src.telco_customer_churn_analysis.s3_utils.joblib.dump") as mock_dump:
        upload_to_s3(dummy_obj, "model.pkl")
        mock_dump.assert_called_once()
        mock_s3.upload_fileobj.assert_called_once()
        captured = capsys.readouterr()
        assert "Uploaded models/model.pkl to S3" in captured.err


## dowload_from_s3 tests

def test_download_from_s3_local(monkeypatch, tmp_path, capsys):
    """Test downloading model locally when USE_S3 is False."""
    dummy_obj = {"x": 5}
    model_path = tmp_path / "model.pkl"
    joblib.dump(dummy_obj, model_path)

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", False)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.MODELS_DIR", str(tmp_path))

    obj = download_from_s3("model.pkl")
    captured = capsys.readouterr()

    assert obj == dummy_obj
    assert "Loaded" in captured.err


def test_download_from_s3_s3(monkeypatch, capsys):
    """Test downloading from S3 when USE_S3 is True."""
    dummy_obj = {"ok": 1}
    buffer = io.BytesIO()
    joblib.dump(dummy_obj, buffer)
    buffer.seek(0)

    mock_s3 = mock.Mock()
    mock_s3.download_fileobj.side_effect = lambda bucket, key, buf: buf.write(buffer.read())

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", True)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.s3", mock_s3)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.S3_BUCKET", "test-bucket")

    obj = download_from_s3("model.pkl")
    captured = capsys.readouterr()

    assert isinstance(obj, dict)
    assert "Downloaded models/model.pkl from S3" in captured.err
    mock_s3.download_fileobj.assert_called_once()


## upload_image_to_s3 tests

def test_upload_image_to_s3_local(monkeypatch, tmp_path, capsys):
    """Test uploading image locally when USE_S3 is False."""
    test_img = tmp_path / "test.png"
    test_img.write_text("fake image")

    static_dir = tmp_path / "static"
    static_dir.mkdir()

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", False)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.STATIC_DIR", str(static_dir))

    with mock.patch("src.telco_customer_churn_analysis.s3_utils.shutil.copy") as mock_copy:
        upload_image_to_s3(str(test_img), "xai/test.png")
        mock_copy.assert_called_once()
        captured = capsys.readouterr()
        assert "Saved image locally" in captured.err


def test_upload_image_to_s3_s3(monkeypatch, capsys, tmp_path):
    """Test uploading image to S3."""
    test_img = tmp_path / "test.png"
    test_img.write_text("fake image")
    mock_s3 = mock.Mock()

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", True)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.s3", mock_s3)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.S3_BUCKET", "test-bucket")

    upload_image_to_s3(str(test_img), "xai/test.png")
    captured = capsys.readouterr()
    assert "Uploaded image xai/test.png to S3" in captured.err
    mock_s3.upload_fileobj.assert_called_once()


## download_image_from_s3 tests

def test_download_image_from_s3_local(monkeypatch, tmp_path, capsys):
    """Test downloading image locally when USE_S3 is False."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", False)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.s3", None)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.STATIC_DIR", str(static_dir))

    result_path = download_image_from_s3("chart.png", "xai/chart.png")
    captured = capsys.readouterr()

    assert result_path == os.path.join("eda", "chart.png")
    assert "Using local image" in captured.err


def test_download_image_from_s3_s3(monkeypatch, tmp_path, capsys):
    """Test downloading image from S3."""
    mock_s3 = mock.Mock()
    static_dir = tmp_path / "static"
    static_dir.mkdir()

    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.USE_S3", True)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.s3", mock_s3)
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.S3_BUCKET", "test-bucket")
    monkeypatch.setattr("src.telco_customer_churn_analysis.s3_utils.STATIC_DIR", str(static_dir))

    result = download_image_from_s3("plot.png", "xai/plot.png")
    captured = capsys.readouterr()

    mock_s3.download_fileobj.assert_called_once()
    assert "Downloaded image xai/plot.png from S3" in captured.err
    assert result == os.path.join("eda", "plot.png")

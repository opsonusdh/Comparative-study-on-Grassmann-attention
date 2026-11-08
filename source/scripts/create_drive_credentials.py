#!/usr/bin/env python3
"""Create a refreshable PyDrive2 credentials JSON for use as a Kaggle Secret.

Usage:
    1. Download an OAuth client JSON from Google Cloud and save it as
       client_secrets.json beside this script.
    2. Run:  python scripts/create_drive_credentials.py
    3. Copy the generated credentials.json contents into a Kaggle Secret named
       GDRIVE_CREDENTIALS_JSON and attach that secret to the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydrive2.auth import GoogleAuth

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "client_secrets.json"
OUT = ROOT / "credentials.json"

if not CLIENT.exists():
    raise SystemExit(
        f"Missing {CLIENT}. Download your Google OAuth client JSON, rename it to "
        "client_secrets.json, and rerun this script."
    )

settings = {
    "client_config_backend": "file",
    "client_config_file": str(CLIENT),
    "save_credentials": True,
    "save_credentials_backend": "file",
    "save_credentials_file": str(OUT),
    "get_refresh_token": True,
    "oauth_scope": ["https://www.googleapis.com/auth/drive"],
}

gauth = GoogleAuth(settings=settings)
gauth.LocalWebserverAuth()

if not OUT.exists():
    gauth.SaveCredentialsFile(str(OUT))

payload = json.loads(OUT.read_text(encoding="utf-8"))
print(f"Created: {OUT}")
print("Credential JSON keys:", ", ".join(sorted(payload.keys())))
print("Now copy the COMPLETE credentials.json contents into the Kaggle Secret:")
print("  GDRIVE_CREDENTIALS_JSON")

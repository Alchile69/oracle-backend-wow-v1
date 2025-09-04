import os
import base64
import json
import tempfile
import firebase_admin
from firebase_admin import credentials, firestore

firebase_b64 = os.environ.get('FIREBASE_CREDENTIALS_B64')
if firebase_b64:
    try:
        cred_json = base64.b64decode(firebase_b64)
        cred_dict = json.loads(cred_json)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(cred_json)
            tmp_path = tmp.name
        cred = credentials.Certificate(tmp_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firestore initialized from Base64")
    except Exception as e:
        print(f"Firebase init failed: {e}")
        db = None
else:
    print("⚠️ No Firebase credentials found")
    db = None


import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

firebase_json = os.environ.get('FIREBASE_CREDENTIALS_JSON')
if firebase_json:
    try:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firestore configured from Railway env")
    except Exception as e:
        print(f"❌ Firebase init error: {e}")
        db = None
else:
    db = None
    print("⚠️ No Firebase credentials found")


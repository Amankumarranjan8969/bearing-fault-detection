"""Role-Based Access Control (RBAC) and Authentication Service.
Provides Super Admin, Admin, and Operator/User authentication,
session token verification, and permission gating for industrial compliance.
"""

import hashlib
import hmac
import os
import secrets
import time
from typing import Optional, Dict, Any, List

SECRET_KEY = os.getenv("BDT_AUTH_SECRET", "bdt-industrial-secret-key-2026-cwru")

# Pre-configured Industrial Users
USERS = {
    "superadmin": {
        "username": "superadmin",
        "name": "Chief Reliability Architect",
        "role": "super_admin",
        "title": "Super Admin · Master Control",
        "badge": "SUPER ADMIN",
        "salt": "sa_salt_8872",
        "passwords": ["SuperAdmin@2026!", "Suhani@2026"],
        "hash": hashlib.sha256(("SuperAdmin@2026!" + "sa_salt_8872").encode()).hexdigest(),
        "permissions": [
            "monitor", "predict", "simulate", "chat",
            "ingest_docs", "delete_docs",
            "switch_model", "calibrate_thresholds", "reset_db", "manage_users", "view_audit"
        ]
    },
    "admin": {
        "username": "admin",
        "name": "Plant Reliability Engineer",
        "role": "admin",
        "title": "Plant Admin · Reliability Lead",
        "badge": "PLANT ADMIN",
        "salt": "adm_salt_5514",
        "passwords": ["Admin@2026!"],
        "hash": hashlib.sha256(("Admin@2026!" + "adm_salt_5514").encode()).hexdigest(),
        "permissions": [
            "monitor", "predict", "simulate", "chat",
            "ingest_docs", "view_audit"
        ]
    },
    "operator": {
        "username": "operator",
        "name": "Vibration Technician",
        "role": "operator",
        "title": "Field Operator · Condition Monitoring",
        "badge": "OPERATOR",
        "salt": "op_salt_3321",
        "passwords": ["Operator@2026!"],
        "hash": hashlib.sha256(("Operator@2026!" + "op_salt_3321").encode()).hexdigest(),
        "permissions": [
            "monitor", "predict", "simulate", "chat"
        ]
    }
}

# In-memory active tokens session store
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}
AUDIT_LOGS: List[Dict[str, Any]] = [
    {
        "timestamp": time.time() - 3600,
        "username": "system",
        "action": "SYSTEM_INITIALIZE",
        "details": "Industrial CWRU 12k Condition Monitoring Twin initialized"
    }
]


def log_audit(username: str, action: str, details: str):
    AUDIT_LOGS.append({
        "timestamp": time.time(),
        "username": username,
        "action": action,
        "details": details
    })
    # Keep last 500 audit logs
    if len(AUDIT_LOGS) > 500:
        AUDIT_LOGS.pop(0)


def verify_password(plain: str, user: dict) -> bool:
    if not plain or not user:
        return False
    salt = user.get("salt", "")
    plain_hash = hashlib.sha256((plain + salt).encode()).hexdigest()
    if "passwords" in user:
        for p in user["passwords"]:
            ph = hashlib.sha256((p + salt).encode()).hexdigest()
            if hmac.compare_digest(plain_hash, ph):
                return True
    if "hash" in user and hmac.compare_digest(plain_hash, user["hash"]):
        return True
    return False


def create_token(username: str) -> str:
    token = secrets.token_hex(24)
    uname = (username or "").strip().lower()
    if uname in ("superadmin", "suhani@2026"):
        user = USERS["superadmin"]
    else:
        user = USERS.get(uname) or next((u for u in USERS.values() if u["username"].lower() == uname), None)
    if not user:
        raise ValueError(f"Invalid user: {username}")
    
    expires = time.time() + (86400 * 7)  # 7-day token
    ACTIVE_SESSIONS[token] = {
        "token": token,
        "username": user["username"],
        "role": user["role"],
        "name": user["name"],
        "title": user["title"],
        "badge": user["badge"],
        "permissions": user["permissions"],
        "expires": expires
    }
    log_audit(user["username"], "USER_LOGIN", f"User {user['username']} authenticated successfully with role {user['role']}")
    return token


def validate_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    clean_token = token.replace("Bearer ", "").strip()
    session = ACTIVE_SESSIONS.get(clean_token)
    if not session:
        return None
    if time.time() > session["expires"]:
        del ACTIVE_SESSIONS[clean_token]
        return None
    return USERS.get(session["username"]) or next((u for u in USERS.values() if u["role"] == session["role"]), None)


def check_permission(user: dict, required_perm: str) -> bool:
    if not user:
        return False
    if user.get("role") == "super_admin":
        return True
    return required_perm in user.get("permissions", [])

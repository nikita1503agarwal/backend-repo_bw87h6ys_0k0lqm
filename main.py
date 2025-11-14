import os
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import jwt
from bson import ObjectId

from database import db
from schemas import (
    User as UserSchema,
    Place as PlaceSchema,
    Activity, CulturalInsight, FoodRecommendation, Review,
    SignupRequest, LoginRequest, GoogleAuthRequest, UpdateProfileRequest,
    NearbyRequest, SearchRequest, OptimizeRouteRequest,
    AnalyticsEvent, BillingActivateRequest
)

# App setup
app = FastAPI(title="Explorer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security / Auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_EXPIRE_HOURS = 48

# Rate limit (very simple per-IP token bucket)
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "120"))  # requests per 10 minutes
_rate_buckets: Dict[str, Dict[str, Any]] = {}

COL_USERS = "user"
COL_PLACES = "place"
COL_ANALYTICS = "analyticsevent"

# Helpers

def oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_token(payload: dict) -> str:
    to_encode = payload.copy()
    to_encode.update({"exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)})
    return jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])  
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def rate_limit_dep(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_buckets.get(ip)
    window = 600.0
    if not bucket or now - bucket["start"] > window:
        _rate_buckets[ip] = {"count": 1, "start": now}
    else:
        if bucket["count"] >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket["count"] += 1

class AuthedUser(BaseModel):
    id: str
    email: EmailStr
    name: str
    is_admin: bool = False
    plan: str = "free"

async def get_current_user(authorization: Optional[str] = Header(None)) -> AuthedUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    data = decode_token(token)
    user = db[COL_USERS].find_one({"_id": oid(data["uid"])})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return AuthedUser(
        id=str(user["_id"]),
        email=user["email"],
        name=user.get("name", ""),
        is_admin=user.get("is_admin", False),
        plan=user.get("plan", "free")
    )

# Math helpers

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Health and test
@app.get("/")
def root():
    return {"app": "Explorer API", "status": "ok"}

@app.get("/test")
def test_database():
    ok = db is not None
    colls = []
    try:
        if db:
            colls = db.list_collection_names()
    except Exception:
        pass
    return {
        "backend": "✅ Running",
        "database": "✅ Connected" if ok else "❌ Not Connected",
        "collections": colls[:10]
    }

# Auth
@app.post("/auth/signup", dependencies=[Depends(rate_limit_dep)])
def signup(body: SignupRequest):
    existing = db[COL_USERS].find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    user_doc = UserSchema(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        plan="free",
        favorites=[],
        provider="local",
        is_admin=False,
    ).model_dump()
    res = db[COL_USERS].insert_one(user_doc)
    token = create_token({"uid": str(res.inserted_id)})
    return {"token": token}

@app.post("/auth/login", dependencies=[Depends(rate_limit_dep)])
def login(body: LoginRequest):
    user = db[COL_USERS].find_one({"email": body.email})
    if not user or not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"uid": str(user["_id"])})
    return {"token": token}

@app.post("/auth/google", dependencies=[Depends(rate_limit_dep)])
def google_auth(body: GoogleAuthRequest):
    # Stub: in production, verify id_token with Google
    email = f"google_{body.id_token[-8:]}@example.com"
    user = db[COL_USERS].find_one({"email": email})
    if not user:
        doc = UserSchema(
            name="Google User",
            email=email,
            password_hash=hash_password(os.urandom(8).hex()),
            plan="free",
            favorites=[],
            provider="google",
            is_admin=False,
        ).model_dump()
        res = db[COL_USERS].insert_one(doc)
        uid = str(res.inserted_id)
    else:
        uid = str(user["_id"]) 
    token = create_token({"uid": uid})
    return {"token": token}

# Users
@app.get("/users/me")
def me(user: AuthedUser = Depends(get_current_user)):
    return user

@app.put("/users/me")
def update_me(body: UpdateProfileRequest, user: AuthedUser = Depends(get_current_user)):
    update: Dict[str, Any] = {}
    if body.name is not None:
        update["name"] = body.name
    if body.plan is not None:
        update["plan"] = body.plan
    if update:
        db[COL_USERS].update_one({"_id": oid(user.id)}, {"$set": update})
    doc = db[COL_USERS].find_one({"_id": oid(user.id)})
    return {
        "id": user.id,
        "email": doc["email"],
        "name": doc.get("name", ""),
        "plan": doc.get("plan", "free")
    }

@app.post("/users/me/favorites")
def add_favorite(place_id: str, user: AuthedUser = Depends(get_current_user)):
    doc = db[COL_USERS].find_one({"_id": oid(user.id)})
    favs: List[str] = doc.get("favorites", [])
    is_premium = doc.get("plan", "free") == "premium"
    if not is_premium and len(favs) >= 10 and place_id not in favs:
        raise HTTPException(status_code=403, detail="Free tier allows up to 10 favorites")
    if place_id not in favs:
        favs.append(place_id)
        db[COL_USERS].update_one({"_id": oid(user.id)}, {"$set": {"favorites": favs}})
    return {"favorites": favs}

@app.delete("/users/me/favorites")
def remove_favorite(place_id: str, user: AuthedUser = Depends(get_current_user)):
    doc = db[COL_USERS].find_one({"_id": oid(user.id)})
    favs: List[str] = [f for f in doc.get("favorites", []) if f != place_id]
    db[COL_USERS].update_one({"_id": oid(user.id)}, {"$set": {"favorites": favs}})
    return {"favorites": favs}

# Places

def weighted_score(distance_km: float, popularity: float, reviews_avg: float, culture_richness: float) -> float:
    # Closer is better: invert distance weight
    dist_component = max(0.0, 1.0 - (distance_km / 25.0))
    return 0.4 * dist_component + 0.35 * (popularity/100.0) + 0.15 * (reviews_avg/5.0) + 0.10 * culture_richness

@app.post("/places/nearby")
def nearby(body: NearbyRequest, user: Optional[AuthedUser] = Depends(lambda: None)):
    # naive scan + compute distance
    query: Dict[str, Any] = {}
    if body.categories:
        query["category"] = {"$in": body.categories}
    items = list(db[COL_PLACES].find(query))
    results = []
    for it in items:
        d = haversine(body.latitude, body.longitude, it.get("latitude"), it.get("longitude"))
        if d <= body.radius_km:
            reviews = it.get("reviews", [])
            avg = sum([r.get("rating", 0) for r in reviews]) / len(reviews) if reviews else 0.0
            culture = 1.0 if it.get("cultural_insights") else 0.0
            score = weighted_score(d, float(it.get("popularity_score", 0)), avg, culture)
            results.append({
                "id": str(it["_id"]),
                "name": it.get("name"),
                "description": it.get("description"),
                "category": it.get("category"),
                "images": it.get("images", []),
                "entry_fee": it.get("entry_fee"),
                "best_hours": it.get("best_hours"),
                "open_now": it.get("open_now"),
                "family_friendly": it.get("family_friendly"),
                "latitude": it.get("latitude"),
                "longitude": it.get("longitude"),
                "distance_km": round(d, 2),
                "score": round(score, 4)
            })
    results.sort(key=lambda x: (-x["score"]))
    return {"results": results}

@app.get("/places/{place_id}")
def get_place(place_id: str):
    it = db[COL_PLACES].find_one({"_id": oid(place_id)})
    if not it:
        raise HTTPException(status_code=404, detail="Place not found")
    it["id"] = str(it.pop("_id"))
    return it

# Search
@app.post("/search")
def search(body: SearchRequest):
    q = body.query.strip()
    regex = {"$regex": q, "$options": "i"}
    items = list(db[COL_PLACES].find({
        "$or": [
            {"name": regex},
            {"description": regex},
            {"category": regex},
            {"activities.title": regex},
            {"cultural_insights.title": regex},
            {"food_recommendations.name": regex},
        ]
    }))
    out = [{"id": str(it["_id"]), "name": it.get("name"), "category": it.get("category"), "images": it.get("images", [])} for it in items]
    return {"results": out}

# Routing optimization (greedy nearest neighbor)
@app.post("/routes/optimize")
def optimize(body: OptimizeRouteRequest):
    origin_lat, origin_lng = body.origin
    ids = body.waypoint_ids
    waypoints = []
    for pid in ids:
        it = db[COL_PLACES].find_one({"_id": oid(pid)})
        if it:
            waypoints.append({"id": pid, "lat": it["latitude"], "lng": it["longitude"]})
    if not waypoints:
        return {"order": [], "distance_km": 0.0}
    visited = []
    cur_lat, cur_lng = origin_lat, origin_lng
    total = 0.0
    remaining = waypoints[:]
    while remaining:
        remaining.sort(key=lambda w: haversine(cur_lat, cur_lng, w["lat"], w["lng"]))
        nxt = remaining.pop(0)
        d = haversine(cur_lat, cur_lng, nxt["lat"], nxt["lng"]) 
        total += d
        visited.append(nxt["id"])
        cur_lat, cur_lng = nxt["lat"], nxt["lng"]
    return {"order": visited, "distance_km": round(total, 2), "mode": body.mode}

# Admin CRUD (guarded)

def require_admin(user: AuthedUser):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

@app.post("/admin/places")
def admin_create_place(body: PlaceSchema, user: AuthedUser = Depends(get_current_user)):
    require_admin(user)
    doc = body.model_dump()
    res = db[COL_PLACES].insert_one(doc)
    return {"id": str(res.inserted_id)}

@app.get("/admin/places")
def admin_list_places(user: AuthedUser = Depends(get_current_user)):
    require_admin(user)
    items = list(db[COL_PLACES].find({}).limit(200))
    for it in items:
        it["id"] = str(it.pop("_id"))
    return {"results": items}

@app.put("/admin/places/{place_id}")
def admin_update_place(place_id: str, body: PlaceSchema, user: AuthedUser = Depends(get_current_user)):
    require_admin(user)
    db[COL_PLACES].update_one({"_id": oid(place_id)}, {"$set": body.model_dump()})
    return {"id": place_id, "updated": True}

@app.delete("/admin/places/{place_id}")
def admin_delete_place(place_id: str, user: AuthedUser = Depends(get_current_user)):
    require_admin(user)
    db[COL_PLACES].delete_one({"_id": oid(place_id)})
    return {"deleted": True}

# Analytics
@app.post("/analytics/view/{place_id}")
def analytics_view(place_id: str, user: Optional[AuthedUser] = Depends(lambda: None)):
    evt = AnalyticsEvent(event_type="view", user_id=(user.id if user else None), place_id=place_id, metadata={"ts": datetime.utcnow().isoformat()}).model_dump()
    db[COL_ANALYTICS].insert_one(evt)
    return {"ok": True}

@app.get("/analytics/insights")
def analytics_insights(user: AuthedUser = Depends(get_current_user)):
    require_admin(user)
    most_saved = db[COL_USERS].aggregate([
        {"$unwind": "$favorites"},
        {"$group": {"_id": "$favorites", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ])
    saved = list(most_saved)
    most_viewed = db[COL_ANALYTICS].aggregate([
        {"$match": {"event_type": "view"}},
        {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ])
    viewed = list(most_viewed)
    return {"most_saved": saved, "most_viewed": viewed}

# Billing
@app.post("/billing/activate")
def billing_activate(body: BillingActivateRequest, user: AuthedUser = Depends(get_current_user)):
    # Stub: in production verify with Stripe/Razorpay
    if not body.token:
        raise HTTPException(status_code=400, detail="Payment token required")
    db[COL_USERS].update_one({"_id": oid(user.id)}, {"$set": {"plan": "premium"}})
    return {"plan": "premium"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

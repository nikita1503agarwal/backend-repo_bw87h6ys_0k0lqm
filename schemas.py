from typing import List, Optional, Literal
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

# Core domain schemas

class Activity(BaseModel):
    time_of_day: Literal["morning", "afternoon", "evening"]
    title: str
    description: Optional[str] = None

class CulturalInsight(BaseModel):
    title: str
    description: str
    category: Literal["traditions", "heritage", "etiquette", "stories"]

class FoodRecommendation(BaseModel):
    name: str
    cuisine: Optional[str] = None
    price_range: Literal["$", "$$", "$$$", "$$$$"] = "$"
    description: Optional[str] = None
    distance_km: Optional[float] = None

class Review(BaseModel):
    user_id: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    created_at: Optional[datetime] = None

class Place(BaseModel):
    name: str
    description: Optional[str] = None
    category: Literal[
        "temple", "park", "cafe", "museum", "mall", "waterfall", "food_street", "monument", "beach", "other"
    ] = "other"
    latitude: float
    longitude: float
    images: List[str] = []
    entry_fee: Optional[float] = None
    best_hours: Optional[str] = None
    popularity_score: float = 0.0
    open_now: Optional[bool] = None
    family_friendly: Optional[bool] = None
    activities: List[Activity] = []
    cultural_insights: List[CulturalInsight] = []
    food_recommendations: List[FoodRecommendation] = []
    reviews: List[Review] = []

class User(BaseModel):
    name: str
    email: EmailStr
    password_hash: str
    plan: Literal["free", "premium"] = "free"
    favorites: List[str] = []
    provider: Literal["local", "google"] = "local"
    is_admin: bool = False

class AnalyticsEvent(BaseModel):
    event_type: Literal["view", "save", "login", "search"]
    user_id: Optional[str] = None
    place_id: Optional[str] = None
    metadata: Optional[dict] = None

# Request/Response models

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    plan: Optional[Literal["free", "premium"]] = None

class NearbyRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = Field(gt=0, le=50)
    categories: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None

class OptimizeRouteRequest(BaseModel):
    origin: List[float]  # [lat, lng]
    mode: Literal["driving", "walking", "biking"] = "driving"
    waypoint_ids: List[str]

class BillingActivateRequest(BaseModel):
    provider: Literal["stripe", "razorpay"]
    token: str

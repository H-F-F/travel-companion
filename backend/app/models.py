from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


BudgetLevel = Literal["economy", "standard", "premium"]
Pace = Literal["relaxed", "normal", "fast"]
PlanningMode = Literal["auto", "candidate"]


class PlanRequest(BaseModel):
    city: str = Field(default="", description="City or address")
    travel_date: date
    days: int = Field(default=1, ge=1, le=14)
    planning_mode: PlanningMode = "auto"
    booking_date: date | None = None
    origin_city: str = ""
    origin_station: str = ""
    origin_station_code: str = ""
    destination_station: str = ""
    destination_station_code: str = ""
    budget_level: BudgetLevel = "standard"
    attraction_styles: list[str] = Field(default_factory=list)
    food_preferences: list[str] = Field(default_factory=list)
    selected_attractions: list[str] = Field(default_factory=list)
    excluded_attractions: list[str] = Field(default_factory=list)
    selected_foods: list[str] = Field(default_factory=list)
    excluded_foods: list[str] = Field(default_factory=list)
    pace: Pace = "normal"
    live_poi: bool = True
    location_lat: float | None = None
    location_lon: float | None = None
    include_hotel: bool = False
    hotel_price_max: int = Field(default=500, ge=100, le=5000)
    selected_hotel_name: str = ""
    selected_transport: dict = Field(default_factory=dict)


class WeatherSummary(BaseModel):
    city: str
    date: date
    max_temp_c: float
    min_temp_c: float
    precipitation_mm: float
    wind_speed_mps: float
    weather_label: str
    tips: list[str]


class RecommendationItem(BaseModel):
    name: str
    category: str
    score: float
    reason: str
    indoor: bool
    est_cost: str
    theme_key: str = ""
    theme_label: str = ""
    theme_tags: list[str] = Field(default_factory=list)


class ItineraryItem(BaseModel):
    time: str
    activity_type: Literal["attraction", "food", "hotel", "arrival"]
    name: str
    reason: str


class PlanResponse(BaseModel):
    weather: WeatherSummary
    attractions: list[RecommendationItem]
    foods: list[RecommendationItem]
    itinerary: list[ItineraryItem]
    budget_hint: str
    planning_mode: PlanningMode = "auto"
    estimated_days: int = 1
    estimated_hours: float = 0
    warnings: list[str] = Field(default_factory=list)
    candidate_attractions: list[RecommendationItem] = Field(default_factory=list)
    candidate_foods: list[RecommendationItem] = Field(default_factory=list)

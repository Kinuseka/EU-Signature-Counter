import asyncio
import datetime
import json
import pandas as pd
import requests
import uvicorn
import math
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Dict, Any, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
API_URL = "https://eci.ec.europa.eu/045/public/api/report/progression"
DEADLINE = datetime.datetime(2025, 7, 31, 23, 59, 59)
FETCH_INTERVAL_SECONDS = 5
HISTORY_MAX_LEN = 10000
RAW_DATA_DAYS = 7  # Keep 7 days of raw data, compress older data
HISTORY_FILE = Path("signature_history.json")
app = FastAPI(
    title="Signature Growth Tracker API",
    description="Provides signature data and predictions.",
)

# Middleware setup
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Memory-Optimized Data Storage ---
# 'history' = recent raw signature counts (last 7 days or up to 10k points)
# 'compressed_history' = older data aggregated by day/hour for memory efficiency
# 'arima_pre' = short-term predictions
# 'sarima_pre' = long-term predictions
signature_data: Dict[str, Any] = {
    "history": [],
    "compressed_history": {},  # Format: {"2025-01-15": {"count": 1440, "sum": 75600, "mean": 52.5, "M2": 12000}}
    "arima_pre": [],
    "sarima_pre": []
}

# We cache our prediction models so we don't have to recalculate them constantly
model_cache = {"model": None, "fit_time": None}

# --- Welford's Algorithm for Online Statistics ---
class OnlineStats:
    """
    Implements Welford's algorithm for computing running mean and variance.
    This allows us to maintain accurate statistics even with compressed data.
    """
    def __init__(self, count: int = 0, mean: float = 0.0, M2: float = 0.0):
        self.count = count
        self.mean = mean
        self.M2 = M2  # Sum of squares of differences from mean
    
    def update(self, value: float):
        """Add a single value to the running statistics."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def combine(self, other: 'OnlineStats') -> 'OnlineStats':
        """Combine two OnlineStats objects using parallel algorithm."""
        if self.count == 0:
            return OnlineStats(other.count, other.mean, other.M2)
        if other.count == 0:
            return OnlineStats(self.count, self.mean, self.M2)
        
        combined_count = self.count + other.count
        delta = other.mean - self.mean
        combined_mean = (self.mean * self.count + other.mean * other.count) / combined_count
        combined_M2 = self.M2 + other.M2 + delta * delta * self.count * other.count / combined_count
        
        return OnlineStats(combined_count, combined_mean, combined_M2)
    
    @property
    def variance(self) -> float:
        """Calculate variance from M2."""
        return self.M2 / self.count if self.count > 1 else 0.0
    
    @property
    def std_dev(self) -> float:
        """Calculate standard deviation."""
        return math.sqrt(self.variance)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "mean": self.mean,
            "M2": self.M2,
            "std_dev": self.std_dev
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'OnlineStats':
        """Create from dictionary."""
        return cls(data["count"], data["mean"], data["M2"])

def compress_historical_data():
    """
    Compress older data points into daily/hourly aggregates to save memory
    while maintaining statistical accuracy using Welford's algorithm.
    """
    if len(signature_data["history"]) <= HISTORY_MAX_LEN:
        return
    
    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=RAW_DATA_DAYS)
    
    # Separate recent and old data
    recent_data = []
    old_data = []
    
    for point in signature_data["history"]:
        if point["time"] >= cutoff_time:
            recent_data.append(point)
        else:
            old_data.append(point)
    
    # If we have old data to compress
    if old_data:
        # Group old data by date for daily compression
        daily_groups = {}
        for point in old_data:
            date_key = point["time"].strftime("%Y-%m-%d")
            if date_key not in daily_groups:
                daily_groups[date_key] = []
            daily_groups[date_key].append(point)
        
        # Compress each day's data
        for date_key, day_points in daily_groups.items():
            if date_key not in signature_data["compressed_history"]:
                # Create new compressed entry for this day
                stats = OnlineStats()
                for point in day_points:
                    stats.update(point["count"])
                signature_data["compressed_history"][date_key] = stats.to_dict()
            else:
                # Merge with existing compressed data for this day
                existing_stats = OnlineStats.from_dict(signature_data["compressed_history"][date_key])
                new_stats = OnlineStats()
                for point in day_points:
                    new_stats.update(point["count"])
                combined_stats = existing_stats.combine(new_stats)
                signature_data["compressed_history"][date_key] = combined_stats.to_dict()
    
    # Keep only recent raw data
    signature_data["history"] = recent_data
    
    # Clean up very old compressed data (older than 31 days)
    thirty_one_days_ago = datetime.datetime.now() - datetime.timedelta(days=31)
    dates_to_remove = []
    for date_key in signature_data["compressed_history"]:
        date = datetime.datetime.strptime(date_key, "%Y-%m-%d")
        if date < thirty_one_days_ago:
            dates_to_remove.append(date_key)
    
    for date_key in dates_to_remove:
        del signature_data["compressed_history"][date_key]
    
    print(f"\nCompressed {len(old_data)} old points into {len(daily_groups)} daily aggregates. Keeping {len(recent_data)} recent points.")

def get_combined_statistics() -> Dict[str, float]:
    """
    Calculate statistics across both compressed historical data and recent raw data.
    Uses Welford's algorithm to maintain accuracy.
    """
    if not signature_data["history"] and not signature_data["compressed_history"]:
        return {}
    
    # Start with compressed data statistics
    combined_stats = OnlineStats()
    
    # Add compressed historical data
    for date_key, compressed_data in signature_data["compressed_history"].items():
        daily_stats = OnlineStats.from_dict(compressed_data)
        combined_stats = combined_stats.combine(daily_stats)
    
    # Add recent raw data
    recent_stats = OnlineStats()
    for point in signature_data["history"]:
        recent_stats.update(point["count"])
    
    if recent_stats.count > 0:
        combined_stats = combined_stats.combine(recent_stats)
    
    if combined_stats.count == 0:
        return {}
    
    return {
        "total_points": combined_stats.count,
        "mean_signatures": combined_stats.mean,
        "std_dev_signatures": combined_stats.std_dev,
        "min_signatures": min([p["count"] for p in signature_data["history"]] + 
                             [data["mean"] for data in signature_data["compressed_history"].values()]) if signature_data["history"] or signature_data["compressed_history"] else 0,
        "max_signatures": max([p["count"] for p in signature_data["history"]] + 
                             [data["mean"] for data in signature_data["compressed_history"].values()]) if signature_data["history"] or signature_data["compressed_history"] else 0,
    }

# --- Data Persistence Functions ---
def save_history():
    """Saves our precious data to disk - including compressed historical data."""
    # JSON doesn't understand Python datetime objects, so we convert them to strings
    data_to_save = {
        "history": [
            {"time": p["time"].isoformat(), "count": p["count"]}
            for p in signature_data["history"]
        ],
        "compressed_history": signature_data["compressed_history"],
        "arima_pre": signature_data["arima_pre"],
        "sarima_pre": signature_data["sarima_pre"]
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(data_to_save, f, indent=2)

def load_history():
    """Loads our data from disk when the server starts up."""
    global signature_data
    if not HISTORY_FILE.exists():
        print("Starting fresh - no history file found.")
        return

    with open(HISTORY_FILE, "r") as f:
        try:
            loaded_data = json.load(f)
            
            # Handle multiple data formats for backward compatibility
            if isinstance(loaded_data, list):
                # Old format: just an array of data points
                signature_data = {
                    "history": [
                        {"time": datetime.datetime.fromisoformat(p["time"]), "count": p["count"]}
                        for p in loaded_data
                    ],
                    "compressed_history": {},
                    "arima_pre": [],
                    "sarima_pre": []
                }
                print(f"Migrated {len(loaded_data)} data points from old format.")
                # Compress old data immediately after loading
                compress_historical_data()
            else:
                # New format: load everything including compressed data
                signature_data = {
                    "history": [
                        {"time": datetime.datetime.fromisoformat(p["time"]), "count": p["count"]}
                        for p in loaded_data.get("history", [])
                    ],
                    "compressed_history": loaded_data.get("compressed_history", {}),
                    "arima_pre": loaded_data.get("arima_pre", []),
                    "sarima_pre": loaded_data.get("sarima_pre", [])
                }
                compressed_days = len(signature_data["compressed_history"])
                print(f"Loaded {len(signature_data['history'])} recent points, {compressed_days} compressed days, {len(signature_data['arima_pre'])} ARIMA, {len(signature_data['sarima_pre'])} SARIMA points.")
                
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading history: {e}. Starting fresh.")
            signature_data = {"history": [], "compressed_history": {}, "arima_pre": [], "sarima_pre": []}

# --- Background Task for Data Fetching ---
async def fetch_signature_data_periodically():
    """This runs in the background, constantly checking for new signature data.
    Think of it as the heartbeat of our application."""
    while True:
        try:
            response = requests.get(API_URL)
            response.raise_for_status()
            data = response.json()
            count = data.get("signatureCount")

            if count is not None:
                now = datetime.datetime.now()
                
                # Don't save duplicate data - if the count hasn't changed, why clutter our database?
                if not signature_data["history"] or signature_data["history"][-1]['count'] != count:
                    signature_data["history"].append({"time": now, "count": count})
                    
                    # Use our new compression logic instead of just truncating
                    compress_historical_data()
                    
                    save_history()  # Backup after every new data point (better safe than sorry!)
                    
                    # Show enhanced status with compressed data info
                    compressed_days = len(signature_data["compressed_history"])
                    total_stats = get_combined_statistics()
                    total_points = total_stats.get("total_points", len(signature_data["history"]))
                    print(f"\rData: {count:,} signatures | Recent: {len(signature_data['history'])} | Compressed: {compressed_days} days | Total: {total_points} points | {now.strftime('%H:%M:%S')}", end="", flush=True)

        except requests.exceptions.RequestException as e:
            print(f"\nError fetching data: {e}")
        
        await asyncio.sleep(FETCH_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup_event():
    """When the server starts, load our saved data and start watching for new signatures."""
    load_history()
    asyncio.create_task(fetch_signature_data_periodically())


# --- Prediction Logic ---
def get_arima_model(series: pd.Series):
    """
    ARIMA models are great for short-term predictions.
    They look at recent trends and say "if this pattern continues..."
    Perfect for minute-by-minute forecasting!
    """
    if len(series) < 5:
        return None  # Need at least some data to work with

    try:
        # ARIMA(1,1,1) means: use 1 past value, difference once, use 1 past error
        # It's a good general-purpose setting for most time series
        model = SARIMAX(series, order=(1, 1, 1))
        fitted_model = model.fit(disp=False)  # disp=False keeps it quiet
        return fitted_model
    except Exception as e:
        print(f"\nARIMA model error: {e}")
        return None

def get_sarima_model(series: pd.Series):
    """
    SARIMA adds seasonality to ARIMA - it can detect daily/weekly patterns.
    Like "people sign more petitions during lunch break" or "weekends are slower".
    We need more data for this (48 hours = 2 full days to detect patterns).
    """
    if len(series) < 48:
        return None  # Need at least 2 days of hourly data

    try:
        # The (1,1,0,24) part means we expect patterns every 24 hours
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 24))
        fitted_model = model.fit(disp=False)
        print(f"\nSARIMA model fitted with {len(series)} hourly points.")
        return fitted_model
    except Exception as e:
        print(f"\nSARIMA model error: {e}")
        return None


# --- API Endpoint ---
@app.get("/api/data")
@limiter.limit("10/second")  # Allow 10 requests per second per IP
async def get_data(request: Request):
    """
    This is the main API endpoint that returns everything:
    - Historical data (what actually happened)
    - Predictions (what we think will happen)
    - Current statistics (how fast things are growing right now)
    - Combined statistics from both raw and compressed data
    """
    if not signature_data["history"]:
        raise HTTPException(status_code=404, detail="No data available yet. Please wait.")

    # --- Prepare Data for Analysis ---
    # Convert our data to pandas DataFrames because they're easier to work with for time series
    history_df = pd.DataFrame(signature_data["history"])
    history_df['time'] = pd.to_datetime(history_df['time'])
    history_df = history_df.set_index('time')

    # Create different frequency series for different types of predictions
    minute_series = history_df['count'].resample('min').last().ffill()  # Minute-level data
    hourly_series = history_df['count'].resample('h').last().ffill()   # Hourly data
    
    # --- ARIMA Predictions (short-term, minute-level) ---
    arima_model = get_arima_model(minute_series)
    if arima_model:
        try:
            # Get predictions for data we already have (to see how accurate our model is)
            fitted_values = arima_model.fittedvalues
            forecast = arima_model.get_forecast(steps=60)  # Look 60 minutes into the future
            prediction_df = forecast.summary_frame(alpha=0.05)
            earliest_actual_time = signature_data["history"][0]['time'] if signature_data["history"] else None
            new_arima_predictions = []
            for idx, value in fitted_values.items():
                if earliest_actual_time is None or idx >= earliest_actual_time:
                    new_arima_predictions.append({
                        "time": idx.isoformat(), 
                        "count": max(0, round(value))
                    })       
            # Add future predictions
            for idx, row in prediction_df.iterrows():
                new_arima_predictions.append({
                    "time": idx.isoformat(), 
                    "count": max(0, round(row['mean']))
                })
            signature_data["arima_pre"] = new_arima_predictions
        except Exception as e:
            print(f"\nARIMA forecast error: {e}")
    arima_predictions = signature_data["arima_pre"]

    # --- SARIMA Predictions (More accurate, but needs more data) ---
    sarima_model = get_sarima_model(hourly_series)
    if sarima_model:
        try:
            # Same logic as ARIMA, but for longer-term patterns
            fitted_values = sarima_model.fittedvalues
            forecast = sarima_model.get_forecast(steps=48)  # Look 48 hours ahead
            prediction_df = forecast.summary_frame(alpha=0.05)
            earliest_actual_time = signature_data["history"][0]['time'] if signature_data["history"] else None
            new_sarima_predictions = []
            # Historical data
            for idx, value in fitted_values.items():
                if earliest_actual_time is None or idx >= earliest_actual_time:
                    new_sarima_predictions.append({
                        "time": idx.isoformat(), 
                        "count": max(0, round(value))
                    })
            
            # Future predictions
            for idx, row in prediction_df.iterrows():
                new_sarima_predictions.append({
                    "time": idx.isoformat(), 
                    "count": max(0, round(row['mean']))
                })
            signature_data["sarima_pre"] = new_sarima_predictions
        except Exception as e:
            print(f"\nSARIMA forecast error: {e}")
            
    sarima_predictions = signature_data["sarima_pre"]
    
    # --- Current Statistics (Enhanced with memory optimization) ---
    current_stats = {
        "current_count": signature_data["history"][-1]['count'],
        "per_second_growth": 0,
        "per_minute_growth": 0,
        "per_day_growth_raw": 0,
        "per_day_growth_rounded": 0,
        "deadline_prediction": 0
    }

    # Growth rate calculations using recent data
    if len(signature_data["history"]) > 1:
        last_entry = signature_data["history"][-1]
        prev_entry = signature_data["history"][-2]
        time_delta = (last_entry['time'] - prev_entry['time']).total_seconds()
        count_delta = last_entry['count'] - prev_entry['count']

        if time_delta > 0:
            per_second = count_delta / time_delta
            current_stats["per_second_growth"] = per_second
            current_stats["per_minute_growth"] = per_second * 60
            per_day = per_second * 60 * 60 * 24
            current_stats["per_day_growth_raw"] = per_day
            current_stats["per_day_growth_rounded"] = round(per_day / 100) * 100

        # Long-term growth calculation using all available data
        first_entry = signature_data["history"][0]
        total_time_delta_min = (last_entry['time'] - first_entry['time']).total_seconds() / 60
        total_count_delta = last_entry['count'] - first_entry['count']
        if total_time_delta_min > 0:
            avg_growth_per_min = total_count_delta / total_time_delta_min
            minutes_to_deadline = (DEADLINE - datetime.datetime.now()).total_seconds() / 60
            if minutes_to_deadline > 0:
                current_stats["deadline_prediction"] = last_entry['count'] + (avg_growth_per_min * minutes_to_deadline)
    four_hours_ago = datetime.datetime.now() - datetime.timedelta(hours=4)
    recent_history = [
        p for p in signature_data["history"] 
        if p["time"] >= four_hours_ago
    ]
    
    # --- Get Combined Historical Statistics ---
    combined_stats = get_combined_statistics()
    
    memory_info = {
        "raw_data_points": len(signature_data["history"]),
        "compressed_days": len(signature_data["compressed_history"]),
        "total_data_points": combined_stats.get("total_points", 0),
        "client_data_points": len(recent_history),
        "memory_efficiency": f"{len(signature_data['history'])} raw + {len(signature_data['compressed_history'])} compressed days"
    }
    
    return {
        "history": [
            {"time": p['time'].isoformat(), "count": p['count']} 
            for p in recent_history
        ],
        "arima_prediction": arima_predictions,
        "sarima_prediction": sarima_predictions,
        "stats": current_stats,
        "combined_stats": combined_stats,  # New: Statistics across all data (raw + compressed)
        "memory_info": memory_info  # New: Memory optimization information
    }

if __name__ == "__main__":
    # To run do "uvicorn server.main:app"
    uvicorn.run(app, host="0.0.0.0", port=8000) 
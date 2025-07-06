# EU Initiative Counter

Real-time signature tracking system with advanced memory optimization and predictive analytics.

## Quick Start

```bash
# Install dependencies
pip install -r server/requirements.txt

# Run the server
cd server
python main.py

# Access API
curl http://localhost:8000/api/data
```

## API Response
```js
{
    'history': [],
    'arima_prediction': [],
    'sarima_prediction': [],
    'stats': {
        'current_count': 0.0f,
        'per_second_growth': 0.0f,
        'per_minute_growth': 0.0f,
        'deadline_prediction': 0.0f,
        'per_day_growth_raw': 0.0f,
        'per_day_grwoth_rounded': 0,
    },
    'memory_info': [],
    'combined_stats': {...}
}
```
from datetime import datetime

def detect_current_season():
    """Detect the current season based on the month (Northern Hemisphere logic)."""
    month = datetime.now().month

    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"

import requests, time, random, math
from datetime import datetime

API_URL = "http://127.0.0.1:8000/sensor"

print("==================================================")
print("🇺🇬 MUKONO DISTRICT VIRTUAL IOT SENSOR ACTIVE")
print("Sending real-time telemetry based on current time.")
print("==================================================\n")

# Base load for the Mukono feeder
base_load = 8500.0 # MW

while True:
    now = datetime.now()
    
    # Get exact time as a decimal (e.g., 14:30 = 14.5)
    time_decimal = now.hour + (now.minute / 60.0)
    
    # 1. SIMULATE TEMPERATURE (Uganda Equatorial Curve)
    # Lowest at 6 AM (22°C), Highest at 3 PM (31°C)
    temp_base = 26.5 - (4.5 * math.cos((time_decimal - 6) / 24 * 2 * math.pi))
    temp = temp_base + random.uniform(-0.3, 0.3) # Add tiny natural variation
    
    # 2. SIMULATE LIGHT (Lux)
    # Daylight between 7 AM (7.0) and 7 PM (19.0)
    if 7.0 <= time_decimal <= 19.0:
        # Peak sun at 1 PM (13.0)
        light_base = 1000 * math.sin((time_decimal - 7) / 12 * math.pi)
        light = max(50, light_base + random.uniform(-50, 50))
    else:
        light = random.uniform(0, 5) # Moonlight/barely any light
        
    # 3. SIMULATE VOLTAGE (Grid stability)
    # 1 in 20 chance of a brownout (voltage sag) to trigger the impact metrics
    if random.random() < 0.05: 
        voltage = random.uniform(195, 215) # Brownout condition!
        print(f"⚡ [GRID EVENT] Voltage sag detected: {voltage:.1f}V")
    else:
        voltage = random.uniform(235, 245)
        
    payload = {
        "temp": round(temp, 1),
        "light": round(light, 1),
        "voltage": round(voltage, 1),
        "humidity": round(random.uniform(60, 85), 1)
    }
    
    try:
        # Send to FastAPI backend
        requests.post(API_URL, json=payload, timeout=2)
        
        # Print to terminal so you can see it working during the demo
        time_str = now.strftime("%H:%M:%S")
        print(f"[{time_str}] Temp: {payload['temp']}°C | Light: {int(payload['light'])} Lux | Voltage: {payload['voltage']}V")
        
    except Exception as e:
        print(f"❌ Backend offline: Make sure uvicorn is running.")

    # Wait 3 seconds before next reading (feels like a real live stream)
    time.sleep(3)
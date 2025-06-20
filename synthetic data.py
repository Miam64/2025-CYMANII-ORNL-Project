import time
import csv
import random
from datetime import datetime

# Config
LOG_INTERVAL = 0.1  # seconds
SIMULATION_DURATION = 10  # seconds (for quick testing)

# Create CSV file
filename = f"simulated_vibration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Vibration"])  # Header

    print(f"Simulated logging to {filename}...")

    start_time = time.time()
    try:
        while time.time() - start_time < SIMULATION_DURATION:
            # Simulated vibration reading (0 = no vibration, 1 = vibration)
            state = random.choices([0, 1], weights=[0.8, 0.2])[0]  # mostly 0s
            timestamp = datetime.now().isoformat()
            writer.writerow([timestamp, state])
            print(f"{timestamp}, Vibration: {state}")
            time.sleep(LOG_INTERVAL)
    except KeyboardInterrupt:
        print("Logging stopped.")
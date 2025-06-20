import time
import csv
import random
from datetime import datetime

# Config
LOG_INTERVAL = 0.1  # seconds
SIMULATION_DURATION = 10  # seconds

filename = f"abnormal_vibration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Vibration"])  # Header

    print(f"Simulated abnormal logging to {filename}...")

    start_time = time.time()
    try:
        while time.time() - start_time < SIMULATION_DURATION:
            # Abnormal vibration simulation: more 1s and spikes
            # e.g., 60% chance vibration, plus random spikes (extra 1s)
            base_state = random.choices([0, 1], weights=[0.4, 0.6])[0]

            # Random spikes: 20% chance to force a 1 (high vibration)
            if random.random() < 0.2:
                base_state = 1

            timestamp = datetime.now().isoformat()
            writer.writerow([timestamp, base_state])
            print(f"{timestamp}, Vibration: {base_state}")
            time.sleep(LOG_INTERVAL)

    except KeyboardInterrupt:
        print("Logging stopped.")
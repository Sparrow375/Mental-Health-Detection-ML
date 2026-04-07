import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class LatLonPoint:
    def __init__(self, lat, lon, timeMs, accuracy):
        self.lat = lat
        self.lon = lon
        self.timeMs = timeMs
        self.accuracy = accuracy

startOfDayMs = 1700000000000
nowMs = startOfDayMs + (24 * 3600_000)
homeLat, homeLon = 40.0, -73.0

def run_simulation(name, lastNightFix, snaps):
    print(f"\n=============================================")
    print(f"SCENARIO: {name}")
    print(f"=============================================")
    
    # 1. Displacement
    filteredSnaps = [s for s in snaps if s.accuracy <= 800.0]
    sortedSnaps = sorted(filteredSnaps, key=lambda s: s.timeMs)
    distKm = 0.0
    lastCell = None
    lastCellLat = lastCellLon = 0.0
    for snap in sortedSnaps:
        cell = f"{snap.lat:.4f},{snap.lon:.4f}"
        if lastCell is None:
            lastCell = cell; lastCellLat = snap.lat; lastCellLon = snap.lon
        elif cell != lastCell:
            segmentKm = haversine(lastCellLat, lastCellLon, snap.lat, snap.lon)
            distKm += segmentKm
            lastCell = cell; lastCellLat = snap.lat; lastCellLon = snap.lon
    print(f"-> Total Distance: {distKm:.2f} km")

    # 2. Home Time
    BRIDGE_CAP_MS = 12 * 3600_000
    DAY_MS = 24 * 3600_000
    def isNearHome(s): return haversine(s.lat, s.lon, homeLat, homeLon) < 0.5
    
    homeSnaps = [s for s in snaps if startOfDayMs <= s.timeMs <= nowMs and s.accuracy <= 800.0]
    homeSnaps = sorted(homeSnaps, key=lambda s: s.timeMs)
    homeTimeMs = 0
    
    if len(homeSnaps) == 0:
        if lastNightFix and isNearHome(lastNightFix):
            homeTimeMs = min(nowMs - startOfDayMs, BRIDGE_CAP_MS)
    else:
        firstSnap = homeSnaps[0]
        midnightNearHome = isNearHome(lastNightFix) if lastNightFix else isNearHome(firstSnap)
        if midnightNearHome:
            homeTimeMs += min(firstSnap.timeMs - startOfDayMs, BRIDGE_CAP_MS)
        
        for i in range(len(homeSnaps) - 1):
            if isNearHome(homeSnaps[i]):
                gap = homeSnaps[i+1].timeMs - homeSnaps[i].timeMs
                homeTimeMs += min(gap, BRIDGE_CAP_MS)
                
        lastSnap = homeSnaps[-1]
        if isNearHome(lastSnap):
            homeTimeMs += min(nowMs - lastSnap.timeMs, BRIDGE_CAP_MS)

    homeHours = homeTimeMs / 3600_000
    print(f"-> Time at Home:   {homeHours:.1f} hours ({(homeHours/24)*100:.1f}%)")

# -------------------------------------------------------------
# Scenario 1: The "Phone Died" Gap (Device offline for 6 hours)
# -------------------------------------------------------------
# Leaves work at 17:00, phone dies instantly. Plugged in at home at 23:00.
last_night = LatLonPoint(homeLat, homeLon, startOfDayMs - 1000, 50.0)
snaps_1 = [
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 100.0), # Home till 08:00
    LatLonPoint(40.045, -73.0, startOfDayMs + (9 * 3600_000), 100.0),    # Work starts
    LatLonPoint(40.045, -73.0, startOfDayMs + (17 * 3600_000), 100.0),   # Leaves work. Phone dies immediately.
    # ... GAP 6 HOURS ...
    LatLonPoint(homeLat, homeLon, startOfDayMs + (23 * 3600_000), 100.0) # Plugged in at home 23:00
]
run_simulation("1. Phone Dead for 6 Hours (Offline Gap)", last_night, snaps_1)

# -------------------------------------------------------------
# Scenario 2: Deep Doze / Zero Pings
# -------------------------------------------------------------
# Patient stays home bed-ridden. Phone sleeps deeply. ZERO GPS fixes logged today.
last_night = LatLonPoint(homeLat, homeLon, startOfDayMs - 1000, 50.0)
snaps_2 = []
run_simulation("2. Zero GPS Pings (Deep Doze/Stationary)", last_night, snaps_2)

# -------------------------------------------------------------
# Scenario 3: The Midnight Shift Worker
# -------------------------------------------------------------
# At work (away from home) from 20:00 last night to 06:00 today.
# Sleeps all day at home (07:00 to 19:00).
# The overnight anchor should CORRECTLY recognize they are NOT at home at midnight.
last_night = LatLonPoint(40.045, -73.0, startOfDayMs - 1000, 50.0) # Left last night AT WORK
snaps_3 = [
    LatLonPoint(40.045, -73.0, startOfDayMs + (6 * 3600_000), 100.0),  # Getting off work at 06:00
    LatLonPoint(homeLat, homeLon, startOfDayMs + (7 * 3600_000), 100.0), # Arrive home 07:00
    LatLonPoint(homeLat, homeLon, startOfDayMs + (19 * 3600_000), 100.0) # Wake up and leave 19:00
]
run_simulation("3. Night Shift Worker (Away at Midnight)", last_night, snaps_3)

# -------------------------------------------------------------
# Scenario 4: GPS Triangulation Ghost Jump
# -------------------------------------------------------------
# Patient stays home all day. At 14:00, terrible cell-tower connection causes 
# GPS to "wobble" and report a location 600m away (just inside 800m limit, so it gets logged).
# It returns to home at 14:05.
last_night = LatLonPoint(homeLat, homeLon, startOfDayMs - 1000, 50.0)
wobbleLat = homeLat + 0.0054 # Approx 600m away
snaps_4 = [
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 100.0),   
    LatLonPoint(wobbleLat, homeLon, startOfDayMs + (14 * 3600_000), 750.0),  # Ghost Jump Out
    LatLonPoint(wobbleLat, homeLon, startOfDayMs + (14.05 * 3600_000), 750.0), # Ghost Jump Still
    LatLonPoint(homeLat, homeLon, startOfDayMs + (15 * 3600_000), 100.0)   # Jump Back
]
run_simulation("4. GPS Ghost Jump (Triangulation wobble)", last_night, snaps_4)

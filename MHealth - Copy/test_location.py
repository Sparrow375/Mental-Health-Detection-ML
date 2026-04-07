import math

# --- Haversine implementation from Android ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- Classes ---
class LatLonPoint:
    def __init__(self, lat, lon, timeMs, accuracy, speed):
        self.lat = lat
        self.lon = lon
        self.timeMs = timeMs
        self.accuracy = accuracy
        self.speed = speed

# --- Test Environment variables ---
startOfDayMs = 1700000000000
nowMs = startOfDayMs + (24 * 3600_000)

homeLat = 40.0
homeLon = -73.0
lastNightFix = LatLonPoint(homeLat, homeLon, startOfDayMs - 1000, 50.0, 0.0) # From yesterday

# --- Test Data: Budget phone snaps (all ~300m accuracy, which previously 
# would have been rejected by the 200m filter) ---
# Patient wakes up at home, stays there for 8 hours (midnight to 08:00)
# Then commutes to work 5km away. Stays there until 17:00.
# Then commutes home.

snaps = [
    # 08:00 (left home)
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 300.0, 0.0),
    # 08:30 (arrive at work, 5km away)
    LatLonPoint(40.045, -73.0, startOfDayMs + (8.5 * 3600_000), 350.0, 0.0),
    # 17:00 (leave work)
    LatLonPoint(40.045, -73.0, startOfDayMs + (17 * 3600_000), 320.0, 0.0),
    # 17.5 (arrive home)
    LatLonPoint(homeLat, homeLon, startOfDayMs + (17.5 * 3600_000), 280.0, 0.0)
]

print("--- EXECUTING BUGFIX ALGORITHM ON BUDGET PHONE DATA ---")

# --- Step 1: Displacement ---
WALK_SPEED_MS = 6.0 / 3.6
filteredSnaps = [s for s in snaps if s.accuracy <= 800.0]
# BEFORE THE FIX, this list would be empty and distance would be 0.0!

sortedSnaps = sorted(filteredSnaps, key=lambda s: s.timeMs)

distKm = 0.0
lastCell = None
lastCellLat = 0.0
lastCellLon = 0.0
lastCellTimeMs = 0

for snap in sortedSnaps:
    cell = f"{snap.lat:.4f},{snap.lon:.4f}"
    if lastCell is None:
        lastCell = cell; lastCellLat = snap.lat; lastCellLon = snap.lon; lastCellTimeMs = snap.timeMs
    elif cell != lastCell:
        segmentKm = haversine(lastCellLat, lastCellLon, snap.lat, snap.lon)
        distKm += segmentKm
        lastCell = cell; lastCellLat = snap.lat; lastCellLon = snap.lon; lastCellTimeMs = snap.timeMs

print(f"Total Displacement using 800m filter: {distKm:.2f} km")

# --- Step 2: Home Time Ratio ---
BRIDGE_CAP_MS = 12 * 3600_000
DAY_MS = 24 * 3600_000

def isNearHome(s):
    return haversine(s.lat, s.lon, homeLat, homeLon) < 0.5

homeSnaps = [s for s in snaps if startOfDayMs <= s.timeMs <= nowMs and s.accuracy <= 800.0]
homeSnaps = sorted(homeSnaps, key=lambda s: s.timeMs)

homeTimeMs = 0

if len(homeSnaps) == 0:
    if lastNightFix and isNearHome(lastNightFix):
        homeTimeMs = min(nowMs - startOfDayMs, BRIDGE_CAP_MS)
else:
    # 1. Midnight bridge using yesterday's fix
    firstSnap = homeSnaps[0]
    midnightNearHome = isNearHome(lastNightFix) if lastNightFix else isNearHome(firstSnap)
    if midnightNearHome:
        homeTimeMs += min(firstSnap.timeMs - startOfDayMs, BRIDGE_CAP_MS)
    
    # 2. Pairs
    for i in range(len(homeSnaps) - 1):
        if isNearHome(homeSnaps[i]):
            gap = homeSnaps[i+1].timeMs - homeSnaps[i].timeMs
            homeTimeMs += min(gap, BRIDGE_CAP_MS)
            
    # 3. Last snap to now
    lastSnap = homeSnaps[-1]
    if isNearHome(lastSnap):
        homeTimeMs += min(nowMs - lastSnap.timeMs, BRIDGE_CAP_MS)

homeRatio = homeTimeMs / DAY_MS
homeHours = homeTimeMs / 3600_000

print(f"Home Time using last-night bridge + 800m filter: {homeHours:.1f} hours ({homeRatio*100:.1f}%)")

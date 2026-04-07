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
    print(f"\n[{name}]")
    
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
    print(f"Dist: {distKm:4.1f}km | Home: {homeHours:4.1f}h ({(homeHours/24)*100:5.1f}%)")


print("\n=== RUNNING 10 EXTREME EDGE CASE SIMULATIONS ===")

anchor_home = LatLonPoint(homeLat, homeLon, startOfDayMs - 1000, 50.0)
anchor_away = LatLonPoint(40.5, -73.0, startOfDayMs - 1000, 50.0)

# 1. Perfect Normal Day
snaps1 = [
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 50.0),
    LatLonPoint(40.1, -73.0, startOfDayMs + (9 * 3600_000), 50.0),  # 11km away
    LatLonPoint(40.1, -73.0, startOfDayMs + (17 * 3600_000), 50.0),
    LatLonPoint(homeLat, homeLon, startOfDayMs + (18 * 3600_000), 50.0),
]
run_simulation("1. Textbook Routine (Office 9-to-5)", anchor_home, snaps1)


# 2. Insomniac Wanderer 
snaps2 = [
    LatLonPoint(homeLat, homeLon, startOfDayMs + (2 * 3600_000), 50.0), # Home until 2 AM
    LatLonPoint(40.01, -73.0, startOfDayMs + (3 * 3600_000), 50.0),     # 1km away at 3 AM
    LatLonPoint(homeLat, homeLon, startOfDayMs + (4 * 3600_000), 50.0), # Back home at 4 AM
]
run_simulation("2. Insomniac Wanderer (02:00-04:00 AM walk)", anchor_home, snaps2)


# 3. GPS Toggled OFF Mid-Day
snaps3 = [
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 50.0), # Home till 8
    LatLonPoint(40.1, -73.0, startOfDayMs + (8.5 * 3600_000), 50.0),    # Arrive Work
    # GPS toggled off for 10 hours...
    LatLonPoint(homeLat, homeLon, startOfDayMs + (18.5 * 3600_000), 50.0), # Toggled on at home at 18:30
]
run_simulation("3. GPS Toggled OFF at Work (10h gap)", anchor_home, snaps3)


# 4. App Installed Mid-Day (No Anchor)
snaps4 = [
    LatLonPoint(40.1, -73.0, startOfDayMs + (14 * 3600_000), 50.0),    # Service starts at 14:00 at work
    LatLonPoint(homeLat, homeLon, startOfDayMs + (18 * 3600_000), 50.0), # Goes home at 18:00
]
run_simulation("4. First Day / App installed at 2:00 PM (No midnight anchor)", None, snaps4)


# 5. Delivery Driver 
snaps5 = [LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 50.0)]
for i in range(12): # 12 stops between 8:00 and 20:00
    snaps5.append(LatLonPoint(40.01 + (i*0.01), -73.0 + (i*0.01), startOfDayMs + ((8 + i) * 3600_000), 50.0))
snaps5.append(LatLonPoint(homeLat, homeLon, startOfDayMs + (20 * 3600_000), 50.0))
run_simulation("5. Delivery Driver (Moving constantly 08:00 to 20:00)", anchor_home, snaps5)


# 6. Ultra-Budget Phone (All GPS > 800m)
snaps6 = [
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 1200.0),
    LatLonPoint(40.1, -73.0, startOfDayMs + (12 * 3600_000), 900.0),
]
run_simulation("6. Broken GPS Antenna (All pings > 800m accuracy)", anchor_home, snaps6)


# 7. Forgetful User (Leaves phone at home, goes to work)
snaps7 = [
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 50.0),
    LatLonPoint(homeLat, homeLon, startOfDayMs + (12 * 3600_000), 50.0),
    LatLonPoint(homeLat, homeLon, startOfDayMs + (18 * 3600_000), 50.0),
]
run_simulation("7. Leaves Phone on Nightstand All Day (Never moves)", anchor_home, snaps7)


# 8. Night Shift - 2nd Day Cross
snaps8 = [
    LatLonPoint(40.1, -73.0, startOfDayMs + (7 * 3600_000), 50.0),  # At work until 07:00
    LatLonPoint(homeLat, homeLon, startOfDayMs + (8 * 3600_000), 50.0), # Arrive home 08:00
    LatLonPoint(homeLat, homeLon, startOfDayMs + (22 * 3600_000), 50.0), # Leave home 22:00
    LatLonPoint(40.1, -73.0, startOfDayMs + (23 * 3600_000), 50.0), # At work 23:00 to midnight
]
run_simulation("8. Night Shift Worker (Crosses midnight at Work)", anchor_away, snaps8)


# 9. Extended Out-of-Town Vacation
snaps9 = [
    LatLonPoint(45.0, -80.0, startOfDayMs + (8 * 3600_000), 50.0),  # In another state
    LatLonPoint(45.0, -80.0, startOfDayMs + (15 * 3600_000), 50.0),
]
anchor_vacation = LatLonPoint(45.0, -80.0, startOfDayMs - 1000, 50.0)
run_simulation("9. On Vacation (Out of state 1000+ km away)", anchor_vacation, snaps9)


# 10. The Micro-Jitter (Phone on desk, GPS drifts 20m constantly)
snaps10 = []
for i in range(20):
    jitterLat = homeLat + (0.0001 if i%2==0 else -0.0001) # Approx 10m drift
    snaps10.append(LatLonPoint(jitterLat, homeLon, startOfDayMs + (12 * 3600_000) + (i * 300_000), 50.0))
# Jitter stays within the `%.4f` 11-meter cluster
run_simulation("10. GPS Micro-Drift (Phone on desk bouncing 10m)", anchor_home, snaps10)

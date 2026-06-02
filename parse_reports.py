import json

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def print_reports(data):
    reports = data.get("analysis_reports", [])
    print("=== Analysis Reports from JSON ===")
    for i, r in enumerate(reports):
        date = r.get("date")
        detected = r.get("anomalyDetected")
        score = r.get("anomalyScore")
        alert = r.get("alertLevel")
        sustained = r.get("sustainedDays")
        proto = r.get("prototypeMatch")
        conf = r.get("prototypeConfidence")
        msg = r.get("anomalyMessage", "")
        print(f"Date: {date} | Score: {score:.4f} | Alert: {alert} | Sustained: {sustained} | Match: {proto} ({conf:.4f})")
        if msg:
            print(f"  Message: {msg}")

if __name__ == "__main__":
    print_reports(load_data(r"F:\Avaneesh\download\testData\26-2-1.json"))

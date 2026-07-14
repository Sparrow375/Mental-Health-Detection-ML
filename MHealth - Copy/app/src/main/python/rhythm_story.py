import math

def generate_rhythm_story(current_data, baseline_means, baseline_stds, s1_report, user_profile, day_number, confidence):
    """
    Generates a human-friendly narrative describing the user's daily circadian rhythm.
    
    Parameters:
    -----------
    current_data: dict of feature_name -> float value
    baseline_means: dict of feature_name -> float mean
    baseline_stds: dict of feature_name -> float stddev
    s1_report: AnomalyReport object or dict (containing alert_level, etc.)
    user_profile: UserProfile object or dict (containing age, profession, country, etc.)
    day_number: int (current monitoring day)
    confidence: float (baseline maturity confidence, 0.0 to 1.0)
    """
    # Extract alert level
    alert_level = "green"
    flagged_features = []
    if s1_report:
        if hasattr(s1_report, "alert_level"):
            alert_level = s1_report.alert_level
        elif isinstance(s1_report, dict):
            alert_level = s1_report.get("alert_level", "green")
            
        if hasattr(s1_report, "flagged_features"):
            flagged_features = s1_report.flagged_features
        elif isinstance(s1_report, dict):
            flagged_features = s1_report.get("flagged_features", [])

    # Extract user profile demographics
    profession = "student"
    gender = "prefer_not_to_say"
    if user_profile:
        if hasattr(user_profile, "employment"):
            profession = user_profile.employment
        elif isinstance(user_profile, dict):
            profession = user_profile.get("employment", "student")
            
        if hasattr(user_profile, "gender"):
            gender = user_profile.gender
        elif isinstance(user_profile, dict):
            gender = user_profile.get("gender", "prefer_not_to_say")

    # Group term for demographics
    role_term = "user"
    if "student" in profession.lower():
        role_term = "student"
    elif "engineer" in profession.lower() or "developer" in profession.lower() or "tech" in profession.lower():
        role_term = "professional"
    elif "doctor" in profession.lower() or "nurse" in profession.lower() or "medical" in profession.lower():
        role_term = "healthcare professional"
    elif profession and profession.lower() != "other":
        role_term = profession.lower()

    # Determine phase
    is_onboarding = day_number <= 7 or confidence < 0.35

    # Safe retrieval helpers
    def get_val(feat, default=0.0):
        return current_data.get(feat, default) if current_data else default

    def get_mean(feat, default=0.0):
        return baseline_means.get(feat, default) if baseline_means else default

    def get_std(feat, default=1.0):
        val = baseline_stds.get(feat, default) if baseline_stds else default
        return val if val > 0.05 else 1.0

    # Key features for story
    sleep_dur = get_val("sleepDurationHours")
    sleep_dur_mean = get_mean("sleepDurationHours")
    sleep_dur_std = get_std("sleepDurationHours")

    screen_time = get_val("screenTimeHours")
    screen_time_mean = get_mean("screenTimeHours")
    screen_time_std = get_std("screenTimeHours")

    steps = get_val("dailyStepCount")
    steps_mean = get_mean("dailyStepCount")

    wake_hour = get_val("wakeTimeHour")
    wake_mean = get_mean("wakeTimeHour")

    sleep_hour = get_val("sleepTimeHour")
    sleep_mean = get_mean("sleepTimeHour")

    social_ratio = get_val("socialAppRatio")
    social_ratio_mean = get_mean("socialAppRatio")

    # Time format helpers
    def format_hour(h):
        h = h % 24
        hour = int(h)
        minute = int((h - hour) * 60)
        meridiem = "AM" if h < 12 or h >= 24 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minute:02d} {meridiem}"

    def circular_diff(v1, v2):
        diff = v1 - v2
        return ((diff + 12.0) % 24.0) - 12.0

    # 1. Onboarding Phase Story
    if is_onboarding:
        sentences = [
            f"Lumen is currently mapping your unique behavioral rhythms."
        ]
        
        # Mention sleep window if logged
        if sleep_dur > 0:
            sentences.append(f"Today, you slept for {sleep_dur:.1f} hours, winding down around {format_hour(sleep_hour)}.")
        else:
            sentences.append("We're tracking your sleep window to build a circadian baseline.")

        # Mention screen time
        if screen_time > 0:
            sentences.append(f"Your device screen time was {screen_time:.1f} hours, with social apps accounting for {int(social_ratio * 100)}% of usage.")
            
        # Add educational onboarding note
        sentences.append("As we gather more daily data, Lumen will begin highlighting subtle shifts and deviations.")
        return " ".join(sentences)

    # 2. Mature Phase Story
    # Check deviations (Z-scores)
    sleep_z = (sleep_dur - sleep_dur_mean) / sleep_dur_std if sleep_dur_mean > 0 else 0.0
    screen_z = (screen_time - screen_time_mean) / screen_time_std if screen_time_mean > 0 else 0.0
    wake_diff = circular_diff(wake_hour, wake_mean)
    sleep_diff = circular_diff(sleep_hour, sleep_mean)

    # Construct the story based on the alert level
    if alert_level == "green":
        # Check if it was an exceptionally calm/perfectly aligned day
        is_very_calm = abs(sleep_z) < 1.0 and abs(screen_z) < 1.0 and abs(wake_diff) < 0.75 and abs(sleep_diff) < 0.75
        
        if is_very_calm:
            sentences = [
                f"Your day followed a highly stable, optimal rhythm.",
                f"You fell asleep within {int(abs(sleep_diff)*60)} minutes of your regular bedtime and slept a balanced {sleep_dur:.1f} hours.",
                f"Screen usage ({screen_time:.1f} hours) and physical activity ({int(steps):,} steps) both matched your typical baseline perfectly.",
                f"This consistency represents a healthy, grounded day for your routine."
            ]
        else:
            # Minor deviations present but overall calm
            sentences = ["Your rhythm remained healthy and stable today, with just a few minor shifts."]
            
            # Describe sleep duration deviation if notable
            if abs(sleep_z) >= 1.0:
                dir_word = "more" if sleep_z > 0 else "less"
                sentences.append(f"You slept {abs(sleep_dur - sleep_dur_mean):.1f} hours {dir_word} than usual.")
            elif abs(sleep_diff) >= 0.75:
                dir_word = "later" if sleep_diff > 0 else "earlier"
                sentences.append(f"You went to sleep about {int(abs(sleep_diff)*60)} minutes {dir_word} than your typical time.")

            # Describe screen time if notable
            if abs(screen_z) >= 1.0:
                dir_word = "higher" if screen_z > 0 else "lower"
                sentences.append(f"Your screen activity was slightly {dir_word} at {screen_time:.1f} hours.")
            
            # Describe steps if notable
            step_diff = steps - steps_mean
            if abs(step_diff) > 2000:
                dir_word = "more active" if step_diff > 0 else "more sedentary"
                sentences.append(f"You were {dir_word} than usual, recording {int(steps):,} steps.")

            sentences.append("Overall, these micro-adjustments are standard variations that do not disrupt your core pattern.")
        
        return " ".join(sentences)

    else:
        # Yellow/Orange/Red alert: narrative explanation of anomaly
        sentences = [
            f"We observed a distinct deviation in your daily rhythm today."
        ]
        
        details = []
        # Group features that deviated significantly
        if sleep_z < -1.5:
            details.append(f"your sleep duration was shortened by {abs(sleep_dur - sleep_dur_mean):.1f} hours")
        elif sleep_z > 1.5:
            details.append(f"your sleep duration was extended by {abs(sleep_dur - sleep_dur_mean):.1f} hours")
            
        if sleep_diff > 1.5:
            details.append(f"your bedtime shifted {int(sleep_diff*60)} minutes later than usual")
        elif sleep_diff < -1.5:
            details.append(f"your bedtime was {int(abs(sleep_diff)*60)} minutes earlier")

        if screen_z > 1.5:
            details.append(f"your screen time spiked to {screen_time:.1f} hours")
        elif screen_z < -1.5:
            details.append(f"your screen usage dropped to {screen_time:.1f} hours")

        if len(details) > 0:
            if len(details) == 1:
                sentences.append(f"Specifically, {details[0]}.")
            else:
                sentences.append(f"Specifically, {', '.join(details[:-1])}, and {details[-1]}.")
        
        # Contextual explanation based on demographics / notes
        if "student" in role_term:
            sentences.append("For a student, sleep and screen shifts can often coincide with study loads or late-night projects.")
        else:
            sentences.append("Such routine shifts can sometimes result from work deadlines, travels, or personal events.")

        sentences.append("Lumen is monitoring this pattern to see if it settles or suggests a sustained shift in your rhythm.")
        return " ".join(sentences)

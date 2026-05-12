personality_vector = {
    # Voice (88+ features from openSMILE, or simplified:)
    'voice': {
        'pitch_mean': float,
        'pitch_std': float,
        'energy_mean': float,
        'speaking_rate': float,
        'pause_rate': float,
        # ... more
    },
    
    # Activity
    'activity': {
        'screen_time_daily': float,
        'unlock_frequency': float,
        'social_app_ratio': float,
        'calls_per_day': float,
        'texts_per_day': float,
        'unique_contacts': int,
        'avg_response_time': float,
    },
    
    # Movement
    'movement': {
        'daily_displacement': float,
        'location_entropy': float,
        'home_time_ratio': float,
        'places_visited': int,
    },
    
    # Circadian
    'circadian': {
        'wake_time_mean': float,
        'wake_time_std': float,
        'sleep_time_mean': float,
        'sleep_duration': float,
    },
    
    # Meta (variance/regularity measures)
    'regularity': {
        'daily_routine_variance': float,
        'week_to_week_consistency': float,
    }
}

def individual_data(learning_pref):
    return {
        "id": str(learning_pref["_id"]),
        "mode": learning_pref["mode"],
        "topic": learning_pref["topic"],
        "subTopic": learning_pref["subTopic"],
        "level": learning_pref["level"],
        "userId": learning_pref["userId"]
    }

def all_data(learning_prefs):
    return [individual_data(learning_pref) for learning_pref in learning_prefs]
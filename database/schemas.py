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

def user_pref_data(prefs):
    return{
         "id": str(prefs["_id"]),
        "topic": prefs["topic"],
        "subTopic": prefs["subTopic"],
        "mode": prefs["mode"],
        "level": prefs["level"],
        "createdAt": prefs["created_at"]
    }

def all_user_pref_data(prefs):
    return [user_pref_data(prefs) for prefs in prefs]
def make_decision(detections: list, lane_info: dict) -> tuple:
  
    DECISION_COLORS = {
        "STOP":         (0, 0, 255),    # Red
        "SLOW DOWN":    (0, 255, 255),  # Yellow
        "MOVE FORWARD": (0, 255, 0),    # Green
    }

    # only care about objects actually on the road 
    road_objects = [obj for obj in detections if obj.get("on_road") == True]

    # if nothing is on the road, just move forward
    if not road_objects:
        return "MOVE FORWARD", DECISION_COLORS["MOVE FORWARD"]

    #  Priority 1: Person on road → always STOP 
    for obj in road_objects:
        if obj["label"] == "person":
            return "STOP", DECISION_COLORS["STOP"]

    # Priority 2: Any object VERY NEAR → STOP
    for obj in road_objects:
        if obj["distance"] == "VERY NEAR":
            return "STOP", DECISION_COLORS["STOP"]

    #  Priority 3: Any object NEAR → SLOW DOWN
    for obj in road_objects:
        if obj["distance"] == "NEAR":
            return "SLOW DOWN", DECISION_COLORS["SLOW DOWN"]


def get_steering(lane_info: dict, frame_width: int = 640) -> str:
   

    lane_center  = lane_info.get("lane_center")
    frame_center = frame_width // 2

    # if no lane info available, go straight
    if lane_center is None:
        return "STRAIGHT (no lane)"

    offset = lane_center - frame_center   # positive = lane is to the right

    #need to tune these according to our frame size
    MILD_THRESHOLD  = 30    # small drift
    SHARP_THRESHOLD = 80    # large drift

    if abs(offset) <= MILD_THRESHOLD:
        return "STRAIGHT"
    elif offset > SHARP_THRESHOLD:
        return "TURN RIGHT (sharp)"
    elif offset > MILD_THRESHOLD:
        return "TURN RIGHT (mild)"
    elif offset < -SHARP_THRESHOLD:
        return "TURN LEFT  (sharp)"
    else:
        return "TURN LEFT  (mild)"
    

def decide(detections: list, lane_info: dict, frame_width: int = 640) -> tuple:
  
    decision, color = make_decision(detections, lane_info)
    steering        = get_steering(lane_info, frame_width)

    return decision, steering, color
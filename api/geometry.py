# Geometry helper functions for calculating shot distance, angle, and player positions
# (Used in both training data extraction and in the serving API)

import math

GOAL_X = 120.0
GOAL_Y_CENTER = 40.0
POST_LEFT = (120.0, 36.0)
POST_RIGHT = (120.0, 44.0)


def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def shot_angle(x, y):
    # Calculates the angle between the shot point and the two goalposts
    # Uses law of cosines (high school math!)
    a = dist(x, y, *POST_LEFT)
    b = dist(x, y, *POST_RIGHT)
    c = dist(*POST_LEFT, *POST_RIGHT)
    denom = 2 * a * b
    if denom == 0:
        return 0.0
    # clamp cos_angle between -1 and 1 to prevent math domain error
    cos_angle = max(-1.0, min(1.0, (a * a + b * b - c * c) / denom))
    return math.acos(cos_angle)


def point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
    # Standard sign test to check if a point is inside a triangle (our shot cone)
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def perpendicular_distance(px, py, x1, y1, x2, y2):
    # Calculates how far a point is from a line (used for goalkeeper positioning)
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den != 0 else 0.0


def freeze_frame_features(x, y, opponents):
    # Extracts features from the 360 freeze frame. 
    # opponents is a list of player dicts. Returns None if data is missing so sklearn's SimpleImputer can handle it!
    if not opponents:
        return {
            'n_opponents_in_cone': None,
            'n_opponents_total': None,
            'nearest_opponent_dist': None,
            'gk_dist_to_goal_center': None,
            'gk_dist_from_shot_line': None,
        }

    # count defenders in the triangle between ball and both goalposts
    n_opp_in_cone = sum(
        1 for p in opponents
        if point_in_triangle(p['x'], p['y'], x, y, *POST_LEFT, *POST_RIGHT)
    )
    # find closest defender
    nearest_opp_dist = min(dist(x, y, p['x'], p['y']) for p in opponents)

    # find the goalkeeper and get their metrics
    gk = next((p for p in opponents if p.get('position_name') == 'Goalkeeper'), None)
    if gk is not None:
        gk_dist_to_goal_center = dist(gk['x'], gk['y'], GOAL_X, GOAL_Y_CENTER)
        gk_dist_from_shot_line = perpendicular_distance(gk['x'], gk['y'], x, y, GOAL_X, GOAL_Y_CENTER)
    else:
        gk_dist_to_goal_center = None
        gk_dist_from_shot_line = None

    return {
        'n_opponents_in_cone': n_opp_in_cone,
        'n_opponents_total': len(opponents),
        'nearest_opponent_dist': nearest_opp_dist,
        'gk_dist_to_goal_center': gk_dist_to_goal_center,
        'gk_dist_from_shot_line': gk_dist_from_shot_line,
    }

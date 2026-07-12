# StatsBomb Open Data: Expected Goals (xG) Shot Event Extractor
# Course: CS ML Project
# Author: Saurabh

import json
import glob
import math
import csv

GOAL_X = 120.0
GOAL_Y_CENTER = 40.0
POST_LEFT = (120.0, 36.0)
POST_RIGHT = (120.0, 44.0)

def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def shot_angle(x, y):
    # Calculations for angle subtended by the goal mouth from where the shot is taken
    a = dist(x, y, *POST_LEFT)
    b = dist(x, y, *POST_RIGHT)
    c = dist(*POST_LEFT, *POST_RIGHT)  # Goal width = 8 yards constant
    denom = 2 * a * b
    if denom == 0:
        return 0.0
    cos_angle = (a * a + b * b - c * c) / denom
    # guard against numerical overflow or rounding issues
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)

def point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
    # Barycentric test for whether a defender is in the triangle between shot and posts
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def freeze_frame_features(x, y, freeze_frame):
    # filter out teammates - we only care about defenders and the goalkeeper
    opponents = [p for p in freeze_frame if not p.get('teammate', False)]
    gk = next((p for p in opponents if p.get('position', {}).get('name') == 'Goalkeeper'), None)

    n_opp_in_cone = 0
    for p in opponents:
        ox, oy = p['location'][0], p['location'][1]
        if point_in_triangle(ox, oy, x, y, *POST_LEFT, *POST_RIGHT):
            n_opp_in_cone += 1

    nearest_opp_dist = min((dist(x, y, p['location'][0], p['location'][1]) for p in opponents), default=None)

    if gk is not None:
        gk_x, gk_y = gk['location'][0], gk['location'][1]
        gk_dist_to_goal_center = dist(gk_x, gk_y, GOAL_X, GOAL_Y_CENTER)
        # perpendicular offset distance from direct shot line
        gk_dist_from_shot_line = perpendicular_distance(gk_x, gk_y, x, y, GOAL_X, GOAL_Y_CENTER)
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

def perpendicular_distance(px, py, x1, y1, x2, y2):
    # math formula for distance from point to a line segment
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den != 0 else 0.0

def process_match(events, match_id, tag, competition_name, season_name):
    # Track the running score so we can get goal difference
    score = {}
    rows = []
    # print(f"Processing match {match_id}") # debug print
    for e in events:
        team = e.get('team', {}).get('name')
        etype = e.get('type', {}).get('name')

        if etype == 'Shot':
            shot = e['shot']
            shot_type = shot.get('type', {}).get('name')
            # Skip penalties - they distort the training data since conversion is constant ~76%
            if shot_type == 'Penalty':
                continue  

            x, y = e['location'][0], e['location'][1]
            outcome = shot.get('outcome', {}).get('name')
            is_goal = 1 if outcome == 'Goal' else 0

            row = {
                'match_id': match_id,
                'tournament': tag,
                'competition': competition_name,
                'season': season_name,
                'team': team,
                'minute': e.get('minute'),
                'period': e.get('period'),
                'x': x,
                'y': y,
                'distance_to_goal': dist(x, y, GOAL_X, GOAL_Y_CENTER),
                'shot_angle_rad': shot_angle(x, y),
                'body_part': shot.get('body_part', {}).get('name'),
                'technique': shot.get('technique', {}).get('name'),
                'shot_type': shot_type,
                'play_pattern': e.get('play_pattern', {}).get('name'),
                'under_pressure': bool(e.get('under_pressure', False)),
                'first_time': bool(shot.get('first_time', False)),
                'open_goal': bool(shot.get('open_goal', False)),
                'deflected': bool(shot.get('deflected', False)),
                'aerial_won': bool(shot.get('aerial_won', False)),
                'follows_dribble': bool(shot.get('follows_dribble', False)),
                'score_diff_before_shot': score.get(team, 0) - sum(v for k, v in score.items() if k != team),
                'statsbomb_xg': shot.get('statsbomb_xg'),  # keep StatsBomb xG for benchmark comparison
                'is_goal': is_goal,
            }

            ff = shot.get('freeze_frame')
            if ff:
                row.update(freeze_frame_features(x, y, ff))
            else:
                row.update({
                    'n_opponents_in_cone': None,
                    'n_opponents_total': None,
                    'nearest_opponent_dist': None,
                    'gk_dist_to_goal_center': None,
                    'gk_dist_from_shot_line': None,
                })

            rows.append(row)

            if is_goal:
                score[team] = score.get(team, 0) + 1

        elif etype == 'Own Goal Against':
            pass # skip own goals for simplicity

    return rows

def main():
    # Load match id mappings
    with open('sb/match_id_map.json') as f:
        match_map = json.load(f)

    id_to_tag = {}
    for tag, ids in match_map.items():
        for mid in ids:
            id_to_tag[mid] = tag

    # Load competition details
    meta = {}
    for f in glob.glob('sb/matches/*.json'):
        tag = f.split('/')[-1].replace('.json', '')
        for m in json.load(open(f)):
            meta[m['match_id']] = (
                m['competition']['competition_name'],
                m['season']['season_name'],
            )

    all_rows = []
    n_matches = 0
    for f in glob.glob('sb/events/*.json'):
        match_id = int(f.split('/')[-1].replace('.json', ''))
        tag = id_to_tag.get(match_id, 'unknown')
        comp_name, season_name = meta.get(match_id, ('unknown', 'unknown'))
        events = json.load(open(f))
        rows = process_match(events, match_id, tag, comp_name, season_name)
        all_rows.extend(rows)
        n_matches += 1

    print(f"Processed {n_matches} matches, extracted {len(all_rows)} non-penalty shots")

    fieldnames = list(all_rows[0].keys())
    with open('sb_out/shots.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    goals = sum(r['is_goal'] for r in all_rows)
    print(f"Goals: {goals} ({100*goals/len(all_rows):.2f}% conversion rate)")

if __name__ == '__main__':
    main()

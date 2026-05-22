#!/usr/bin/env python3
"""
ML Predictor for HKOI Team Formation Test
Uses pairwise comparisons (Bradley-Terry model) to estimate skill
Fair comparison - only compares contestants who participated in same contests
Excludes HKGOI (girls-only) from training data
"""
import json
import numpy as np
from collections import defaultdict
from itertools import combinations

# Load historical data
with open('data.json') as f:
    data = json.load(f)

contests = data['contests']
person_stats = data['person_stats']

# Convert contests list to dict by ID
contests_by_id = {}
for c in contests:
    contests_by_id[c['id']] = c

# Contest 480 and 481 contestants (2-day TFT)
# These will be combined - average rank across both days
contest_480 = [
    "3lo", "Archso", "E10619", "E10709", "Ikea", "NET221142", "WYK22F22", "WYK22F26",
    "WYK22L19", "WYK22R42", "WYK23F31", "WYK23F32", "WYK23R16", "WYK23X35", "WYK24X26",
    "Whaphark", "XWK220284", "bstc-21003", "bstc-21049", "bstc-22009", "ccsc_fangzecheng",
    "chyim", "clcheung", "cpu-s2022035", "creeper_computer", "cyjpang", "dbs23072020",
    "dbsCosmicCrusader", "dbsMarco", "dbs_kayth", "dbscarsonho", "dbscarychan", "dbsculver0412",
    "dbsgc9987", "dbsptl", "dgs221195", "dgs231031", "hcpoon", "hkgoi202526-13",
    "hkoi202425-01", "hkoi202425-14", "jdkmaths", "jltan", "kyeung", "lfngan", "lihonglin",
    "liumz0413", "lkcss_it", "mfong", "mst-s2210037", "pufflet233", "qc15701", "rsu", "rzhou",
    "s200146", "s20126plk", "s20146", "s201701122", "s201901024", "s20192", "s202010089",
    "s202110288", "s202210391", "s2023085", "s20251", "s220039", "s220082", "s220204",
    "s250038", "s250301", "s25218", "sjc-jacklam", "sms27098", "sms27113", "sms28128",
    "sp20226361", "tmastercoding", "twg-210083", "twg-210110", "twg-220030", "twg-240101",
    "wy_23918", "wy_24084", "wy_24125", "wy_24180", "wy_24215", "wy_24641", "ylmass_s20210012",
    "ywgs217", "ywgs265"
]

# 2-day TFT format: combine results from both days
# When results are available, calculate average rank across both days
tft_day1_contest = '480'
tft_day2_contest = '481'

# All TFT contests
all_tft_contests = []
for cid, c in contests_by_id.items():
    if 'Team Formation' in c.get('name', ''):
        all_tft_contests.append(cid)

# Exclude EGOI TFTs (472, 441) as they are different competition format
egoi_tft_contests = ['441', '472']

# All competitions except HKGOI and EGOI (including team events)
all_contests = list(contests_by_id.keys())
hkgoi_contests = [cid for cid, c in contests_by_id.items() if 'HKGOI' in c.get('name', '').upper()]
non_hkgoi_contests = [cid for cid in all_contests if cid not in hkgoi_contests and cid not in egoi_tft_contests]

# Training contests: all except HKGOI, EGOI, and team events
training_contests = non_hkgoi_contests

print(f"All TFT contests: {all_tft_contests}")
print(f"EGOI TFT contests excluded: {egoi_tft_contests}")
print(f"HKGOI contests excluded: {hkgoi_contests}")
print(f"Total training contests (excluding HKGOI/EGOI): {len(training_contests)}")

# Build pairwise comparison data using 2-contest averages
# For each pair of 2 contests, calculate average rank and do head-to-head
# For team events, each team member gets the team's rank
pairwise_wins = defaultdict(lambda: defaultdict(int))
pairwise_contests = defaultdict(lambda: defaultdict(int))

print("\nBuilding pairwise comparison data using 2-contest averages...")
print(f"Generating all pairs of {len(training_contests)} training contests...")

# Function to get individual ranks from a contest (including team members)
def get_individual_ranks(cid):
    """Get individual ranks from a contest, including team members"""
    c = contests_by_id.get(cid, {})
    rankings = c.get('rankings', [])
    
    # Check if this is a team contest
    is_team_contest = 'team' in c.get('name', '').lower() or 'formation' in c.get('name', '').lower()
    
    participants = {}
    
    if is_team_contest:
        # For team contests, get team members and assign team rank to each member
        teams = c.get('teams', [])
        for team in teams:
            team_rank = team.get('rank')
            if team_rank is not None:
                members = team.get('members', [])
                for member in members:
                    try:
                        participants[member] = int(team_rank)
                    except:
                        pass
    else:
        # For individual contests, get individual ranks
        for entry in rankings:
            handle = entry.get('handle', '')
            rank = entry.get('rank_normalized')
            if rank is not None:
                try:
                    participants[handle] = int(rank)
                except:
                    pass
    
    return participants

# Generate all pairs of 2 contests
from itertools import combinations
contest_pairs = list(combinations(training_contests, 2))
print(f"Total contest pairs: {len(contest_pairs)}")

for cid1, cid2 in contest_pairs:
    participants1 = get_individual_ranks(cid1)
    participants2 = get_individual_ranks(cid2)
    
    # Find contestants who participated in both contests
    common_participants = set(participants1.keys()) & set(participants2.keys())
    
    if len(common_participants) < 2:
        continue
    
    # Calculate average rank for each contestant
    avg_ranks = {}
    for handle in common_participants:
        avg_ranks[handle] = (participants1[handle] + participants2[handle]) / 2
    
    # Do head-to-head comparisons based on average rank
    handles = list(avg_ranks.keys())
    for i, a in enumerate(handles):
        for b in handles[i+1:]:
            avg_a = avg_ranks[a]
            avg_b = avg_ranks[b]
            
            pairwise_contests[a][b] += 1
            pairwise_contests[b][a] += 1
            
            if avg_a < avg_b:  # Lower average rank = better
                pairwise_wins[a][b] += 1
            else:
                pairwise_wins[b][a] += 1

print(f"Completed pairwise comparisons using 2-contest averages")

# Bradley-Terry model using iterative algorithm
def bradley_terry_scores(pairwise_wins, pairwise_contests, iterations=100):
    """
    Estimate Bradley-Terry skill scores using iterative algorithm.
    Returns log-odds skill scores for each player.
    """
    # Get all players
    players = set()
    for a in pairwise_wins:
        for b in pairwise_wins[a]:
            players.add(a)
            players.add(b)
    
    if not players:
        return {}
    
    # Initialize scores
    scores = {p: 0.0 for p in players}
    
    for _ in range(iterations):
        new_scores = {}
        for p in players:
            # Calculate expected wins
            expected = 0
            total_games = 0
            
            for opponent in players:
                if opponent == p:
                    continue
                games = pairwise_contests[p].get(opponent, 0)
                if games > 0:
                    # Expected wins against this opponent
                    p_win_prob = np.exp(scores[p]) / (np.exp(scores[p]) + np.exp(scores[opponent]))
                    expected += games * p_win_prob
                    total_games += games
            
            # Update score based on actual vs expected
            actual_wins = sum(pairwise_wins[p].values())
            if total_games > 0:
                # Newton-Raphson update
                new_scores[p] = scores[p] + (actual_wins - expected) / max(total_games, 1)
            else:
                new_scores[p] = scores[p]
        
        scores = new_scores
    
    return scores

# Calculate Bradley-Terry scores
print("Calculating Bradley-Terry skill scores...")
skill_scores = bradley_terry_scores(pairwise_wins, pairwise_contests)

# Also calculate head-to-head win rates for top contestants
def get_head_to_head_stats(username):
    """Get head-to-head statistics for a contestant"""
    stats = {
        'total_contests': 0,
        'total_wins': 0,
        'total_comparisons': 0,
        'head_to_head': {}
    }
    
    for opponent in pairwise_wins.get(username, {}):
        wins = pairwise_wins[username][opponent]
        total = pairwise_contests[username][opponent]
        if total > 0:
            stats['head_to_head'][opponent] = {
                'wins': wins,
                'total': total,
                'win_rate': wins / total
            }
            stats['total_wins'] += wins
            stats['total_comparisons'] += total
    
    stats['total_contests'] = len(set(
        cid for cid in training_contests
        for entry in contests_by_id.get(cid, {}).get('rankings', [])
        if entry.get('handle') == username
    ))
    
    return stats

# Calculate predictions using Bradley-Terry scores
print("\nCalculating predictions...")

predictions = []
for username in contest_480:
    skill = skill_scores.get(username, 0.0)
    h2h = get_head_to_head_stats(username)
    
    predictions.append({
        'username': username,
        'skill_score': skill,
        'h2h_stats': h2h
    })

# Filter out contestants with less than 3 contests
predictions = [p for p in predictions if p['h2h_stats']['total_contests'] >= 3]

# Calculate top 1, top 4 and top 8 probabilities using Bradley-Terry model
def calculate_top_k_probabilities(predictions, k_values=[1, 4, 8], num_simulations=100000):
    """
    Calculate probability of each contestant finishing in top k using Monte Carlo simulation.
    Uses Bradley-Terry pairwise win probabilities.
    """
    usernames = [p['username'] for p in predictions]
    skills = {p['username']: p['skill_score'] for p in predictions}
    n = len(usernames)
    
    # Calculate pairwise win probabilities
    def win_prob(a, b):
        """Probability that a beats b based on Bradley-Terry model"""
        skill_a = skills.get(a, 0)
        skill_b = skills.get(b, 0)
        return np.exp(skill_a) / (np.exp(skill_a) + np.exp(skill_b))
    
    # Monte Carlo simulation
    top_k_counts = {k: defaultdict(int) for k in k_values}
    
    for _ in range(num_simulations):
        # Simulate tournament: for each pair, determine winner based on win probability
        wins = defaultdict(int)
        
        for i, a in enumerate(usernames):
            for b in usernames[i+1:]:
                if np.random.random() < win_prob(a, b):
                    wins[a] += 1
                else:
                    wins[b] += 1
        
        # Rank by wins (higher = better)
        ranked = sorted(usernames, key=lambda x: wins[x], reverse=True)
        
        # Count top k finishes
        for k in k_values:
            for username in ranked[:k]:
                top_k_counts[k][username] += 1
    
    # Convert to probabilities
    probabilities = {}
    for username in usernames:
        probabilities[username] = {}
        for k in k_values:
            probabilities[username][f'top_{k}'] = top_k_counts[k][username] / num_simulations
    
    return probabilities

print("\nCalculating top 1, top 4 and top 8 probabilities (Monte Carlo simulation)...")
top_k_probs = calculate_top_k_probabilities(predictions)

# Add probabilities to predictions
for pred in predictions:
    pred['top_1_prob'] = top_k_probs[pred['username']]['top_1']
    pred['top_4_prob'] = top_k_probs[pred['username']]['top_4']
    pred['top_8_prob'] = top_k_probs[pred['username']]['top_8']

# Sort by skill score (higher = better)
predictions.sort(key=lambda x: x['skill_score'], reverse=True)

# Print top 50 predictions
print("\n" + "="*130)
print("ML PREDICTION - TOP 50 FOR TEAM FORMATION TEST (Contest 480/481)")
print("="*130)
print(f"{'Rank':<6} {'Username':<20} {'Skill':<10} {'H2H Win%':<10} {'1st%':<15} {'Top 4%':<15} {'Top 8%':<15} {'Contests':<10}")
print("-"*130)

for i, pred in enumerate(predictions[:50], 1):
    h2h_win_rate = (pred['h2h_stats']['total_wins'] / pred['h2h_stats']['total_comparisons'] * 100 
                   if pred['h2h_stats']['total_comparisons'] > 0 else 0)
    
    print(f"{i:<6} {pred['username']:<20} {pred['skill_score']:<10.2f} "
          f"{h2h_win_rate:<10.1f} {pred['top_1_prob']*100:<15.10f} {pred['top_4_prob']*100:<15.10f} {pred['top_8_prob']*100:<15.10f} {pred['h2h_stats']['total_contests']:<10}")

# Print methodology
print("\n" + "="*100)
print("PREDICTION METHODOLOGY")
print("="*100)
print(f"""
Training Data:
- HKGOI contests excluded: {hkgoi_contests}
- EGOI contests excluded: {egoi_tft_contests}
- Total training contests (including team events): {len(training_contests)}

Algorithm: Bradley-Terry Model
- Uses pairwise comparisons between contestants
- Only compares contestants who participated in the same contest
- Fair comparison - no bias from different contest participation
- Estimates skill scores based on head-to-head win rates

Features:
1. Bradley-Terry skill score (main predictor)
   - Based on head-to-head performance against other contestants
   - Fair comparison - only uses contests where both participated
   - Team events: each member gets team's rank for comparison

Key insight: Bradley-Terry model provides fair comparison by only
considering contests where both contestants participated. This avoids
bias from different contest participation patterns.
Team events are included - each team member gets the team's rank.
""")

# Show some interesting head-to-head matchups
print("="*100)
print("INTERESTING HEAD-TO-HEAD MATCHUPS (Top 10)")
print("="*100)

top_10 = [p['username'] for p in predictions[:10]]
for i, a in enumerate(top_10):
    for b in top_10[i+1:]:
        h2h_a = pairwise_wins[a].get(b, 0)
        h2h_b = pairwise_wins[b].get(a, 0)
        total = pairwise_contests[a].get(b, 0)
        if total > 0:
            print(f"{a} vs {b}: {h2h_a}-{h2h_b} ({total} contests)")

# Save predictions to file
with open('predictions_480.json', 'w') as f:
    json.dump({
        'contest': 480,
        'predictions': [{
            'rank': i+1,
            'username': p['username'],
            'skill_score': p['skill_score'],
            'h2h_win_rate': (p['h2h_stats']['total_wins'] / p['h2h_stats']['total_comparisons'] * 100 
                           if p['h2h_stats']['total_comparisons'] > 0 else 0),
            'top_1_prob': p['top_1_prob'],
            'top_4_prob': p['top_4_prob'],
            'top_8_prob': p['top_8_prob'],
            'contests': p['h2h_stats']['total_contests']
        } for i, p in enumerate(predictions)]
    }, f, indent=2)

print("\nPredictions saved to predictions_480.json")

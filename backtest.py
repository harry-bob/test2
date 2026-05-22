#!/usr/bin/env python3
"""
Backtest the prediction model on Mini Competition 0 contests.
Uses historical data to predict results and compare with actual outcomes.
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

# Mini Competition 0 contests to backtest
minicomp0_contests = ['320', '358', '387', '431', '463']

# Exclude HKGOI and EGOI
hkgoi_contests = [cid for cid, c in contests_by_id.items() if 'HKGOI' in c.get('name', '').upper()]
egoi_tft_contests = ['441', '472']

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

def bradley_terry_scores(pairwise_wins, pairwise_contests, iterations=100):
    """
    Estimate Bradley-Terry skill scores using iterative algorithm.
    """
    players = set()
    for a in pairwise_wins:
        for b in pairwise_wins[a]:
            players.add(a)
            players.add(b)
    
    if not players:
        return {}
    
    scores = {p: 0.0 for p in players}
    
    for _ in range(iterations):
        new_scores = {}
        for p in players:
            expected = 0
            total_games = 0
            
            for opponent in players:
                if opponent == p:
                    continue
                games = pairwise_contests[p].get(opponent, 0)
                if games > 0:
                    p_win_prob = np.exp(scores[p]) / (np.exp(scores[p]) + np.exp(scores[opponent]))
                    expected += games * p_win_prob
                    total_games += games
            
            actual_wins = sum(pairwise_wins[p].values())
            if total_games > 0:
                new_scores[p] = scores[p] + (actual_wins - expected) / max(total_games, 1)
            else:
                new_scores[p] = scores[p]
        
        scores = new_scores
    
    return scores

def predict_contest(training_contests, target_contest_id):
    """
    Predict results for a target contest using training data.
    Returns predicted rankings for participants in target contest.
    """
    # Build pairwise comparison data using 2-contest averages
    pairwise_wins = defaultdict(lambda: defaultdict(int))
    pairwise_contests = defaultdict(lambda: defaultdict(int))
    
    # Generate all pairs of 2 contests from training data
    contest_pairs = list(combinations(training_contests, 2))
    
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
    
    # Calculate Bradley-Terry scores
    skill_scores = bradley_terry_scores(pairwise_wins, pairwise_contests)
    
    # Get participants in target contest
    target_participants = get_individual_ranks(target_contest_id)
    
    # Create predictions for target contest participants
    predictions = []
    for handle in target_participants:
        skill = skill_scores.get(handle, 0.0)
        actual_rank = target_participants[handle]
        predictions.append({
            'handle': handle,
            'skill_score': skill,
            'actual_rank': actual_rank
        })
    
    # Sort by skill score (higher = better, so lower predicted rank)
    predictions.sort(key=lambda x: x['skill_score'], reverse=True)
    
    # Add predicted rank
    for i, pred in enumerate(predictions):
        pred['predicted_rank'] = i + 1
    
    return predictions

def calculate_accuracy(predictions, k_values=[1, 3, 5, 10]):
    """
    Calculate accuracy metrics for predictions.
    """
    results = {}
    
    for k in k_values:
        # Get top k predicted and actual
        top_k_predicted = set(p['handle'] for p in predictions[:k])
        top_k_actual = set(p['handle'] for p in sorted(predictions, key=lambda x: x['actual_rank'])[:k])
        
        # Calculate overlap
        overlap = len(top_k_predicted & top_k_actual)
        results[f'top_{k}_accuracy'] = overlap / k
    
    # Calculate rank correlation (Spearman)
    n = len(predictions)
    if n > 1:
        predicted_ranks = {p['handle']: p['predicted_rank'] for p in predictions}
        actual_ranks = {p['handle']: p['actual_rank'] for p in predictions}
        
        d_squared_sum = 0
        for handle in predicted_ranks:
            if handle in actual_ranks:
                d = predicted_ranks[handle] - actual_ranks[handle]
                d_squared_sum += d * d
        
        spearman = 1 - (6 * d_squared_sum) / (n * (n * n - 1))
        results['spearman_correlation'] = spearman
    else:
        results['spearman_correlation'] = 0
    
    return results

# Run backtesting
print("="*80)
print("BACKTESTING PREDICTION MODEL ON MINI COMPETITION 0")
print("="*80)

all_results = []

for i, target_contest_id in enumerate(minicomp0_contests):
    target_contest = contests_by_id.get(target_contest_id, {})
    target_name = target_contest.get('name', 'Unknown')
    
    # Use all contests before the target contest as training data
    training_contests = []
    for c in contests:
        cid = c['id']
        if cid == target_contest_id:
            break
        if cid not in hkgoi_contests and cid not in egoi_tft_contests:
            training_contests.append(cid)
    
    print(f"\n{'='*80}")
    print(f"Backtest {i+1}: Predicting {target_name} (Contest {target_contest_id})")
    print(f"Training data: {len(training_contests)} contests")
    print(f"{'='*80}")
    
    # Make predictions
    predictions = predict_contest(training_contests, target_contest_id)
    
    if not predictions:
        print("No predictions available (not enough training data)")
        continue
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predictions)
    
    # Print results
    print(f"\nTop 10 Predicted vs Actual:")
    print(f"{'Predicted':<10} {'Handle':<20} {'Skill':<10} {'Actual':<10}")
    print("-"*50)
    
    for j, pred in enumerate(predictions[:10]):
        print(f"#{pred['predicted_rank']:<9} {pred['handle']:<20} {pred['skill_score']:<10.2f} #{pred['actual_rank']}")
    
    print(f"\nAccuracy Metrics:")
    print(f"  Top 1 Accuracy: {accuracy['top_1_accuracy']*100:.1f}%")
    print(f"  Top 3 Accuracy: {accuracy['top_3_accuracy']*100:.1f}%")
    print(f"  Top 5 Accuracy: {accuracy['top_5_accuracy']*100:.1f}%")
    print(f"  Top 10 Accuracy: {accuracy['top_10_accuracy']*100:.1f}%")
    print(f"  Spearman Correlation: {accuracy['spearman_correlation']:.3f}")
    
    all_results.append({
        'contest': target_name,
        'contest_id': target_contest_id,
        'accuracy': accuracy,
        'predictions': predictions
    })

# Summary
print("\n" + "="*80)
print("BACKTESTING SUMMARY")
print("="*80)

if all_results:
    avg_metrics = defaultdict(float)
    for result in all_results:
        for metric, value in result['accuracy'].items():
            avg_metrics[metric] += value
    
    n = len(all_results)
    print(f"\nAverage Metrics across {n} backtests:")
    print(f"  Top 1 Accuracy: {avg_metrics['top_1_accuracy']/n*100:.1f}%")
    print(f"  Top 3 Accuracy: {avg_metrics['top_3_accuracy']/n*100:.1f}%")
    print(f"  Top 5 Accuracy: {avg_metrics['top_5_accuracy']/n*100:.1f}%")
    print(f"  Top 10 Accuracy: {avg_metrics['top_10_accuracy']/n*100:.1f}%")
    print(f"  Spearman Correlation: {avg_metrics['spearman_correlation']/n:.3f}")
    
    # Show some interesting predictions
    print("\n" + "="*80)
    print("INTERESTING FINDINGS")
    print("="*80)
    
    for result in all_results:
        predictions = result['predictions']
        if len(predictions) >= 3:
            # Find best predicted contestant
            best_predicted = predictions[0]
            actual_winner = min(predictions, key=lambda x: x['actual_rank'])
            
            print(f"\n{result['contest']}:")
            print(f"  Best Predicted: {best_predicted['handle']} (Predicted #{best_predicted['predicted_rank']}, Actual #{best_predicted['actual_rank']})")
            print(f"  Actual Winner: {actual_winner['handle']} (Predicted #{actual_winner['predicted_rank']}, Actual #{actual_winner['actual_rank']})")

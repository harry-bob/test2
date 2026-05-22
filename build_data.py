#!/usr/bin/env python3
"""
Convert raw YAML contest data to a single JSON file for the website.
Handles non-standard YAML with comma-separated fields.
"""
import yaml
import json
import os
import re
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path(__file__).parent / "raw_data"
OUTPUT = Path(__file__).parent / "data.json"

def parse_ranking_line(line):
    """Parse a line like: - rank: 1, contestant: handle - name, field1: val1, field2: val2, total: val"""
    result = {}
    # Remove leading "- " and split by ", "
    line = line.strip()
    if line.startswith("- "):
        line = line[2:]
    
    parts = []
    current = ""
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"' and not in_quotes:
            in_quotes = True
            current += ch
        elif ch == '"' and in_quotes:
            in_quotes = False
            current += ch
        elif ch == ',' and not in_quotes:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        parts.append(current.strip())
    
    for part in parts:
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip()
        val = val.strip().strip('"')
        result[key] = val
    
    return result

def load_snapshot(path):
    """Load a snapshot YAML file with custom parsing."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    result = {}
    in_rankings = False
    rankings = []
    
    for line in content.split("\n"):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Top level fields
        if stripped.startswith("url:"):
            result["url"] = stripped[4:].strip()
        elif stripped.startswith("name:"):
            result["name"] = stripped[5:].strip()
        elif stripped.startswith("date:"):
            result["date"] = stripped[5:].strip()
        elif stripped.startswith("duration:"):
            result["duration"] = stripped[9:].strip()
        elif stripped.startswith("tasks:"):
            in_rankings = False
            result["tasks"] = []
        elif stripped.startswith("rankings_top20:") or stripped.startswith("rankings:"):
            in_rankings = True
            result["rankings_top20"] = []
        elif stripped.startswith("- ") and in_rankings:
            ranking = parse_ranking_line(stripped)
            if ranking:
                rankings.append(ranking)
        elif in_rankings and stripped and not stripped.startswith("-") and not stripped.startswith("tasks") and not stripped.startswith("url") and not stripped.startswith("name") and not stripped.startswith("date") and not stripped.startswith("duration"):
            # continuation of previous ranking? skip
            pass
        elif stripped.startswith("- ") and not in_rankings:
            # task list item
            task = stripped[2:].strip()
            if "tasks" in result and isinstance(result["tasks"], list):
                result["tasks"].append(task)
    
    result["rankings_top20"] = rankings
    return result

def load_teams(path):
    """Load a teams YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    result = {"teams": []}
    current_team = None
    
    for line in content.split("\n"):
        stripped = line.strip()
        
        if stripped.startswith("contest_id:"):
            result["contest_id"] = stripped[11:].strip()
        elif stripped.startswith("contest_name:"):
            result["contest_name"] = stripped[13:].strip()
        elif stripped.startswith("date:"):
            result["date"] = stripped[5:].strip()
        elif stripped.startswith("- team_name:"):
            if current_team:
                result["teams"].append(current_team)
            team_name = stripped[12:].strip().strip('"')
            current_team = {"name": team_name, "members": []}
        elif stripped.startswith("- ") and current_team and not stripped.startswith("- team_name") and not stripped.startswith("- contest"):
            member = stripped[2:].strip()
            if member and not member.startswith("[") and "members" not in member:
                current_team["members"].append(member)
    
    if current_team:
        result["teams"].append(current_team)
    
    return result

def main():
    contests = []
    person_stats = defaultdict(lambda: {
        "contests": [],
        "contest_names": [],
        "total_score": 0.0,
        "wins": 0,
        "top3": 0,
        "best_rank": 999,
    })
    team_stats = defaultdict(lambda: {
        "contests": [],
        "contest_names": [],
        "members": [],
        "wins": 0,
        "top3": 0,
        "best_rank": 999,
    })

    # Process snapshot files
    snapshot_files = sorted(RAW_DIR.glob("contest_*_snapshot.yaml"))
    print(f"Found {len(snapshot_files)} snapshot files")
    
    for path in snapshot_files:
        cid = path.stem.split("_")[1]
        try:
            data = load_snapshot(path)
        except Exception as e:
            print(f"  Error loading {path.name}: {e}")
            continue

        contest = {
            "id": cid,
            "name": data.get("name", "Unknown"),
            "url": data.get("url", ""),
            "date": data.get("date", ""),
            "duration": data.get("duration", ""),
            "tasks": data.get("tasks", []),
            "rankings": [],
        }

        for r in data.get("rankings_top20", []):
            rank = r.get("rank", "")
            contestant = r.get("contestant", "")
            total = r.get("total", "")
            
            handle = contestant.split(" - ")[0] if " - " in contestant else contestant
            
            # Parse scores
            scores = {}
            for key, val in r.items():
                if key in ("rank", "contestant", "handle", "total", "solved", "penalty"):
                    continue
                scores[key] = val
            
            ranking_entry = {
                "rank": rank,
                "contestant": contestant,
                "handle": handle,
                "total": total,
                "scores": scores,
            }
            
            contest["rankings"].append(ranking_entry)
            
            # Update person stats
            # Only numeric rank counts; ** means unranked/special
            rank_num = None
            if rank is not None:
                try:
                    rank_num = int(rank)
                except (ValueError, TypeError):
                    rank_num = None  # ** or other non-numeric = no rank
            
            ps = person_stats[handle]
            ps["contests"].append(cid)
            ps["contest_names"].append(data.get("name", "Unknown"))
            try:
                ps["total_score"] += float(total)
            except (ValueError, TypeError):
                pass
            # Only rank 1 is a win
            if rank_num == 1:
                ps["wins"] += 1
            # Top 3 only counts if numeric rank
            if rank_num is not None and rank_num <= 3:
                ps["top3"] += 1
            # Best rank only from numeric ranks
            if rank_num is not None and rank_num < ps["best_rank"]:
                ps["best_rank"] = rank_num
            
            # Store the raw rank and normalized rank
            ranking_entry["rank_normalized"] = rank_num

        contests.append(contest)

    # Process team files
    team_files = sorted(RAW_DIR.glob("contest_*_teams.yaml"))
    print(f"Found {len(team_files)} team files")
    
    for path in team_files:
        try:
            data = load_teams(path)
        except Exception as e:
            print(f"  Error loading {path.name}: {e}")
            continue
        
        cid = data.get("contest_id", "")
        
        # Find matching contest
        contest = next((c for c in contests if c["id"] == cid), None)
        if not contest:
            continue
        
        contest["has_teams"] = True
        contest["teams"] = []
        
        for t in data.get("teams", []):
            team_name = t.get("name", "Unknown")
            members = t.get("members", [])
            
            # Find team ranking
            team_rank = None
            team_total = None
            for r in contest.get("rankings", []):
                if r["contestant"] == team_name:
                    team_rank = r["rank"]
                    team_total = r["total"]
                    break
            
            team_data = {
                "name": team_name,
                "members": members,
                "rank": team_rank,
                "total": team_total,
            }
            contest["teams"].append(team_data)
            
            # Update team stats
            ts = team_stats[team_name]
            ts["contests"].append(cid)
            ts["contest_names"].append(data.get("contest_name", "Unknown"))
            ts["members"] = list(set(ts["members"] + members))
            
            # Only numeric rank counts
            rank_num = None
            if team_rank is not None:
                try:
                    rank_num = int(team_rank)
                except (ValueError, TypeError):
                    rank_num = None
            
            # Only rank 1 is a win
            if rank_num == 1:
                ts["wins"] += 1
            # Top 3 only counts if numeric rank
            if rank_num is not None and rank_num <= 3:
                ts["top3"] += 1
            # Best rank only from numeric ranks
            if rank_num is not None and rank_num < ts["best_rank"]:
                ts["best_rank"] = rank_num
            
            # Also add team score to each member's individual stats
            try:
                team_score = float(team_total) if team_total else 0
            except (ValueError, TypeError):
                team_score = 0
            
            for member in members:
                # Extract handle from member string like "s20192 - hopefuI"
                member_handle = member.split(" - ")[0] if " - " in member else member
                ps = person_stats[member_handle]
                ps["contests"].append(cid)
                ps["contest_names"].append(data.get("contest_name", "Unknown"))
                ps["total_score"] += team_score
                # Only rank 1 is a win
                if rank_num == 1:
                    ps["wins"] += 1
                # Top 3 only counts if numeric rank
                if rank_num is not None and rank_num <= 3:
                    ps["top3"] += 1
                # Best rank only from numeric ranks
                if rank_num is not None and rank_num < ps["best_rank"]:
                    ps["best_rank"] = rank_num

    # Filter out contests with only 1 participant
    filtered_contests = []
    for contest in contests:
        if len(contest["rankings"]) > 1:
            filtered_contests.append(contest)
        else:
            print(f"  Removing contest {contest['id']} ({contest['name']}) - only {len(contest['rankings'])} participant(s)")
    contests = filtered_contests
    
    # Build output
    output = {
        "contests": contests,
        "person_stats": {},
        "team_stats": {},
    }
    
    for handle, stats in person_stats.items():
        output["person_stats"][handle] = {
            "contests": list(set(stats["contests"])),
            "contest_names": list(set(stats["contest_names"])),
            "contest_count": len(set(stats["contests"])),
            "total_score": round(stats["total_score"], 3),
            "wins": stats["wins"],
            "top3": stats["top3"],
            "best_rank": stats["best_rank"] if stats["best_rank"] != 999 else None,
        }
    
    for team_name, stats in team_stats.items():
        output["team_stats"][team_name] = {
            "contests": list(set(stats["contests"])),
            "contest_names": list(set(stats["contest_names"])),
            "members": list(set(stats["members"])),
            "contest_count": len(set(stats["contests"])),
            "wins": stats["wins"],
            "top3": stats["top3"],
            "best_rank": stats["best_rank"] if stats["best_rank"] != 999 else None,
        }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(contests)} contests to {OUTPUT}")
    print(f"  {len(person_stats)} unique persons")
    print(f"  {len(team_stats)} unique teams")

if __name__ == "__main__":
    main()

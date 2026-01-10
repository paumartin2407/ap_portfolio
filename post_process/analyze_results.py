import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- REGEX CONFIGURATION ---
REGEX_INFO = {
    "solved": re.compile(r"(Solution found|Plan found|SOLVED_SATISFICING|SOLVED_OPTIMALLY)", re.IGNORECASE),
    "cost": re.compile(r"(?:Plan cost|Metric value|Quality):\s*([\d\.]+)"),
    "length": re.compile(r"(?:Plan length|Steps):\s*(\d+)"),
    "expanded": re.compile(r"Expanded\s+(\d+)\s+state"),
    "time_duration": re.compile(r"Duration:\s+([\d\.]+)\s*s"),
    "time_search": re.compile(r"Search time:\s+([\d\.]+)\s*s"),
    "time_total": re.compile(r"Total time:\s+([\d\.]+)\s*s")
}

def parse_log_file(filepath):
    data = {"solved": False, "cost": np.nan, "length": np.nan, "expanded": np.nan, "time": np.nan}
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if REGEX_INFO["solved"].search(content):
                data["solved"] = True
                c = REGEX_INFO["cost"].search(content)
                if c: data["cost"] = float(c.group(1))
                
                l = REGEX_INFO["length"].search(content)
                if l: data["length"] = float(l.group(1))
                
                # Fallback: If cost is missing but length exists, assume cost = length (unit cost)
                if pd.isna(data["cost"]) and pd.notna(data["length"]):
                    data["cost"] = data["length"]

                e = REGEX_INFO["expanded"].search(content)
                if e: data["expanded"] = float(e.group(1))

            d = REGEX_INFO["time_duration"].search(content)
            if d: data["time"] = float(d.group(1))
            else:
                t = REGEX_INFO["time_total"].search(content)
                if t: data["time"] = float(t.group(1))
                else:
                    s = REGEX_INFO["time_search"].search(content)
                    if s: data["time"] = float(s.group(1))
    except: pass
    return data

def process_logs(input_dir):
    results = []
    print(f"--- Scanning directory: {os.path.abspath(input_dir)} ---")
    if not os.path.exists(input_dir):
        return pd.DataFrame()

    for planner_name in os.listdir(input_dir):
        planner_path = os.path.join(input_dir, planner_name)
        if not os.path.isdir(planner_path): continue

        for domain_name in os.listdir(planner_path):
            domain_path = os.path.join(planner_path, domain_name)
            if not os.path.isdir(domain_path): continue
            
            for filename in os.listdir(domain_path):
                filepath = os.path.join(domain_path, filename)
                if not os.path.isfile(filepath) or filename.startswith("."): continue
                
                problem_name = filename
                for ext in [".pddl", ".log", ".txt", ".out"]:
                    problem_name = problem_name.replace(ext, "")

                metrics = parse_log_file(filepath)
                entry = {
                    "Domain": domain_name,
                    "Problem": problem_name,
                    "Planner": planner_name,
                    **metrics
                }
                results.append(entry)

    return pd.DataFrame(results)

def calculate_leaderboard(df, output_dir):
    print("\n--- Calculating Leaderboard (Speed & Quality) ---")
    
    # Pre-processing
    df["time"] = df["time"].clip(lower=0.01) # Avoid div/0
    
    # ---------------------------
    # 1. CALCULATE SPEED SCORE
    # ---------------------------
    min_times = df[df["solved"]==True].groupby(["Domain", "Problem"])["time"].min().reset_index()
    min_times.rename(columns={"time": "best_time"}, inplace=True)
    
    df = pd.merge(df, min_times, on=["Domain", "Problem"], how="left")
    
    df["ipc_score_speed"] = df.apply(
        lambda x: (x["best_time"] / x["time"]) if x["solved"] and pd.notnull(x["best_time"]) else 0, 
        axis=1
    )

    # ---------------------------
    # 2. CALCULATE QUALITY SCORE
    # ---------------------------
    # Get MINIMUM cost found by any planner for each problem
    min_costs = df[df["solved"]==True].groupby(["Domain", "Problem"])["cost"].min().reset_index()
    min_costs.rename(columns={"cost": "best_cost"}, inplace=True)
    
    df = pd.merge(df, min_costs, on=["Domain", "Problem"], how="left")
    
    # Quality Score = Best_Cost / Your_Cost (0 if unsolved)
    # If cost is 0 (rare, but possible in some domains), handle gracefully
    def calc_quality(row):
        if not row["solved"] or pd.isna(row["best_cost"]) or pd.isna(row["cost"]):
            return 0
        if row["cost"] == 0: return 1 # Avoid div/0 if cost is 0
        return row["best_cost"] / row["cost"]

    df["ipc_score_quality"] = df.apply(calc_quality, axis=1)

    # ---------------------------
    # 3. AGGREGATE
    # ---------------------------
    leaderboard = df.groupby("Planner").agg(
        Total_Solved=('solved', 'sum'),
        Coverage_Pct=('solved', 'mean'), # mean of boolean is %
        IPC_Speed_Score=('ipc_score_speed', 'sum'),
        IPC_Quality_Score=('ipc_score_quality', 'sum'),
        Avg_Time=('time', lambda x: x[df['solved']].mean()),
        Avg_Cost=('cost', lambda x: x[df['solved']].mean())
    ).sort_values("IPC_Quality_Score", ascending=False)
    
    # Formatting
    leaderboard["Coverage_Pct"] = (leaderboard["Coverage_Pct"] * 100).round(1)
    leaderboard = leaderboard.round(2)
    
    csv_path = f"{output_dir}/leaderboard.csv"
    leaderboard.to_csv(csv_path)
    print(f"🏆 Leaderboard saved to: {csv_path}")
    print(leaderboard)
    return leaderboard

def generate_report_assets(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    calculate_leaderboard(df, output_dir)
    
    # Detailed CSVs
    if "solved" in df.columns:
        df.pivot_table(index="Domain", columns="Planner", values="solved", aggfunc="sum").to_csv(f"{output_dir}/domain_coverage.csv")
    
    # Boxplots
    df_solved = df[df["solved"] == True].copy()
    if df_solved.empty: return

    sns.set_theme(style="whitegrid")
    metrics = [("time", "Time (s)"), ("length", "Plan Length"), ("cost", "Plan Cost")]
    
    for metric, title in metrics:
        if metric in df_solved.columns and df_solved[metric].notnull().any():
            plt.figure(figsize=(12, 6))
            try:
                sns.boxplot(data=df_solved, x="Domain", y=metric, hue="Planner", palette="Set2")
                plt.title(f"{title} Comparison (Solved Only)")
                plt.xticks(rotation=45, ha='right')
                if metric == "time": plt.yscale("log")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/plot_{metric}.png")
                plt.close()
            except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", default="results_analysis")
    args = parser.parse_args()
    
    df = process_logs(args.dir)
    if not df.empty:
        generate_report_assets(df, args.out)
    else:
        print("❌ No logs found.")
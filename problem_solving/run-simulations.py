import os

# Set Java memory limit for planners like ENHSP
os.environ["JAVA_TOOL_OPTIONS"] = "-Xmx10g"

import argparse
import time
import sys
import csv
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader
from unified_planning.engines import CompilationKind

# --- GLOBAL CONFIGURATION ---
LOG_DIR = "logs"

PLANNERS = [
    "enhsp",
    "lpg",
    "fast-downward", 
    "symk"           
]

# --- TIME CONFIGURATION ---
TIMEOUT_SEC = 300     # 5 minutes limit

def get_all_problems(domain_dir):
    files = [
        f for f in os.listdir(domain_dir) 
        if f.endswith(".pddl") and f != "domain.pddl"
    ]
    return sorted(files)

def load_existing_results():
    """
    Reads existing CSVs to skip already solved problems.
    Structure: { 'planner_name': {('domain', 'problem'), ...} }
    """
    completed = {p: set() for p in PLANNERS}
    for planner in PLANNERS:
        csv_path = os.path.join(LOG_DIR, planner, "results.csv")
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) 
                    for row in reader:
                        if len(row) >= 2:
                            p_name = row[0]
                            d_name = row[1]
                            completed[planner].add((d_name, p_name))
            except Exception:
                pass
    return completed

def write_csv_result(planner_name, domain_name, prob_name, solved, duration, note=""):
    planner_dir = os.path.join(LOG_DIR, planner_name)
    os.makedirs(planner_dir, exist_ok=True)
    csv_path = os.path.join(planner_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)
    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["name", "domain", "solved", "time-seconds", "note"])
            writer.writerow([prob_name, domain_name, solved, f"{duration:.4f}", note])
    except Exception as e:
        print(f"⚠️  Error writing CSV: {e}")

def run_benchmarks_up(root_path, dry_run=True):
    print(f"--- Scanning ALL benchmarks in: {os.path.abspath(root_path)} ---")
    
    if not dry_run:
        completed_tasks = load_existing_results()
    else:
        completed_tasks = {p: set() for p in PLANNERS}

    reader = PDDLReader()
    
    for dirpath, _, filenames in os.walk(root_path):
        if "domain.pddl" in filenames:
            domain_name = os.path.basename(dirpath)
            domain_file = os.path.join(dirpath, "domain.pddl")
            problems = get_all_problems(dirpath)
            if not problems: continue
            problems = problems[:10]

            print(f"📂 Domain Found: {domain_name}")
            
            # Filter problems
            problems_to_run = []
            for p_file in problems:
                if dry_run:
                    problems_to_run.append(p_file)
                else:
                    # If any planner hasn't done it, we queue it
                    if any((domain_name, p_file) not in completed_tasks[pl] for pl in PLANNERS):
                        problems_to_run.append(p_file)
            
            if not problems_to_run:
                print(f"   ✨ All problems solved. Skipping domain.")
                continue

            print(f"   Queued {len(problems_to_run)} problems.")
            
            for prob_file in problems_to_run:
                prob_path = os.path.join(dirpath, prob_file)

                # --- 1. Parse Problem ---
                try:
                    if not dry_run:
                        problem = reader.parse_problem(domain_file, prob_path)
                    else:
                        problem = None
                except Exception as e:
                    print(f"      ❌ Parse Error in {prob_file}: {e}")
                    continue

                # --- 2. Run Planners ---
                for planner_name in PLANNERS:
                    if (domain_name, prob_file) in completed_tasks[planner_name]:
                        continue

                    # Manual Skip for known broken files
                    if domain_name == "data-network-sat18-strips" and prob_file == "p10.pddl":
                        continue

                    if dry_run:
                        print(f"      [DRY RUN] {prob_file} -> {planner_name}")
                        continue

                    print(f"      🚀 {prob_file} -> {planner_name} ... ", end="", flush=True)
                    start_time = time.time()
                    
                    try:
                        # --- 3. PRE-CHECK: Support ---
                        # We use a temp planner instance just to check support
                        supported = False
                        try:
                            with OneshotPlanner(name=planner_name) as temp_planner:
                                if temp_planner.supports(problem.kind):
                                    supported = True
                        except:
                            # If we can't even check support (e.g. planner not installed), fail gracefully
                            supported = False

                        final_problem = problem
                        
                        # --- 4. COMPILATION (Automatic Fixes) ---
                        # If unsupported OR if it has conditional effects (which trip up many planners), try compiling
                        if not supported or problem.kind.has_conditional_effects():
                            try:
                                with Compiler(problem_kind=problem.kind, compilation_kind=CompilationKind.CONDITIONAL_EFFECTS) as compiler:
                                    res = compiler.compile(problem, compilation_kind=CompilationKind.CONDITIONAL_EFFECTS)
                                    final_problem = res.problem
                                    # Re-check support on the compiled problem
                                    with OneshotPlanner(name=planner_name) as temp_planner:
                                        if temp_planner.supports(final_problem.kind):
                                            supported = True
                            except Exception as comp_err:
                                # Compilation failed, proceed with original problem and see if it works or fails
                                pass

                        if not supported:
                            # Last ditch: just try running it. If it fails, the catch block below handles it.
                            # But we log a warning.
                            pass 

                        # --- 5. SOLVE ---
                        with OneshotPlanner(name=planner_name) as planner:
                            result = planner.solve(final_problem, timeout=TIMEOUT_SEC)
                            duration = time.time() - start_time
                            
                            if result.status.name in ['SOLVED_SATISFICING', 'SOLVED_OPTIMALLY']:
                                print(f"✅ Solved ({duration:.2f}s)")
                                write_csv_result(planner_name, domain_name, prob_file, True, duration)
                            else:
                                print(f"❌ {result.status.name} ({duration:.2f}s)")
                                write_csv_result(planner_name, domain_name, prob_file, False, duration, result.status.name)
                            
                            completed_tasks[planner_name].add((domain_name, prob_file))

                    except Exception as e:
                        duration = time.time() - start_time
                        err_msg = str(e).split('\n')[0] # Keep it short
                        print(f"⚠️  Error: {err_msg}")
                        write_csv_result(planner_name, domain_name, prob_file, False, duration, f"ERROR: {err_msg}")
                        completed_tasks[planner_name].add((domain_name, prob_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dir", default=".")
    args = parser.parse_args()
    
    run_benchmarks_up(args.dir, dry_run=not args.run)

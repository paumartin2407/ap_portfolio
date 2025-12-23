import os
import argparse
import time
import sys
import csv
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

# --- GLOBAL CONFIGURATION ---
LOG_DIR = "logs"

# SELECCIÓN DE PLANIFICADORES (Aries reemplaza a Tamer)
PLANNERS = [
    "symk",           # Simbólico
    "aries"           # Constraint Programming / SAT (Nuevo)
]

# --- CONFIGURACIÓN DE TIEMPO ---
TIMEOUT_SEC = 300     # 5 minutos límite por problema

def get_all_problems(domain_dir):
    """
    Devuelve TODOS los archivos .pddl ordenados alfabéticamente.
    """
    files = [
        f for f in os.listdir(domain_dir) 
        if f.endswith(".pddl") and f != "domain.pddl"
    ]
    return sorted(files)

def write_csv_result(planner_name, domain_name, prob_name, solved, duration):
    """
    Escribe el resultado en logs/<planner>/results.csv
    """
    planner_dir = os.path.join(LOG_DIR, planner_name)
    os.makedirs(planner_dir, exist_ok=True)
    
    csv_path = os.path.join(planner_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)
    
    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Cabecera si el archivo es nuevo
            if not file_exists:
                writer.writerow(["name", "domain", "solved", "time-seconds"])
            
            # Datos
            writer.writerow([prob_name, domain_name, solved, f"{duration:.4f}"])
            
    except Exception as e:
        print(f"⚠️  Error escribiendo en CSV: {e}")

def run_benchmarks_up(root_path, dry_run=True):
    print(f"--- Scanning ALL benchmarks in: {os.path.abspath(root_path)} ---")
    print(f"--- Planners: {PLANNERS} ---")
    print(f"--- Timeout: {TIMEOUT_SEC}s ---")
    print(f"--- Output: {LOG_DIR}/<planner>/results.csv ---\n")
    
    domains_found = 0
    reader = PDDLReader()

    # Recorrer carpetas
    for dirpath, dirnames, filenames in os.walk(root_path):
        
        # Detectar dominio
        if "domain.pddl" in filenames:
            domain_name = os.path.basename(dirpath)
            domain_file = os.path.join(dirpath, "domain.pddl")
            
            problems = get_all_problems(dirpath)
            
            if not problems:
                continue

            domains_found += 1
            print(f"📂 Domain Found ({domains_found}): {domain_name}")
            print(f"   Queued {len(problems)} problems.")
            
            for prob_file in problems:
                prob_path = os.path.join(dirpath, prob_file)

                # --- 1. Parsear Problema (Validación) ---
                try:
                    if not dry_run:
                        problem = reader.parse_problem(domain_file, prob_path)
                    else:
                        problem = None 
                except Exception as e:
                    print(f"      ❌ Error parsing {prob_file} (Skipping): {e}")
                    continue

                # --- 2. Ejecutar Planificadores ---
                for planner_name in PLANNERS:
                    if dry_run:
                        print(f"      [DRY RUN] {prob_file} -> {planner_name}")
                    else:
                        print(f"      🚀 {prob_file} -> {planner_name} ... ", end="", flush=True)
                        start_time = time.time()
                        solved = False
                        
                        try:
                            # Unified Planning gestiona el proceso y el timeout
                            with OneshotPlanner(name=planner_name) as planner:
                                result = planner.solve(problem, timeout=TIMEOUT_SEC)
                                duration = time.time() - start_time
                                
                                # Verificar éxito
                                if result.status.name in ['SOLVED_SATISFICING', 'SOLVED_OPTIMALLY']:
                                    print(f"✅ Solved ({duration:.2f}s)")
                                    solved = True
                                else:
                                    # Puede ser TIMEOUT, UNSOLVABLE, o MEMORY_LIMIT
                                    print(f"❌ {result.status.name} ({duration:.2f}s)")
                                    solved = False
                                
                                write_csv_result(planner_name, domain_name, prob_file, solved, duration)

                        except Exception as e:
                            duration = time.time() - start_time
                            print(f"⚠️ Crash/Error ({duration:.2f}s)")
                            print(f"      >>> ERROR: {e}")  # <--- ESTO ES LO IMPORTANTE
                            write_csv_result(planner_name, domain_name, prob_file, False, duration)

            print("-" * 40)

    if domains_found == 0:
        print("\n❌ No domains found.")
    else:
        print(f"\n--- Complete. Processed {domains_found} domains. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmarks with Aries/FD/LPG/SymK")
    parser.add_argument("--run", action="store_true", help="Actually run the planners")
    parser.add_argument("--dir", default=".", help="Root directory of benchmarks")
    
    args = parser.parse_args()
    
    is_dry_run = not args.run
    
    if not is_dry_run:
        os.makedirs(LOG_DIR, exist_ok=True)
    
    run_benchmarks_up(args.dir, dry_run=is_dry_run)

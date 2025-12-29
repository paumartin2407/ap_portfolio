import os

os.environ["JAVA_TOOL_OPTIONS"] = "-Xmx10g"

import argparse
import time
import sys
import csv
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

# --- GLOBAL CONFIGURATION ---
LOG_DIR = "logs"

PLANNERS = [
    "enhsp",
    "fast-downward"
]

# --- CONFIGURACIÓN DE TIEMPO ---
TIMEOUT_SEC = 300     # 5 minutos límite por problema

def get_all_problems(domain_dir):
    files = [
        f for f in os.listdir(domain_dir)
        if f.endswith(".pddl") and f != "domain.pddl"
    ]
    return sorted(files)

def load_existing_results():
    """
    Lee todos los CSVs existentes y devuelve un diccionario de Sets.
    Estructura: { 'planner_name': {('domain', 'problem'), ...} }
    """
    completed = {p: set() for p in PLANNERS}

    for planner in PLANNERS:
        csv_path = os.path.join(LOG_DIR, planner, "results.csv")
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) # Saltar cabecera
                    for row in reader:
                        if len(row) >= 2:
                            # CSV format: name, domain, solved, time
                            p_name = row[0]
                            d_name = row[1]
                            completed[planner].add((d_name, p_name))
            except Exception as e:
                print(f"⚠️ Error leyendo logs previos de {planner}: {e}")

    return completed

def write_csv_result(planner_name, domain_name, prob_name, solved, duration):
    planner_dir = os.path.join(LOG_DIR, planner_name)
    os.makedirs(planner_dir, exist_ok=True)

    csv_path = os.path.join(planner_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)

    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["name", "domain", "solved", "time-seconds"])
            writer.writerow([prob_name, domain_name, solved, f"{duration:.4f}"])

    except Exception as e:
        print(f"⚠️  Error escribiendo en CSV: {e}")

def run_benchmarks_up(root_path, dry_run=True):
    print(f"--- Scanning ALL benchmarks in: {os.path.abspath(root_path)} ---")
    print(f"--- Planners: {PLANNERS} ---")
    print(f"--- Timeout: {TIMEOUT_SEC}s ---")
    print(f"--- Output: {LOG_DIR}/<planner>/results.csv ---\n")

    # --- PASO NUEVO: Cargar historial ---
    if not dry_run:
        print("--- Loading existing results to skip duplicates... ---")
        completed_tasks = load_existing_results()
    else:
        completed_tasks = {p: set() for p in PLANNERS}

    domains_found = 0
    reader = PDDLReader()

    for dirpath, dirnames, filenames in os.walk(root_path):

        if "domain.pddl" in filenames:
            domain_name = os.path.basename(dirpath)
            domain_file = os.path.join(dirpath, "domain.pddl")
            problems = get_all_problems(dirpath)

            if not problems: continue

            domains_found += 1
            print(f"📂 Domain Found ({domains_found}): {domain_name}")

            # Filtrar problemas: ¿Hay algún planificador que AÚN no lo haya hecho?
            # Si todos los planificadores ya hicieron este problema, ni lo parseamos.
            problems_to_run = []
            for p_file in problems:
                needs_run = False
                for pl in PLANNERS:
                    if (domain_name, p_file) not in completed_tasks[pl]:
                        needs_run = True
                        break
                if needs_run or dry_run:
                    problems_to_run.append(p_file)

            if not problems_to_run:
                print(f"   ✨ All {len(problems)} problems already solved in this domain. Skipping.")
                continue

            print(f"   Queued {len(problems_to_run)} problems (Skipped {len(problems)-len(problems_to_run)}).")

            for prob_file in problems_to_run:
                prob_path = os.path.join(dirpath, prob_file)

                # --- 1. Parsear Problema ---
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

                    # CHEQUEO DE SKIP
                    if (domain_name, prob_file) in completed_tasks[planner_name]:
                        # Opcional: imprimir que se salta, o dejarlo en silencio para limpiar output
                        # print(f"      ⏭️   {prob_file} -> {planner_name} (Already done)")
                        continue

                    if dry_run:
                        print(f"      [DRY RUN] {prob_file} -> {planner_name}")
                    else:
                        print(f"      🚀 {prob_file} -> {planner_name} ... ", end="", flush=True)
                        start_time = time.time()
                        solved = False

                        try:
                            with OneshotPlanner(name=planner_name) as planner:
                                result = planner.solve(problem, timeout=TIMEOUT_SEC)
                                duration = time.time() - start_time

                                if result.status.name in ['SOLVED_SATISFICING', 'SOLVED_OPTIMALLY']:
                                    print(f"✅ Solved ({duration:.2f}s)")
                                    solved = True
                                else:
                                    print(f"❌ {result.status.name} ({duration:.2f}s)")
                                    if result.log_messages:
                                        for log in result.log_messages:
                                            print(f"      [Log]: {log.message}")
                                    solved = False

                                # Escribir resultado y actualizar memoria local por si acaso
                                write_csv_result(planner_name, domain_name, prob_file, solved, duration)
                                completed_tasks[planner_name].add((domain_name, prob_file))

                        except Exception as e:
                            duration = time.time() - start_time
                            print(f"⚠️ Crash/Error ({duration:.2f}s)")
                            print(f"      >>> 🛑 DETALLE DEL ERROR: {e}")
                            write_csv_result(planner_name, domain_name, prob_file, False, duration)
                            completed_tasks[planner_name].add((domain_name, prob_file))

            print("-" * 40)

    if domains_found == 0:
        print("\n❌ No domains found.")
    else:
        print(f"\n--- Complete. Processed {domains_found} domains. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmarks with Resume capability")
    parser.add_argument("--run", action="store_true", help="Actually run the planners")
    parser.add_argument("--dir", default=".", help="Root directory of benchmarks")

    args = parser.parse_args()

    is_dry_run = not args.run

    if not is_dry_run:
        os.makedirs(LOG_DIR, exist_ok=True)

    run_benchmarks_up(args.dir, dry_run=is_dry_run)

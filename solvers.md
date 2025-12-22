# Solver Portfolio (8 Fast Downward configs, satisficing-only)

This portfolio defines the set of planners (Fast Downward configurations) we consider. The goal is **diversity**: each configuration is chosen to be *as orthogonal as possible* to the others in terms of (i) **search control** and (ii) **heuristic signal**.

Unless otherwise noted, commands assume the standard Fast Downward invocation:

- `domain.pddl` is the PDDL domain file
- `problem.pddl` is the PDDL problem instance

---

## 1) LAMA-first baseline

**Why (orthogonality):** Landmark-based guidance + preferred operators; a very strong general-purpose baseline.

```bash
./fast-downward.py --alias lama-first domain.pddl problem.pddl
```

---

## 2) IPC LAMA-2011 satisficing

**Why (orthogonality vs 1):** Competition-tuned configuration; often differs in phases/parameters versus `lama-first`.

```bash
./fast-downward.py --alias seq-sat-lama-2011 domain.pddl problem.pddl
```

---

## 3) FF-style GBFS (classic greedy with FF heuristic)

**Why (orthogonality):** Pure delete-relaxation / relaxed-plan heuristic control; tends to behave very differently from landmark-heavy LAMA on many domains.

```bash
./fast-downward.py domain.pddl problem.pddl \
  --evaluator "hff=ff()" \
  --search "lazy_greedy([hff], preferred=[hff])"
```

---

## 4) Enforced Hill-Climbing with FF (EHC-FF)

**Why (orthogonality):** Local-search-ish behavior (FF’s classic EHC idea). This can succeed fast where GBFS stalls, and vice versa.

```bash
./fast-downward.py domain.pddl problem.pddl \
  --evaluator "hff=ff()" \
  --search "ehc(hff, preferred=[hff])"
```

---

## 5) Greedy with CEA heuristic

**Why (orthogonality):** CEA is a different relaxation-based heuristic than FF/add; it changes plateau and dead-end behavior.

```bash
./fast-downward.py domain.pddl problem.pddl \
  --evaluator "hcea=cea()" \
  --search "lazy_greedy([hcea], preferred=[hcea])"
```

---

## 6) Greedy with Causal-Graph heuristic (CG)

**Why (orthogonality):** CG is a non delete-relaxation heuristic family (very different signal than FF/CEA/landmarks). Fast Downward is historically built around causal-graph heuristic ideas.

```bash
./fast-downward.py domain.pddl problem.pddl \
  --evaluator "hcg=cg()" \
  --search "lazy_greedy([hcg])"
```

---

## 7) Enforced Hill-Climbing with Goalcount (EHC-GC)

**Why (orthogonality):** Extremely cheap heuristic; sometimes surprisingly effective for easy/structured tasks, and it has a different failure mode than FF/CEA.

```bash
./fast-downward.py domain.pddl problem.pddl \
  --evaluator "hgc=goalcount()" \
  --search "ehc(hgc)"
```

---

## 8) Weighted A* (wastar) with FF heuristic

**Why (orthogonality):** Introduces a more systematic $g + w \cdot h$ bias (less “pure greedy”), often rescuing cases where greedy search thrashes.

```bash
./fast-downward.py domain.pddl problem.pddl \
  --evaluator "hff=ff()" \
  --search "eager_wastar([hff], w=5, preferred=[hff])"
```

---

# Why this set is “maximally orthogonal”

This lineup is deliberately spread across two axes.

## Search control diversity

- Greedy best-first (lazy): (3), (5), (6)
- Enforced hill-climbing: (4), (7)
- Weighted A*: (8)
- IPC-tuned / landmark-driven configurations: (1), (2)

## Heuristic signal diversity

- Landmarks (LAMA family): (1), (2)
- Relaxation / relaxed-plan heuristics: FF in (3), (4), (8); CEA in (5)
- Causal-graph family: (6)
- Very cheap STRIPS heuristic: goalcount in (7)

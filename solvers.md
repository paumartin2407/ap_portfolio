# Solver Portfolio (3 Fast Downward configs, satisficing-only)

This portfolio defines the set of planners (Fast Downward configurations) we consider.

Unless otherwise noted, commands assume the standard Fast Downward invocation:

- `domain.pddl` is the PDDL domain file
- `problem.pddl` is the PDDL problem instance

---

## 1) LAMA-first baseline

```bash
./fast-downward.py --alias lama-first domain.pddl problem.pddl
```

---

## 2) FD autotune (seq-sat-fd-autotune-1)

```bash
./fast-downward.py --alias seq-sat-fd-autotune-1 domain.pddl problem.pddl
```

---

## 3) FD autotune (seq-sat-fd-autotune-2)

```bash
./fast-downward.py --alias seq-sat-fd-autotune-2 domain.pddl problem.pddl
```

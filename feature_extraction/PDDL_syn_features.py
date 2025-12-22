from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import statistics
import re
import math

SExp = Union[str, List["SExp"]]


def strip_pddl_comments(text: str) -> str:
    # Remove ';' comments to end of line.
    return re.sub(r";[^\n]*", "", text)


def tokenize_sexp(text: str) -> List[str]:
    # Surround parentheses with spaces, split on whitespace.
    text = text.replace("(", " ( ").replace(")", " ) ")
    return [t for t in text.split() if t]


def parse_sexp(tokens: List[str]) -> SExp:
    if not tokens:
        raise ValueError("Unexpected EOF while reading S-expression.")
    tok = tokens.pop(0)
    if tok == "(":
        lst: List[SExp] = []
        while tokens and tokens[0] != ")":
            lst.append(parse_sexp(tokens))
        if not tokens:
            raise ValueError("Missing ')' in S-expression.")
        tokens.pop(0)  # consume ')'
        return lst
    if tok == ")":
        raise ValueError("Unexpected ')'")
    return tok.lower()


def parse_pddl_file(path: Path) -> SExp:
    text = strip_pddl_comments(path.read_text(encoding="utf-8", errors="ignore"))
    tokens = tokenize_sexp(text)
    sexp = parse_sexp(tokens)
    if tokens:
        # Some files contain multiple top-level expressions; handle that if needed.
        rest = []
        while tokens:
            rest.append(parse_sexp(tokens))
        return [sexp] + rest
    return sexp


def find_sections(tree: SExp, key: str) -> List[SExp]:
    """Return all sublists whose first element is `key` (case-insensitive already)."""
    found = []

    def rec(node: SExp):
        if isinstance(node, list) and node:
            head = node[0]
            if isinstance(head, str) and head == key:
                found.append(node)
            for ch in node:
                rec(ch)

    rec(tree)
    return found


def iter_domain_actions(domain_tree: SExp) -> List[SExp]:
    # Actions typically look like: (:action name :parameters (...) :precondition (...) :effect (...))
    # We find all lists starting with ':action'.
    actions = []
    for node in find_sections(domain_tree, ":action"):
        actions.append(node)
    return actions


def count_atoms(expr: SExp) -> int:
    """Count atomic predicates in a logical expression tree."""
    if isinstance(expr, str):
        return 0
    if not expr:
        return 0
    head = expr[0] if isinstance(expr[0], str) else None
    if head in {"and", "or"}:
        return sum(count_atoms(e) for e in expr[1:])
    if head in {"not"}:
        # not(<atom or subexpr>)
        return count_atoms(expr[1]) if len(expr) > 1 else 0
    if head in {"imply", "forall", "exists"}:
        # Rough: skip quantifier vars/typed lists and count inside body.
        # Typical: (forall (?x - t) <body>)
        return count_atoms(expr[-1]) if len(expr) >= 2 else 0
    if head == "when":
        # (when <cond> <eff>)
        # for preconditions we don't expect 'when', but handle anyway:
        return count_atoms(expr[1]) + count_atoms(expr[2]) if len(expr) >= 3 else 0

    # Otherwise treat as an atom: (pred arg1 arg2 ...)
    return 1


def has_negative_precondition(expr: SExp) -> bool:
    if isinstance(expr, str):
        return False
    if not expr:
        return False
    head = expr[0] if isinstance(expr[0], str) else None
    if head == "not":
        return True
    return any(has_negative_precondition(e) for e in expr)


def effect_add_del_counts(effect: SExp) -> Tuple[int, int, bool]:
    """
    Return (add_count, del_count, has_conditional_effect).
    - delete effect: (not (p ...))
    - conditional effect: (when <cond> <eff>)
    """
    add = 0
    dele = 0
    conditional = False

    def rec(e: SExp):
        nonlocal add, dele, conditional
        if isinstance(e, str) or not e:
            return
        head = e[0] if isinstance(e[0], str) else None
        if head == "and":
            for x in e[1:]:
                rec(x)
            return
        if head == "when":
            conditional = True
            # effects can be nested in the "then" part
            if len(e) >= 3:
                rec(e[2])
            return
        if head == "not":
            # delete
            dele += 1
            return
        # otherwise: add atom
        add += 1

    rec(effect)
    return add, dele, conditional


@dataclass
class PDDLSyntacticFeatures:
    # Domain-level
    num_predicates: int
    num_action_schemas: int
    has_derived_predicates: int
    has_conditional_effects: int
    has_negative_preconditions: int

    precond_atoms_min: int
    precond_atoms_mean: float
    precond_atoms_max: int

    add_eff_min: int
    add_eff_mean: float
    add_eff_max: int

    del_eff_min: int
    del_eff_mean: float
    del_eff_max: int

    # Problem-level
    num_objects: int
    num_object_types: int
    objects_by_type_total: int
    objects_by_type_min: int
    objects_by_type_max: int
    objects_by_type_mean: float
    objects_by_type_std: float
    objects_by_type_entropy: float
    objects_by_type_gini: float
    objects_by_type_top1_frac: float
    objects_by_type_top2_frac: float
    objects_by_type_top3_frac: float
    init_atoms: int
    goal_atoms: int


def _object_type_stats(objects_by_type: Dict[str, int]) -> Tuple[
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Compute domain-agnostic summary stats from per-type object counts."""
    if not objects_by_type:
        return (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    counts: List[int] = []
    for v in objects_by_type.values():
        try:
            iv = int(v)
        except Exception:
            continue
        if iv < 0:
            continue
        counts.append(iv)

    n_types = len(objects_by_type)
    total = sum(counts) if counts else 0
    cmin = min(counts) if counts else 0
    cmax = max(counts) if counts else 0
    mean = (total / len(counts)) if counts else 0.0
    var = 0.0
    if len(counts) > 1:
        var = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = math.sqrt(var)

    # Entropy of proportions (natural log).
    ent = 0.0
    if total > 0:
        for c in counts:
            if c <= 0:
                continue
            p = c / total
            ent -= p * math.log(p)

    # Gini coefficient.
    gini = 0.0
    if total > 0 and len(counts) > 1:
        xs = sorted(counts)
        n = len(xs)
        cum = 0
        for i, x in enumerate(xs, start=1):
            cum += i * x
        gini = (2 * cum) / (n * total) - (n + 1) / n
        gini = max(0.0, min(1.0, gini))

    top = sorted(counts, reverse=True)
    top1 = (top[0] / total) if total > 0 and len(top) >= 1 else 0.0
    top2 = ((top[0] + top[1]) / total) if total > 0 and len(top) >= 2 else top1
    top3 = ((top[0] + top[1] + top[2]) / total) if total > 0 and len(top) >= 3 else top2

    return (
        int(n_types),
        int(total),
        int(cmin),
        int(cmax),
        float(mean),
        float(std),
        float(ent),
        float(gini),
        float(top1),
        float(top2),
        float(top3),
    )


def extract_pddl_syntactic(domain_path: Path, problem_path: Path) -> PDDLSyntacticFeatures:
    dom = parse_pddl_file(domain_path)
    prob = parse_pddl_file(problem_path)

    # Predicates
    pred_sections = find_sections(dom, ":predicates")
    num_predicates = 0
    if pred_sections:
        # (:predicates (p ?x) (q ?x ?y) ...)
        num_predicates = sum(
            1 for item in pred_sections[0][1:] if isinstance(item, list) and item
        )

    # Actions
    actions = iter_domain_actions(dom)
    num_action_schemas = len(actions)

    precond_counts = []
    add_counts = []
    del_counts = []
    any_cond_eff = False
    any_neg_pre = False

    for act in actions:
        # Extract :precondition and :effect blocks.
        # act looks like [':action', name, ':parameters', [...], ':precondition', [...], ':effect', [...], ...]
        pre = None
        eff = None
        i = 0
        while i < len(act):
            if act[i] == ":precondition" and i + 1 < len(act):
                pre = act[i + 1]
            if act[i] == ":effect" and i + 1 < len(act):
                eff = act[i + 1]
            i += 1

        if pre is not None:
            precond_counts.append(count_atoms(pre))
            any_neg_pre = any_neg_pre or has_negative_precondition(pre)
        else:
            precond_counts.append(0)

        if eff is not None:
            a, d, cond = effect_add_del_counts(eff)
            add_counts.append(a)
            del_counts.append(d)
            any_cond_eff = any_cond_eff or cond
        else:
            add_counts.append(0)
            del_counts.append(0)

    # Derived predicates
    has_derived = 1 if find_sections(dom, ":derived") else 0

    # Problem: objects
    obj_sections = find_sections(prob, ":objects")
    num_objects = 0
    objects_by_type: Dict[str, int] = {}
    if obj_sections:
        # parse typed list: a b c - type1 d e - type2 ...
        raw = obj_sections[0][1:]
        cur_names = []
        cur_type = "untyped"
        j = 0
        while j < len(raw):
            tok = raw[j]
            if tok == "-":
                if j + 1 < len(raw):
                    cur_type = str(raw[j + 1])
                    objects_by_type[cur_type] = objects_by_type.get(cur_type, 0) + len(cur_names)
                    num_objects += len(cur_names)
                    cur_names = []
                    j += 2
                    cur_type = "untyped"
                    continue
            # symbol
            if isinstance(tok, str):
                cur_names.append(tok)
            j += 1
        # remaining untyped names
        if cur_names:
            objects_by_type["untyped"] = objects_by_type.get("untyped", 0) + len(cur_names)
            num_objects += len(cur_names)

    (
        num_object_types,
        objects_by_type_total,
        objects_by_type_min,
        objects_by_type_max,
        objects_by_type_mean,
        objects_by_type_std,
        objects_by_type_entropy,
        objects_by_type_gini,
        objects_by_type_top1_frac,
        objects_by_type_top2_frac,
        objects_by_type_top3_frac,
    ) = _object_type_stats(objects_by_type)

    # Problem: init
    init_sections = find_sections(prob, ":init")
    init_atoms = 0
    if init_sections:
        for item in init_sections[0][1:]:
            # Ignore numeric assignments like (= (f ...) 3) by skipping head '='
            if isinstance(item, list) and item and item[0] == "=":
                continue
            if isinstance(item, list):
                init_atoms += 1

    # Problem: goal
    goal_sections = find_sections(prob, ":goal")
    goal_atoms = 0
    if goal_sections and len(goal_sections[0]) >= 2:
        goal_expr = goal_sections[0][1]
        goal_atoms = count_atoms(goal_expr)

    def stats(vals: List[int]) -> Tuple[int, float, int]:
        if not vals:
            return 0, 0.0, 0
        return min(vals), float(statistics.mean(vals)), max(vals)

    pre_min, pre_mean, pre_max = stats(precond_counts)
    add_min, add_mean, add_max = stats(add_counts)
    del_min, del_mean, del_max = stats(del_counts)

    return PDDLSyntacticFeatures(
        num_predicates=num_predicates,
        num_action_schemas=num_action_schemas,
        has_derived_predicates=has_derived,
        has_conditional_effects=1 if any_cond_eff else 0,
        has_negative_preconditions=1 if any_neg_pre else 0,
        precond_atoms_min=pre_min,
        precond_atoms_mean=pre_mean,
        precond_atoms_max=pre_max,
        add_eff_min=add_min,
        add_eff_mean=add_mean,
        add_eff_max=add_max,
        del_eff_min=del_min,
        del_eff_mean=del_mean,
        del_eff_max=del_max,
        num_objects=num_objects,
        num_object_types=num_object_types,
        objects_by_type_total=objects_by_type_total,
        objects_by_type_min=objects_by_type_min,
        objects_by_type_max=objects_by_type_max,
        objects_by_type_mean=objects_by_type_mean,
        objects_by_type_std=objects_by_type_std,
        objects_by_type_entropy=objects_by_type_entropy,
        objects_by_type_gini=objects_by_type_gini,
        objects_by_type_top1_frac=objects_by_type_top1_frac,
        objects_by_type_top2_frac=objects_by_type_top2_frac,
        objects_by_type_top3_frac=objects_by_type_top3_frac,
        init_atoms=init_atoms,
        goal_atoms=goal_atoms,
    )

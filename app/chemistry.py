"""Reaction network description, validation and compilation.

A network is a JSON-serialisable dict::

    {
      "species": ["A", "B", "C"],
      "reactions": [
        {"name": "r1",
         "stoichiometry": {"A": -1, "B": 1},
         "orders": {"A": 1.0},            # forward orders (defaults below)
         "reversible": false,
         "reverse_orders": {}},           # only when reversible
        ...
      ]
    }

Each reaction owns one forward rate constant; reversible reactions own an
additional reverse constant.  Each constant follows its own Arrhenius law
``k = A * exp(-Ea / (R*T))``, so a network with ``n`` irreversible reactions
has ``n`` parameter pairs (A, Ea); a reversible reaction contributes two.

Default forward order of a species is ``-min(stoich, 0)`` (reactant side),
default reverse order is ``max(stoich, 0)`` (product side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

R_GAS = 8.314462618  # J mol^-1 K^-1


class ValidationError(Exception):
    """Field-level validation problems.

    Each element is ``(field_path, message)``, e.g.
    ``("reactions[1].orders.B", "reaction references undefined species 'B'")``.
    """

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{f}: {m}" for f, m in errors))

    def to_dict(self) -> list[dict]:
        return [{"field": f, "message": m} for f, m in self.errors]


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


@dataclass(frozen=True)
class RateLaw:
    """Mass-action monomial: k * prod_i c[species_i] ** order_i."""

    label: str  # e.g. "r1.forward"
    order_idx: tuple[int, ...]
    order_pow: tuple[float, ...]

    def power(self, c: np.ndarray) -> float:
        p = 1.0
        for idx, order in zip(self.order_idx, self.order_pow):
            p *= c[idx] ** order
        return p


@dataclass(frozen=True)
class CompiledNetwork:
    species: tuple[str, ...]
    species_idx: dict[str, int]
    reactions: tuple[str, ...]
    reversible: tuple[bool, ...]
    # stoichiometry rows indexed by reaction (forward direction), columns species
    stoich: np.ndarray
    forward_laws: tuple[RateLaw, ...]
    reverse_laws: tuple  # RateLaw | None per reaction

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_constants(self) -> int:
        """Number of temperature-dependent rate constants (params pairs)."""
        return len(self.reactions) + sum(1 for r in self.reversible if r)

    def rate_constant_labels(self) -> list[str]:
        labels = []
        for i, name in enumerate(self.reactions):
            labels.append(f"{name}.forward")
            if self.reversible[i]:
                labels.append(f"{name}.reverse")
        return labels

    def _constant_pairs(self, kvec: np.ndarray):
        """Yield (reaction_index, k_forward, k_reverse_or_None) from flat k."""
        j = 0
        for i, rev in enumerate(self.reversible):
            kf = kvec[j]
            j += 1
            kr = kvec[j] if rev else None
            if rev:
                j += 1
            yield i, kf, kr

    def reaction_rates(self, c: np.ndarray, kvec: np.ndarray) -> np.ndarray:
        """Net (forward - reverse) rate of each reaction."""
        r = np.zeros(len(self.reactions))
        for i, kf, kr in self._constant_pairs(kvec):
            v = kf * self.forward_laws[i].power(c)
            if kr is not None:
                v -= kr * self.reverse_laws[i].power(c)
            r[i] = v
        return r

    def rhs(self, c: np.ndarray, kvec: np.ndarray) -> np.ndarray:
        """dc/dt = stoich^T @ reaction_rates."""
        return self.reaction_rates(c, kvec) @ self.stoich


def validate_network(net: dict) -> None:
    """Raise :class:`ValidationError` listing every bad field."""
    errors: list[tuple[str, str]] = []
    if not isinstance(net, dict):
        raise ValidationError([("", "network must be an object")])

    species = net.get("species")
    names: list[str] = []
    if not isinstance(species, list) or not species:
        errors.append(("species", "must be a non-empty list"))
    else:
        for i, s in enumerate(species):
            if not isinstance(s, str) or not s:
                errors.append((f"species[{i}]", "must be a non-empty string"))
                continue
            if s in names:
                errors.append((f"species[{i}]", f"duplicate species '{s}'"))
            names.append(s)

    reactions = net.get("reactions")
    if not isinstance(reactions, list) or not reactions:
        errors.append(("reactions", "must be a non-empty list"))
        reactions = []

    name_set = set(names)
    seen: set[str] = set()
    for i, rx in enumerate(reactions):
        p = f"reactions[{i}]"
        if not isinstance(rx, dict):
            errors.append((p, "must be an object"))
            continue

        rname = rx.get("name", f"reaction_{i}")
        if not isinstance(rname, str) or not rname:
            errors.append((f"{p}.name", "must be a non-empty string"))
        elif rname in seen:
            errors.append((f"{p}.name", f"duplicate reaction name '{rname}'"))
        else:
            seen.add(rname)

        stoich = rx.get("stoichiometry")
        if not isinstance(stoich, dict) or not stoich:
            errors.append((f"{p}.stoichiometry", "must be a non-empty object"))
        else:
            for sp, v in stoich.items():
                if sp not in name_set:
                    errors.append((f"{p}.stoichiometry.{sp}", f"undefined species '{sp}'"))
                if not _is_finite_number(v) or float(v) == 0.0:
                    errors.append((f"{p}.stoichiometry.{sp}", "must be a non-zero finite number"))

        for key, allow_missing in (("orders", True), ("reverse_orders", True)):
            orders = rx.get(key)
            if orders is None:
                orders = {}
            if not isinstance(orders, dict):
                errors.append((f"{p}.{key}", "must be an object"))
                continue
            for sp, v in orders.items():
                if sp not in name_set:
                    errors.append((f"{p}.{key}.{sp}", f"undefined species '{sp}'"))
                elif not _is_finite_number(v) or float(v) < 0.0:
                    errors.append((f"{p}.{key}.{sp}", "must be a non-negative finite number"))

        rev = rx.get("reversible", False)
        if not isinstance(rev, bool):
            errors.append((f"{p}.reversible", "must be true or false"))
        elif rev and isinstance(stoich, dict) and not any(float(v) > 0 for v in stoich.values()):
            errors.append((f"{p}.reversible", "reverse reaction needs at least one product (positive stoichiometry)"))

    if errors:
        raise ValidationError(errors)


def compile_network(net: dict) -> CompiledNetwork:
    """Validate and compile a network description."""
    validate_network(net)
    names = list(net["species"])
    idx = {s: i for i, s in enumerate(names)}
    n = len(names)

    reaction_names: list[str] = []
    reversible: list[bool] = []
    stoich_rows: list[np.ndarray] = []
    fwd_laws: list[RateLaw] = []
    rev_laws: list[RateLaw | None] = []

    def build_law(rname, direction, orders: dict, stoich: dict, forward_default: bool):
        oi, op = [], []
        for sp, sv in stoich.items():
            if sp in orders:
                v = float(orders[sp])
            elif forward_default:
                v = float(-min(float(sv), 0.0))
            else:
                v = float(max(float(sv), 0.0))
            if v != 0.0:
                oi.append(idx[sp])
                op.append(v)
        return RateLaw(f"{rname}.{direction}", tuple(oi), tuple(op))

    for i, rx in enumerate(net["reactions"]):
        rname = rx.get("name", f"reaction_{i}")
        rev = bool(rx.get("reversible", False))
        row = np.zeros(n)
        for sp, v in rx["stoichiometry"].items():
            row[idx[sp]] += float(v)

        reaction_names.append(rname)
        reversible.append(rev)
        stoich_rows.append(row)
        fwd_laws.append(build_law(rname, "forward", rx.get("orders") or {}, rx["stoichiometry"], True))
        if rev:
            rev_laws.append(build_law(rname, "reverse", rx.get("reverse_orders") or {}, rx["stoichiometry"], False))
        else:
            rev_laws.append(None)

    return CompiledNetwork(
        tuple(names), idx, tuple(reaction_names), tuple(reversible),
        np.asarray(stoich_rows, dtype=float), tuple(fwd_laws), tuple(rev_laws),
    )

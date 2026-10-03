"""Reaction-network description and kinetics.

A network is described by the caller with:

* ``components``: list of component names;
* ``reactions``: elementary reactions, each with stoichiometric coefficients
  and per-component reaction orders, optionally reversible.

Every reaction contributes one or two *channels* (a forward channel, and a
reverse channel when ``reversible`` is true).  Each channel has its own
Arrhenius parameters (pre-exponential factor ``A`` and activation energy
``Ea``) and its own rate law

    rate_channel = k_channel(T) * prod_i c_i ** order_i          [concentration / time]

with

    k_channel(T) = A * exp(-Ea / (R * T)).

The component balance is

    dc_i/dt = sum_channel nu_channel,i * rate_channel.

Keeping forward and reverse rates as independent channels (instead of a single
subtracted net rate) makes the integration well behaved near equilibrium and
gives every Arrhenius parameter its own estimable quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Universal gas constant in J mol^-1 K^-1.
R_GAS = 8.31446261815324


@dataclass(frozen=True)
class Channel:
    """One directed elementary step (forward or reverse of a reaction)."""

    reaction_id: str
    direction: str  # "forward" | "reverse"
    label: str
    nu: tuple[float, ...]          # stoichiometric change per event
    orders: tuple[float, ...]      # reaction order w.r.t. each component


@dataclass
class Network:
    components: list[str]
    reactions: list[dict]
    component_index: dict[str, int] = field(init=False)
    channels: list[Channel] = field(init=False)

    def __post_init__(self) -> None:
        self.component_index = {name: i for i, name in enumerate(self.components)}
        n = len(self.components)
        channels: list[Channel] = []
        for rxn in self.reactions:
            rid = str(rxn["id"])
            stoich = rxn.get("stoichiometry", {})
            orders = rxn.get("orders", {})
            rev_orders = rxn.get("reverse_orders", {})

            def vectors(order_map: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
                nu = [0.0] * n
                od = [0.0] * n
                for comp, coef in stoich.items():
                    nu[self.component_index[comp]] = float(coef)
                for comp, p in order_map.items():
                    od[self.component_index[comp]] = float(p)
                return tuple(nu), tuple(od)

            f_nu, f_ord = vectors(orders)
            channels.append(Channel(rid, "forward", f"{rid}:forward", f_nu, f_ord))
            if rxn.get("reversible"):
                # Reverse event undoes the stoichiometric change; reverse
                # orders default to the reverse-side stoichiometry unless
                # explicitly given.
                r_nu = tuple(-v for v in f_nu)
                r_ord_map = rev_orders if rev_orders else {
                    # Reverse reaction consumes the forward products; assume
                    # elementary order = forward product coefficient.
                    comp: coef for comp, coef in stoich.items() if coef > 0
                }
                _, r_ord = vectors(r_ord_map)
                channels.append(Channel(rid, "reverse", f"{rid}:reverse", r_nu, r_ord))
        self.channels = channels

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def stoich_matrix(self):
        import numpy as np

        return np.array([ch.nu for ch in self.channels], dtype=float)

    def rate_constants(self, temperature, ln_a, ea):
        """Arrhenius rate constants.

        Parameters are accepted as log-pre-exponential ``ln_a`` so that fitting
        works in log space.  ``ea`` is in J/mol.
        """
        import numpy as np

        ln_a = np.asarray(ln_a, dtype=float)
        ea = np.asarray(ea, dtype=float)
        return np.exp(ln_a - ea / (R_GAS * float(temperature)))

    def rates(self, concentration, temperature, ln_a, ea):
        """Channel reaction rates at a state point."""
        import numpy as np

        c = np.maximum(np.asarray(concentration, dtype=float), 0.0)
        k = self.rate_constants(temperature, ln_a, ea)
        n_ch = self.n_channels
        r = np.ones(n_ch)
        for j, ch in enumerate(self.channels):
            for i, p in enumerate(ch.orders):
                if p != 0.0:
                    # np.pow without the 0**0 warning; 0**p == 0 for p > 0.
                    r[j] *= c[i] ** p
        return k * r

    def rhs(self, concentration, temperature, ln_a, ea):
        import numpy as np

        rates = self.rates(concentration, temperature, ln_a, ea)
        nu = self.stoich_matrix()
        return nu.T @ rates

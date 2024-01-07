from .nc_symbols import NCSymbols
from .normalize_basis import anti_comm, comm, normalize, single
from .symbol import Symbol


def particle_sort_order(symbols):
    ann_symbols = tuple(sym for sym in symbols.symbols if isinstance(sym, AnnihilationSymbol))
    non_ann_symbols = tuple(sym for sym in symbols.symbols if not isinstance(sym, AnnihilationSymbol))

    return (
        symbols.__class__.__name__,
        -len(ann_symbols),
        tuple((sym.name, not sym.is_conjugate) for sym in ann_symbols),
        -len(non_ann_symbols),
        tuple(sym.name for sym in non_ann_symbols),
    )


def particle_normalize_rule(seq):
    if len(seq) < 2:
        return None

    el1 = seq[0]
    el2 = seq[1]

    if len(seq) >= 3:
        el1 = seq[0]
        el2 = seq[1]
        el3 = seq[2]

        match el1, el2, el3:  # a† a a† -> a† ; a a† a -> a; rule may not be needed with normal ordering
            case AnnihilationSymbol(name1, conj1), AnnihilationSymbol(name2, conj2), AnnihilationSymbol(
                name3, conj3
            ) if name1 == name2 == name3 and conj1 == (not conj2) == conj3:
                return None, 3, (el1,)

    match el1, el2:
        # order by name
        case AnnihilationSymbol(name1), AnnihilationSymbol(name2) if name1 > name2:
            return anti_comm(el1, el2)

        # a a -> 0 ; a† a† -> 0
        case AnnihilationSymbol(name1, conj1), AnnihilationSymbol(name2, conj2) if name1 == name2 and conj1 == conj2:
            return 0

        # normal order: a a† -> 1 - a† a
        case AnnihilationSymbol(name1, conj1), AnnihilationSymbol(
            name2, conj2
        ) if name1 == name2 and not conj1 and conj2:
            return ((tuple(), 1), ((el2, el1), -1)), 2, None


def particle_normalize(basis_factor):
    basis_factor = {basis.symbols: factor for basis, factor in basis_factor.items()}  # unpack NCSymbols

    normalized_basis_factor = normalize(basis_factor, rule=particle_normalize_rule)

    basis_factor = {  # re-wrap NYCymbols
        NCSymbols(basis, sort_order=particle_sort_order): factor for basis, factor in normalized_basis_factor.items()
    }

    return basis_factor


# @dataclass(unsafe_hash=True, order=True)
class AnnihilationSymbol(Symbol):
    def _needs_conjugate(self):
        return True

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        printer.pretty(self.color(self.name + ("†" if self.is_conjugate else "")))

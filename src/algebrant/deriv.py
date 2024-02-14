from collections import Counter
from numbers import Number

from .algebra import Algebra
from .clifford import CliffordBasis
from .deriv_symbol import DerivSymbol
from .symbol import Symbol
from .symbols import Symbols


def deriv(term: Algebra | Number, param: str):
    if isinstance(term, Number):  # usually should not happen due to optimization
        return 0

    if not isinstance(term, Algebra):
        raise ValueError(f"Cannot derive type {type(term)} with value {term}")

    result_terms = []

    for basis, factor in term.basis_factor.items():
        match basis:
            case CliffordBasis():
                new_part = term._create({basis: deriv(param, factor)})
                result_terms.append(new_part)  # derivative of factor only
            case Symbols():
                symbol_powers = list(basis.symbol_powers)
                for i, (symbol, power) in enumerate(symbol_powers):
                    assert power != 0

                    match symbol:
                        case Symbol(name) if name == param:
                            new_symbol_powers = symbol_powers[:i] + [(symbol, power - 1)] + symbol_powers[i + 1 :]
                            new_part = term._create({Symbols(new_symbol_powers): factor * power})
                            result_terms.append(new_part)
                        case DerivSymbol(
                            deriv_symbol, parameters, derivatives
                        ) if not parameters or param in parameters:
                            new_derivatives = Counter(dict(derivatives))
                            new_derivatives[param] += 1

                            new_deriv_symbol = DerivSymbol(deriv_symbol, parameters, frozenset(new_derivatives.items()))

                            new_symbol_powers = Counter(
                                dict(
                                    symbol_powers[:i] + [(symbol, power - 1)] + symbol_powers[i + 1 :]
                                )  # assumes symbols don't change and don't collide
                            )
                            new_symbol_powers[new_deriv_symbol] += 1

                            new_part = term._create({Symbols(frozenset(new_symbol_powers.items())): factor * power})
                            result_terms.append(new_part)
                        case _:
                            raise ValueError(f"Unknown symbol {symbol} of type {type(symbol)}")
            case _:
                raise ValueError(f"Unknown basis type {type(basis)} with value {basis}")

    return sum(result_terms)

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal, LiteralString, Sequence

from algebrant.algebra import Algebra
from algebrant.creation import sym
from algebrant.graded.graded_symbol import GradedMultivector, GradedSymbol, get_valid_grades
from algebrant.repr_printer import ReprPrinter

"""
* What is contraction is zero since too many contractions?
"""


def _get_max_grade(elem) -> int:
    grade = elem.grade
    if grade is not None:
        return grade

    return elem.max_grade  # TODO


@dataclass(repr=False)
class GradedSymbolProduct:
    def __init__(self, elems: "Sequence[WedgeSortable | Wedge]") -> None:
        self.elems = elems

        if len(elems) == 1:
            self.grade = elems[0].grade
        else:
            self.grade = None

        self.is_odd = sum(elem.is_odd for elem in elems) % 2

    def _repr_pretty_(self, printer, cycle) -> None:
        if cycle:
            printer.text("...")
            return

        printer.text("(")
        for i, elem in enumerate(self.elems):
            if i > 0:
                printer.text(" ")

            if not isinstance(elem, GradedSymbol):
                printer.text("(")

            printer.pretty(elem)

            if not isinstance(elem, GradedSymbol):
                printer.text(")")
        printer.text(")")

    def __hash__(self) -> int:
        return hash(tuple(hash(e) for e in self.elems))

    @property
    def max_grade(self) -> int:
        return sum(_get_max_grade(elem) for elem in self.elems)


def is_vector(elem: Wedge) -> bool:
    """
    Check if the element is a vector.
    """
    return elem.grade == 1


def calc_vector_contraction_single(vector: Wedge, other: Wedge) -> dict[Wedge, Any]:
    if not is_vector(vector):
        raise ValueError("Vector must be grade 1")

    result: dict[Wedge, Any] = defaultdict(float)

    sign_involution = 1

    for i, elem in enumerate(other.elems):
        if isinstance(elem, LeftContraction):
            contraction, contr_coef = make_leftcontraction(
                [vector.elems[0]] + list(elem.elems)
            )

        else:
            contraction, contr_coef = make_leftcontraction([vector.elems[0], elem])

        if contraction is None:
            continue

        scalar_text = contraction.test_scalar_expression()

        if scalar_text is not None:
            new_wedge, wedge_coef = make_wedge(
                list(other.elems[:i]) + list(other.elems[i + 1 :])
            )
            if new_wedge is None:
                continue

            coef = sign_involution * wedge_coef * contr_coef * sym(scalar_text)  # TODO
        else:
            new_wedge, wedge_coef = make_wedge(
                list(other.elems[:i]) + [contraction] + list(other.elems[i + 1 :])
            )
            if new_wedge is None:
                continue

            coef = sign_involution * wedge_coef * contr_coef

        result[new_wedge] += coef

        if elem.is_odd:
            sign_involution *= -1

    return result


def calc_vector_contraction_basis_factor(
    vector, basis_factor
) -> defaultdict[Wedge, Any]:
    if not is_vector(vector):
        raise ValueError("Vector must be grade 1")

    result: dict[Wedge, Any] = defaultdict(float)

    for basis, factor in basis_factor.items():
        elem_contr = calc_vector_contraction_single(vector, basis)
        for basis2, factor2 in elem_contr.items():
            result[basis2] += factor * factor2

    return result


def calc_wedge_single(elem1: Wedge, elem2: Wedge) -> dict[Wedge, int]:
    parts1 = list(elem1.elems)
    parts2 = list(elem2.elems)

    wedge, coef = make_wedge(parts1 + parts2)
    if wedge is None:
        return {}

    return {wedge: coef}


def calc_wedge_basis_factor(elem, basis_factor2):
    result = defaultdict(float)
    for basis, factor in basis_factor2.items():
        wedge, wedge_coef = calc_wedge_single(elem, basis)
        if wedge is None:
            continue

        result[wedge] += factor * wedge_coef

    return result


def calc_product(factors: list[GradedSymbol]) -> dict[Wedge, Any]:
    if not factors:
        return {Wedge([]): 1}

    grouped_factors: list[Wedge] = []
    counter = itertools.count()
    for _, group in itertools.groupby(
        factors, key=lambda x: next(counter) if x.grade == 1 else -1
    ):
        group = list(group)
        if len(group) > 1:
            grouped_factors.append(Wedge([GradedSymbolProduct(group)]))
        else:
            grouped_factors.append(Wedge([group[0]]))

    result: dict[Wedge, Any] = defaultdict(float)
    result[grouped_factors[0]] = 1

    for elem in grouped_factors[1:]:
        if elem.is_unity():
            continue

        new_result: dict[Wedge, Any] = defaultdict(float)

        for basis, factor in result.items():
            if basis.is_unity():
                new_result[elem] += factor

            elif is_vector(elem):
                sign_involution = {True: -1, False: 1}[basis.is_odd]

                new_basis_factor = calc_vector_contraction_single(elem, basis)
                for basis2, factor2 in new_basis_factor.items():
                    new_result[basis2] += factor * factor2 * -sign_involution

                new_basis_factor = calc_wedge_single(elem, basis)
                for basis2, factor2 in new_basis_factor.items():
                    new_result[basis2] += factor * factor2 * sign_involution

            elif is_vector(basis):
                new_basis_factor = calc_vector_contraction_single(basis, elem)
                for basis2, factor2 in new_basis_factor.items():
                    new_result[basis2] += factor * factor2

                new_basis_factor = calc_wedge_single(basis, elem)
                for basis2, factor2 in new_basis_factor.items():
                    new_result[basis2] += factor * factor2

            else:
                elem1 = basis
                elem2 = elem

                if len(elem1.elems) == 1:
                    elem1 = elem1.elems[0]

                if len(elem2.elems) == 1:
                    elem2 = elem2.elems[0]

                new_result[Wedge([GradedSymbolProduct([elem1, elem2])])] += factor

        result = new_result

    return result


def graded(algebra):
    result: dict[GradedMultivector, Any] = defaultdict(float)
    for basis, factor in algebra.basis_factor.items():
        if isinstance(basis, GradedMultivector):
            product = calc_product(basis.symbols).items()
            for basis2, factor2 in product:  # type: ignore
                result[GradedMultivector((basis2,) if basis2.elems else tuple())] += (
                    factor * factor2
                )
        else:
            result[GradedMultivector((basis,) if basis.elems else tuple())] += factor

    result = {basis: factor for basis, factor in result.items() if factor != 0}

    return Algebra(
        result,
        op_prio=algebra.op_prio,
        basis_class=GradedMultivector,  # TODO?
    )

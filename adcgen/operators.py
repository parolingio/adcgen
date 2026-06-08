from collections.abc import Sequence
from functools import cached_property
from typing import Any

from sympy import Add, Expr, Rational, Mul, factorial, latex, Symbol
from sympy.physics.secondquant import Fd, F

from .misc import cached_member
from .indices import Indices, Index, get_symbols
from .rules import Rules
from .sympy_objects import AntiSymmetricTensor
from .logger import logger
from .tensor_names import tensor_names


class Operators:
    """
    Constructs operators, like the zeroth and first order Hamiltonian or
    arbitrary N-particle operators.

    Parameters
    ----------
    variant : str, optional
        Defines the partitioning of the Hamiltonian.
        (default: the MP Hamiltonian)
    """
    def __init__(self, variant: str = "mp"):
        self._indices = Indices()
        self._variant = variant

    @cached_property
    def hamiltonian(self) -> Expr:
        """Constructs the full electronic Hamiltonian."""
        h = Add(self.mp_h0()[0], self.mp_h1()[0])
        assert isinstance(h, Expr)
        return h

    @cached_property
    def h0(self) -> tuple[Expr, Rules | None]:
        """Constructs the zeroth order Hamiltonian."""
        if self._variant == 'mp':
            return self.mp_h0()
        elif self._variant == 're':
            return self.re_h0()
        elif self._variant == 'remp':
            return self.remp_h0()
        else:
            raise NotImplementedError(
                f"H0 not implemented for {self._variant}"
            )

    @cached_property
    def h1(self) -> tuple[Expr, Rules | None]:
        """Constructs the first order Hamiltonian."""
        if self._variant == 'mp':
            return self.mp_h1()
        elif self._variant == 're':
            return self.re_h1()
        elif self._variant == 'remp':
            return self.remp_h1()
        else:
            raise NotImplementedError(
                f"H1 not implented for {self._variant}"
            )

    @cached_member
    def operator(self, n_create: int, n_annihilate: int) -> tuple[Expr, None]:
        """
        Constructs an arbitrary second quantized operator placing creation
        operators to the left of annihilation operators.

        Parameters
        ----------
        n_create : int
            The number of creation operators. Placed left of the annihilation
            operators.
        n_annihilate : int
            The number of annihilation operators. Placed right of the creation
            operators.
        """
        # generate general indices for the operator
        idx = self._indices.get_generic_indices(
            general=n_create + n_annihilate
        )
        idx = idx[("general", "")]
        create = idx[:n_create]
        annihilate = idx[n_create:]
        name = tensor_names.operator

        pref = Rational(1, Mul(factorial(n_create), factorial(n_annihilate)))
        d = AntiSymmetricTensor(name, create, annihilate)
        op = self.excitation_operator(creation=create,
                                      annihilation=annihilate,
                                      reverse_annihilation=True)
        return pref * d * op, None

    def excitation_operator(
            self, creation: Sequence[Index] | Sequence[str] | None = None,
            annihilation: Sequence[Index] | Sequence[str] | None = None,
            reverse_annihilation: bool = True
            ) -> Expr:
        """
        Creates an arbitrary string of second quantized excitation operators.
        Operators are concatenated as [creation * annihilation]

        Parameters
        ----------
        creation : Sequence[Index] | Sequence[str], optional
            Each of the provided indices is placed on a creation operator.
            Operators are concatenated in the provided order:
            [i, j] -> Fd(i) * Fd(j).
        annihilation : Sequence[Index] | Sequence[str], optional
            Each of the provided indices is placed on a annihilation operator.
            Operators are concatenated in the provided order:
            [i, j] -> F(i) * F(j)
        reverse_annihilation : bool, optional
            If set, the order of the annihilation operators is reversed, i.e.
            [i, j] -> F(j) * F(i)
            (default: True).
        """
        res = []
        if creation is not None:
            res.extend(Fd(s) for s in get_symbols(creation))
        if annihilation is not None:
            symbols = get_symbols(annihilation)
            if reverse_annihilation:
                symbols = reversed(symbols)
            res.extend(F(s) for s in symbols)
        expr = Mul(*res)
        assert isinstance(expr, Expr)
        return expr

    @staticmethod
    def mp_h0() -> tuple[Expr, None]:
        """Constructs the zeroth order MP-Hamiltonian."""
        idx_cls = Indices()
        p, q = idx_cls.get_indices("pq")[("general", "")]
        f = AntiSymmetricTensor(tensor_names.fock, (p,), (q,))
        pq = Mul(Fd(p), F(q))
        h0 = Mul(f, pq)
        assert isinstance(h0, Expr)
        logger.debug(f"H0 = {latex(h0)}")
        return h0, None

    @staticmethod
    def mp_h1() -> tuple[Expr, None]:
        """Constructs the first order MP-Hamiltonian."""
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices("pqrs")[("general", "")]
        # get an occ index for 1 particle part of H1
        occ = idx_cls.get_generic_indices(occ=1)[("occ", "")][0]
        v1 = AntiSymmetricTensor(tensor_names.eri, (p, occ), (q, occ))
        pq = Mul(Fd(p), F(q))
        v2 = AntiSymmetricTensor(tensor_names.eri, (p, q), (r, s))
        pqsr = Mul(Fd(p), Fd(q), F(s), F(r))
        h1 = Add(Mul(-v1, pq), Rational(1, 4) * v2 * pqsr)
        assert isinstance(h1, Expr)
        logger.debug(f"H1 = {latex(h1)}")
        return h1, None

    @staticmethod
    def re_h0() -> tuple[Expr, Rules]:
        """Constructs the zeroth order RE-Hamiltonian."""
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices('pqrs')[("general", "")]
        # get an occ index for 1 particle part of H0
        occ = idx_cls.get_generic_indices(occ=1)[("occ", "")][0]

        f = AntiSymmetricTensor(tensor_names.fock, (p,), (q,))
        piqi = AntiSymmetricTensor(tensor_names.eri, (p, occ), (q, occ))
        pqrs = AntiSymmetricTensor(tensor_names.eri, (p, q), (r, s))
        op_pq = Mul(Fd(p), F(q))
        op_pqsr = Mul(Fd(p), Fd(q), F(s), F(r))

        h0 = Add(
            Mul(f, op_pq), -Mul(piqi, op_pq), Rational(1, 4) * pqrs * op_pqsr
        )
        assert isinstance(h0, Expr)
        logger.debug(f"H0 = {latex(h0)}")
        # construct the rules for forbidden blocks in H0
        # we are not in a real orbital basis!! -> More canonical blocks
        rules = Rules(forbidden_tensor_blocks={
            tensor_names.fock: ('ov', 'vo'),
            tensor_names.eri: ('ooov', 'oovv', 'ovvv', 'ovoo', 'vvoo', 'vvov')
        })
        return h0, rules

    @staticmethod
    def re_h1() -> tuple[Expr, Rules]:
        """Constructs the first order RE-Hamiltonian."""
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices('pqrs')[("general", "")]
        # get an occ index for 1 particle part of H0
        occ = idx_cls.get_generic_indices(occ=1)[("occ", "")][0]

        f = AntiSymmetricTensor(tensor_names.fock, (p,), (q,))
        piqi = AntiSymmetricTensor(tensor_names.eri, (p, occ), (q, occ))
        pqrs = AntiSymmetricTensor(tensor_names.eri, (p, q), (r, s))
        op_pq = Mul(Fd(p), F(q))
        op_pqsr = Mul(Fd(p), Fd(q), F(s), F(r))

        h1 = Add(
            Mul(f, op_pq), -Mul(piqi, op_pq), Rational(1, 4) * pqrs * op_pqsr
        )
        assert isinstance(h1, Expr)
        logger.debug(f"H1 = {latex(h1)}")
        # construct the rules for forbidden blocks in H1
        rules = Rules(forbidden_tensor_blocks={
            tensor_names.fock: ['oo', 'vv'],
            tensor_names.eri: ['oooo', 'ovov', 'vvvv']
        })
        return h1, rules

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Operators):
            return self._variant == other._variant
        return False

   # @staticmethod
   # def remp_h0() -> tuple[Expr, None]:
   #     """Constructs the zeroth-order REMP Hamiltonian."""
   #     from .operators import Operators
   #     remp_A = Symbol("A")
   #     rules = Rules(forbidden_tensor_blocks={
   #         tensor_names.fock: ('ov', 'vo'),
   #         tensor_names.eri: ('ooov', 'oovv', 'ovvv', 'ovoo', 'vvoo', 'vvov')
   #     })
   #     return (remp_A*Operators.mp_h0()[0] + (1-remp_A)*Operators.re_h0()[0],
   #             rules)

    @staticmethod
    def remp_h0() -> tuple[Expr, None]:
        """Constructs the zeroth-order REMP Hamiltonian."""
        idx_cls = Indices()
        i, j, k, l = idx_cls.get_generic_indices(occ=4)[("occ", "")][:4]
        a, b, c, d = idx_cls.get_generic_indices(virt=4)[("virt", "")][:4]
        remp_A = Symbol("A")

        f = tensor_names.fock
        V = tensor_names.eri

        E0 = Add(
                AntiSymmetricTensor(f, (i,), (i,)),
                Mul(Add(-1, remp_A), AntiSymmetricTensor(V, (i, j), (i,j)),
                    Rational(1,2))
                )
        h0 = Add(
                E0,
                Mul(remp_A, AntiSymmetricTensor(f, (i,), (a,)), Fd(i), F(a)),
                Mul(remp_A, AntiSymmetricTensor(f, (a,), (i,)), Fd(a), F(i)),
                Mul(AntiSymmetricTensor(f, (a,), (b,)), Fd(a), F(b)),
                Mul(-1, AntiSymmetricTensor(f, (i,), (j,)), F(j), Fd(i)),
                Mul(Add(1, -remp_A), AntiSymmetricTensor(V, (i,j,), (k,l,)),
                      F(k), F(l), Fd(j), Fd(i), Rational(1,4)),
                Mul(Add(1, -remp_A), AntiSymmetricTensor(V, (a,b,), (c,d,)),
                      Fd(a), Fd(b), F(d), F(c), Rational(1,4)),
                Mul(Add(1, -remp_A), AntiSymmetricTensor(V, (i,a,), (j,b,)),
                      Fd(a), F(j), F(b), Fd(i))
                )
        assert isinstance(h0, Expr)
        logger.debug(f"H0 = {latex(h0)}")
        return h0, None

    @staticmethod
    def remp_h1() -> tuple[Expr, None]:
        """Constructs the first-order REMP Hamiltonian."""
        idx_cls = Indices()
        i, j, k, l = idx_cls.get_generic_indices(occ=4)[("occ", "")][:4]
        a, b, c, d = idx_cls.get_generic_indices(virt=4)[("virt", "")][:4]
        remp_A = Symbol("A")

        f = tensor_names.fock
        V = tensor_names.eri

        E1 = - Mul(remp_A, AntiSymmetricTensor(V, (i, j), (i,j)))
        h1 = Add(
                E1,
                Mul(Add(1, -remp_A),
                      AntiSymmetricTensor(f, (i,), (a,)), Fd(i), F(a)), # F_{ov}
                Mul(Add(1, -remp_A),
                      AntiSymmetricTensor(f, (a,), (i,)), Fd(a), F(i)), # F_{vo}
                Mul(AntiSymmetricTensor(V, (i,j,), (a,b,)),
                      Fd(i), Fd(j), F(b), F(a), Rational(1,4)), # -2
                Mul(AntiSymmetricTensor(V, (i,j,), (k,a,)),
                      F(k), F(a), Fd(j), Fd(i), Rational(1,2)), # -1
                Mul(AntiSymmetricTensor(V, (i,a,), (b,c,)),
                      Fd(a), Fd(i), F(b), F(c), Rational(1,2)), # -1
                Mul(remp_A, AntiSymmetricTensor(V, (i,j,), (k,l,)),
                      F(k), F(l), Fd(j), Fd(i), Rational(1,4)), #  0
                Mul(remp_A, AntiSymmetricTensor(V, (a,b,), (c,d,)),
                      Fd(a), Fd(b), F(d), F(c), Rational(1,4)), #  0
                Mul(remp_A, AntiSymmetricTensor(V, (i,a,), (j,b,)),
                      Fd(a), F(j), F(b), Fd(i)),                #  0
                Mul(AntiSymmetricTensor(V, (k,a,), (i,j,)),
                      F(i), F(j), Fd(a), Fd(k), Rational(1,2)), # +1
                Mul(AntiSymmetricTensor(V, (b,c,), (i,a,)),
                      Fd(c), Fd(b), F(i), F(a), Rational(1,2)), # +1
                Mul(AntiSymmetricTensor(V, (a,b,), (i,j,)),
                      Fd(a), Fd(b), F(j), F(i), Rational(1,4))  # +2
                )

        assert isinstance(h1, Expr)
        logger.debug(f"H1 = {latex(h1)}")
        return h1, None

  #  @staticmethod
  #  def remp_h1() -> tuple[Expr, None]:
  #      """Constructs the first-order REMP Hamiltonian."""
  #      from .operators import Operators
  #      remp_A = Symbol("A")
  #      rules = Rules(forbidden_tensor_blocks={
  #          tensor_names.fock: ['oo', 'vv']
  #      })
  #      return (remp_A*Operators.mp_h1()[0] + (1-remp_A)*Operators.re_h1()[0],
  #              rules)

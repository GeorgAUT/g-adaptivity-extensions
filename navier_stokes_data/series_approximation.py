from abc import ABC, abstractmethod
import json
import numpy as np
from firedrake import *

__all__ = ["FourierExpansion2d", "GaussianExpansion2d", "load_expansion"]


class Expansion2d(ABC):

    def __init__(self, nxmodes, nymodes, xm=0.0, xp=1.0, ym=0.0, yp=1.0):
        """Initialise object

        The grid of points is given by (x_i,x_j) where

            x_i = xm + i/(2*nxmodes)*(xp-xm) for i = 0,1,...,nxmodes
            y_j = ym + (j+1/2)/(2*nymodes)*(yp-ym) for j = 0,1,...,nymodes-1

        :arg nxmodes: number of Fourier modes in x-direction is 2*nxfourier+1
        :arg nymodes: number of Fourier modes in y-direction is 2*nyfourier
        :arg xm: lower bound of domain in x-direction
        :arg xp: upper bound of domain in x-direction
        :arg ym: lower bound of domain in y-direction
        :arg yp: upper bound of domain in y-direction
        """
        self.nxmodes = nxmodes
        self.nymodes = nymodes
        self.xm = xm
        self.xp = xp
        self.ym = ym
        self.yp = yp
        # construct matrix
        points = self._interpolation_points()
        n = (2 * self.nxmodes + 1) * 2 * self.nymodes
        self.mass_matrix = np.empty((n, n))
        for ell in range(2 * self.nymodes):
            for k in range(2 * self.nxmodes + 1):
                fourier_idx = ell * (2 * self.nxmodes + 1) + k
                for spatial_idx in range(n):
                    xn, yn = points[spatial_idx, :]
                    self.mass_matrix[spatial_idx, fourier_idx] = self._evaluate_basis(
                        k, ell, xn, yn
                    )
        self._coefficients = None

    def _interpolation_points(self):
        """Construct regular grid of interpolation points"""
        x_grid = self.xm + np.arange(2 * self.nxmodes + 1) * (self.xp - self.xm) / (
            2 * self.nxmodes
        )
        y_grid = self.ym + (np.arange(2 * self.nymodes) + 0.5) * (self.yp - self.ym) / (
            2 * self.nymodes
        )
        return np.asarray(np.meshgrid(x_grid, y_grid, indexing="ij")).T.reshape([-1, 2])

    def approximate(self, u):
        """Construct series expansion of a function

        :arg u: function to expans
        """
        fs = u.function_space()
        n_components = fs.value_size
        if len(fs) > 1:
            raise ValueError("mixed spaces not supported")
        n = (2 * self.nxmodes + 1) * 2 * self.nymodes
        # Construct grid of interpolation points
        points = self._interpolation_points()
        # evaluate function at points
        u_at_points = np.zeros((n, n_components))
        for j, p in enumerate(points):
            try:
                u_at_points[j, :] = u.at(p)
            except:
                pass
        self._coefficients = np.linalg.solve(self.mass_matrix, u_at_points).T

    @abstractmethod
    def as_expression(self, x, y):
        """Return series expansion of approximation as an expression

        :arg x: spatial coordinate x
        :arg y: spatial coordinate y
        """

    def save(self, filename):
        """Save to file

        :arg filename: name of file to write to
        """
        if self._coefficients is None:
            raise RuntimeError(
                "Call expand(u) first to construct expansion that can be saved"
            )

        metadata = dict(
            nxmodes=self.nxmodes,
            nymodes=self.nymodes,
            xm=self.xm,
            xp=self.xp,
            ym=self.ym,
            yp=self.yp,
        )
        data = dict(type=type(self).__name__, metadata=metadata, coefficients=[])
        for component in range(self._coefficients.shape[0]):
            data["coefficients"].append(list(self._coefficients[component, :]))
        with open(filename, "w", encoding="utf8") as f:
            json.dump(data, f, indent=4)


class FourierExpansion2d(Expansion2d):
    """Class for constructing an expression for the Fourier-expansin of a function

    It is assumed that the function has zero boundary conditions at y=ym, y=yp
    """

    def __init__(self, nxmodes, nymodes, xm=0.0, xp=1.0, ym=0.0, yp=1.0):
        """Initialise object

        :arg nxmodes: number of Fourier modes in x-direction is 2*nxfourier+1
        :arg nymodes: number of Fourier modes in y-direction is 2*nyfourier
        :arg xm: lower bound of domain in x-direction
        :arg xp: upper bound of domain in x-direction
        :arg ym: lower bound of domain in y-direction
        :arg yp: upper bound of domain in y-direction
        """
        super().__init__(nxmodes, nymodes, xm, xp, ym, yp)

    def _evaluate_basis(self, k, ell, x, y):
        """Evaluate basis function with Fourier index (k,ell)

        :arg k: Fourier index in x-direction
        :arg ell: Fourier index in y-direction
        :arg x: x-coordinate of point to evaluate
        :arg y: y-coordinate of point to evaluate
        """
        if k <= self.nxmodes:
            phix = np.cos((x - self.xm) / (self.xp - self.xm) * np.pi * k)
        else:
            phix = np.sin(
                (x - self.xm) / (self.xp - self.xm) * np.pi * (k - self.nxmodes)
            )
        phiy = np.sin((y - self.ym) / (self.yp - self.ym) * np.pi * (ell + 1))
        return phix * phiy

    def as_expression(self, x, y):
        """Return series expansion of approximation as an expression

        :arg x: spatial coordinate x
        :arg y: spatial coordinate y
        """

        if self._coefficients is None:
            raise RuntimeError("Call expand(u) first to construct expansion")

        n_components = self._coefficients.shape[0]
        expressions = []
        for component in range(n_components):
            expression = 0
            for ell in range(2 * self.nymodes):
                phiy = sin((y - self.ym) / (self.yp - self.ym) * np.pi * (ell + 1))
                for k in range(2 * self.nxmodes + 1):
                    fourier_idx = ell * (2 * self.nxmodes + 1) + k
                    if k <= self.nxmodes:
                        phix = cos((x - self.xm) / (self.xp - self.xm) * np.pi * k)
                    else:
                        phix = sin(
                            (x - self.xm)
                            / (self.xp - self.xm)
                            * np.pi
                            * (k - self.nxmodes)
                        )
                    expression += (
                        self._coefficients[component, fourier_idx] * phiy * phix
                    )
            expressions.append(expression)
        if n_components > 1:
            return as_vector(expressions)
        else:
            return expressions[0]


class GaussianExpansion2d(Expansion2d):
    """Class for constructing an expression for the Gaussian-expansin of a function"""

    def __init__(self, nxmodes, nymodes, xm=0.0, xp=1.0, ym=0.0, yp=1.0):
        """Initialise object

        :arg nxmodes: number of modes in x-direction is 2*nxmodes+1
        :arg nymodes: number of modes in y-direction is 2*nymodes
        :arg xm: lower bound of domain in x-direction
        :arg xp: upper bound of domain in x-direction
        :arg ym: lower bound of domain in y-direction
        :arg yp: upper bound of domain in y-direction
        """
        super().__init__(nxmodes, nymodes, xm, xp, ym, yp)

    def _evaluate_basis(self, k, ell, x, y):
        """Evaluate basis function with index (k,ell)

        :arg k: index in x-direction
        :arg ell: index in y-direction
        :arg x: x-coordinate of point to evaluate
        :arg y: y-coordinate of point to evaluate
        """
        xcentre = self.xm + k / (2 * self.nxmodes) * (self.xp - self.xm)
        ycentre = self.ym + (ell + 0.5) / (2 * self.nymodes) * (self.yp - self.ym)
        sigmax = (self.xp - self.xm) / (2 * self.nxmodes)
        sigmay = (self.yp - self.ym) / (2 * self.nymodes)
        phi = np.exp(
            -0.5 * (((x - xcentre) / (sigmax)) ** 2 + ((y - ycentre) / (sigmay)) ** 2)
        )
        return phi

    def as_expression(self, x, y):
        """Return series expansion of approximation as an expression

        :arg x: spatial coordinate x
        :arg y: spatial coordinate y
        """

        if self._coefficients is None:
            raise RuntimeError("Call expand(u) first to construct expansion")

        sigmax = (self.xp - self.xm) / (2 * self.nxmodes)
        sigmay = (self.yp - self.ym) / (2 * self.nymodes)

        n_components = self._coefficients.shape[0]
        expressions = []
        for component in range(n_components):
            expression = 0
            for ell in range(2 * self.nymodes):
                sub_expression = 0
                for k in range(2 * self.nxmodes + 1):
                    fourier_idx = ell * (2 * self.nxmodes + 1) + k
                    xcentre = self.xm + k / (2 * self.nxmodes) * (self.xp - self.xm)
                    ycentre = self.ym + (ell + 0.5) / (2 * self.nymodes) * (
                        self.yp - self.ym
                    )

                    phi = exp(
                        -0.5
                        * (
                            ((x - xcentre) / (sigmax)) ** 2
                            + ((y - ycentre) / (sigmay)) ** 2
                        )
                    )
                    sub_expression += self._coefficients[component, fourier_idx] * phi
                expression += sub_expression
            expressions.append(expression)
        if n_components > 1:
            return as_vector(expressions)
        else:
            return expressions[0]


def load_expansion(filename):
    """Load an expansion from a file"""
    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)
        if data["type"] == "FourierExpansion2d":
            ExpansionType = FourierExpansion2d
        elif data["type"] == "GaussianExpansion2d":
            ExpansionType = GaussianExpansion2d
        else:
            raise RuntimeError("Unknown expansion type: " + data["type"])
        metadata = data["metadata"]
        expansion = ExpansionType(
            metadata["nxmodes"],
            metadata["nymodes"],
            metadata["xm"],
            metadata["xp"],
            metadata["ym"],
            metadata["yp"],
        )
        expansion._coefficients = np.array(data["coefficients"])
        return expansion

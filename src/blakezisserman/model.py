
from .core import solve1d, solve2d

from typing import Literal, Annotated, Optional
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator, model_validator

import numpy as np



# Set default parameter choices
defaults1D = { 
  "epsilon":       1e-2,
  "fun_rtol":      1e-3,
  "bound_cond":    'none',
  "maxiter":       100,
  "pcg_maxiter_u": 100000,
  "pcg_maxiter_z": 10000,
  "log_level":     'WARNING',
  }

defaults2D = { 
  "epsilon":       1e-2,
  "fun_rtol":      1e-3,
  "maxiter":       100,
  "pcg_maxiter_s": 10000,
  "pcg_maxiter_z": 10000,
  "pcg_maxiter_u": 100000,
  "log_level":     'WARNING',
  }



# Define constrained types using Annotated
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat    = Annotated[float, Field(gt=0.0)]
PositiveInt      = Annotated[int, Field(ge=1)]
NumpyArray       = Annotated[np.ndarray, PlainSerializer(lambda x: x.tolist(), return_type=list)]


# Internal data-class to handle parameter checking for the solve method
class SolveParameters1D(BaseModel):

    llambda: NonNegativeFloat = Field(..., description="Functional parameter weighting the fidelity term.")
    eta: NonNegativeFloat = Field(..., description="Functional parameter weighting the size of the gradient discontinuity set.")
    epsilon: PositiveFloat = Field(defaults1D['epsilon'], description="Discretization measure theoretic parameter.")
    bound_cond: Literal['none', 'null-neumann', 'periodic'] = Field(defaults1D['bound_cond'], description="Boundary conditions type for the differential operators.")
    fun_rtol: PositiveFloat = Field(defaults1D['fun_rtol'], description="Convergence tolerance for relative variation of the objective functional.")
    maxiter: PositiveInt = Field(defaults1D['maxiter'], description="Maximum number of external iterations.")
    pcg_maxiter_z: PositiveInt = Field(defaults1D['pcg_maxiter_z'], description="Max internal iterations for the PCG solver in z.")
    pcg_maxiter_u: PositiveInt = Field(defaults1D['pcg_maxiter_u'], description="Max internal iterations for the PCG solver in u.")
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(defaults1D['log_level'], description="Logging level verbosity.")



class BlakeZisserman1D(BaseModel):
    # Enable Pydantic to accept non-standard Python types (numpy)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State attributes
    values: NumpyArray
    breaks: Optional[NumpyArray] = None

    # Pydantic validator
    @field_validator('values')
    @classmethod
    def check_values(cls, arr: NumpyArray) -> NumpyArray:
        if not arr.ndim <= 2:
            raise ValueError(f"Input array must be at most 2-dimensional, but got {arr.ndim} dimensions.")
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    # Pydantic validator
    @field_validator('breaks')
    @classmethod
    def check_breaks(cls, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return arr
        if not arr.ndim == 1:
            raise ValueError(f"Input array must be 1-dimensional, but got {arr.ndim} dimensions.")
        return arr

    @model_validator(mode='after')
    def check_dimensions_match(self) -> 'ArrayComparisonModel':
        breaks = self.breaks
        values = self.values
        if breaks is None:
            return self
        if breaks.shape[0] != values.shape[0]:
            raise ValueError(f"Input arrays must have the same size along axis=0, but got size {breaks.shape} and {values.shape}.")
        return self



    def solve(
        self, 
        llambda: float, 
        eta: float, 
        epsilon: float = defaults1D['epsilon'], 
        bound_cond: str = defaults1D['bound_cond'],
        fun_rtol: float = defaults1D['fun_rtol'], 
        maxiter: int = defaults1D['maxiter'], 
        pcg_maxiter_z: int = defaults1D['pcg_maxiter_z'], 
        pcg_maxiter_u: int = defaults1D['pcg_maxiter_u'],
        log_level: str = defaults1D['log_level']
    ):
        
        
        # Automatically validate all arguments using Pydantic parameters before processing
        params = SolveParameters1D(
            llambda=llambda, eta=eta, epsilon=epsilon, bound_cond=bound_cond, fun_rtol=fun_rtol, maxiter=maxiter, pcg_maxiter_z=pcg_maxiter_z, pcg_maxiter_u=pcg_maxiter_u, log_level=log_level
        )

        # Launch solver
        out = solve1d(    
              self.values,
              self.breaks,
              params.llambda,
              params.eta, 
              params.epsilon, 
              params.bound_cond,
              params.fun_rtol, 
              params.maxiter, 
              params.pcg_maxiter_z, 
              params.pcg_maxiter_u,
              params.log_level
            )

        return out



# Internal data-class to handle parameter checking for the solve method
class SolveParameters2D(BaseModel):

    mu: NonNegativeFloat = Field(..., description="Functional parameter weighting the fidelity term.")
    alpha: NonNegativeFloat = Field(..., description="Functional parameter weighting the size of the discontinuity set.")
    beta: NonNegativeFloat = Field(..., description="Functional parameter weighting the size of the gradient discontinuity set.")
    delta: NonNegativeFloat = Field(..., description="Functional parameter weighting the Hessian component.")
    epsilon: PositiveFloat = Field(defaults2D['epsilon'], description="Discretization measure theoretic parameter.")
    fun_rtol: PositiveFloat = Field(defaults2D['fun_rtol'], description="Convergence tolerance for relative variation of the objective functional.")
    maxiter: PositiveInt = Field(defaults2D['maxiter'], description="Maximum number of external iterations.")
    pcg_maxiter_s: PositiveInt = Field(defaults2D['pcg_maxiter_s'], description="Max internal iterations for the PCG solver in s.")
    pcg_maxiter_z: PositiveInt = Field(defaults2D['pcg_maxiter_z'], description="Max internal iterations for the PCG solver in z.")
    pcg_maxiter_u: PositiveInt = Field(defaults2D['pcg_maxiter_u'], description="Max internal iterations for the PCG solver in u.")
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(defaults2D['log_level'], description="Logging level verbosity.")



class BlakeZisserman2D(BaseModel):
    # Enable Pydantic to accept non-standard Python types (numpy)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State attributes
    values: NumpyArray

    # Pydantic validator
    @field_validator('values')
    @classmethod
    def check_values(cls, arr: NumpyArray) -> NumpyArray:
        if not arr.ndim <= 3:
            raise ValueError(f"Input array must be at most 3-dimensional, but got {arr.ndim} dimensions.")
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr



    def solve(
        self, 
        mu: float, 
        alpha: float,
        beta: float,
        delta: float, 
        epsilon: float = defaults2D['epsilon'], 
        fun_rtol: float = defaults2D['fun_rtol'], 
        maxiter: int = defaults2D['maxiter'],
        pcg_maxiter_s: int = defaults2D['pcg_maxiter_s'], 
        pcg_maxiter_z: int = defaults2D['pcg_maxiter_z'], 
        pcg_maxiter_u: int = defaults2D['pcg_maxiter_u'],
        log_level: str = defaults2D['log_level']
    ):
        
        
        # Automatically validate all arguments using Pydantic parameters before processing
        params = SolveParameters2D(
            mu=mu, alpha=alpha, beta=beta, delta=delta, epsilon=epsilon, fun_rtol=fun_rtol, maxiter=maxiter, pcg_maxiter_s=pcg_maxiter_s, pcg_maxiter_z=pcg_maxiter_z, pcg_maxiter_u=pcg_maxiter_u, log_level=log_level
        )

        # Launch solver
        out = solve2d(    
              self.values,
              params.mu,
              params.alpha,
              params.beta,
              params.delta, 
              params.epsilon, 
              params.fun_rtol, 
              params.maxiter, 
              params.pcg_maxiter_s,
              params.pcg_maxiter_z, 
              params.pcg_maxiter_u,
              params.log_level
            )

        return out

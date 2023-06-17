import numpy as np
from typing import Callable, Union

NPValue = Union[float, np.ndarray]
NPValueToValueFn = Callable[[NPValue], NPValue]
NPValueToScalarFn = Callable[[NPValue], float]

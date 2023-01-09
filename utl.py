import numbers
import numpy as np

"""
Helpers
"""

def rvec(*args) -> np.ndarray:
    """
    Creates a row vector of type float64
    """
    return np.array(args, dtype=np.float64)

def cvec(*args) -> np.ndarray:
    """
    Creates a column vector of type float64
    """
    return rvec(args).reshape(len(args), 1)

def is_symmetric(A: np.ndarray) -> bool:
    """
    Check ob die n x n Matrix A symmetrisch ist
    True falls A == A.T    
    """
    assert_square(A)
    return np.allclose(A, A.T)


"""
Assertions
"""

def assert_eq_shape(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        raise Exception("shape mismatch")
    return True

def assert_square(A: np.ndarray):
    if len(A.shape) != 2:
        raise Exception(f'unexpected number of dimensions in array: {len(A.shape)}')

    if A.shape[0] == A.shape[1]:
        return True
    else:
        raise Exception("A is not square")

def assert_is_vec(v: np.ndarray):
    if (len(v.shape) == 2 and v.shape[1] != 1) \
        or len(v.shape) > 2:
        raise Exception("b is not a vector")
    return True

def assert_dimensions_match(A: np.ndarray, b: np.ndarray):
    """
    Check if A is square & matches b in dimensions (helpful for Gauss e.g.)
    """
    assert_square(A)

    n = A.shape[0]

    assert_is_vec(b)    
    
    if b.shape[0] != n:
        raise Exception("A & b have shape mismatch")

def assert_normed(v: np.ndarray):
    assert_is_vec(v)

    length = np.linalg.norm(v, 2)

    if not np.isclose(length, 1):
        raise Exception('vector is not normed')
    return True



import unittest

class UtlTest(unittest.TestCase):
    def test_cvec(self):
        tvec = cvec(1,2)

        self.assertEqual(tvec.shape[0], 2)
        self.assertEqual(tvec.shape[1], 1)

        self.assertFalse(isinstance(tvec[0,0], numbers.Integral))
        self.assertTrue(isinstance(tvec[0, 0], numbers.Real))

        self.assertEqual(tvec[0,0], 1.)
        self.assertEqual(tvec[1,0], 2.)

    def test_rvec(self):
        tvec = rvec(1,2)

        self.assertEqual(len(tvec.shape), 1)
        self.assertEqual(tvec.shape[0], 2)

        self.assertFalse(isinstance(tvec[0], numbers.Integral))
        self.assertTrue(isinstance(tvec[0], numbers.Real))

        self.assertEqual(tvec[0], 1.)
        self.assertEqual(tvec[1], 2.)

    def test_eq_shape(self):
        v1 = rvec(1,2)
        v2 = rvec(3,4)

        v3 = cvec(1,2)
        
        self.assertTrue(assert_eq_shape(v1, v2))

        with self.assertRaises(Exception):
            assert_eq_shape(v1, v3)


if __name__ == '__main__':
    unittest.main()
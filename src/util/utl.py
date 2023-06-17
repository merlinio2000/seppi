import numbers
import numpy as np

"""
Abbruchkriterien
"""

class AbbruchKriteriumHandler:
    ''' 
    Interface für generisches überprüfen des Abbruchkriteriums
    Returns:
        bool: true=weitermachen, false=abbruch
    '''
    def __call__(self, curr_i: int, curr_x: np.ndarray, delta: np.ndarray) -> bool:
        raise Exception('implementation required')

class AbbruchKriteriumNIterationen(AbbruchKriteriumHandler):
    ''' 
    AbbruchKriteriumHandler der nach n Iterationen abbricht
    (i = n-1 weil i bei 0 beginnt)
    '''
    def __call__(self, curr_i: int, curr_x: np.ndarray, delta: np.ndarray) -> bool: 
        return curr_i <= self.n

    def __init__(self, n: int):
        self.n = n

"""
Helpers
"""

def convert_float(num_str: str, from_base: int, to_base: int, places: int = 10) -> str:
    num_str = num_str.upper()
    # supports up to hex
    base_symbols = '0123456789ABCDEF'
    ret = ''
    bef = aft = ''

    if "." not in num_str: 
        bef = num_str
    else: 
        bef, aft = num_str.split(".")

    before_int = int(bef, from_base)
    before_str = ''
    while before_int > 0:
        before_str += base_symbols[before_int%to_base]
        before_int //= to_base

    ret += before_str[::-1]

    if "." not in num_str: 
        return ret 


    after_int = int(aft, from_base)
    after_places = int(np.log10(after_int)) + 1
    after_float = after_int / 10**after_places

    ret += '.'

    for _ in range(places):
        after_float *= to_base
        int_part = int(after_float)
        ret += base_symbols[int_part]
        after_float -= int_part

    return ret

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

def is_diagonaldominant(A: np.ndarray) -> bool:
    # spaltenweise check
    assert_square(A)
    n = A.shape[0]
    result = True
    for i in range(n):
        diag = np.abs(A[i, i])
        reihen_sum = np.sum(np.abs(A[:, i])) - diag
        result = diag > reihen_sum
        if not result:
            break
    
    if result:
        return True

    #reihenweise check
    result = False
    for i in range(n):
        diag = np.abs(A[i, i])
        reihen_sum = np.sum(np.abs(A[i, :])) - diag
        result = diag > reihen_sum
        if not result:
            break

    return result


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
        raise Exception(f'vector is not normed, {length=}')
    return True

"""
ANSI Escape sequences für farbige schrift im Terminal
"""
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



import unittest

class UtlTest(unittest.TestCase):
    def test_convert_HS2014_A1b(self):
        x = np.sqrt(3)

        binary = convert_float(str(x), 10, 2, places=6)

        self.assertEqual(binary, '1.101110')

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

    def test_is_diagonaldominant_spalte(self):
        A = np.array([[5, 4],
                    [3, 4]])
        self.assertTrue(is_diagonaldominant(A))

        A = np.array([[14, 4, 2],
                    [10, 8, 4],
                    [4, 4, 10]])
        self.assertFalse(is_diagonaldominant(A))

        A = np.array([[-8, 12.2, -1],
                    [6, -20, -3.5],
                    [0, 3, 20]])
        self.assertTrue(is_diagonaldominant(A))
    
    def test_is_diagonaldominant_zeile(self):
        A = np.array([[-8, 3],
                    [10, 11]])
        self.assertTrue(is_diagonaldominant(A))

        A = np.array([[-5, 0.5, 4],
                        [7, 7, 0],
                        [3, -8, 11]])
        self.assertFalse(is_diagonaldominant(A))

if __name__ == '__main__':
    unittest.main()

import numpy as np
from amplitudes.color_interference import color_gram_multi_quark_lines

def test_color_gram_single_line_ng1():
    # For one gluon: sum_a Tr(T^a T^a) = (Nc^2-1)/2
    for Nc in (2,3,5):
        val = color_gram_multi_quark_lines([[1]], [[1]], Nc=Nc)
        assert abs(val - (Nc*Nc-1)/2) < 1e-9

def test_color_gram_two_lines_single_generator_vanishes():
    # If a single gluon label sits alone in each of two separate traces:
    #   sum_a Tr(T^a) Tr(T^a) = 0 because Tr(T^a)=0.
    for Nc in (3,4,7):
        v = color_gram_multi_quark_lines([[1],[]], [[],[1]], Nc=Nc)
        assert abs(v) < 1e-12

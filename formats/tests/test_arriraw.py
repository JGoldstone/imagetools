import unittest
import numpy as np

from extract_photosite_region import chunk_bounds_and_photosite_buf_start, fill_photosites_from_chunks
from extract_photosite_region import fill_result_from_photosites, inverse_qlut  # , offset_in_photosite_buf

TWO_CHUNK_STRIDE = 16


class MyTestCase(unittest.TestCase):
    def test_chunk_math_at_origin_for_UHD1_lj_0ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 0, 0)
        self.assertEqual(0, s)
        self.assertEqual(0, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_lj_1ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 1, 0)
        self.assertEqual(0, s)
        self.assertEqual(1, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_lj_2ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 2, 0)
        self.assertEqual(0, s)
        self.assertEqual(1, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_lj_3ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 3, 0)
        self.assertEqual(0, s)
        self.assertEqual(1, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_lj_4ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 4, 0)
        self.assertEqual(0, s)
        self.assertEqual(1, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_lj_7ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 7, 0)
        self.assertEqual(0, s)
        self.assertEqual(1, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_lj_8ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 8, 0)
        self.assertEqual(0, s)
        self.assertEqual(1, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_lj_9ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 9, 0)
        self.assertEqual(0, s)
        self.assertEqual(2, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_second_line_lj_1ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 0, 1, 1)
        self.assertEqual(2, s)
        self.assertEqual(3, e)
        self.assertEqual(0, o)

    def test_chunk_math_at_origin_for_UHD1_second_line_offset_3ps(self):
        s, e, o = chunk_bounds_and_photosite_buf_start(TWO_CHUNK_STRIDE, 1, 4, 1)
        self.assertEqual(2, s)
        self.assertEqual(3, e)
        self.assertEqual(1, o)

    # def test_offset_in_photosite_buf(self):
    #     min_x = 7
    #     y = 2
    #     stride = 20
    #     self.assertEqual(7, offset_in_photosite_buf(min_x, y, stride))

    def apply_inv_qlut_algo(self, x):
        q = x // 512
        o = x % 512
        return ((1024 + 2 * o + 1) << (q - 2)) - 1

    def test_inverse_qlut(self):
        buf = np.arange(8).reshape(2, 4)
        buf[0][1] = 2000
        buf[1][3] = 3000
        buf_ref = np.arange(8).reshape(2, 4)
        buf_ref[0][1] = self.apply_inv_qlut_algo(buf[0][1])
        buf_ref[1][3] = self.apply_inv_qlut_algo(buf[1][3])
        # q0 = buf[0][1] // 512
        # o0 = buf[0][1] % 512
        # buf_ref[0][1] = (((1024 + 2 * o0 + 1) << (q0 - 2)) - 1) / 1
        # q1 = buf[1][3] // 512
        # o1 = buf[1][3] % 512
        # buf_ref[1][3] = (((1024 + 2 * o1 + 1) << (q1 - 2)) - 1) / 1
        buf = inverse_qlut(buf)
        # comparison = buf_ref == buf
        self.assertTrue(buf[0][1] == buf_ref[0][1])
        self.assertTrue(buf[1][3] == buf_ref[1][3])

    def test_filling_first_four_photosites_from_chunk(self):
        chunk_buf = [0x654321cb, 0xa9870000, 0x00000000]
        pb_buf = [0x0000] * 8
        pb_buf_ref = [0x0321, 0x0654, 0x0987, 0x0cba, 0x0000, 0x0000, 0x0000, 0x0000]
        fill_photosites_from_chunks(chunk_buf, 1, pb_buf)
        for ref, test in zip(pb_buf_ref, pb_buf):
            self.assertEqual(ref, test)

    def test_filling_second_four_photosites_from_chunk(self):
        chunk_buf = [0x00000000, 0x00006543, 0x21cba987]
        pb_buf = [0x0000] * 8
        pb_buf_ref = [0x0000, 0x0000, 0x0000, 0x0000, 0x0321, 0x0654, 0x0987, 0x0cba]
        fill_photosites_from_chunks(chunk_buf, 1, pb_buf)
        for ref, test in zip(pb_buf_ref, pb_buf):
            self.assertEqual(ref, test)

    def test_filling_result_from_photosites(self):
        photosite_buf = [0x0321, 0x0654, 0x0987, 0x0cba, 0x0000, 0x0000, 0x0000, 0x0000]
        photosite_buf_offset = 1
        min_x = 2
        max_x = 5
        result_buf_offset = 3
        result_y = 2
        result_buf = np.zeros((4, 8))
        result_buf_ref = np.zeros((4, 8))
        result_buf_ref[result_y][3] = 0x0654
        result_buf_ref[result_y][4] = 0x0987
        result_buf_ref[result_y][5] = 0x0cba
        fill_result_from_photosites(min_x, max_x, photosite_buf, photosite_buf_offset,
                                    result_buf, result_buf_offset, result_y)
        comparison = result_buf_ref == result_buf
        self.assertTrue(comparison.all())


if __name__ == '__main__':
    unittest.main()

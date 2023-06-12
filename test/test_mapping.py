import unittest

from src.semantic_mapping_node import *


class FindClosestDataTestCase(unittest.TestCase):
    # Test cases
    # 1. If the queue is empty.
    # 2. If the time stamp is never greater than target_stamp.
    # 3. If the time stamp is all greater than target_stamp.
    # 4. If the closest time stamp is the one that smaller than the target_stamp.
    # 5. If the closest time stamp is the one that larger than the target_stamp.
    class _RosHeader:
        """ Mimic the header of ros message """

        def __init__(self, stamp):
            self.stamp = stamp

    def test_case_1(self):
        stamp_list = []
        target_stamp = 7

        output = self._test_engine(stamp_list, target_stamp)
        self.assertIsNone(output["closest_stamp"])

    def test_case_2(self):
        stamp_list = [1, 3, 4, 6, 9, 10, 12, 15, 19, 23, 30]
        target_stamp = 31

        output = self._test_engine(stamp_list, target_stamp)
        self.assertEqual(30, output["closest_stamp"])
        self.assertTrue(len(output["queue"]) == 0)

    def test_case_3(self):
        stamp_list = [1, 3, 4, 6, 9, 10, 12, 15, 19, 23, 30]
        target_stamp = 0.5

        output = self._test_engine(stamp_list, target_stamp)
        self.assertEqual(1, output["closest_stamp"])

        # Check if the queue contains the correct elements
        queue = output["queue"]
        start_idx = 1
        self.assertTrue(len(queue) == (len(stamp_list) - start_idx))

        for i in range(start_idx, len(stamp_list)):
            curr_stamp = queue.popleft()[0].stamp
            self.assertEqual(stamp_list[i], curr_stamp)

    def test_case_4(self):
        stamp_list = [1, 3, 4, 6, 9, 10, 12, 15, 19, 23, 30]
        target_stamp = 7

        output = self._test_engine(stamp_list, target_stamp)
        self.assertEqual(6, output["closest_stamp"])

        # Check if the queue contains the correct elements
        queue = output["queue"]
        start_idx = 4
        self.assertTrue(len(queue) == (len(stamp_list) - start_idx))

        for i in range(start_idx, len(stamp_list)):
            curr_stamp = queue.popleft()[0].stamp
            self.assertEqual(stamp_list[i], curr_stamp)

    def test_case_5(self):
        stamp_list = [1, 3, 4, 6, 9, 10, 12, 15, 19, 23, 30]
        target_stamp = 11.9

        output = self._test_engine(stamp_list, target_stamp)
        self.assertEqual(12, output["closest_stamp"])

        # Check if the queue contains the correct elements
        queue = output["queue"]
        start_idx = 7  # The start idx of the remain elements in queue
        self.assertTrue(len(queue) == (len(stamp_list) - start_idx))

        for i in range(start_idx, len(stamp_list)):
            curr_stamp = queue.popleft()[0].stamp
            self.assertEqual(stamp_list[i], curr_stamp)

    def _test_engine(self, stamp_list, target_stamp):
        queue = deque()

        dummy_data = ""
        for l in stamp_list:
            queue.append((self._RosHeader(l), dummy_data))

        self_place_holder = 0
        closest_data = SemanticMapping._find_closest_data(self_place_holder, queue, target_stamp)

        closest_stamp = None if closest_data is None else closest_data[0].stamp

        output_dict = {
            "closest_stamp": closest_stamp,
            "queue": queue,
        }
        return output_dict


if __name__ == '__main__':
    unittest.main()

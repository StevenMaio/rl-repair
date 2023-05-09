from unittest import TestCase

from src.utils.data_struct.CircularList import CircularList


class TestCircularList(TestCase):
    def test_add(self):
        test_list = CircularList(3)
        test_list.add(0)
        test_list.add(1)
        test_list.add(2)
        self.assertIn(0, test_list)
        self.assertIn(1, test_list)
        self.assertIn(2, test_list)

        test_list.add(3)
        self.assertIn(3, test_list)
        self.assertNotIn(0, test_list)

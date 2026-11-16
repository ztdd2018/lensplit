import unittest

from lensplit.core.model import LogicalGroup
from lensplit.core.numbering import (
    MergeMode,
    NumberingStyle,
    format_groups,
    format_ordinal,
)


class NumberingStyleTests(unittest.TestCase):
    def test_decimal_padding_and_affixes(self):
        self.assertEqual(
            format_ordinal(1, NumberingStyle(width=3, suffix="#")), "001#"
        )
        self.assertEqual(format_ordinal(10, NumberingStyle(width=4)), "0010")
        self.assertEqual(
            format_ordinal(2, NumberingStyle(width=3, prefix="S", suffix="#")),
            "S002#",
        )

    def test_alphabetic_sequences_and_padding(self):
        self.assertEqual(
            format_ordinal(1, NumberingStyle(width=3, sequence="lower")), "00a"
        )
        lower = NumberingStyle(width=0, sequence="lower")
        upper = NumberingStyle(width=0, sequence="upper")
        expected = {1: "a", 2: "b", 26: "z", 27: "aa", 28: "ab", 52: "az", 53: "ba"}
        for ordinal, label in expected.items():
            with self.subTest(ordinal=ordinal):
                self.assertEqual(format_ordinal(ordinal, lower), label)
                self.assertEqual(format_ordinal(ordinal, upper), label.upper())

    def test_start_offset_applies_before_formatting(self):
        style = NumberingStyle(start=5, width=3, sequence="lower")
        self.assertEqual(format_ordinal(1, style), "00e")
        self.assertEqual(format_ordinal(23, style), "0aa")


class GroupNumberingTests(unittest.TestCase):
    def setUp(self):
        self.groups = [
            LogicalGroup("g1", ["s1", "s2"]),
            LogicalGroup("g2", ["s3"]),
            LogicalGroup("g3", ["s4"]),
        ]
        self.style = NumberingStyle(width=3, prefix="S", suffix="#")

    def test_combined_wraps_affixes_only_around_the_outer_label(self):
        self.assertEqual(
            format_groups(self.groups, self.style, MergeMode.COMBINED),
            ["S001+002#", "S003#", "S004#"],
        )

    def test_sequential_renumbers_by_logical_group(self):
        self.assertEqual(
            format_groups(self.groups, self.style, "sequential"),
            ["S001#", "S002#", "S003#"],
        )

    def test_first_preserves_original_ordinals_after_merged_groups(self):
        self.assertEqual(
            format_groups(self.groups, self.style, MergeMode.FIRST),
            ["S001#", "S003#", "S004#"],
        )


if __name__ == "__main__":
    unittest.main()

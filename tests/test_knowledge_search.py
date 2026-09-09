import unittest

from tools.knowledge_tool import search_knowledge


class KnowledgeSearchTests(unittest.TestCase):
    def test_chinese_query_without_spaces_finds_black_screen_incident(self):
        results = search_knowledge("安卓2.3.1多人副本黑屏是不是已知问题")

        serialized = str(results)
        self.assertIn("INC-101", serialized)

    def test_hud_incident_is_retrievable(self):
        results = search_knowledge("2.2.8进入战斗后HUD消失", top_k=3)

        serialized = str(results)
        self.assertIn("INC-102", serialized)

    def test_invalid_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "非空字符串"):
            search_knowledge("")


if __name__ == "__main__":
    unittest.main()

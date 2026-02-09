import unittest
import os
from fsm_core.fsm_generator import NR_RRC_FSM_Extractor

class TestFSMGenerator(unittest.TestCase):
    def setUp(self):
        # Create a dummy markdown file for testing
        self.test_md = "test_spec.md"
        with open(self.test_md, "w") as f:
            f.write("# 5.3.3.1 General\nUE in RRC_IDLE enters RRC_CONNECTED.\n")
            f.write("# 5.3.8.3 Release\nUE in RRC_CONNECTED enters RRC_IDLE.\n")

    def tearDown(self):
        if os.path.exists(self.test_md):
            os.remove(self.test_md)

    def test_extraction(self):
        extractor = NR_RRC_FSM_Extractor(self.test_md)
        extractor.segment_by_header()
        extractor.extract_logic()
        
        self.assertTrue(len(extractor.transitions) >= 2)
        states = [t['to'] for t in extractor.transitions]
        self.assertIn("RRC_CONNECTED", states)
        self.assertIn("RRC_IDLE", states)

if __name__ == "__main__":
    unittest.main()

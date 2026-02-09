import unittest
import os
import sys

# Add project root to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fsm_core.fsm_generator import AutoFSMExtractor

class TestFSMGenerator(unittest.TestCase):
    def setUp(self):
        # Create a dummy markdown file for testing using absolute path based on this file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_md = os.path.join(self.base_dir, "test_spec.md")
        with open(self.test_md, "w", encoding='utf-8') as f:
            f.write("### 4.2.1 UE states and state transitions\n")
            f.write("- RRC_IDLE\n")
            f.write("- RRC_CONNECTED\n")
            f.write("- RRC_INACTIVE\n")
            f.write("### 5.3.3 RRC connection establishment\n")
            f.write("The UE in RRC_IDLE transitions to RRC_CONNECTED.\n")

    def tearDown(self):
        if os.path.exists(self.test_md):
            os.remove(self.test_md)

    def test_extraction(self):
        extractor = AutoFSMExtractor(self.test_md)
        self.assertTrue(extractor.load_document())
        extractor.discover_states()
        extractor.extract_transitions()
        
        self.assertIn("RRC_IDLE", extractor.states)
        self.assertIn("RRC_CONNECTED", extractor.states)
        self.assertTrue(any(t['from'] == "RRC_IDLE" and t['to'] == "RRC_CONNECTED" for t in extractor.transitions))

if __name__ == "__main__":
    unittest.main()

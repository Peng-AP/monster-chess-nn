import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "books" /
    "gate_mixed_v20_v21_v21b_gen7_gen9_p8_20260816.partitions.json")


class BookPartitionManifestContracts(unittest.TestCase):
    def test_pinned_book_and_blocks_remain_valid(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        book_path = ROOT / manifest["book"]
        digest = hashlib.sha256(book_path.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["book_sha256"])
        book = json.loads(book_path.read_text(encoding="utf-8"))
        entries = book["entries"]
        self.assertEqual(len(entries), manifest["entries"])
        identities = {
            (row["fen"], bool(row["half"]), int(row["turn_count"]))
            for row in entries
        }
        self.assertEqual(len(identities), len(entries))
        sources = {row["source_model"] for row in entries}
        self.assertEqual(len(sources), 5)
        self.assertEqual(
            {sum(row["source_model"] == source for row in entries)
             for source in sources}, {240})

        used = []
        for block in manifest["blocks"]:
            self.assertLess(block["start"], block["end"])
            self.assertLessEqual(block["end"], len(entries))
            used.extend(range(block["start"], block["end"]))
        self.assertEqual(len(used), len(set(used)))


if __name__ == "__main__":
    unittest.main()

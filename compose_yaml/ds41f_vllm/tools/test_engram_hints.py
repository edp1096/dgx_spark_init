import unittest
import test_engram_reader as original
from engram_reader import AdvisedReader
class AdvisedTests(original.ReaderTests):
    reader_class=AdvisedReader
if __name__=="__main__":unittest.main(defaultTest="AdvisedTests")

from unittest import TestCase

from export.portable.stark_s import build_starks as portable_stark
from lib.models.stark import build_starks as stark
from lib.test.parameter.stark_s import parameters


class ExportStarkSTest(TestCase):
    def setUp(self) -> None:
        self.params = parameters('baseline').cfg

    def test_load(self):
        portable = portable_stark(self.params)
        self.assertIsNotNone(portable)
        original = stark(self.params)
        self.assertIsNotNone(original)

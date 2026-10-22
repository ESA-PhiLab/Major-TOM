"""docs/ARCHIVES.md's products table matches collections.py (offline)."""
import pytest

pytest.importorskip("pystac_client")
from majortom.build import catalogue                             # noqa: E402


def test_products_table_is_current():
    text = catalogue.DOC.read_text()
    assert catalogue.updated(text) == text, "run: python -m majortom.build.catalogue"


def test_every_product_has_a_producer_page():
    from majortom.build.collections import COLLECTIONS, PRODUCT_PAGES

    assert set(COLLECTIONS) <= set(PRODUCT_PAGES), set(COLLECTIONS) - set(PRODUCT_PAGES)

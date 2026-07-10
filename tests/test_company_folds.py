"""Tests for item 5: leave-companies-out fold construction (_company_folds)."""
from analyzer.correlation import _company_folds, _CV_FOLDS


def test_folds_are_disjoint_and_cover_all():
    companies = [f"C{i}" for i in range(23)]
    folds = _company_folds(companies)
    # Each company appears in exactly one fold.
    seen = [c for f in folds for c in f]
    assert sorted(seen) == sorted(set(companies))
    assert len(seen) == len(set(seen))  # no company in two folds
    # No two folds overlap.
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            assert folds[i].isdisjoint(folds[j])


def test_fold_assignment_is_deterministic():
    companies = [f"C{i}" for i in range(23)]
    a = _company_folds(companies)
    b = _company_folds(list(reversed(companies)))  # order-independent
    assert a == b


def test_default_k_folds():
    companies = [f"C{i}" for i in range(50)]
    folds = _company_folds(companies)
    assert len(folds) == _CV_FOLDS


def test_fewer_companies_than_k_drops_empty_folds():
    folds = _company_folds(["A", "B"])
    assert all(len(f) > 0 for f in folds)
    assert len(folds) == 2

from lanka_news_lens.brief import split_cited
from lanka_news_lens.text import clean, tokens
from lanka_news_lens.weak_labels import weak_label


def test_tokens_keep_sinhala_marks_and_zwj():
    # ශ්‍රී needs ZWJ (U+200D); vowel signs must stay attached
    assert tokens("ශ්‍රී ලංකාව") == ["ශ්‍රී", "ලංකාව"]
    assert tokens("அமைச்சர் கைது!") == ["அமைச்சர்", "கைது"]


def test_clean_strips_bom_and_nbsp():
    assert clean("බි﻿ඳ වැටීම") == "බිඳ වැටීම"


def test_split_cited_one_per_line():
    text = "Act No.9 of 2024 takes effect [1][2].\nNo citation here.\nBad id [9]."
    sents, rejected = split_cited(text, 3)
    assert sents == [("Act No.9 of 2024 takes effect.", [1, 2])]
    assert rejected == 2


def test_split_cited_rejects_foreign_script_leak():
    assert split_cited("Bill 生效 today [1].", 2) == ([], 1)


def test_weak_label_section_and_keywords():
    assert weak_label("adalk", "sports", "x", "", "si") == ("sports", "section")
    assert weak_label("dailymirrorlk", "breaking-news", "Two arrested with heroin in Colombo", "Police arrested",
                      "en")[0] == "crime"
    assert weak_label("x", "", "Nothing matches here", "", "en") == (None, None)


def test_fix_sinhala_rejoins_conjuncts():
    from lanka_news_lens.brief import fix_sinhala

    assert fix_sinhala("ප් රකාශය") == "ප්\u200dරකාශය"
    assert fix_sinhala("වික් රමසිංහ මහතා") == "වික්\u200dරමසිංහ මහතා"
    assert fix_sinhala("මහජන ආරක් ෂක අමාත්‍ය") == "මහජන ආරක්ෂක අමාත්‍ය"
    assert fix_sinhala("දිස්ත් රික්කය") == "දිස්ත්\u200dරික්කය"
    # real word breaks after a hal are kept
    assert fix_sinhala("අධිකරණය විසින් රක්ෂිත") == "අධිකරණය විසින් රක්ෂිත"
    assert fix_sinhala("ක් ලෙස") == "ක් ලෙස"
    assert fix_sinhala("පත් කිරීම") == "පත් කිරීම"

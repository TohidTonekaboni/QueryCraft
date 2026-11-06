"""Language detection, Persian preprocessing and the Gemma translation prompt."""

import re
from typing import Literal

_PERSIAN_LETTERS = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]")
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")  # harakat, superscript alef, tatweel
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_CHARS = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "ە": "ه", "ۀ": "ه", "؟": "?", "٪": "%", "،": ","})

# Schema vocabulary, so the translation matches table/column names.
GLOSSARY = (
    "محصول=product, دسته‌بندی/دسته=category, فروش/فروخته=sold/sales, قیمت=price, "
    "گران‌ترین=most expensive, ارزان‌ترین=cheapest, موجودی=stock, تعداد=quantity/count, "
    "درآمد=revenue, میانگین=average, مجموع=total"
)


def detect_language(text: str) -> Literal["fa", "en"]:
    """Persian if the text contains any Persian/Arabic-script letter (covers mixed text)."""
    return "fa" if _PERSIAN_LETTERS.search(text) else "en"


def normalize_persian(text: str) -> str:
    """Unify Arabic/Persian letter variants, digits and punctuation; drop diacritics."""
    text = _DIACRITICS.sub("", text).translate(_CHARS).translate(_DIGITS)
    text = text.replace("‌", " ").replace("‏", "").replace("‎", "")  # ZWNJ -> space, RTL marks
    return re.sub(r"\s+", " ", text).strip()


def build_translation_prompt(persian: str) -> str:
    return (
        "Translate this Persian question about an online store database into English. "
        f"Use these terms where they apply: {GLOSSARY}.\n"
        "Keep numbers and product names unchanged. Output only the English question, nothing else.\n\n"
        f"Persian: {persian}\nEnglish:"
    )


def clean_translation(output: str) -> str:
    """First non-empty line, without an 'English:' prefix or surrounding quotes."""
    for line in output.splitlines():
        line = line.strip()
        if line:
            line = re.sub(r"^(english|translation)\s*:\s*", "", line, flags=re.I)
            return line.strip(" \"'`“”«»*")
    return ""

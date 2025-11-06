# preproc_runtime.py
import re, multiprocessing
from typing import List, Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# --- diccionarios/regex ---
EMOTICONS_Y_EMOJIS: Dict[str, str] = {
    ">:(": "emojiEnojo", ">:-(": "emojiEnojo", "😠": "emojiEnojo", "😡": "emojiEnojo", "😤": "emojiEnojo", "🤬": "emojiEnojo",
    ":)": "emojiSonrisa", ":-)": "emojiSonrisa", "😄": "emojiSonrisa", "😃": "emojiSonrisa", "😊": "emojiSonrisa",
    ";)": "emojiGuiño", ";-)": "emojiGuiño",
    ":(": "emojiTriste", ":-(": "emojiTriste", "😢": "emojiTriste", "😭": "emojiTriste",
    ":P": "emojiLengua", ":-P": "emojiLengua", "😛": "emojiLengua", "😜": "emojiLengua",
    ":o": "emojiSorprendido", ":-O": "emojiSorprendido", "😮": "emojiSorprendido", "😲": "emojiSorprendido",
    "😂": "emojiRisa", "🤣": "emojiRisa", "xD": "emojiRisa", "XD": "emojiRisa",
    "❤": "emojiAmor", "<3": "emojiAmor", "</3": "emojiCorazonRoto", "😍": "emojiEnamorado",
    "😨": "emojiMiedo", "😱": "emojiAsustado",
    "🎉": "emojiFiesta", "🥳": "emojiFiesta",
}

emoji_unicode_pattern = (
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF]+"
)

def build_emoticon_regex(emoticon_dict: Dict[str, str]) -> re.Pattern:
    escaped = sorted([re.escape(e) for e in emoticon_dict.keys()], key=len, reverse=True)
    emoticon_pattern = "|".join(escaped) if escaped else ""
    if emoticon_pattern:
        combined_pattern = f"(?:{emoticon_pattern})|{emoji_unicode_pattern}"
    else:
        combined_pattern = emoji_unicode_pattern
    return re.compile(combined_pattern)

EMOTICON_REGEX = build_emoticon_regex(EMOTICONS_Y_EMOJIS)

def emoji_process_keep(text: str, keep: bool = True) -> str:
    if text is None:
        return ""
    def repl(match):
        token = match.group(0)
        return EMOTICONS_Y_EMOJIS.get(token, "emoji_otro") if keep else ""
    return EMOTICON_REGEX.sub(repl, text)

# --- Transformadores picklables ---
class ElongationNormalizerV5(BaseEstimator, TransformerMixin):
    def __init__(self):  # <-- CORREGIDO
        pass
    def fit(self, X, y=None): return self
    def transform(self, X: List[str]) -> List[str]:
        return [re.sub(r'(.)\1{2,}', r'\1\1', "" if x is None else str(x)) for x in X]

class BaselineCleanerV5(BaseEstimator, TransformerMixin):
    def __init__(self):  # <-- CORREGIDO
        pass
    def fit(self, X, y=None): return self
    def transform(self, X: List[str]) -> List[str]:
        out = []
        for x in X:
            t = "" if x is None else str(x).lower()
            t = re.sub(r'https?://\S+|www\.\S+', '', t)  # URLs
            t = re.sub(r'@\w+', '', t)                  # @mentions
            t = t.replace('#', '')                      # quitar '#'
            out.append(t)
        return out

class EmojiHandlerV5(BaseEstimator, TransformerMixin):
    def __init__(self, keep_emojis: bool = True):  # <-- CORREGIDO
        self.keep_emojis = keep_emojis
    def fit(self, X, y=None): return self
    def transform(self, X: List[str]) -> List[str]:
        return [emoji_process_keep(x, keep=self.keep_emojis) for x in X]

_SPACY_CACHE_V5 = {}
class SpaCyTokenizerV5(BaseEstimator, TransformerMixin):
    def __init__(self,                      # <-- CORREGIDO
                 lemmatize: bool = False,
                 drop_stopwords: bool = False,
                 drop_punct: bool = True,
                 model: str = "en_core_web_sm",
                 n_threads: Optional[int] = None):
        self.lemmatize = lemmatize
        self.drop_stopwords = drop_stopwords
        self.drop_punct = drop_punct
        self.model = model
        self.n_threads = n_threads or multiprocessing.cpu_count()

    def _ensure_nlp(self):
        import spacy
        global _SPACY_CACHE_V5
        if self.model not in _SPACY_CACHE_V5:
            _SPACY_CACHE_V5[self.model] = spacy.load(self.model, disable=["ner", "textcat"])
        return _SPACY_CACHE_V5[self.model]

    def fit(self, X, y=None):
        self._ensure_nlp()
        return self

    def transform(self, X: List[str]) -> List[str]:
        nlp = self._ensure_nlp()
        docs_in = [("" if x is None else str(x)) for x in X]
        docs = nlp.pipe(docs_in, n_process=self.n_threads, batch_size=1000)
        out: List[str] = []
        for doc in docs:
            toks = []
            for tok in doc:
                if self.drop_stopwords and tok.is_stop:
                    continue
                if self.drop_punct and tok.is_punct:
                    continue
                toks.append(tok.lemma_ if self.lemmatize else tok.text)
            out.append(" ".join(toks))
        return out

# (opcional) Builder del pipeline de texto si lo necesitas en otro lado
def build_text_pipeline_v5(
    lemmatize: bool = False,
    drop_stopwords: bool = False,
    drop_punct: bool = True,
    normalizar_alargamientos: bool = True,
    keep_emojis: bool = True,
    tfidf_params: Optional[dict] = None,
    spacy_model: str = "en_core_web_sm"
) -> Pipeline:
    if tfidf_params is None:
        tfidf_params = dict(ngram_range=(2, 2), min_df=2)
    steps = []
    if normalizar_alargamientos:
        steps.append(("elongations", ElongationNormalizerV5()))
    steps.append(("emojis",   EmojiHandlerV5(keep_emojis=keep_emojis)))
    steps.append(("baseline", BaselineCleanerV5()))
    steps.append(("spacy",    SpaCyTokenizerV5(
        lemmatize=lemmatize,
        drop_stopwords=drop_stopwords,
        drop_punct=drop_punct,
        model=spacy_model
    )))
    steps.append(("tfidf",    TfidfVectorizer(**tfidf_params)))
    return Pipeline(steps)


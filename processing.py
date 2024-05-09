import pydash as py_
import re
from sentence_transformers import SentenceTransformer

sub_ci = lambda x, y: py_.partial(re.sub, x, y, flags=re.IGNORECASE)
sub_cs = lambda x, y: py_.partial(re.sub, x, y)
emoticons = r"\(>\.>\)|\(\^\.\^ゞ\)|\(\^_\^\)Y|:\-\)|;\-@|;\-\^|\(>\.<\)\(\^\.\^\)|\(\^_\-\)/\~\~|:\^|;\(|\(\^_\^\)/|\(ToT\)|:\-\^|\(\^\^ゞ|:\-=|:\-\#|;\-\[|\(>_>\)|:\-D|\(>\.<\)|\(\^o\^\)丿|:\-\.|:P|\(\^_\^\)\-☆|\(\^_\^\)w|;\\|:\-o|;\-C|;\-S|\(\^_\^\)v|:\-C|\(>\.<\)b|\(\*_\*\)|\(\-_\-;\)|;P|;=|\(\^_\-\)b|\(\^o\^\)|:\-P|:\#|\(\*\^\.\^\*\)|>:\[|\(\^_\-\)/\~|:\$|\(\^ω\^\)|:\-\{|:'\-\(|\(\^_\-\)\-☆|\(\-_\-\)|x\-\)|:\-X|:X|\(\*O\*\)|\(\*\^_\^\*\)|\(<_<\)|\(ーー;\)|;\-\#|:\*|;\-P|;\-!|:@|\(\^_\-\)Y|:/|\(\^_\-\)W|:\-0|\(\~_\~\)|;/|:!|;\-D|X\-\)|;\-/|;\-=|\(@_@\)|\(°\~°\)|\(\^_\^メ\)|:'\(|8\-\)|\(°u°\)|;\-\(|:\-\(|:\\|:D|;\-\\|\(>_<\)|\(\^ε\^\)|\(\^_\^\)b|:O|\(\^з\^\)|:\-\&|:=|O:\-\)|\(\^\.\^\)|:\-!|;'\-\)|\('\-'\)|\(\._\.\)|:\-<|;O|\(\^人\^\)|\(\^_\^\)|\(°\-°\)|:'\)|;\-\)|\(\^\-\^\)|;\-\$|\(\^\-\^\)b|\(,_,\)|\(\^_\-\)w|;\-\&|;D|:\-\||\(°_°\)|:S|:\-\\|>:D|;\-\{|\(\^\.\^\)y|\(\^_\-\)d|\(°\.°\)|\(\^_\^\)/\~|:\-\[|:\-/|\(\^_\^\*\)|:\&|;\-<|;'\)|:\)|;\)|;\*|\(\^_\-\)|:\-O|;'\-\(|:\-S|;\-O|:\(|B\-\)|\(\~_\^\)|;@|\(\^\-\^ゝ゛\)|\(\^_\^\)W|;\^|;S|\(°o°\)|\(\^O\^\)|\(\*o\*\)|\(>﹏<\)|;\||;\&|\(\^_\^\)/\~\~|:\||>:\)|\(\^_\-\)/|:\-\*|0:\-\)|;\$|;!|;\-\||;\#|\(\^_\^'\)|:\-\$|:\-@|\(≧∇≦\)|\(T_T\)|\(\*\^0\^\*\)|;\-\*"
abbr_to_slug_cs = {
    r"([A-Z][a-z]+)\.(?: ?(\d+) ?\.( [A-Z]))": r"\1[dot] \2[dot] \3",  # Fig. 6. The | Fig. 6.ctct
    r"([A-Z][a-z]+) ?(\d+)\. ?( [A-Z])": r"\1 \2[dot] \3",
    r"([A-Z][a-z]+)\.": r"\1[dot]",  # Sentence with one word that starts with captial letter.
}
abbr_to_slug_ci = {
    r"et\.? al\.": "[etal]",
    r"vs\.": "[vs]",
    r"etc\.": "[etc]",
    r"Eq\.": "[Eq]",
}
slug_to_abbr = {
    r"\[dot\]": ".",
    r"\[etc\]": "etc.",
    r"\[vs\]": "vs.",
    r"\[fig\]": "fig",
    r"\[tab\]": "tab",
    r"\[ie\]": "i.e.",
    r"\[sec\]": "sec.",
    r"\[eq\]": "eq.",
    r"\[eg\]": "e.g.",
    r"\[ellipsis\]": "...",
    r"\[aka\]": "a.k.a.",
    r"\[etal\]": "et al.",
}
general = [
    r"\( *(?:[a-zA-Z_& \.,*-]+\d{4};?)+ *\)",  # citations (Asic et al., 1234)
    r" ?\[\d+( ?, ?\d+)*\]( ?,? ?\[\d+( ?, ?\d+)*\])*",  # citations [1,2]; [1]
    r"\(\d+\)( ?, ?\(\d+\))*",  # equation numbers (1), (2)
]
sentence_splitter = py_.flow(
    py_.deburr,
    lambda x: py_.reduce_(
        py_.chain(re.findall(r"\b(?:[a-zA-Z]+\.){1,}[a-zA-Z]\.", x))
        .apply(set)
        .map_(lambda x: (re.sub("\.", "\.", x), re.sub("\.", "[dot]", x)))
        .from_pairs()
        .value()
        .items(),
        lambda p, c: re.sub(c[0], c[1], p),
        x,
    ),  # a.k.a. i.i.d. e.g. i.e.
    *py_.map_(general, lambda x: sub_ci(x, "")),
    sub_ci(emoticons, ""),
    sub_ci(r",\. ([A-Z0-9])", r". \1"),  # cwercwer,. The -> cwercwer. The
    sub_ci(r",\. ?([a-z0-9])", r", \1"),  # cwercwer,. cewrc -> cwercwer, cwerc
    sub_ci(r"(\w+)@(\w+)\.(\w+)", r"\1@\2[dot]"),
    sub_ci(r"[\"'] *(.*)([\.\!\?]) *[\"']", r'"\1\"\2'),
    sub_ci(r" *([\.,:])", r"\1"),
    sub_ci(r"\.{3}", "[ellipsis]"),
    sub_ci(r"\.{2}", "."),
    sub_ci(r"\.{4,}", ""),
    sub_ci(r"(?:, ?){2,}", ""),
    sub_ci(r"([^ \(\.,])\(", r"\1 ("),
    sub_ci(r"\)([^ \)\.,:])", r") \1"),
    sub_ci(r"\/{2,} ", ""),
    sub_ci(r"(\d+)(?:\.(\d+))+", r"\1[dot]\2"),
    *py_.map_(abbr_to_slug_cs.items(), lambda x: sub_cs(x[0], x[1])),
    *py_.map_(abbr_to_slug_ci.items(), lambda x: sub_ci(x[0], x[1])),
    sub_ci(r"(?:\[dot] ){2,}", "[dot]"),
    sub_ci(
        r"arXiv:(\d+)\.(\w+) ?(?:\[(\w+)\.(\w+)\])?", r"arXiv:\1[dot]\2 [\3[dot]\4]"
    ),
    sub_ci(r"\(([^\)]*?)\.([^\)]*?)\)", r"(\1[dot]\2)"),
    sub_ci(r"\[([^\]]*?)\.([^\]]*?)\]", r"[\1[dot]\2]"),
    sub_ci(r"\{([^\}]*?)\.([^\}]*?)\}", r"{\1[dot]\2}"),
    sub_ci(r"\"([^\"]*?)\.([^\"]*?)\"", r"\"\1[dot]\2\""),
    sub_ci(r"\'([^\']*?)\.([^\']*?)\'", r"'\1[dot]\2'"),
    sub_ci(r"\b\d+(\.\d+)*", lambda match: match.group(0).replace(".", "[dot]")),
    py_.strings.clean,
    sub_ci(r" \)", ")"),
    sub_ci(r"\( ", "("),
    py_.partial(re.findall, r"[^\.\!\?]*[\.\!\?]"),
    py_.partial(py_.reject, predicate=lambda x: len(x.split(" ")) < 4),
    *py_.map_(
        slug_to_abbr.items(), lambda x: lambda y: py_.map_(y, sub_ci(x[0], x[1]))
    ),
    lambda x: py_.map_(x, lambda y: py_.strings.trim(y)),
)


def group_sentences(sentences, max_tokens=100, overlap=1):
    chunks = []
    tokens_amount = 0
    chunk = []
    for sentence in sentences:
        if tokens_amount < max_tokens:
            chunk.append(sentence)
            tokens_amount += len(py_.strings.words(sentence))
        else:
            chunks.append(chunk)
            chunk = chunk[len(chunk) - overlap :] + [sentence]
            tokens_amount = py_.reduce_(
                chunk,
                lambda total, sentence: len(py_.strings.words(sentence)) + total,
                0,
            )
    else:
        chunks.append(chunk)

    return py_.chain(chunks[bool(overlap) :]).map_(lambda x: " ".join(x)).value()


pretty_print_with_zeros = lambda number, length: str(number).zfill(length)

embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder='temp/transformers/model/')


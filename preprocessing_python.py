"""
NLP Preprocessing Pipeline for Context-Aware Video Comment Analysis
Implements all steps without using built-in NLP libraries (no nltk, spacy, etc.)
Only standard Python allowed.
"""

# ─────────────────────────────────────────────
#  EMOJI MAP  (unicode codepoint → sentiment token)
# ─────────────────────────────────────────────
EMOJI_MAP = {
    # ── Positive ──────────────────────────────
    "\U0001F600": "emoji_positive",   # 😀 grinning face
    "\U0001F601": "emoji_positive",   # 😁 beaming face
    "\U0001F602": "emoji_positive",   # 😂 face with tears of joy
    "\U0001F603": "emoji_positive",   # 😃 grinning face with big eyes
    "\U0001F604": "emoji_positive",   # 😄 grinning face with smiling eyes
    "\U0001F605": "emoji_positive",   # 😅 grinning face with sweat
    "\U0001F606": "emoji_positive",   # 😆 grinning squinting face
    "\U0001F609": "emoji_positive",   # 😉 winking face
    "\U0001F60A": "emoji_positive",   # 😊 smiling face with smiling eyes
    "\U0001F60B": "emoji_positive",   # 😋 face savoring food
    "\U0001F60D": "emoji_positive",   # 😍 smiling face with heart-eyes
    "\U0001F618": "emoji_positive",   # 😘 face blowing a kiss
    "\U0001F617": "emoji_positive",   # 😗 kissing face
    "\U0001F619": "emoji_positive",   # 😙 kissing face with smiling eyes
    "\U0001F61A": "emoji_positive",   # 😚 kissing face with closed eyes
    "\U0001F642": "emoji_positive",   # 🙂 slightly smiling face
    "\U0001F917": "emoji_positive",   # 🤗 hugging face
    "\U0001F929": "emoji_positive",   # 🤩 star-struck
    "\U0001F970": "emoji_positive",   # 🥰 smiling face with hearts
    "\U0001F973": "emoji_positive",   # 🥳 partying face
    "\U00002764": "emoji_positive",   # ❤  red heart
    "\U0001F495": "emoji_positive",   # 💕 two hearts
    "\U0001F496": "emoji_positive",   # 💖 sparkling heart
    "\U0001F497": "emoji_positive",   # 💗 growing heart
    "\U0001F498": "emoji_positive",   # 💘 heart with arrow
    "\U0001F49A": "emoji_positive",   # 💚 green heart
    "\U0001F49B": "emoji_positive",   # 💛 yellow heart
    "\U0001F49C": "emoji_positive",   # 💜 purple heart
    "\U0001F49E": "emoji_positive",   # 💞 revolving hearts
    "\U0001F44D": "emoji_positive",   # 👍 thumbs up
    "\U0001F44F": "emoji_positive",   # 👏 clapping hands
    "\U0001F525": "emoji_positive",   # 🔥 fire (hype)
    "\U00002B50": "emoji_positive",   # ⭐ star
    "\U0001F31F": "emoji_positive",   # 🌟 glowing star
    "\U0001F3C6": "emoji_positive",   # 🏆 trophy
    "\U0001F389": "emoji_positive",   # 🎉 party popper
    "\U0001F38A": "emoji_positive",   # 🎊 confetti ball
    "\U0001F4AF": "emoji_positive",   # 💯 hundred points
    "\U0001F64C": "emoji_positive",   # 🙌 raising hands
    "\U0001F91D": "emoji_positive",   # 🤝 handshake
    "\U0001F60E": "emoji_positive",   # 😎 smiling face with sunglasses
    "\U0001F4AA": "emoji_positive",   # 💪 flexed biceps
    "\U0001F440": "emoji_positive",   # 👀 eyes (interested/excited)
    "\U0001F64B": "emoji_positive",   # 🙋 person raising hand
    "\U0001F90D": "emoji_positive",   # 🤍 white heart
    "\U0001F9E1": "emoji_positive",   # 🧡 orange heart

    # ── Negative ──────────────────────────────
    "\U0001F614": "emoji_negative",   # 😔 pensive face
    "\U0001F615": "emoji_negative",   # 😕 confused face
    "\U0001F61E": "emoji_negative",   # 😞 disappointed face
    "\U0001F61F": "emoji_negative",   # 😟 worried face
    "\U0001F620": "emoji_negative",   # 😠 angry face
    "\U0001F621": "emoji_negative",   # 😡 pouting face
    "\U0001F622": "emoji_negative",   # 😢 crying face
    "\U0001F623": "emoji_negative",   # 😣 persevering face
    "\U0001F624": "emoji_negative",   # 😤 face with steam from nose
    "\U0001F625": "emoji_negative",   # 😥 sad but relieved
    "\U0001F626": "emoji_negative",   # 😦 frowning face with open mouth
    "\U0001F627": "emoji_negative",   # 😧 anguished face
    "\U0001F628": "emoji_negative",   # 😨 fearful face
    "\U0001F629": "emoji_negative",   # 😩 weary face
    "\U0001F62B": "emoji_negative",   # 😫 tired face
    "\U0001F62D": "emoji_negative",   # 😭 loudly crying face
    "\U0001F630": "emoji_negative",   # 😰 anxious face with sweat
    "\U0001F631": "emoji_negative",   # 😱 face screaming in fear
    "\U0001F633": "emoji_negative",   # 😳 flushed face
    "\U0001F635": "emoji_negative",   # 😵 dizzy face
    "\U0001F641": "emoji_negative",   # 🙁 slightly frowning face
    "\U0001F44E": "emoji_negative",   # 👎 thumbs down
    "\U0001F4A9": "emoji_negative",   # 💩 pile of poo
    "\U0001F915": "emoji_negative",   # 🤕 face with head-bandage
    "\U0001F912": "emoji_negative",   # 🤒 face with thermometer
    "\U0001F616": "emoji_negative",   # 😖 confounded face
    "\U0001F62C": "emoji_negative",   # 😬 grimacing face
    "\U0001F97A": "emoji_negative",   # 🥺 pleading face
    "\U0001F494": "emoji_negative",   # 💔 broken heart

    # ── Neutral / Surprise ────────────────────
    "\U0001F610": "emoji_neutral",    # 😐 neutral face
    "\U0001F611": "emoji_neutral",    # 😑 expressionless face
    "\U0001F636": "emoji_neutral",    # 😶 face without mouth
    "\U0001F644": "emoji_neutral",    # 🙄 face with rolling eyes
    "\U0001F643": "emoji_neutral",    # 🙃 upside-down face
    "\U0001F60F": "emoji_neutral",    # 😏 smirking face
    "\U0001F913": "emoji_neutral",    # 🤓 nerd face
    "\U0001F914": "emoji_neutral",    # 🤔 thinking face
    "\U0001F928": "emoji_neutral",    # 🤨 face with raised eyebrow
    "\U0001F62E": "emoji_neutral",    # 😮 face with open mouth (surprise)
    "\U0001F62F": "emoji_neutral",    # 😯 hushed face
    "\U0001F632": "emoji_neutral",    # 😲 astonished face
    "\U0001F611": "emoji_neutral",    # 😑 expressionless
    "\U0001F9D0": "emoji_neutral",    # 🧐 face with monocle
    "\U0001F607": "emoji_neutral",    # 😇 smiling face with halo
    "\U0001F920": "emoji_neutral",    # 🤠 cowboy hat face
    "\U0001F921": "emoji_neutral",    # 🤡 clown face
    "\U0001F47B": "emoji_neutral",    # 👻 ghost
    "\U0001F918": "emoji_neutral",    # 🤘 sign of horns (could be hype)
    "\U0001F926": "emoji_neutral",    # 🤦 face palm
    "\U0001F937": "emoji_neutral",    # 🤷 shrug

    # ── Laughter / Humor ─────────────────────
    "\U0001F923": "emoji_laughter",   # 🤣 rolling on floor laughing
    "\U0001FAE0": "emoji_laughter",   # 🫠 melting face (irony/humor)

    # ── Disgust ───────────────────────────────
    "\U0001F922": "emoji_disgust",    # 🤢 nauseated face
    "\U0001F92E": "emoji_disgust",    # 🤮 face vomiting
    "\U0001F910": "emoji_disgust",    # 🤐 zipper-mouth face

    # ── Shock / Wow ───────────────────────────
    "\U0001F92F": "emoji_shock",      # 🤯 exploding head
    "\U0001F975": "emoji_shock",      # 🥵 hot face
    "\U0001F976": "emoji_shock",      # 🥶 cold face
}


# ─────────────────────────────────────────────
#  STEP 1 – Normalize URLs and Mentions
# ─────────────────────────────────────────────
def normalize_urls_and_mentions(text: str) -> str:
    """
    Replace URLs with __URL__ and @mentions with __MENTION__.
    """
    # --- simple state-machine URL detector ---
    tokens = text.split(" ")
    result = []
    url_prefixes = ("http://", "https://", "www.")
    for tok in tokens:
        lower = tok.lower()
        is_url = False
        for prefix in url_prefixes:
            if lower.startswith(prefix):
                is_url = True
                break
        if is_url:
            result.append("__URL__")
        elif tok.startswith("@") and len(tok) > 1:
            result.append("__MENTION__")
        else:
            result.append(tok)
    return " ".join(result)


# ─────────────────────────────────────────────
#  STEP 2 – Preserve & Map Emojis
# ─────────────────────────────────────────────
def map_emojis(text: str) -> str:
    """
    Replace each known emoji with its sentiment token.
    Unknown emojis are replaced with emoji_unknown.
    Walks the string codepoint by codepoint to handle multi-codepoint sequences.
    """
    result = []
    i = 0
    chars = list(text)
    while i < len(chars):
        cp = chars[i]
        # Check for known emoji
        if cp in EMOJI_MAP:
            result.append(" " + EMOJI_MAP[cp] + " ")
            i += 1
            continue
        # Detect any emoji-range codepoint not in our map
        cp_ord = ord(cp)
        if (
            0x1F600 <= cp_ord <= 0x1FAFF  # Emoticons & supplemental
            or 0x2600 <= cp_ord <= 0x27BF   # Misc symbols & dingbats
            or 0xFE00 <= cp_ord <= 0xFE0F   # Variation selectors
            or 0x1F300 <= cp_ord <= 0x1F5FF  # Misc symbols & pictographs
            or 0x1F900 <= cp_ord <= 0x1F9FF  # Supplemental symbols
            or 0x1FA00 <= cp_ord <= 0x1FA6F  # Chess / other
        ):
            result.append(" emoji_unknown ")
            i += 1
            continue
        result.append(cp)
        i += 1
    return "".join(result)


# ─────────────────────────────────────────────
#  STEP 3 – Retain Hashtag Text
# ─────────────────────────────────────────────
def retain_hashtag_text(text: str) -> str:
    """
    Strip the '#' symbol but keep the hashtag word so it participates
    in downstream tokenization as regular text.
    e.g. #Amazing → Amazing
    """
    tokens = text.split(" ")
    result = []
    for tok in tokens:
        if tok.startswith("#") and len(tok) > 1:
            result.append(tok[1:])   # drop '#', keep word
        else:
            result.append(tok)
    return " ".join(result)


# ─────────────────────────────────────────────
#  STEP 4 – Preserve Original Casing
# ─────────────────────────────────────────────
def preserve_casing(text: str) -> str:
    """
    No-op transformation: original casing is intentionally preserved
    for downstream models (ALL-CAPS signals emotion intensity).
    Returns text unchanged.
    """
    return text   # identity – explicit stage for pipeline clarity


# ─────────────────────────────────────────────
#  STEP 5 – Remove Extra Spaces
# ─────────────────────────────────────────────
def remove_extra_spaces(text: str) -> str:
    """
    Collapse multiple consecutive whitespace characters into a single space.
    Implemented without re.sub by splitting on any whitespace and rejoining.
    """
    parts = text.split()            # split() handles tabs, newlines, etc.
    return " ".join(parts).strip()


# ─────────────────────────────────────────────
#  STEP 6 – Detect Character Elongation
# ─────────────────────────────────────────────
def detect_elongation(text: str, threshold: int = 3) -> str:
    """
    Detects elongated characters (e.g. 'sooooo', 'noooo', 'hahaha').
    Tags the token with __ELONG__ suffix and normalises the repeated char.
    e.g. 'sooooo' → 'so__ELONG__'
    threshold: minimum run length to count as elongation.
    """
    tokens = text.split(" ")
    result = []
    for token in tokens:
        new_token, elongated = _compress_elongation(token, threshold)
        if elongated:
            result.append(new_token + "__ELONG__")
        else:
            result.append(token)
    return " ".join(result)


def _compress_elongation(token: str, threshold: int) -> tuple:
    """
    Returns (compressed_token, was_elongated).
    Scans for runs of the same character >= threshold.
    """
    if len(token) < threshold:
        return token, False

    compressed = []
    i = 0
    elongated = False
    while i < len(token):
        char = token[i]
        run_len = 1
        while i + run_len < len(token) and token[i + run_len].lower() == char.lower():
            run_len += 1
        if run_len >= threshold:
            elongated = True
            compressed.append(char)   # keep only one instance
        else:
            compressed.append(token[i: i + run_len])
        i += run_len
    return "".join(compressed), elongated


# ─────────────────────────────────────────────
#  STEP 7 – Tokenization
# ─────────────────────────────────────────────
def tokenize(text: str) -> list:
    """
    Whitespace + punctuation-aware tokenizer.
    Rules:
      - Split on whitespace first.
      - Punctuation at the end/start of a word is separated as its own token.
      - Contractions and hyphenated words are kept together.
      - Special pipeline tokens (__URL__, __MENTION__, __ELONG__, emoji_*) are never split.
    """
    PUNCT = set(".,!?;:\"'()[]{}…")
    # characters we never want to split inside special tokens
    SPECIAL_PREFIXES = ("__", "emoji_")

    raw_tokens = text.split()
    tokens = []

    for raw in raw_tokens:
        # Never split sentinel/emoji tokens
        is_special = any(raw.startswith(p) for p in SPECIAL_PREFIXES)
        if is_special or raw.endswith("__ELONG__"):
            tokens.append(raw)
            continue

        # Strip leading punctuation
        leading = []
        while raw and raw[0] in PUNCT:
            leading.append(raw[0])
            raw = raw[1:]

        # Strip trailing punctuation
        trailing = []
        while raw and raw[-1] in PUNCT:
            trailing.insert(0, raw[-1])
            raw = raw[:-1]

        if leading:
            tokens.extend(leading)
        if raw:
            tokens.append(raw)
        if trailing:
            tokens.extend(trailing)

    return [t for t in tokens if t]


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────
def preprocess(text: str, elongation_threshold: int = 3) -> dict:
    """
    Run the full preprocessing pipeline and return intermediate stages
    plus the final token list.

    Returns a dict with keys:
        raw, after_normalize, after_emoji, after_hashtag,
        after_casing, after_spaces, after_elongation, tokens
    """
    stages = {"raw": text}

    s = normalize_urls_and_mentions(text)
    stages["after_normalize"] = s

    s = map_emojis(s)
    stages["after_emoji"] = s

    s = retain_hashtag_text(s)
    stages["after_hashtag"] = s

    s = preserve_casing(s)
    stages["after_casing"] = s

    s = remove_extra_spaces(s)
    stages["after_spaces"] = s

    s = detect_elongation(s, threshold=elongation_threshold)
    stages["after_elongation"] = s

    stages["tokens"] = tokenize(s)
    return stages


# ─────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_comments = [
        "OMG this part at 2:30 was AMAZINGGG 😍🔥 #bestmoment @creator check https://t.co/xyz",
        "noooooo why did he do that 😭😭 I was sooooo not expecting it!!",
        "lmaoooo 🤣🤣 this is the funniest thing everrr #viral @user123",
        "Honestly... this scene hits different 😔 no cap",
        "WOW WOW WOW 💯 best video of 2024 #mustwatch",
    ]

    for comment in test_comments:
        print("=" * 65)
        result = preprocess(comment)
        for stage, value in result.items():
            label = stage.upper().replace("_", " ").ljust(22)
            print(f"  {label}: {value}")
    print("=" * 65)
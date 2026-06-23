/*
 * nlp_preprocessor.c
 * ─────────────────────────────────────────────────────────────
 * NLP Preprocessing Pipeline  –  C Implementation
 *
 * Pipeline steps  (mirrors the Python version exactly):
 *   1. Normalize URLs / Mentions
 *   2. Preserve & Map Emojis  (UTF-8 aware, same 80+ emoji map)
 *   3. Retain Hashtag Text    (strip '#', keep word)
 *   4. Preserve Original Casing  (identity pass)
 *   5. Remove Extra Spaces
 *   6. Detect Character Elongation
 *   7. Tokenization
 *
 * Allowed headers: <stdio.h>, <string.h>, <ctype.h>
 * ─────────────────────────────────────────────────────────────
 */

#include "nlp_preprocessor.h"

/* ════════════════════════════════════════════════════════════
 *  EMOJI MAP
 *  Each entry stores:
 *    - utf8_bytes : the raw UTF-8 byte sequence of the emoji
 *    - token      : the sentiment token to emit
 * ════════════════════════════════════════════════════════════ */
typedef struct {
    unsigned char utf8_bytes[8]; /* up to 4 UTF-8 bytes + null  */
    const char   *token;
} EmojiEntry;

/*
 * Helper macro: build a 4-byte UTF-8 sequence from a Unicode codepoint.
 * All emoji in our map are in the range U+1F300..U+1FAFF  (4 bytes)
 * or U+2600..U+27BF / U+2B50 (3 bytes).
 * We store the actual UTF-8 bytes directly for simple memcmp matching.
 */

static const EmojiEntry EMOJI_MAP[] = {

    /* ── Positive ─────────────────────────────────────────── */
    /* 😀 U+1F600 */ {{0xF0,0x9F,0x98,0x80}, "emoji_positive"},
    /* 😁 U+1F601 */ {{0xF0,0x9F,0x98,0x81}, "emoji_positive"},
    /* 😂 U+1F602 */ {{0xF0,0x9F,0x98,0x82}, "emoji_positive"},
    /* 😃 U+1F603 */ {{0xF0,0x9F,0x98,0x83}, "emoji_positive"},
    /* 😄 U+1F604 */ {{0xF0,0x9F,0x98,0x84}, "emoji_positive"},
    /* 😅 U+1F605 */ {{0xF0,0x9F,0x98,0x85}, "emoji_positive"},
    /* 😆 U+1F606 */ {{0xF0,0x9F,0x98,0x86}, "emoji_positive"},
    /* 😉 U+1F609 */ {{0xF0,0x9F,0x98,0x89}, "emoji_positive"},
    /* 😊 U+1F60A */ {{0xF0,0x9F,0x98,0x8A}, "emoji_positive"},
    /* 😋 U+1F60B */ {{0xF0,0x9F,0x98,0x8B}, "emoji_positive"},
    /* 😍 U+1F60D */ {{0xF0,0x9F,0x98,0x8D}, "emoji_positive"},
    /* 😘 U+1F618 */ {{0xF0,0x9F,0x98,0x98}, "emoji_positive"},
    /* 😗 U+1F617 */ {{0xF0,0x9F,0x98,0x97}, "emoji_positive"},
    /* 😙 U+1F619 */ {{0xF0,0x9F,0x98,0x99}, "emoji_positive"},
    /* 😚 U+1F61A */ {{0xF0,0x9F,0x98,0x9A}, "emoji_positive"},
    /* 🙂 U+1F642 */ {{0xF0,0x9F,0x99,0x82}, "emoji_positive"},
    /* 🤗 U+1F917 */ {{0xF0,0x9F,0xA4,0x97}, "emoji_positive"},
    /* 🤩 U+1F929 */ {{0xF0,0x9F,0xA4,0xA9}, "emoji_positive"},
    /* 🥰 U+1F970 */ {{0xF0,0x9F,0xA5,0xB0}, "emoji_positive"},
    /* 🥳 U+1F973 */ {{0xF0,0x9F,0xA5,0xB3}, "emoji_positive"},
    /* ❤  U+2764  */ {{0xE2,0x9D,0xA4,0x00}, "emoji_positive"},   /* 3-byte */
    /* 💕 U+1F495 */ {{0xF0,0x9F,0x92,0x95}, "emoji_positive"},
    /* 💖 U+1F496 */ {{0xF0,0x9F,0x92,0x96}, "emoji_positive"},
    /* 💗 U+1F497 */ {{0xF0,0x9F,0x92,0x97}, "emoji_positive"},
    /* 💘 U+1F498 */ {{0xF0,0x9F,0x92,0x98}, "emoji_positive"},
    /* 💚 U+1F49A */ {{0xF0,0x9F,0x92,0x9A}, "emoji_positive"},
    /* 💛 U+1F49B */ {{0xF0,0x9F,0x92,0x9B}, "emoji_positive"},
    /* 💜 U+1F49C */ {{0xF0,0x9F,0x92,0x9C}, "emoji_positive"},
    /* 💞 U+1F49E */ {{0xF0,0x9F,0x92,0x9E}, "emoji_positive"},
    /* 👍 U+1F44D */ {{0xF0,0x9F,0x91,0x8D}, "emoji_positive"},
    /* 👏 U+1F44F */ {{0xF0,0x9F,0x91,0x8F}, "emoji_positive"},
    /* 🔥 U+1F525 */ {{0xF0,0x9F,0x94,0xA5}, "emoji_positive"},
    /* ⭐ U+2B50  */ {{0xE2,0xAD,0x90,0x00}, "emoji_positive"},   /* 3-byte */
    /* 🌟 U+1F31F */ {{0xF0,0x9F,0x8C,0x9F}, "emoji_positive"},
    /* 🏆 U+1F3C6 */ {{0xF0,0x9F,0x8F,0x86}, "emoji_positive"},
    /* 🎉 U+1F389 */ {{0xF0,0x9F,0x8E,0x89}, "emoji_positive"},
    /* 🎊 U+1F38A */ {{0xF0,0x9F,0x8E,0x8A}, "emoji_positive"},
    /* 💯 U+1F4AF */ {{0xF0,0x9F,0x92,0xAF}, "emoji_positive"},
    /* 🙌 U+1F64C */ {{0xF0,0x9F,0x99,0x8C}, "emoji_positive"},
    /* 🤝 U+1F91D */ {{0xF0,0x9F,0xA4,0x9D}, "emoji_positive"},
    /* 😎 U+1F60E */ {{0xF0,0x9F,0x98,0x8E}, "emoji_positive"},
    /* 💪 U+1F4AA */ {{0xF0,0x9F,0x92,0xAA}, "emoji_positive"},
    /* 👀 U+1F440 */ {{0xF0,0x9F,0x91,0x80}, "emoji_positive"},
    /* 🙋 U+1F64B */ {{0xF0,0x9F,0x99,0x8B}, "emoji_positive"},
    /* 🤍 U+1F90D */ {{0xF0,0x9F,0xA4,0x8D}, "emoji_positive"},
    /* 🧡 U+1F9E1 */ {{0xF0,0x9F,0xA7,0xA1}, "emoji_positive"},

    /* ── Negative ─────────────────────────────────────────── */
    /* 😔 U+1F614 */ {{0xF0,0x9F,0x98,0x94}, "emoji_negative"},
    /* 😕 U+1F615 */ {{0xF0,0x9F,0x98,0x95}, "emoji_negative"},
    /* 😞 U+1F61E */ {{0xF0,0x9F,0x98,0x9E}, "emoji_negative"},
    /* 😟 U+1F61F */ {{0xF0,0x9F,0x98,0x9F}, "emoji_negative"},
    /* 😠 U+1F620 */ {{0xF0,0x9F,0x98,0xA0}, "emoji_negative"},
    /* 😡 U+1F621 */ {{0xF0,0x9F,0x98,0xA1}, "emoji_negative"},
    /* 😢 U+1F622 */ {{0xF0,0x9F,0x98,0xA2}, "emoji_negative"},
    /* 😣 U+1F623 */ {{0xF0,0x9F,0x98,0xA3}, "emoji_negative"},
    /* 😤 U+1F624 */ {{0xF0,0x9F,0x98,0xA4}, "emoji_negative"},
    /* 😥 U+1F625 */ {{0xF0,0x9F,0x98,0xA5}, "emoji_negative"},
    /* 😦 U+1F626 */ {{0xF0,0x9F,0x98,0xA6}, "emoji_negative"},
    /* 😧 U+1F627 */ {{0xF0,0x9F,0x98,0xA7}, "emoji_negative"},
    /* 😨 U+1F628 */ {{0xF0,0x9F,0x98,0xA8}, "emoji_negative"},
    /* 😩 U+1F629 */ {{0xF0,0x9F,0x98,0xA9}, "emoji_negative"},
    /* 😫 U+1F62B */ {{0xF0,0x9F,0x98,0xAB}, "emoji_negative"},
    /* 😭 U+1F62D */ {{0xF0,0x9F,0x98,0xAD}, "emoji_negative"},
    /* 😰 U+1F630 */ {{0xF0,0x9F,0x98,0xB0}, "emoji_negative"},
    /* 😱 U+1F631 */ {{0xF0,0x9F,0x98,0xB1}, "emoji_negative"},
    /* 😳 U+1F633 */ {{0xF0,0x9F,0x98,0xB3}, "emoji_negative"},
    /* 😵 U+1F635 */ {{0xF0,0x9F,0x98,0xB5}, "emoji_negative"},
    /* 🙁 U+1F641 */ {{0xF0,0x9F,0x99,0x81}, "emoji_negative"},
    /* 👎 U+1F44E */ {{0xF0,0x9F,0x91,0x8E}, "emoji_negative"},
    /* 💩 U+1F4A9 */ {{0xF0,0x9F,0x92,0xA9}, "emoji_negative"},
    /* 🤕 U+1F915 */ {{0xF0,0x9F,0xA4,0x95}, "emoji_negative"},
    /* 🤒 U+1F912 */ {{0xF0,0x9F,0xA4,0x92}, "emoji_negative"},
    /* 😖 U+1F616 */ {{0xF0,0x9F,0x98,0x96}, "emoji_negative"},
    /* 😬 U+1F62C */ {{0xF0,0x9F,0x98,0xAC}, "emoji_negative"},
    /* 🥺 U+1F97A */ {{0xF0,0x9F,0xA5,0xBA}, "emoji_negative"},
    /* 💔 U+1F494 */ {{0xF0,0x9F,0x92,0x94}, "emoji_negative"},

    /* ── Neutral / Surprise ───────────────────────────────── */
    /* 😐 U+1F610 */ {{0xF0,0x9F,0x98,0x90}, "emoji_neutral"},
    /* 😑 U+1F611 */ {{0xF0,0x9F,0x98,0x91}, "emoji_neutral"},
    /* 😶 U+1F636 */ {{0xF0,0x9F,0x98,0xB6}, "emoji_neutral"},
    /* 🙄 U+1F644 */ {{0xF0,0x9F,0x99,0x84}, "emoji_neutral"},
    /* 🙃 U+1F643 */ {{0xF0,0x9F,0x99,0x83}, "emoji_neutral"},
    /* 😏 U+1F60F */ {{0xF0,0x9F,0x98,0x8F}, "emoji_neutral"},
    /* 🤓 U+1F913 */ {{0xF0,0x9F,0xA4,0x93}, "emoji_neutral"},
    /* 🤔 U+1F914 */ {{0xF0,0x9F,0xA4,0x94}, "emoji_neutral"},
    /* 🤨 U+1F928 */ {{0xF0,0x9F,0xA4,0xA8}, "emoji_neutral"},
    /* 😮 U+1F62E */ {{0xF0,0x9F,0x98,0xAE}, "emoji_neutral"},
    /* 😯 U+1F62F */ {{0xF0,0x9F,0x98,0xAF}, "emoji_neutral"},
    /* 😲 U+1F632 */ {{0xF0,0x9F,0x98,0xB2}, "emoji_neutral"},
    /* 🧐 U+1F9D0 */ {{0xF0,0x9F,0xA7,0x90}, "emoji_neutral"},
    /* 😇 U+1F607 */ {{0xF0,0x9F,0x98,0x87}, "emoji_neutral"},
    /* 🤠 U+1F920 */ {{0xF0,0x9F,0xA4,0xA0}, "emoji_neutral"},
    /* 🤡 U+1F921 */ {{0xF0,0x9F,0xA4,0xA1}, "emoji_neutral"},
    /* 👻 U+1F47B */ {{0xF0,0x9F,0x91,0xBB}, "emoji_neutral"},
    /* 🤘 U+1F918 */ {{0xF0,0x9F,0xA4,0x98}, "emoji_neutral"},
    /* 🤦 U+1F926 */ {{0xF0,0x9F,0xA4,0xA6}, "emoji_neutral"},
    /* 🤷 U+1F937 */ {{0xF0,0x9F,0xA4,0xB7}, "emoji_neutral"},

    /* ── Laughter / Humor ─────────────────────────────────── */
    /* 🤣 U+1F923 */ {{0xF0,0x9F,0xA4,0xA3}, "emoji_laughter"},
    /* 🫠 U+1FAE0 */ {{0xF0,0x9F,0xAB,0xA0}, "emoji_laughter"},

    /* ── Disgust ──────────────────────────────────────────── */
    /* 🤢 U+1F922 */ {{0xF0,0x9F,0xA4,0xA2}, "emoji_disgust"},
    /* 🤮 U+1F92E */ {{0xF0,0x9F,0xA4,0xAE}, "emoji_disgust"},
    /* 🤐 U+1F910 */ {{0xF0,0x9F,0xA4,0x90}, "emoji_disgust"},

    /* ── Shock / Wow ──────────────────────────────────────── */
    /* 🤯 U+1F92F */ {{0xF0,0x9F,0xA4,0xAF}, "emoji_shock"},
    /* 🥵 U+1F975 */ {{0xF0,0x9F,0xA5,0xB5}, "emoji_shock"},
    /* 🥶 U+1F976 */ {{0xF0,0x9F,0xA5,0xB6}, "emoji_shock"},
};

/* Number of entries in EMOJI_MAP */
#define EMOJI_MAP_SIZE  (int)(sizeof(EMOJI_MAP) / sizeof(EMOJI_MAP[0]))


/* ════════════════════════════════════════════════════════════
 *  UTF-8 UTILITIES
 * ════════════════════════════════════════════════════════════ */

/*
 * utf8_seq_len()
 * Given the first byte of a UTF-8 sequence, return the total byte length
 * of that sequence (1-4).  Returns 1 for any invalid/continuation byte.
 */
static int utf8_seq_len(unsigned char first_byte)
{
    if ((first_byte & 0x80) == 0x00) return 1; /* 0xxxxxxx  ASCII         */
    if ((first_byte & 0xE0) == 0xC0) return 2; /* 110xxxxx  2-byte seq    */
    if ((first_byte & 0xF0) == 0xE0) return 3; /* 1110xxxx  3-byte seq    */
    if ((first_byte & 0xF8) == 0xF0) return 4; /* 11110xxx  4-byte seq    */
    return 1; /* continuation or invalid byte */
}

/*
 * decode_utf8_codepoint()
 * Decodes up to 4 bytes from `src` into a Unicode codepoint.
 * Returns the codepoint; writes byte length into *len.
 */
static unsigned int decode_utf8_codepoint(const unsigned char *src, int *len)
{
    *len = utf8_seq_len(src[0]);
    switch (*len) {
        case 1: return src[0];
        case 2: return ((src[0] & 0x1F) << 6)  | (src[1] & 0x3F);
        case 3: return ((src[0] & 0x0F) << 12) | ((src[1] & 0x3F) << 6)  | (src[2] & 0x3F);
        case 4: return ((src[0] & 0x07) << 18) | ((src[1] & 0x3F) << 12) | ((src[2] & 0x3F) << 6) | (src[3] & 0x3F);
        default: return src[0];
    }
}

/*
 * is_emoji_range()
 * Returns 1 if the codepoint falls in a known emoji Unicode range.
 * Mirrors the Python range checks.
 */
static int is_emoji_range(unsigned int cp)
{
    return (cp >= 0x1F600 && cp <= 0x1FAFF) ||  /* Emoticons + supplemental */
           (cp >= 0x2600  && cp <= 0x27BF)  ||  /* Misc symbols & dingbats  */
           (cp >= 0xFE00  && cp <= 0xFE0F)  ||  /* Variation selectors      */
           (cp >= 0x1F300 && cp <= 0x1F5FF) ||  /* Misc symbols/pictographs */
           (cp >= 0x1F900 && cp <= 0x1F9FF) ||  /* Supplemental symbols     */
           (cp >= 0x1FA00 && cp <= 0x1FA6F);    /* Chess / other            */
}


/* ════════════════════════════════════════════════════════════
 *  INTERNAL HELPERS
 * ════════════════════════════════════════════════════════════ */

/* safe_strcat: append src to dst, never exceeding dst_max-1 chars total */
static void safe_strcat(char *dst, const char *src, int dst_max)
{
    int cur = (int)strlen(dst);
    int rem = dst_max - cur - 1;
    if (rem <= 0) return;
    strncat(dst, src, (size_t)rem);
}

/* str_starts_with: returns 1 if haystack begins with needle (case-sensitive) */
static int str_starts_with(const char *haystack, const char *needle)
{
    return strncmp(haystack, needle, strlen(needle)) == 0;
}

/* str_starts_with_ci: case-insensitive version */
static int str_starts_with_ci(const char *haystack, const char *needle)
{
    size_t n = strlen(needle);
    size_t h = strlen(haystack);
    if (h < n) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)haystack[i]) != tolower((unsigned char)needle[i]))
            return 0;
    }
    return 1;
}

/* is_punct: returns 1 if ch is a punctuation character we split on */
static int is_punct(char ch)
{
    return (ch == '.' || ch == ',' || ch == '!' || ch == '?' ||
            ch == ';' || ch == ':' || ch == '"' || ch == '\'' ||
            ch == '(' || ch == ')' || ch == '[' || ch == ']' ||
            ch == '{' || ch == '}');
}


/* ════════════════════════════════════════════════════════════
 *  STEP 1 – Normalize URLs and Mentions
 *  Mirrors: normalize_urls_and_mentions() in Python
 * ════════════════════════════════════════════════════════════ */
static void normalize_urls_and_mentions(const char *text, char *out, int out_max)
{
    char buf[MAX_TEXT];
    strncpy(buf, text, MAX_TEXT - 1);
    buf[MAX_TEXT - 1] = '\0';

    out[0] = '\0';

    char *tok = strtok(buf, " ");
    int first = 1;

    while (tok != NULL) {
        if (!first) safe_strcat(out, " ", out_max);
        first = 0;

        /* Check URL prefixes (case-insensitive) */
        if (str_starts_with_ci(tok, "http://")  ||
            str_starts_with_ci(tok, "https://") ||
            str_starts_with_ci(tok, "www.")) {
            safe_strcat(out, "__URL__", out_max);
        }
        /* Check @mention */
        else if (tok[0] == '@' && tok[1] != '\0') {
            safe_strcat(out, "__MENTION__", out_max);
        }
        else {
            safe_strcat(out, tok, out_max);
        }

        tok = strtok(NULL, " ");
    }
}


/* ════════════════════════════════════════════════════════════
 *  STEP 2 – Preserve & Map Emojis
 *  Mirrors: map_emojis() in Python
 *  Walks byte-by-byte; handles both 3-byte and 4-byte UTF-8 emoji.
 * ════════════════════════════════════════════════════════════ */
static void map_emojis(const char *text, char *out, int out_max)
{
    const unsigned char *src = (const unsigned char *)text;
    out[0] = '\0';

    int i = 0;
    int text_len = (int)strlen(text);

    while (i < text_len) {
        int seq_len = 0;
        unsigned int cp = decode_utf8_codepoint(src + i, &seq_len);

        /* ── Try to match against our EMOJI_MAP ── */
        int matched = 0;
        for (int e = 0; e < EMOJI_MAP_SIZE; e++) {
            const unsigned char *eb = EMOJI_MAP[e].utf8_bytes;
            /* Determine how many bytes this map entry uses (stop at 0x00) */
            int entry_len = 0;
            while (entry_len < 4 && eb[entry_len] != 0x00) entry_len++;

            if (seq_len == entry_len &&
                memcmp(src + i, eb, (size_t)entry_len) == 0) {
                safe_strcat(out, " ", out_max);
                safe_strcat(out, EMOJI_MAP[e].token, out_max);
                safe_strcat(out, " ", out_max);
                matched = 1;
                break;
            }
        }

        if (!matched) {
            /* Unknown emoji in range → emit emoji_unknown */
            if (is_emoji_range(cp)) {
                safe_strcat(out, " emoji_unknown ", out_max);
            } else {
                /* Regular character – append raw bytes */
                char tmp[8] = {0};
                memcpy(tmp, src + i, (size_t)seq_len);
                safe_strcat(out, tmp, out_max);
            }
        }

        i += seq_len;
    }
}


/* ════════════════════════════════════════════════════════════
 *  STEP 3 – Retain Hashtag Text
 *  Mirrors: retain_hashtag_text() in Python
 *  Strips '#' prefix, keeps the rest of the word.
 * ════════════════════════════════════════════════════════════ */
static void retain_hashtag_text(const char *text, char *out, int out_max)
{
    char buf[MAX_TEXT];
    strncpy(buf, text, MAX_TEXT - 1);
    buf[MAX_TEXT - 1] = '\0';

    out[0] = '\0';

    char *tok = strtok(buf, " ");
    int first = 1;

    while (tok != NULL) {
        if (!first) safe_strcat(out, " ", out_max);
        first = 0;

        if (tok[0] == '#' && tok[1] != '\0') {
            safe_strcat(out, tok + 1, out_max);   /* skip '#' */
        } else {
            safe_strcat(out, tok, out_max);
        }

        tok = strtok(NULL, " ");
    }
}


/* ════════════════════════════════════════════════════════════
 *  STEP 4 – Preserve Original Casing  (identity pass)
 *  Mirrors: preserve_casing() in Python
 * ════════════════════════════════════════════════════════════ */
static void preserve_casing(const char *text, char *out, int out_max)
{
    strncpy(out, text, (size_t)(out_max - 1));
    out[out_max - 1] = '\0';
}


/* ════════════════════════════════════════════════════════════
 *  STEP 5 – Remove Extra Spaces
 *  Mirrors: remove_extra_spaces() in Python
 *  Collapses any run of whitespace into one space.
 * ════════════════════════════════════════════════════════════ */
static void remove_extra_spaces(const char *text, char *out, int out_max)
{
    int i = 0, j = 0;
    int in_space = 0;
    int len = (int)strlen(text);

    /* Skip leading spaces */
    while (i < len && isspace((unsigned char)text[i])) i++;

    while (i < len && j < out_max - 1) {
        if (isspace((unsigned char)text[i])) {
            if (!in_space) {
                out[j++] = ' ';
                in_space = 1;
            }
        } else {
            out[j++] = text[i];
            in_space = 0;
        }
        i++;
    }

    /* Trim trailing space */
    if (j > 0 && out[j - 1] == ' ') j--;
    out[j] = '\0';
}


/* ════════════════════════════════════════════════════════════
 *  STEP 6 – Detect Character Elongation
 *  Mirrors: detect_elongation() + _compress_elongation() in Python
 *
 *  Scans each whitespace-separated token.
 *  If any character repeats >= ELONG_THRESHOLD times consecutively,
 *  compress the run to 1 char and append __ELONG__.
 *
 *  Special tokens (starting with __ or emoji_) are never modified.
 * ════════════════════════════════════════════════════════════ */

/* compress_token: works on a single token in-place.
 * Returns 1 if elongation was detected, 0 otherwise. */
static int compress_token(const char *tok, char *out_tok, int out_max)
{
    int len = (int)strlen(tok);
    int elongated = 0;
    int j = 0;
    int i = 0;

    if (len < ELONG_THRESHOLD) {
        strncpy(out_tok, tok, (size_t)(out_max - 1));
        out_tok[out_max - 1] = '\0';
        return 0;
    }

    while (i < len && j < out_max - 1) {
        char ch = tok[i];
        int  run = 1;
        /* Count how many consecutive identical chars follow */
        while (i + run < len &&
               tolower((unsigned char)tok[i + run]) == tolower((unsigned char)ch)) {
            run++;
        }
        if (run >= ELONG_THRESHOLD) {
            elongated = 1;
            out_tok[j++] = ch;   /* keep only one copy */
        } else {
            /* Copy the whole run verbatim */
            for (int k = 0; k < run && j < out_max - 1; k++)
                out_tok[j++] = tok[i + k];
        }
        i += run;
    }
    out_tok[j] = '\0';
    return elongated;
}

static void detect_elongation(const char *text, char *out, int out_max)
{
    char buf[MAX_TEXT];
    strncpy(buf, text, MAX_TEXT - 1);
    buf[MAX_TEXT - 1] = '\0';

    out[0] = '\0';

    char *tok = strtok(buf, " ");
    int first = 1;

    while (tok != NULL) {
        if (!first) safe_strcat(out, " ", out_max);
        first = 0;

        /* Never touch special sentinel / emoji tokens */
        int is_special = str_starts_with(tok, "__") ||
                         str_starts_with(tok, "emoji_");

        if (is_special) {
            safe_strcat(out, tok, out_max);
        } else {
            char compressed[MAX_TOKEN_LEN];
            int elong = compress_token(tok, compressed, MAX_TOKEN_LEN);
            safe_strcat(out, compressed, out_max);
            if (elong) safe_strcat(out, "__ELONG__", out_max);
        }

        tok = strtok(NULL, " ");
    }
}


/* ════════════════════════════════════════════════════════════
 *  STEP 7 – Tokenization
 *  Mirrors: tokenize() in Python
 *
 *  Rules:
 *   - Split on whitespace.
 *   - Leading/trailing punctuation is peeled off as its own token.
 *   - Special tokens (__…, emoji_…, …__ELONG__) are never split.
 * ════════════════════════════════════════════════════════════ */
static void tokenize(const char *text, TokenList *tl)
{
    tl->count = 0;

    char buf[MAX_TEXT];
    strncpy(buf, text, MAX_TEXT - 1);
    buf[MAX_TEXT - 1] = '\0';

    char *tok = strtok(buf, " ");

    while (tok != NULL && tl->count < MAX_TOKENS) {

        /* Special token? → add as-is */
        int is_special = str_starts_with(tok, "__")    ||
                         str_starts_with(tok, "emoji_");
        /* Elongation tag at end? → also add as-is */
        int elong_len   = (int)strlen("__ELONG__");
        int tok_len     = (int)strlen(tok);
        int has_elong   = (tok_len >= elong_len) &&
                          (strcmp(tok + tok_len - elong_len, "__ELONG__") == 0);

        if (is_special || has_elong) {
            strncpy(tl->tokens[tl->count], tok, MAX_TOKEN_LEN - 1);
            tl->tokens[tl->count][MAX_TOKEN_LEN - 1] = '\0';
            tl->count++;
            tok = strtok(NULL, " ");
            continue;
        }

        /* Peel leading punctuation */
        while (*tok && is_punct(*tok) && tl->count < MAX_TOKENS) {
            tl->tokens[tl->count][0] = *tok;
            tl->tokens[tl->count][1] = '\0';
            tl->count++;
            tok++;
        }

        /* Peel trailing punctuation: find end, walk back */
        int tlen = (int)strlen(tok);
        int trail_start = tlen;
        while (trail_start > 0 && is_punct((unsigned char)tok[trail_start - 1]))
            trail_start--;

        /* Add the core word (between the peeled puncts) */
        if (trail_start > 0 && tl->count < MAX_TOKENS) {
            int copy_len = trail_start < MAX_TOKEN_LEN - 1
                           ? trail_start : MAX_TOKEN_LEN - 1;
            strncpy(tl->tokens[tl->count], tok, (size_t)copy_len);
            tl->tokens[tl->count][copy_len] = '\0';
            tl->count++;
        }

        /* Add trailing punctuation tokens one-by-one */
        for (int k = trail_start; k < tlen && tl->count < MAX_TOKENS; k++) {
            tl->tokens[tl->count][0] = tok[k];
            tl->tokens[tl->count][1] = '\0';
            tl->count++;
        }

        tok = strtok(NULL, " ");
    }
}


/* ════════════════════════════════════════════════════════════
 *  FULL PIPELINE  –  preprocess()
 * ════════════════════════════════════════════════════════════ */
void preprocess(const char *input, PipelineResult *r)
{
    /* Stage 0: raw */
    strncpy(r->raw, input, MAX_TEXT - 1);
    r->raw[MAX_TEXT - 1] = '\0';

    /* Stage 1: normalize URLs / mentions */
    normalize_urls_and_mentions(r->raw, r->after_normalize, MAX_TEXT);

    /* Stage 2: map emojis */
    map_emojis(r->after_normalize, r->after_emoji, MAX_TEXT);

    /* Stage 3: retain hashtag text */
    retain_hashtag_text(r->after_emoji, r->after_hashtag, MAX_TEXT);

    /* Stage 4: preserve casing (identity) */
    preserve_casing(r->after_hashtag, r->after_casing, MAX_TEXT);

    /* Stage 5: remove extra spaces */
    remove_extra_spaces(r->after_casing, r->after_spaces, MAX_TEXT);

    /* Stage 6: detect elongation */
    detect_elongation(r->after_spaces, r->after_elongation, MAX_TEXT);

    /* Stage 7: tokenize */
    tokenize(r->after_elongation, &r->tokens);
}


/* ════════════════════════════════════════════════════════════
 *  PRINT HELPER
 * ════════════════════════════════════════════════════════════ */
void print_pipeline_result(const PipelineResult *r)
{
    printf("  %-22s: %s\n", "RAW",              r->raw);
    printf("  %-22s: %s\n", "AFTER NORMALIZE",  r->after_normalize);
    printf("  %-22s: %s\n", "AFTER EMOJI",      r->after_emoji);
    printf("  %-22s: %s\n", "AFTER HASHTAG",    r->after_hashtag);
    printf("  %-22s: %s\n", "AFTER CASING",     r->after_casing);
    printf("  %-22s: %s\n", "AFTER SPACES",     r->after_spaces);
    printf("  %-22s: %s\n", "AFTER ELONGATION", r->after_elongation);

    printf("  %-22s: [", "TOKENS");
    for (int i = 0; i < r->tokens.count; i++) {
        printf("'%s'", r->tokens.tokens[i]);
        if (i < r->tokens.count - 1) printf(", ");
    }
    printf("]\n");
}

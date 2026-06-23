/*
 * nlp_preprocessor.h
 * ─────────────────────────────────────────────────────────────
 * NLP Preprocessing Pipeline for Context-Aware Video Comment Analysis
 * Headers + shared types used by both the preprocessor and CSV reader.
 *
 * Allowed headers: <stdio.h>, <string.h>, <ctype.h>
 * ─────────────────────────────────────────────────────────────
 */

#ifndef NLP_PREPROCESSOR_H
#define NLP_PREPROCESSOR_H

#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* ── Buffer sizes ─────────────────────────────────────────── */
#define MAX_TEXT      4096   /* max characters in one comment  */
#define MAX_TOKEN_LEN  256   /* max characters in one token    */
#define MAX_TOKENS     512   /* max tokens per comment         */

/* ── Elongation run threshold ─────────────────────────────── */
#define ELONG_THRESHOLD 3

/* ── Token list returned by the tokenizer ────────────────── */
typedef struct {
    char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    int  count;
} TokenList;

/* ── Pipeline stage results (mirrors Python dict) ─────────── */
typedef struct {
    char raw            [MAX_TEXT];
    char after_normalize[MAX_TEXT];
    char after_emoji    [MAX_TEXT];
    char after_hashtag  [MAX_TEXT];
    char after_casing   [MAX_TEXT];   /* identity copy */
    char after_spaces   [MAX_TEXT];
    char after_elongation[MAX_TEXT];
    TokenList tokens;
} PipelineResult;

/* ── Public API ───────────────────────────────────────────── */
void preprocess(const char *input, PipelineResult *out);
void print_pipeline_result(const PipelineResult *r);

#endif /* NLP_PREPROCESSOR_H */

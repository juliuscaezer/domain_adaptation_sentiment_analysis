/*
 * csv_reader.c
 * ─────────────────────────────────────────────────────────────
 * Reads a CSV file of video comments and runs every comment
 * through the full NLP preprocessing pipeline.
 *
 * Expected CSV format (first row is header, ignored):
 *
 *   comment_id, video_id, video_title, timestamp, comment,
 *   sentiment, sarcasm_flag, confidence_level
 *
 *   Col 0: comment_id
 *   Col 1: video_id
 *   Col 2: video_title
 *   Col 3: timestamp
 *   Col 4: comment          ← fed into the NLP pipeline
 *   Col 5: sentiment        ← ground truth label (positive/negative/neutral)
 *   Col 6: sarcasm_flag     ← ground truth sarcasm (True/False)
 *   Col 7: confidence_level ← annotation confidence (high/medium/low)
 *
 * Allowed headers: <stdio.h>, <string.h>, <ctype.h>
 *
 * Build:
 *   gcc -o csv_reader csv_reader.c nlp_preprocessor.c -Wall -Wextra
 *
 * Usage:
 *   ./csv_reader comments.csv
 * ─────────────────────────────────────────────────────────────
 */

#include "nlp_preprocessor.h"

/* ── CSV parsing constants ─────────────────────────────────── */
#define MAX_LINE 8192 /* max bytes per CSV line          */
#define MAX_FIELDS 8  /* max columns expected in CSV     */
#define MAX_FIELD_LEN MAX_TEXT

/* ════════════════════════════════════════════════════════════
 *  CSV PARSER
 *  Handles:
 *   - Quoted fields  ("some, text")
 *   - Embedded commas inside quotes
 *   - Escaped quotes ("")
 *  Returns the number of fields parsed.
 * ════════════════════════════════════════════════════════════ */
static int parse_csv_line(const char *line,
                          char fields[][MAX_FIELD_LEN],
                          int max_fields)
{
    int field_count = 0;
    int fi = 0; /* index within current field */
    int in_quotes = 0;

    const char *p = line;

    /* initialise all fields to empty */
    for (int k = 0; k < max_fields; k++)
        fields[k][0] = '\0';

    while (*p != '\0' && *p != '\n' && *p != '\r')
    {

        if (in_quotes)
        {
            if (*p == '"')
            {
                /* Peek ahead: "" is an escaped quote inside the field */
                if (*(p + 1) == '"')
                {
                    if (fi < MAX_FIELD_LEN - 1)
                        fields[field_count][fi++] = '"';
                    p += 2;
                }
                else
                {
                    /* Closing quote */
                    in_quotes = 0;
                    p++;
                }
            }
            else
            {
                if (fi < MAX_FIELD_LEN - 1)
                    fields[field_count][fi++] = *p;
                p++;
            }
        }
        else
        {
            if (*p == '"')
            {
                in_quotes = 1;
                p++;
            }
            else if (*p == ',')
            {
                /* End of this field */
                fields[field_count][fi] = '\0';
                fi = 0;
                field_count++;
                if (field_count >= max_fields)
                    break;
                p++;
            }
            else
            {
                if (fi < MAX_FIELD_LEN - 1)
                    fields[field_count][fi++] = *p;
                p++;
            }
        }
    }

    /* Commit the last field */
    if (field_count < max_fields)
    {
        fields[field_count][fi] = '\0';
        field_count++;
    }

    return field_count;
}

/* ════════════════════════════════════════════════════════════
 *  PRINT SEPARATOR
 * ════════════════════════════════════════════════════════════ */
static void print_separator(void)
{
    printf("================================================================\n");
}

/* ════════════════════════════════════════════════════════════
 *  PROCESS ONE COMMENT
 *  Prints all metadata fields + the full pipeline result.
 *
 *  Parameters map directly to the 8 CSV columns:
 *   comment_id, video_id, video_title, timestamp,
 *   comment, sentiment, sarcasm_flag, confidence_level
 * ════════════════════════════════════════════════════════════ */
static void process_comment(const char *comment_id,
                            const char *video_id,
                            const char *video_title,
                            const char *timestamp,
                            const char *comment,
                            const char *sentiment,
                            const char *sarcasm_flag,
                            const char *confidence_level)
{
    print_separator();

    /* ── Metadata ── */
    printf("  Comment ID       : %s\n", comment_id);
    printf("  Video ID         : %s\n", video_id);
    printf("  Video Title      : %s\n", video_title);
    printf("  Timestamp        : %s\n", timestamp);

    /* ── Ground truth labels ── */
    printf("  Sentiment        : %s\n", sentiment);
    printf("  Sarcasm Flag     : %s\n", sarcasm_flag);
    printf("  Confidence Level : %s\n", confidence_level);
    printf("  ---\n");

    /* ── NLP pipeline ── */
    PipelineResult result;
    preprocess(comment, &result);
    print_pipeline_result(&result);
}

/* ════════════════════════════════════════════════════════════
 *  MAIN – open CSV, skip header, process each row
 * ════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{

    // Print the args
    for (int i = 0; i < argc; i++)
    {
        printf("Arg %d: %s\n", i, argv[i]);
    }

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <comments.csv>\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open file '%s'\n", argv[1]);
        return 1;
    }

    char line[MAX_LINE];
    char fields[MAX_FIELDS][MAX_FIELD_LEN];
    int row = 0;
    int processed = 0;

    printf("\n");
    printf("  NLP Preprocessing Pipeline  –  CSV Reader\n");
    printf("  File : %s\n", argv[1]);
    printf("\n");

    while (fgets(line, MAX_LINE, fp) != NULL)
    {

        /* Skip the header row */
        if (row == 0)
        {
            row++;
            continue;
        }

        /* Skip blank lines */
        int blank = 1;
        for (int k = 0; line[k] != '\0'; k++)
        {
            if (!isspace((unsigned char)line[k]))
            {
                blank = 0;
                break;
            }
        }
        if (blank)
        {
            row++;
            continue;
        }

        int nfields = parse_csv_line(line, fields, MAX_FIELDS);

        /*
         * Expected columns (0-indexed):
         *   0: comment_id
         *   1: video_id
         *   2: video_title
         *   3: timestamp
         *   4: comment          ← fed into the NLP pipeline
         *   5: sentiment        ← ground truth label
         *   6: sarcasm_flag     ← ground truth sarcasm
         *   7: confidence_level ← annotation confidence
         */
        if (nfields < 8)
        {
            fprintf(stderr,
                    "  [Warning] Row %d has only %d field(s) (expected 8), skipping.\n",
                    row, nfields);
            row++;
            continue;
        }

        process_comment(fields[0], fields[1], fields[2], fields[3],
                        fields[4], fields[5], fields[6], fields[7]);
        processed++;
        row++;
    }

    fclose(fp);

    print_separator();
    printf("\n  Done. Processed %d comment(s) from %d data row(s).\n\n",
           processed, row - 1);

    return 0;
}

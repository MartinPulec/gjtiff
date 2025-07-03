#include "shunting_yard.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>

enum { INIT_SZ =  32 };
#define is_operand_char(ch) (isalnum(ch) || (ch) == '_' || (ch) == '.')
#define operator_prec(ch)                                                      \
        ((ch) == ',' ? 1                                                       \
                     : (((ch) == '+' || (ch) == '-')                           \
                            ? 2                                                \
                            : (((ch) == '*' || (ch) == '/') ? 3 : 0)))

static void realloc_helper(void **var, int n, int req_sz, int *cur_sz) {
    if (req_sz <= *cur_sz) {
        return;
    }
    *cur_sz = req_sz * 2;
    *var = realloc(*var, (size_t) *cur_sz * n);
}

typedef enum {
    EXPECT_OPERAND,
    EXPECT_OPERATOR
} State;

// Shunting-Yard: infix → postfix
// expr      : input C-string, e.g. "(a+b)/(a-b)"
// output[]  : caller-allocated array of char* of size MAX_TOKENS
// *out_cnt  : returns the number of tokens filled in output[]
char **to_postfix(const char *expr, int *out_cnt) {
    int out_sz = INIT_SZ;
    char **output = (char **) malloc(out_sz * sizeof *output);
    int opst_len = INIT_SZ;
    char **op_stack = (char **)malloc(sizeof *op_stack * opst_len);
    int top_op = -1;
    int out_top = 0;
    int i = 0;
    const int n = (int)strlen(expr);

    State state = EXPECT_OPERAND;
    int parenthesis = 0;

    while (i < n) {
            realloc_helper((void **)&output, sizeof(char *), (out_top + 1),
                           &out_sz);
            realloc_helper((void **)&op_stack, sizeof(char *), (top_op + 2),
                           &opst_len);

            char c = expr[i++];
            if (isspace(c)) {
                    continue;
            }
            if (is_operand_char(c)) {
                    if (state == EXPECT_OPERATOR) {
                            fprintf(stderr, "Expecting operator!\n");
                            goto error;
                    }
                    state = EXPECT_OPERATOR;
                    // read an alphanumeric operand
                    int buflen = INIT_SZ;
                    char *buf = malloc(buflen);
                    int len = 0;
                    buf[len++] = c;
                    while (i < n && is_operand_char(expr[i])) {
                            realloc_helper((void **)&buf, sizeof(char), len + 2,
                                           &buflen);
                            buf[len++] = expr[i++];
                    }
                    buf[len] = '\0';
                    output[out_top++] = buf;
                    continue;
            }
            if (c == '(') {
                    if (state == EXPECT_OPERATOR) {
                            fprintf(
                                stderr,
                                "Left parenthsis where operator expected!\n");
                            goto error;
                    }
                    parenthesis++;
                    op_stack[++top_op] = strdup("(");
                    continue;
            }
            if (c == ')') {
                    if (state == EXPECT_OPERAND) {
                            fprintf(
                                stderr,
                                "Right parenthsis where operator expected!\n");
                            goto error;
                    }
                    // pop until '('
                    while (top_op >= 0 && strcmp(op_stack[top_op], "(") != 0) {
                            realloc_helper((void **)&output, sizeof(char *),
                                           (out_top + 1), &out_sz);
                            output[out_top++] = op_stack[top_op--];
                    }
                    if (top_op == -1) {
                            fprintf(stderr, "Error: unmatched ')'!\n");
                            goto error;
                    }
                    parenthesis--;
                    free(op_stack[top_op--]); // drop the "("
                    continue;
            }
            int prec = operator_prec(c);
            if (prec != 0) {
                    if (state == EXPECT_OPERAND) {
                            fprintf(stderr, "Expecting operand!\n");
                            goto error;
                    }
                    state = EXPECT_OPERAND;
                    char tmp[2] = {c, 0};
                    // pop any ops ≥ our precedence
                    while (top_op >= 0 && strcmp(op_stack[top_op], "(") != 0) {
                            char op2 = op_stack[top_op][0];
                            int p2 = operator_prec(op2);
                            if (p2 >= prec) {
                                    realloc_helper((void **)&output,
                                                   sizeof(char *),
                                                   (out_top + 1), &out_sz);
                                    output[out_top++] = op_stack[top_op--];
                            } else {
                                    break;
                            }
                    }
                    op_stack[++top_op] = strdup(tmp);
                    continue;
            }
            fprintf(stderr, "Unknown operator '%c'\n", c);
error:
            while (top_op >= 0) {
                    free(op_stack[top_op--]);
            }
            free((void *)op_stack);
            while (out_top > 0) {
                    free(output[--out_top]);
            }
            free((void *)output);
            return NULL;
    }

    if (parenthesis != 0) {
            fprintf(stderr, "Unmatched parenthesis!\n");
            goto error;
    }

    if (state == EXPECT_OPERAND) {
            fprintf(stderr, "Expecting operator!\n");
            goto error;
    }

    // drain remaining operators
    while (top_op >= 0) {
            realloc_helper((void **)&output, sizeof(char *), (out_top + 1),
                           &out_sz);
            output[out_top++] = op_stack[top_op--];
    }
    free((void *) op_stack);

    *out_cnt = out_top;
    return output;
}

#ifdef SHUNTING_DEMO
// demo
int main(int argc, char *argv[])
{
        const char *expr = argc == 1 ? "(b3+b8)/(b3-b8)" : argv[1];
        int np = 0;

        char **post = to_postfix(expr, &np);
        if (post == NULL) {
                return 1;
        }

        printf("Postfix: ");
        for (int i = 0; i < np; i++) {
                printf("%s ", post[i]);
                free(post[i]);
        }
        free(post);
        printf("\n");
        return 0;
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>

#define is_operand_char(ch) (isalnum(ch) || (ch) == '_' || (ch) == '.')

static void realloc_helper(void **var, int n, int req_sz, int *cur_sz) {
    if (req_sz <= *cur_sz) {
        return;
    }
    *cur_sz = req_sz * 2;
    *var = realloc(*var, (size_t) *cur_sz * n);
}

// Shunting-Yard: infix → postfix
// expr      : input C-string, e.g. "(a+b)/(a-b)"
// output[]  : caller-allocated array of char* of size MAX_TOKENS
// *out_cnt  : returns the number of tokens filled in output[]
char **to_postfix(const char *expr, int *out_cnt) {
    int out_sz = 32;
    char **output = malloc(out_sz * sizeof *output);
    int opst_len = 32;
    char **op_stack = malloc(sizeof *op_stack * opst_len);
    int  top_op   = -1;
    int  out_top  = 0;
    int  i        = 0, n = (int)strlen(expr);

    while (i < n) {
        realloc_helper((void **) &output, sizeof(char *), (out_top + 1), &out_sz);
        realloc_helper((void **) &op_stack, sizeof(char *), (top_op + 1), &opst_len);

        if (isspace(expr[i])) {
            i++;
        }
        else if (is_operand_char(expr[i])) {
            // read an alphanumeric operand
            int buflen = 32;
            char *buf = malloc(buflen);
            int  len = 0;
            while (i < n && is_operand_char(expr[i])) {
                realloc_helper((void **) &buf, sizeof(char), len + 2, &buflen);
                buf[len++] = expr[i++];
            }
            buf[len] = '\0';
            output[out_top++] = buf;
        }
        else {
            char c = expr[i++];
            if (c == '(') {
                op_stack[++top_op] = strdup("(");
            }
            else if (c == ')') {
                // pop until '('
                while (top_op >= 0 && strcmp(op_stack[top_op], "(") != 0) {
                        realloc_helper((void **)&output, sizeof(char *),
                                       (out_top + 1), &out_sz);
                        output[out_top++] = op_stack[top_op--];
                }
                free(op_stack[top_op--]);  // drop the "("
            }
            else if (strchr("+-*/,", c)) {
                int prec = (c=='+'||c=='-') ? 2 : (c==',' ? 1 : 3);
                char tmp[2] = { c, 0 };
                // pop any ops ≥ our precedence
                while (top_op >= 0 && strcmp(op_stack[top_op], "(") != 0) {
                    char op2 = op_stack[top_op][0];
                    int  p2  = (op2=='+'||op2=='-') ? 2 : (c==',' ? 1 : 3);
                    if (p2 >= prec) {
                        realloc_helper((void **)&output, sizeof(char *),
                                       (out_top + 1), &out_sz);
                        output[out_top++] = op_stack[top_op--];
                    } else break;
                }
                op_stack[++top_op] = strdup(tmp);
            }
            else {
                fprintf(stderr, "Unknown char '%c'\n", c);
                while (top_op >= 0) {
                        free(op_stack[top_op--]);
                }
                free(op_stack);
                while (out_top > 0) {
                        free(output[--out_top]);
                }
                free(output);
                return NULL;
            }
        }
    }

    // drain remaining operators
    while (top_op >= 0) {
            realloc_helper((void **) &output, sizeof(char *), (out_top + 1), &out_sz);
            output[out_top++] = op_stack[top_op--];
    }

    free(op_stack);

    *out_cnt = out_top;
    return output;
}

#ifdef SHUNTING_DEMO
// demo
int main(int argc, char *argv[]) {
    const char *expr = argc == 1 ? "(b3+b8)/(b3-b8)" : argv[1];
    int   np;

    char **post = to_postfix(expr,  &np);
    if (post == NULL) {
        return 1;
    }

    printf("Postfix: ");
    for (int i = 0; i < np; i++) {
        printf("%s ", post[i]);
        free(post[i]);
    }
    printf("\n");
    return 0;
}
#endif

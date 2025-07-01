#ifndef SHUNTING_YARD_H_AFC3E21D_6A47_478A_80F3_A409BCB09C32
#define SHUNTING_YARD_H_AFC3E21D_6A47_478A_80F3_A409BCB09C32

// Shunting-Yard: infix → postfix
// expr      : input C-string, e.g. "(a+b)/(a-b)"
// returns   : array of tokens in post-fix notation
// *out_cnt  : returns the number of tokens returned
#ifdef _cplusplus
extern "C"
#endif
char **to_postfix(const char *expr, int *out_cnt);

#endif // SHUNTING_YARD_H_AFC3E21D_6A47_478A_80F3_A409BCB09C32

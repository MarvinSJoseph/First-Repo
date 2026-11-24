/*
 * calc.c - A command-line scientific calculator written in standard C11.
 *
 * FEATURES
 *   - Arithmetic:      +  -  *  /  %  ^ (or **) with normal precedence and
 *                      parentheses, e.g.  (2 + 3) * 4 ^ 2 % 7
 *   - Logarithms:      ln(x), log10(x), log(x) (= log10), log(b, x), log_b(x)
 *                      e.g.  log(2, 8)  or  log_2(8)  -> 3
 *   - Combinatorics:   n!  fact(n)  nPr(n, r)  nCr(n, r)  (aliases P, C,
 *                      perm, comb)
 *   - Extras:          ans (previous result), memory (m+, m-, mr, mc, mem),
 *                      constants pi and e, sqrt(x), abs(x), pow(x, y),
 *                      help screen, one-shot mode from the command line.
 *
 * BUILD (the math library must be linked explicitly with gcc/clang: -lm)
 *   gcc -std=c11 -Wall -Wextra -pedantic -O2 -o calc calc.c -lm
 *
 * USAGE
 *   ./calc               interactive loop (type "help" for commands)
 *   ./calc "nCr(5, 2)"   evaluate one expression and exit (status 0 = ok)
 *
 * NUMERIC MODEL AND LIMITS
 *   All expression values are IEEE-754 doubles (about 15-16 significant
 *   digits, max ~1.8e308). Integers are represented exactly only up to
 *   2^53 = 9,007,199,254,740,992.
 *
 *   Factorials, permutations and combinations are computed twice:
 *     1. exactly, in unsigned long long (64-bit, max 18,446,744,073,709,551,615),
 *        with an overflow check on every multiplication, and
 *     2. approximately, in double.
 *   The double result is what the calculator uses. If the exact integer is
 *   larger than 2^53 (so the double may be rounded) the exact value is shown
 *   as a note; if it does not even fit in 64 bits, a note says the result is
 *   a floating-point approximation. Results beyond the double range (e.g.
 *   171! and up) are reported as overflow errors rather than printed as inf.
 *
 *   Reference points:  20! is the largest factorial that fits in 64 bits;
 *                      170! is the largest that fits in a double.
 */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------ */
/* Configuration constants                                                  */
/* ------------------------------------------------------------------------ */

#define LINE_MAX_LEN   1024   /* longest accepted input line               */
#define IDENT_MAX_LEN  32     /* longest identifier (function/constant)    */
#define MSG_LEN        256    /* size of error / warning message buffers   */
#define MAX_WARNINGS   4      /* warnings kept per evaluation              */
#define MAX_ARGS       2      /* most arguments any function takes         */

#define INT_TOLERANCE  1e-9   /* how close to an integer counts as integer */
#define MAX_EXACT_DBL  9007199254740992.0 /* 2^53: exact-integer limit     */
#define MAX_FACT_DBL   170ULL /* largest n with n! < DBL_MAX               */

/* M_PI is POSIX, not ISO C, so define our own constants. */
static const double CALC_PI = 3.14159265358979323846;
static const double CALC_E  = 2.71828182845904523536;

/* ------------------------------------------------------------------------ */
/* Status codes returned by every math operation                            */
/* ------------------------------------------------------------------------ */

typedef enum {
    CALC_OK = 0,
    CALC_ERR_DIV_ZERO,     /* x / 0, x % 0, 0 ^ negative                    */
    CALC_ERR_OVERFLOW,     /* result outside the range of double            */
    CALC_ERR_LOG_DOMAIN,   /* log of zero or a negative number              */
    CALC_ERR_LOG_BASE,     /* log base <= 0 or == 1                         */
    CALC_ERR_COMPLEX_POW,  /* negative base with a non-integer exponent     */
    CALC_ERR_NEG_SQRT,     /* square root of a negative number              */
    CALC_ERR_NOT_INTEGER,  /* combinatorics argument is not a whole number  */
    CALC_ERR_NEGATIVE,     /* combinatorics argument is negative            */
    CALC_ERR_R_GT_N,       /* nPr / nCr with r > n                          */
    CALC_ERR_TOO_LARGE     /* combinatorics argument above supported limit  */
} CalcStatus;

/* Human-readable explanation for each status code. */
static const char *status_message(CalcStatus st)
{
    switch (st) {
    case CALC_OK:              return "ok";
    case CALC_ERR_DIV_ZERO:    return "division by zero";
    case CALC_ERR_OVERFLOW:    return "result is too large to represent (overflow)";
    case CALC_ERR_LOG_DOMAIN:  return "logarithm is only defined for positive numbers";
    case CALC_ERR_LOG_BASE:    return "logarithm base must be positive and not equal to 1";
    case CALC_ERR_COMPLEX_POW: return "negative base with a non-integer exponent has no real result";
    case CALC_ERR_NEG_SQRT:    return "square root of a negative number has no real result";
    case CALC_ERR_NOT_INTEGER: return "argument must be a whole number";
    case CALC_ERR_NEGATIVE:    return "argument must not be negative";
    case CALC_ERR_R_GT_N:      return "r must not be greater than n";
    case CALC_ERR_TOO_LARGE:   return "argument is too large (limit is 2^53)";
    }
    return "unknown error";
}

/* ------------------------------------------------------------------------ */
/* Small numeric helpers                                                    */
/* ------------------------------------------------------------------------ */

/* True if x is (within a small tolerance) a whole number. */
static bool is_integer(double x)
{
    return isfinite(x) && fabs(x - round(x)) < INT_TOLERANCE;
}

/* Map a computed double to OK, or to OVERFLOW if it became inf/nan. */
static CalcStatus check_finite(double r)
{
    return isfinite(r) ? CALC_OK : CALC_ERR_OVERFLOW;
}

/* Multiply two unsigned long longs; returns false (and leaves *out alone)
 * if the product would overflow. */
static bool mul_u64(unsigned long long a, unsigned long long b,
                    unsigned long long *out)
{
    if (a != 0 && b > ULLONG_MAX / a)
        return false;
    *out = a * b;
    return true;
}

/* Greatest common divisor (Euclid). Used to keep nCr intermediates small. */
static unsigned long long gcd_u64(unsigned long long a, unsigned long long b)
{
    while (b != 0) {
        unsigned long long t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* ------------------------------------------------------------------------ */
/* Basic arithmetic                                                         */
/*   Each operation stores its result in *out and returns a status code.    */
/* ------------------------------------------------------------------------ */

static CalcStatus op_add(double a, double b, double *out)
{
    *out = a + b;
    return check_finite(*out);
}

static CalcStatus op_sub(double a, double b, double *out)
{
    *out = a - b;
    return check_finite(*out);
}

static CalcStatus op_mul(double a, double b, double *out)
{
    *out = a * b;
    return check_finite(*out);
}

static CalcStatus op_div(double a, double b, double *out)
{
    if (b == 0.0)
        return CALC_ERR_DIV_ZERO;
    *out = a / b;
    return check_finite(*out);
}

/* Floating-point remainder (fmod). Like C's %, the sign of the result
 * follows the dividend: -7 % 3 = -1. Works for non-integers: 5.5 % 2 = 1.5. */
static CalcStatus op_mod(double a, double b, double *out)
{
    if (b == 0.0)
        return CALC_ERR_DIV_ZERO;
    *out = fmod(a, b);
    return check_finite(*out);
}

static CalcStatus op_pow(double base, double exponent, double *out)
{
    if (base == 0.0 && exponent < 0.0)
        return CALC_ERR_DIV_ZERO;          /* 0^-n = 1/0^n */
    if (base < 0.0 && !is_integer(exponent))
        return CALC_ERR_COMPLEX_POW;       /* e.g. (-8)^0.5 */
    *out = pow(base, exponent);
    return check_finite(*out);
}

static CalcStatus op_sqrt(double x, double *out)
{
    if (x < 0.0)
        return CALC_ERR_NEG_SQRT;
    *out = sqrt(x);
    return CALC_OK;
}

static CalcStatus op_abs(double x, double *out)
{
    *out = fabs(x);
    return CALC_OK;
}

/* ------------------------------------------------------------------------ */
/* Logarithms                                                               */
/* ------------------------------------------------------------------------ */

static CalcStatus op_ln(double x, double *out)
{
    if (x <= 0.0)
        return CALC_ERR_LOG_DOMAIN;
    *out = log(x);
    return CALC_OK;
}

static CalcStatus op_log10(double x, double *out)
{
    if (x <= 0.0)
        return CALC_ERR_LOG_DOMAIN;
    *out = log10(x);
    return CALC_OK;
}

/* log_b(x) = ln(x) / ln(b), valid for x > 0, b > 0, b != 1. */
static CalcStatus op_log_base(double base, double x, double *out)
{
    if (base <= 0.0 || base == 1.0)
        return CALC_ERR_LOG_BASE;
    if (x <= 0.0)
        return CALC_ERR_LOG_DOMAIN;
    *out = log(x) / log(base);
    return CALC_OK;
}

/* ------------------------------------------------------------------------ */
/* Combinatorics                                                            */
/*   These take an optional `note` buffer that receives a precision warning */
/*   (exact value > 2^53, or value > 64 bits). note[0] == '\0' means none.  */
/* ------------------------------------------------------------------------ */

/* Validate that x is a usable count (whole, non-negative, <= 2^53) and
 * convert it to unsigned long long. */
static CalcStatus validate_count(double x, unsigned long long *out)
{
    if (!isfinite(x))
        return CALC_ERR_TOO_LARGE;
    if (x < 0.0)
        return CALC_ERR_NEGATIVE;
    if (!is_integer(x))
        return CALC_ERR_NOT_INTEGER;
    if (x > MAX_EXACT_DBL)
        return CALC_ERR_TOO_LARGE;
    *out = (unsigned long long)llround(x);
    return CALC_OK;
}

/* Validate an (n, r) pair for nPr / nCr. */
static CalcStatus validate_n_r(double n_in, double r_in,
                               unsigned long long *n, unsigned long long *r)
{
    CalcStatus st = validate_count(n_in, n);
    if (st != CALC_OK)
        return st;
    st = validate_count(r_in, r);
    if (st != CALC_OK)
        return st;
    if (*r > *n)
        return CALC_ERR_R_GT_N;
    return CALC_OK;
}

/* Choose the best double to return and write a precision note if needed.
 *   fits   - the exact value fit in unsigned long long
 *   exact  - the exact value (valid only if fits)
 *   approx - the floating-point approximation                           */
static double finish_combinatoric(bool fits, unsigned long long exact,
                                  double approx, char *note, size_t note_len)
{
    if (note != NULL && note_len > 0)
        note[0] = '\0';

    if (fits) {
        if ((double)exact > MAX_EXACT_DBL && note != NULL)
            snprintf(note, note_len,
                     "exact value is %llu (above 2^53, so the displayed "
                     "double may be rounded)", exact);
        return (double)exact;   /* correctly rounded from the exact value */
    }
    if (note != NULL)
        snprintf(note, note_len,
                 "result exceeds unsigned long long (max %llu); shown as a "
                 "floating-point approximation", ULLONG_MAX);
    return approx;
}

/* n! for whole n in [0, 170]. */
static CalcStatus comb_factorial(double n_in, double *out,
                                 char *note, size_t note_len)
{
    unsigned long long n;
    CalcStatus st = validate_count(n_in, &n);
    if (st != CALC_OK)
        return st;
    if (n > MAX_FACT_DBL)
        return CALC_ERR_OVERFLOW;            /* 171! > DBL_MAX */

    unsigned long long exact = 1;
    bool fits = true;
    double approx = 1.0;

    for (unsigned long long i = 2; i <= n; i++) {
        approx *= (double)i;
        if (fits && !mul_u64(exact, i, &exact))
            fits = false;                    /* keep going in double only */
    }
    *out = finish_combinatoric(fits, exact, approx, note, note_len);
    return CALC_OK;
}

/* nPr = n! / (n - r)! = n * (n-1) * ... * (n-r+1). */
static CalcStatus comb_npr(double n_in, double r_in, double *out,
                           char *note, size_t note_len)
{
    unsigned long long n, r;
    CalcStatus st = validate_n_r(n_in, r_in, &n, &r);
    if (st != CALC_OK)
        return st;

    unsigned long long exact = 1;
    bool fits = true;
    double approx = 1.0;

    for (unsigned long long i = 0; i < r; i++) {
        unsigned long long factor = n - i;
        approx *= (double)factor;
        if (!isfinite(approx))
            return CALC_ERR_OVERFLOW;        /* stops long loops early too */
        if (fits && !mul_u64(exact, factor, &exact))
            fits = false;
    }
    *out = finish_combinatoric(fits, exact, approx, note, note_len);
    return CALC_OK;
}

/* nCr = n! / (r! (n - r)!), computed multiplicatively without factorials:
 *   C = prod_{i=1..k} (n - k + i) / i,   where k = min(r, n - r).
 * After step i the running value is C(n-k+i, i), always an integer, so the
 * exact path divides out gcd(value, i) first to avoid intermediate overflow. */
static CalcStatus comb_ncr(double n_in, double r_in, double *out,
                           char *note, size_t note_len)
{
    unsigned long long n, r;
    CalcStatus st = validate_n_r(n_in, r_in, &n, &r);
    if (st != CALC_OK)
        return st;

    unsigned long long k = (r < n - r) ? r : n - r;   /* symmetry */
    unsigned long long exact = 1;
    bool fits = true;
    double approx = 1.0;

    for (unsigned long long i = 1; i <= k; i++) {
        unsigned long long top = n - k + i;
        approx = approx * (double)top / (double)i;
        if (!isfinite(approx))
            return CALC_ERR_OVERFLOW;
        if (fits) {
            /* exact * top is divisible by i; split i between the factors. */
            unsigned long long g = gcd_u64(exact, i);
            unsigned long long a = exact / g;
            unsigned long long b = top / (i / g);
            if (!mul_u64(a, b, &exact))
                fits = false;
        }
    }
    *out = finish_combinatoric(fits, exact, approx, note, note_len);
    return CALC_OK;
}

/* ------------------------------------------------------------------------ */
/* Calculator state: previous answer and memory register                    */
/* ------------------------------------------------------------------------ */

typedef struct {
    double ans;   /* result of the last successful evaluation */
    double mem;   /* memory register (m+, m-, mr, mc)          */
} CalcState;

/* ------------------------------------------------------------------------ */
/* Recursive-descent expression parser                                      */
/*                                                                          */
/* Grammar (lowest to highest precedence):                                  */
/*   expr    := term   { ('+' | '-') term }                                 */
/*   term    := unary  { ('*' | '/' | '%') unary }                          */
/*   unary   := ('+' | '-') unary | power                                   */
/*   power   := postfix [ ('^' | '**') unary ]      (right-associative)     */
/*   postfix := primary { '!' }                                             */
/*   primary := number | constant | function '(' args ')' | '(' expr ')'    */
/*                                                                          */
/* Consequences: -2^2 = -4, 2^3^2 = 2^9 = 512, 2^-1 = 0.5, 3!^2 = 36.       */
/* Parsing stops at the first error; later calls become no-ops.             */
/* ------------------------------------------------------------------------ */

typedef struct {
    const char      *pos;                 /* current read position        */
    const CalcState *state;               /* for ans / mem lookups        */
    bool             failed;              /* an error has been recorded   */
    char             error[MSG_LEN];
    int              nwarn;
    char             warnings[MAX_WARNINGS][MSG_LEN];
} Parser;

/* Record the first error (later errors are ignored). */
static void parser_fail(Parser *p, const char *fmt, ...)
{
    if (p->failed)
        return;
    p->failed = true;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(p->error, sizeof p->error, fmt, ap);
    va_end(ap);
}

static void parser_warn(Parser *p, const char *msg)
{
    if (msg[0] != '\0' && p->nwarn < MAX_WARNINGS) {
        snprintf(p->warnings[p->nwarn], MSG_LEN, "%s", msg);
        p->nwarn++;
    }
}

/* Convert an operation's status into a parser error. Returns true on OK. */
static bool check_status(Parser *p, CalcStatus st, const char *what)
{
    if (st == CALC_OK)
        return true;
    parser_fail(p, "%s: %s", what, status_message(st));
    return false;
}

static void skip_ws(Parser *p)
{
    while (isspace((unsigned char)*p->pos))
        p->pos++;
}

/* If the next non-space character is c, consume it and return true. */
static bool accept(Parser *p, char c)
{
    skip_ws(p);
    if (*p->pos == c) {
        p->pos++;
        return true;
    }
    return false;
}

static double parse_expr(Parser *p);
static double parse_unary(Parser *p);

/* Built-in functions. min/max arity let log() take 1 or 2 arguments. */
typedef enum {
    FN_LN, FN_LOG10, FN_LOG, FN_SQRT, FN_ABS, FN_POW,
    FN_FACT, FN_NPR, FN_NCR
} FnId;

typedef struct {
    const char *name;
    int         min_args;
    int         max_args;
    FnId        id;
} FnInfo;

static const FnInfo FUNCTIONS[] = {
    { "ln",        1, 1, FN_LN    },
    { "log10",     1, 1, FN_LOG10 },
    { "log",       1, 2, FN_LOG   },   /* log(x) = log10, log(b, x) */
    { "sqrt",      1, 1, FN_SQRT  },
    { "abs",       1, 1, FN_ABS   },
    { "pow",       2, 2, FN_POW   },
    { "fact",      1, 1, FN_FACT  },
    { "factorial", 1, 1, FN_FACT  },
    { "nPr",       2, 2, FN_NPR   },
    { "npr",       2, 2, FN_NPR   },
    { "P",         2, 2, FN_NPR   },
    { "perm",      2, 2, FN_NPR   },
    { "nCr",       2, 2, FN_NCR   },
    { "ncr",       2, 2, FN_NCR   },
    { "C",         2, 2, FN_NCR   },
    { "comb",      2, 2, FN_NCR   },
};

static const FnInfo *find_function(const char *name)
{
    for (size_t i = 0; i < sizeof FUNCTIONS / sizeof FUNCTIONS[0]; i++)
        if (strcmp(FUNCTIONS[i].name, name) == 0)
            return &FUNCTIONS[i];
    return NULL;
}

/* Parse "( expr {, expr} )" into args[]. Returns the argument count, or -1
 * on error. */
static int parse_args(Parser *p, const char *fname, double args[MAX_ARGS])
{
    if (!accept(p, '(')) {
        parser_fail(p, "expected '(' after function name '%s'", fname);
        return -1;
    }
    int count = 0;
    do {
        double v = parse_expr(p);
        if (p->failed)
            return -1;
        if (count >= MAX_ARGS) {
            parser_fail(p, "%s: too many arguments", fname);
            return -1;
        }
        args[count++] = v;
    } while (accept(p, ','));

    if (!accept(p, ')')) {
        parser_fail(p, "%s: expected ')' to close the argument list", fname);
        return -1;
    }
    return count;
}

/* Run a built-in function on already-parsed arguments. */
static double call_function(Parser *p, const FnInfo *fn,
                            const double *a, int argc)
{
    double r = 0.0;
    char note[MSG_LEN] = "";
    CalcStatus st = CALC_OK;

    switch (fn->id) {
    case FN_LN:    st = op_ln(a[0], &r);    break;
    case FN_LOG10: st = op_log10(a[0], &r); break;
    case FN_LOG:
        st = (argc == 1) ? op_log10(a[0], &r) : op_log_base(a[0], a[1], &r);
        break;
    case FN_SQRT:  st = op_sqrt(a[0], &r);  break;
    case FN_ABS:   st = op_abs(a[0], &r);   break;
    case FN_POW:   st = op_pow(a[0], a[1], &r); break;
    case FN_FACT:  st = comb_factorial(a[0], &r, note, sizeof note); break;
    case FN_NPR:   st = comb_npr(a[0], a[1], &r, note, sizeof note); break;
    case FN_NCR:   st = comb_ncr(a[0], a[1], &r, note, sizeof note); break;
    }
    if (!check_status(p, st, fn->name))
        return 0.0;
    parser_warn(p, note);
    return r;
}

/* Handle a name: constant, memory, log_<base>(x), or function call. */
static double parse_identifier(Parser *p, const char *name)
{
    /* Constants and stored values. */
    if (strcmp(name, "ans") == 0) return p->state->ans;
    if (strcmp(name, "mem") == 0) return p->state->mem;
    if (strcmp(name, "pi")  == 0) return CALC_PI;
    if (strcmp(name, "e")   == 0) return CALC_E;

    double args[MAX_ARGS];

    /* Custom-base logarithm written as log_<base>(x), e.g. log_2(8).
     * (Use log(b, x) for non-integer or computed bases.) */
    if (strncmp(name, "log_", 4) == 0 && name[4] != '\0') {
        char *end;
        double base = strtod(name + 4, &end);
        if (*end != '\0' || *p->pos == '.') {   /* e.g. log_x or log_2.5 */
            parser_fail(p, "invalid log base in '%s' (use log(b, x) for "
                           "non-integer bases)", name);
            return 0.0;
        }
        int argc = parse_args(p, name, args);
        if (argc < 0)
            return 0.0;
        if (argc != 1) {
            parser_fail(p, "%s: expected 1 argument, got %d", name, argc);
            return 0.0;
        }
        double r = 0.0;
        check_status(p, op_log_base(base, args[0], &r), name);
        return r;
    }

    const FnInfo *fn = find_function(name);
    if (fn == NULL) {
        parser_fail(p, "unknown name '%s' (type 'help' for a list)", name);
        return 0.0;
    }
    int argc = parse_args(p, name, args);
    if (argc < 0)
        return 0.0;
    if (argc < fn->min_args || argc > fn->max_args) {
        if (fn->min_args == fn->max_args)
            parser_fail(p, "%s: expected %d argument(s), got %d",
                        name, fn->min_args, argc);
        else
            parser_fail(p, "%s: expected %d to %d arguments, got %d",
                        name, fn->min_args, fn->max_args, argc);
        return 0.0;
    }
    return call_function(p, fn, args, argc);
}

static double parse_primary(Parser *p)
{
    skip_ws(p);
    if (p->failed)
        return 0.0;

    char c = *p->pos;

    /* Number literal, e.g. 42, 3.14, .5, 6.02e23 */
    if (isdigit((unsigned char)c) || c == '.') {
        char *end;
        errno = 0;
        double v = strtod(p->pos, &end);
        if (end == p->pos) {
            parser_fail(p, "invalid number");
            return 0.0;
        }
        if (errno == ERANGE && isinf(v)) {
            parser_fail(p, "number literal is too large");
            return 0.0;
        }
        p->pos = end;
        return v;
    }

    /* Parenthesised sub-expression. */
    if (c == '(') {
        p->pos++;
        double v = parse_expr(p);
        if (!accept(p, ')'))
            parser_fail(p, "missing closing ')'");
        return v;
    }

    /* Identifier: [A-Za-z_][A-Za-z0-9_]* */
    if (isalpha((unsigned char)c) || c == '_') {
        char name[IDENT_MAX_LEN];
        size_t len = 0;
        while (isalnum((unsigned char)*p->pos) || *p->pos == '_') {
            if (len + 1 >= sizeof name) {
                parser_fail(p, "name is too long");
                return 0.0;
            }
            name[len++] = *p->pos++;
        }
        name[len] = '\0';
        return parse_identifier(p, name);
    }

    if (c == '\0')
        parser_fail(p, "unexpected end of expression");
    else
        parser_fail(p, "unexpected character '%c'", c);
    return 0.0;
}

/* postfix := primary { '!' } */
static double parse_postfix(Parser *p)
{
    double v = parse_primary(p);
    while (!p->failed && accept(p, '!')) {
        char note[MSG_LEN] = "";
        if (!check_status(p, comb_factorial(v, &v, note, sizeof note), "!"))
            return 0.0;
        parser_warn(p, note);
    }
    return v;
}

/* power := postfix [ ('^' | '**') unary ]  -- right-associative because the
 * exponent is parsed by parse_unary, which itself reaches parse_power. */
static double parse_power(Parser *p)
{
    double base = parse_postfix(p);
    if (p->failed)
        return 0.0;

    skip_ws(p);
    bool is_pow = false;
    if (p->pos[0] == '^') {
        p->pos += 1;
        is_pow = true;
    } else if (p->pos[0] == '*' && p->pos[1] == '*') {
        p->pos += 2;
        is_pow = true;
    }
    if (!is_pow)
        return base;

    double exponent = parse_unary(p);
    if (p->failed)
        return 0.0;
    double r = 0.0;
    check_status(p, op_pow(base, exponent, &r), "^");
    return r;
}

/* unary := ('+' | '-') unary | power */
static double parse_unary(Parser *p)
{
    if (p->failed)
        return 0.0;
    if (accept(p, '-'))
        return -parse_unary(p);
    if (accept(p, '+'))
        return parse_unary(p);
    return parse_power(p);
}

/* term := unary { ('*' | '/' | '%') unary } */
static double parse_term(Parser *p)
{
    double left = parse_unary(p);
    while (!p->failed) {
        skip_ws(p);
        char op = *p->pos;
        if (op != '*' && op != '/' && op != '%')
            break;
        p->pos++;
        double right = parse_unary(p);
        if (p->failed)
            break;

        CalcStatus st;
        if (op == '*')      st = op_mul(left, right, &left);
        else if (op == '/') st = op_div(left, right, &left);
        else                st = op_mod(left, right, &left);

        char what[2] = { op, '\0' };
        if (!check_status(p, st, what))
            break;
    }
    return left;
}

/* expr := term { ('+' | '-') term } */
static double parse_expr(Parser *p)
{
    double left = parse_term(p);
    while (!p->failed) {
        skip_ws(p);
        char op = *p->pos;
        if (op != '+' && op != '-')
            break;
        p->pos++;
        double right = parse_term(p);
        if (p->failed)
            break;

        CalcStatus st = (op == '+') ? op_add(left, right, &left)
                                    : op_sub(left, right, &left);
        char what[2] = { op, '\0' };
        if (!check_status(p, st, what))
            break;
    }
    return left;
}

/* Evaluate a complete expression. Returns true on success and stores the
 * value in *result. Errors and warnings are left in *p for the caller. */
static bool evaluate(const char *text, const CalcState *state,
                     Parser *p, double *result)
{
    memset(p, 0, sizeof *p);
    p->pos = text;
    p->state = state;

    double v = parse_expr(p);
    skip_ws(p);
    if (!p->failed && *p->pos != '\0')
        parser_fail(p, "unexpected '%c' (missing operator?)", *p->pos);
    if (!p->failed && !isfinite(v))
        parser_fail(p, "%s", status_message(CALC_ERR_OVERFLOW));
    if (p->failed)
        return false;

    *result = (v == 0.0) ? 0.0 : v;    /* normalise -0 to 0 */
    return true;
}

/* ------------------------------------------------------------------------ */
/* Output and user interface                                                */
/* ------------------------------------------------------------------------ */

/* Print integers without a decimal point (while exact), others in %.15g. */
static void print_number(double v)
{
    if (is_integer(v) && fabs(v) < 1e15)
        printf("%.0f", round(v));
    else
        printf("%.15g", v);
}

static void print_help(void)
{
    puts(
        "Enter an expression, or one of the commands below.\n"
        "\n"
        "Operators (highest precedence first):\n"
        "  n!                 factorial (postfix)\n"
        "  x ^ y, x ** y      power (right-associative: 2^3^2 = 512)\n"
        "  -x, +x             unary sign (-2^2 = -4)\n"
        "  x * y, x / y, x % y  multiply, divide, remainder (fmod)\n"
        "  x + y, x - y       add, subtract\n"
        "  ( ... )            grouping\n"
        "\n"
        "Functions:\n"
        "  ln(x)  log10(x)  log(x)=log10(x)  log(b, x)  log_b(x) e.g. log_2(8)\n"
        "  sqrt(x)  abs(x)  pow(x, y)\n"
        "  fact(n)            n!             (0 <= n <= 170)\n"
        "  nPr(n, r)  P(n, r)  perm(n, r)    permutations n!/(n-r)!\n"
        "  nCr(n, r)  C(n, r)  comb(n, r)    combinations n!/(r!(n-r)!)\n"
        "\n"
        "Values:  ans (last result)  mem (memory)  pi  e\n"
        "\n"
        "Commands:\n"
        "  m+ / m-            add / subtract ans to / from memory\n"
        "  mr                 recall memory (sets ans = mem)\n"
        "  mc                 clear memory\n"
        "  help               show this text\n"
        "  quit, exit, q      leave the calculator\n"
        "\n"
        "Examples:  2 + 3 * 4   (1 + 2)^3 % 5   log(2, 1024)   10!\n"
        "           nCr(52, 5)   ans / 2   ln(e^3)   sqrt(ans) * pi");
}

/* Trim leading and trailing whitespace in place; returns the trimmed start. */
static char *trim(char *s)
{
    while (isspace((unsigned char)*s))
        s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1]))
        *--end = '\0';
    return s;
}

/* Case-insensitive string equality (ASCII). */
static bool equals_ci(const char *a, const char *b)
{
    for (; *a && *b; a++, b++)
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return false;
    return *a == *b;
}

/* Evaluate one expression, print the result or error, update ans. */
static bool run_expression(const char *text, CalcState *state)
{
    Parser p;
    double result;

    if (!evaluate(text, state, &p, &result)) {
        printf("error: %s\n", p.error);
        return false;
    }
    printf("= ");
    print_number(result);
    putchar('\n');
    for (int i = 0; i < p.nwarn; i++)
        printf("  note: %s\n", p.warnings[i]);
    state->ans = result;
    return true;
}

/* Handle a built-in command. Returns true if `line` was a command.
 * Sets *quit when the user asks to leave. */
static bool run_command(const char *line, CalcState *state, bool *quit)
{
    if (equals_ci(line, "quit") || equals_ci(line, "exit") ||
        equals_ci(line, "q")) {
        *quit = true;
    } else if (equals_ci(line, "help") || equals_ci(line, "?")) {
        print_help();
    } else if (equals_ci(line, "m+") || equals_ci(line, "m-")) {
        double r;
        CalcStatus st = (line[1] == '+') ? op_add(state->mem, state->ans, &r)
                                         : op_sub(state->mem, state->ans, &r);
        if (st != CALC_OK) {
            printf("error: memory: %s\n", status_message(st));
        } else {
            state->mem = r;
            printf("mem = ");
            print_number(state->mem);
            putchar('\n');
        }
    } else if (equals_ci(line, "mr")) {
        state->ans = state->mem;
        printf("= ");
        print_number(state->mem);
        putchar('\n');
    } else if (equals_ci(line, "mc")) {
        state->mem = 0.0;
        puts("memory cleared");
    } else {
        return false;
    }
    return true;
}

/* Read one line into buf. Returns false at end of input. Over-long lines
 * are discarded entirely and reported, so they are never half-evaluated. */
static bool read_line(char *buf, size_t size, bool *too_long)
{
    *too_long = false;
    if (fgets(buf, (int)size, stdin) == NULL)
        return false;

    char *nl = strchr(buf, '\n');
    if (nl != NULL) {
        *nl = '\0';
    } else if (!feof(stdin)) {
        int ch;
        while ((ch = getchar()) != '\n' && ch != EOF)
            ;                                 /* drain rest of the line */
        *too_long = true;
    }
    return true;
}

static void repl(CalcState *state)
{
    char buf[LINE_MAX_LEN];

    puts("C11 Scientific Calculator - type 'help' for usage, 'quit' to exit.");
    for (;;) {
        printf("> ");
        fflush(stdout);

        bool too_long;
        if (!read_line(buf, sizeof buf, &too_long)) {
            putchar('\n');                    /* EOF (Ctrl-D / Ctrl-Z) */
            break;
        }
        if (too_long) {
            printf("error: input longer than %d characters ignored\n",
                   LINE_MAX_LEN - 2);
            continue;
        }

        char *line = trim(buf);
        if (*line == '\0')
            continue;

        bool quit = false;
        if (run_command(line, state, &quit)) {
            if (quit)
                break;
            continue;
        }
        run_expression(line, state);
    }
}

int main(int argc, char *argv[])
{
    CalcState state = { 0.0, 0.0 };

    /* One-shot mode: join all arguments into one expression. */
    if (argc > 1) {
        char expr[LINE_MAX_LEN] = "";
        size_t used = 0;
        for (int i = 1; i < argc; i++) {
            int n = snprintf(expr + used, sizeof expr - used, "%s%s",
                             (i > 1) ? " " : "", argv[i]);
            if (n < 0 || (size_t)n >= sizeof expr - used) {
                fprintf(stderr, "error: expression is too long\n");
                return EXIT_FAILURE;
            }
            used += (size_t)n;
        }
        return run_expression(expr, &state) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    repl(&state);
    return EXIT_SUCCESS;
}

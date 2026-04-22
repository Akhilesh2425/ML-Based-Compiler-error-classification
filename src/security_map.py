
RISK_NONE   = "None"
RISK_LOW    = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH   = "High"
RISK_CRIT   = "Critical"

LEXICAL_SECURITY = {
    "risk"        : RISK_NONE,
    "risk_color"  : "green",
    "cwe"         : None,
    "description" : "Lexical errors are caught during tokenization. "
                    "The source file is rejected before any code is generated, "
                    "so no security vulnerability can result.",
    "recommendation": "Fix the invalid character, token, or literal and recompile.",
}

SYNTAX_SECURITY = {
    "risk"        : RISK_LOW,
    "risk_color"  : "yellow",
    "cwe"         : "CWE-707",
    "description" : "Syntax errors prevent compilation in standard cases. "
                    "However, missing braces around if/else bodies (dangling else) "
                    "or macro expansion issues can introduce logic flaws if the "
                    "code is corrected carelessly.",
    "recommendation": "Carefully review the surrounding logic when adding missing "
                      "braces or semicolons — the fix can accidentally change "
                      "control flow.",
    "examples": [
        {
            "pattern"    : "missing brace if else",
            "risk"       : RISK_MEDIUM,
            "cwe"        : "CWE-483",
            "name"       : "Incorrect Block Delimitation",
            "description": "Missing braces around if/else can cause unintended "
                           "statements to execute or be skipped.",
        },
        {
            "pattern"    : "macro",
            "risk"       : RISK_MEDIUM,
            "cwe"        : "CWE-1025",
            "name"       : "Macro Expansion Side Effects",
            "description": "Syntax errors inside macros can lead to unexpected "
                           "token pasting or expression evaluation order.",
        },
    ],
}


SEMANTIC_RULES = [
    
    {
        "patterns"      : ["array subscript", "array bounds", "above array bounds",
                           "below array bounds", "buffer"],
        "risk"          : RISK_CRIT,
        "cwe"           : "CWE-125 / CWE-787",
        "name"          : "Out-of-Bounds Read / Write (Buffer Overflow)",
        "description"   : "Array index is out of bounds. Reading past the end of a "
                          "buffer can leak sensitive data. Writing past the end can "
                          "corrupt memory, overwrite return addresses, and allow "
                          "arbitrary code execution.",
        "recommendation": "Always validate array indices against the array length. "
                          "Use safe functions (strncpy, snprintf) instead of strcpy/sprintf. "
                          "Consider enabling compiler sanitizers: -fsanitize=address.",
    },


    {
        "patterns"      : ["null pointer", "dereferencing", "dereference",
                           "pointer to incomplete", "incompatible pointer"],
        "risk"          : RISK_HIGH,
        "cwe"           : "CWE-476",
        "name"          : "NULL Pointer Dereference",
        "description"   : "A pointer may be NULL when dereferenced. This causes a "
                          "segmentation fault (denial of service). In some embedded "
                          "or kernel contexts, NULL dereference can be exploited for "
                          "privilege escalation.",
        "recommendation": "Always check if malloc/calloc/fopen returns NULL before "
                          "using the pointer. Use defensive NULL checks.",
    },
    {
        "patterns"      : ["use after free", "dangling pointer", "freed"],
        "risk"          : RISK_CRIT,
        "cwe"           : "CWE-416",
        "name"          : "Use After Free",
        "description"   : "Memory is used after it has been freed. Attackers can "
                          "reclaim the freed memory and control what is placed there, "
                          "leading to arbitrary code execution.",
        "recommendation": "Set pointers to NULL immediately after free(). "
                          "Use tools like Valgrind or AddressSanitizer to detect "
                          "use-after-free bugs.",
    },
    {
        "patterns"      : ["memory leak", "not freed", "allocated but"],
        "risk"          : RISK_MEDIUM,
        "cwe"           : "CWE-401",
        "name"          : "Memory Leak",
        "description"   : "Allocated memory is never freed. Repeated leaks can "
                          "exhaust heap memory, causing denial of service.",
        "recommendation": "Every malloc() must have a corresponding free(). "
                          "Use tools like Valgrind to find leaks.",
    },


    {
        "patterns"      : ["integer overflow", "overflow in expression",
                           "integer constant overflow"],
        "risk"          : RISK_HIGH,
        "cwe"           : "CWE-190",
        "name"          : "Integer Overflow",
        "description"   : "An arithmetic operation produces a value too large for "
                          "its type, wrapping around to a small or negative number. "
                          "This can bypass length checks and lead to buffer overflows.",
        "recommendation": "Use unsigned arithmetic carefully. Validate inputs before "
                          "arithmetic. Use safe integer libraries or compiler flags "
                          "like -ftrapv.",
    },
    {
        "patterns"      : ["division by zero", "divide by zero"],
        "risk"          : RISK_MEDIUM,
        "cwe"           : "CWE-369",
        "name"          : "Division by Zero",
        "description"   : "Division by a zero value causes a crash (SIGFPE), "
                          "which can be triggered by attacker-controlled input, "
                          "resulting in denial of service.",
        "recommendation": "Always validate the divisor is non-zero before division, "
                          "especially when using user-supplied values.",
    },
    {
        "patterns"      : ["signed", "unsigned", "signedness",
                           "comparison of signed and unsigned"],
        "risk"          : RISK_MEDIUM,
        "cwe"           : "CWE-195",
        "name"          : "Signed/Unsigned Comparison Error",
        "description"   : "Comparing signed and unsigned integers can produce "
                          "unexpected results. A negative signed value compared to "
                          "an unsigned value may be treated as a very large number, "
                          "bypassing length or bounds checks.",
        "recommendation": "Explicitly cast values to a consistent type before "
                          "comparison. Prefer size_t for sizes and lengths.",
    },


    {
        "patterns"      : ["incompatible types", "type mismatch",
                           "invalid conversion", "cannot convert"],
        "risk"          : RISK_MEDIUM,
        "cwe"           : "CWE-704",
        "name"          : "Type Confusion / Incorrect Type Conversion",
        "description"   : "Incorrect type conversions can cause data to be "
                          "misinterpreted — a float treated as an int, or a "
                          "pointer cast to an integer — potentially leading to "
                          "information disclosure or memory corruption.",
        "recommendation": "Use explicit casts only when safe. Avoid casting between "
                          "incompatible pointer types. Review all implicit conversions.",
    },
    {
        "patterns"      : ["assignment makes integer from pointer",
                           "assignment makes pointer from integer",
                           "cast from pointer to integer"],
        "risk"          : RISK_HIGH,
        "cwe"           : "CWE-587",
        "name"          : "Assignment of a Fixed Address to a Pointer",
        "description"   : "Assigning an integer to a pointer (or vice versa) without "
                          "a proper cast can corrupt pointer values, leading to "
                          "invalid memory accesses.",
        "recommendation": "Never assign raw integers to pointers. Use proper casting "
                          "and ensure pointer arithmetic is intentional.",
    },


    {
        "patterns"      : ["format string", "format argument", "format specifier",
                           "too few arguments to function 'printf'",
                           "too few arguments to function 'fprintf'",
                           "too few arguments to function 'sprintf'"],
        "risk"          : RISK_CRIT,
        "cwe"           : "CWE-134",
        "name"          : "Uncontrolled Format String",
        "description"   : "If a user-controlled string is passed as a format "
                          "argument to printf/sprintf/fprintf without a format "
                          "specifier, an attacker can read from or write to "
                          "arbitrary memory locations.",
        "recommendation": "Always use a literal format string: printf('%s', input) "
                          "instead of printf(input). Enable -Wformat-security.",
    },
    
    {
        "patterns"      : ["implicit declaration of function",
                           "implicit function declaration"],
        "risk"          : RISK_MEDIUM,
        "cwe"           : "CWE-242",
        "name"          : "Use of Implicitly Declared Function",
        "description"   : "In older C standards, calling an undeclared function "
                          "assumes it returns int. If the function actually returns "
                          "a pointer, the truncation can corrupt the pointer value, "
                          "leading to invalid memory accesses.",
        "recommendation": "Always include the correct header for every function used. "
                          "Compile with -Wimplicit-function-declaration -Werror.",
    },
    {
        "patterns"      : ["undeclared", "undeclared identifier",
                           "use of undeclared"],
        "risk"          : RISK_LOW,
        "cwe"           : "CWE-457",
        "name"          : "Use of Uninitialized Variable",
        "description"   : "Using an undeclared or uninitialized variable can cause "
                          "the program to read garbage values from the stack, "
                          "potentially leaking sensitive data.",
        "recommendation": "Always initialize variables at declaration. "
                          "Compile with -Wuninitialized.",
    },


    {
        "patterns"      : ["too many arguments", "too few arguments"],
        "risk"          : RISK_LOW,
        "cwe"           : "CWE-628",
        "name"          : "Incorrect Function Argument Count",
        "description"   : "Passing wrong number of arguments can cause the function "
                          "to read garbage values from the stack, producing "
                          "unpredictable behavior.",
        "recommendation": "Match the function signature exactly. "
                          "Review the function prototype in the header file.",
    },


    {
        "patterns"      : ["conflicting types", "redeclaration", "redefinition"],
        "risk"          : RISK_LOW,
        "cwe"           : "CWE-694",
        "name"          : "Use of Multiple Declarations with Different Types",
        "description"   : "When a function or variable is declared with conflicting "
                          "types, the behavior is undefined. Different translation "
                          "units may use different type assumptions, corrupting data.",
        "recommendation": "Ensure all declarations of the same symbol use identical "
                          "types. Use a shared header file.",
    },
    
    {
        "patterns"      : ["void expression", "invalid use of void",
                           "return makes integer from pointer"],
        "risk"          : RISK_LOW,
        "cwe"           : "CWE-252",
        "name"          : "Unchecked Return Value",
        "description"   : "Ignoring or misusing a function's return value (especially "
                          "error codes from malloc, fopen, read, write) means errors "
                          "go undetected, potentially causing the program to continue "
                          "with invalid state.",
        "recommendation": "Always check return values of functions that can fail. "
                          "Use -Wunused-result compiler flag.",
    },
]


SEMANTIC_DEFAULT = {
    "risk"          : RISK_LOW,
    "risk_color"    : "yellow",
    "cwe"           : "CWE-710",
    "name"          : "General Semantic Error",
    "description"   : "This semantic error may indicate incorrect type usage, "
                      "undefined behavior, or logic issues that could lead to "
                      "unpredictable program behavior under certain inputs.",
    "recommendation": "Review the flagged code carefully and ensure all types, "
                      "declarations, and function signatures are correct.",
}

RISK_COLORS = {
    RISK_NONE  : "green",
    RISK_LOW   : "yellow",
    RISK_MEDIUM: "orange",
    RISK_HIGH  : "red",
    RISK_CRIT  : "red",
}



def get_security_info(label: str, error_message: str) -> dict:
    
    label = label.lower().strip()
    msg   = error_message.lower()

    if label == "lexical":
        info = dict(LEXICAL_SECURITY)
        info["name"] = "No Security Risk"
        return info

    if label == "syntax":
        info = dict(SYNTAX_SECURITY)
        info["name"] = "Minimal Risk — Code Won't Compile"
        # Check for more specific syntax risk
        for ex in SYNTAX_SECURITY.get("examples", []):
            if any(p in msg for p in ex["pattern"].split()):
                info["risk"]        = ex["risk"]
                info["cwe"]         = ex["cwe"]
                info["name"]        = ex["name"]
                info["description"] = ex["description"]
                break
        info["risk_color"] = RISK_COLORS.get(info["risk"], "yellow")
        return info # Semantic — match rules in order (most specific first)
    for rule in SEMANTIC_RULES:
        if any(p in msg for p in rule["patterns"]):
            return {
                "risk"          : rule["risk"],
                "risk_color"    : RISK_COLORS.get(rule["risk"], "yellow"),
                "cwe"           : rule["cwe"],
                "name"          : rule["name"],
                "description"   : rule["description"],
                "recommendation": rule["recommendation"],
            }# No specific rule matched
    result = dict(SEMANTIC_DEFAULT)
    result["risk_color"] = RISK_COLORS.get(result["risk"], "yellow")
    return result


def get_risk_emoji(risk: str) -> str:
    """Returns an emoji for the risk level."""
    return {
        RISK_NONE  : "✅",
        RISK_LOW   : "🟡",
        RISK_MEDIUM: "🟠",
        RISK_HIGH  : "🔴",
        RISK_CRIT  : "🚨",
    }.get(risk, "⚪")


def print_security_report(label: str, error_message: str) -> None:
    """Prints a formatted security report to the terminal."""
    info  = get_security_info(label, error_message)
    emoji = get_risk_emoji(info["risk"])
    print(f"\n{'='*60}")
    print(f"  Security Risk Assessment")
    print(f"{'='*60}")
    print(f"  Error   : {error_message}")
    print(f"  Label   : {label.upper()}")
    print(f"  Risk    : {emoji}  {info['risk']}")
    if info.get("name"):
        print(f"  Type    : {info['name']}")
    if info.get("cwe"):
        print(f"  CWE     : {info['cwe']}")
    print(f"\n  Description:")
    print(f"  {info['description']}")
    print(f"\n  Recommendation:")
    print(f"  {info['recommendation']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_cases = [
        ("lexical",  "error: stray '\\' in program"),
        ("syntax",   "error: expected ';' before '}' token"),
        ("semantic", "error: array subscript is above array bounds"),
        ("semantic", "error: division by zero"),
        ("semantic", "error: incompatible types when assigning to type int from type char"),
        ("semantic", "error: implicit declaration of function 'gets'"),
        ("semantic", "error: too few arguments to function 'printf'"),
        ("semantic", "error: assignment makes integer from pointer without cast"),
        ("semantic", "error: undeclared identifier 'x'"),
    ]

    print("\nSecurity Mapping Demo — Compiler Error Classification")
    print("Roll: 24CSB0A25")
    for label, msg in test_cases:
        print_security_report(label, msg)
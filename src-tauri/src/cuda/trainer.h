#define MAX_FUNCTIONS 8

typedef struct {
    int starting_functions;
    float selection_fraction;
    float mutation_probability;
    float mutation_strength;
    float function_addition_probability;
    float function_subtraction_probability;
} HyperParameters;

typedef enum {
    Sine,
    SawTooth,
    Count,
} WaveFunction;

typedef struct {
    WaveFunction function_type;
    float scale;
    float x_translation;
} FunctionCoefficients;

typedef struct {
    FunctionCoefficients functions[MAX_FUNCTIONS];
    unsigned int functions_len;
    float fitness;
} Agent;


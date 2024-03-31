#include <curand_kernel.h>
#include "trainer.h"

// TODO: Research if numel is nescesarry

__device__ float frac(float x) {
    return x - floorf(x);
}

__device__ void fcarray_remove(FunctionCoefficients functions[], int index, unsigned int* length) {
    for (int i = index + 1; i < *length; i++) {
        functions[i - 1] = functions[i];
    }
    
    *length -= 1;
}

__device__ float array_to_local(float x, int goal_res) {
    return x / static_cast<float>(goal_res) * 2.f - 1.f;
}

__device__ float function_coefficients_eval(FunctionCoefficients* fc, float x) {
    switch (fc->function_type) {
        case Sine:
            return sin(x + fc->x_translation) * fc->scale;
        case SawTooth:
            return frac(x + fc->x_translation) * fc->scale;
    }
}

__device__ FunctionCoefficients function_coefficients_rand(curandState *curandState) {
    FunctionCoefficients fc;

    fc.function_type = static_cast<WaveFunction>(curand(curandState) % WaveFunction::Count);
    fc.scale = curand_uniform(curandState) * 2.f - 1.f;
    fc.x_translation = curand_uniform(curandState) * 2.f - 1.f;

    return fc;
}

__device__ Agent agent_rand(HyperParameters* params, curandState *curandState) {
    Agent agent;
    agent.fitness = 0.f;
    agent.functions_len = params->starting_functions;

    for (int i = 0; i < params->starting_functions; i++) {
        agent.functions[i] = function_coefficients_rand(curandState);
    }

    return agent;
};

__device__ float agent_evaluate(Agent* agent, float x) {
    float evaluation = 0.f;
    for (int i = 0; i < agent->functions_len; i++) {
        evaluation += function_coefficients_eval(&agent->functions[i], x);
    }

    return evaluation;
}

__device__ void agent_compute_fitness(Agent* agent, float *goal, int goal_res) {
    agent->fitness = 0.f;
    for (int i = 0; i < goal_res; i++) {
        float x = array_to_local(static_cast<float>(i), goal_res);
        agent->fitness -= abs(agent_evaluate(agent, x) - goal[i]);
    }
}

__device__ Agent agent_crossover(Agent* parent_a, Agent* parent_b, curandState *curandState) {
    int minimum = min(parent_a->functions_len, parent_b->functions_len);
    Agent agent;
    agent.fitness = 0.f;
    agent.functions_len = minimum;

    for (int i = 0; i < minimum; i++) {
        FunctionCoefficients* f1 = &parent_a->functions[i];
        FunctionCoefficients* f2 = &parent_b->functions[i];

        if (curand_uniform(curandState) > 0.5) {
            agent.functions[i] = *f1;
        } else {
            agent.functions[i] = *f2;
        }
    }

    return agent;
}

__device__ void agent_mutate(Agent* agent, HyperParameters* params, curandState *curandState) {
    if (agent->functions_len < MAX_FUNCTIONS && curand_uniform(curandState) < params->function_addition_probability) {
        agent->functions[agent->functions_len] = function_coefficients_rand(curandState);
        agent->functions_len++;
    }

    if (agent->functions_len > 1 && curand_uniform(curandState) < params->function_subtraction_probability) {
        fcarray_remove(agent->functions, curand(curandState) % agent->functions_len, &agent->functions_len);
    }

    for (int i = 0; i < agent->functions_len; i++) {
        if (curand_uniform(curandState) < params->mutation_probability) {
            agent->functions[i].scale += curand_normal(curandState) * params->mutation_strength;
        }

        if (curand_uniform(curandState) < params->mutation_probability) {
            agent->functions[i].x_translation += curand_normal(curandState) * params->mutation_strength;
        }
    }

    // TODO: Maybe mutate function type?
}

extern "C" __global__ void init_kernel(curandState *state, Agent *agents, HyperParameters params, float* goal, int goal_res, unsigned long long seed, int numel) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < numel) {    
        curand_init(seed, id, 0, &state[id]);

        agents[id] = agent_rand(&params, &state[id]);
        agent_compute_fitness(&agents[id], goal, goal_res);

        // agents[id].fitness = 5.f;
    }
}

extern "C" __global__ void step_kernel(curandState *state, Agent *agents, HyperParameters params, float* goal, int goal_res, int agents_len, int numel) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < numel) {
        int top = agents_len - numel;

        // TODO: `top +` is for debugging purposes only
        Agent* parent_a = &agents[curand(&state[id]) % top];
        Agent* parent_b = &agents[curand(&state[id]) % top];

        // Crossover
        Agent child = agent_crossover(parent_a, parent_b, &state[id]);

        // Mutation
        agent_mutate(&child, &params, &state[id]);
        // Evaluate
        agent_compute_fitness(&child, goal, goal_res);

        agents[id + top] = child;
        // agents[id] = child;
    }
}

// https://gist.github.com/mre/1392067
extern "C" __global__ void step_sort_kernel(Agent *agents, int j, int k, int numel) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numel) {
        unsigned int ixj = i^j;

        // TODO: This might sort the wrong way backwards
        if (ixj > i) {
            if ((i & k) == 0) {
                if (agents[i].fitness < agents[ixj].fitness) {
                    Agent temp = agents[i];
                    agents[i] = agents[ixj];
                    agents[ixj] = temp;
                }
            }
            if ((i & k) != 0) {
                if (agents[i].fitness > agents[ixj].fitness) {
                    Agent temp = agents[i];
                    agents[i] = agents[ixj];
                    agents[ixj] = temp;
                }
            }
        }
    }
}

extern "C" __global__ void output_kernel(Agent *agents, float *buff, int goal_res, int index, int numel) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < numel) {
        buff[id] = agent_evaluate(&agents[index], array_to_local(static_cast<float>(id), goal_res));
    }
}
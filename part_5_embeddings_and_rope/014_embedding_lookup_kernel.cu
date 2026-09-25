#include "../model.cuh"

__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // total number of output elements
    int total = seq_len * embed_dim;

    // ignore extra threads outside the output
    if (idx < total) {
        // idx belongs to which token
        int token_pos = idx / embed_dim;

        // which feature inside that token's embedding?
        int dim = idx % embed_dim;

        // taking real vocab id
        int token_id = token_ids[token_pos];

        // lookup the corresponding value in the embedding table
        out[idx] = weight[token_id * embed_dim + dim];
    }
}

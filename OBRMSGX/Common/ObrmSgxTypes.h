#pragma once

#include <stdint.h>
#include "VectorFormat.h"

#define OBRM_BLOCK_INTERNAL_NODE 1u
#define OBRM_BLOCK_LEAF_NODE 2u
#define OBRM_BLOCK_DATA 3u
#define OBRM_BLOCK_DUMMY 4u

#pragma pack(push, 1)

struct ObrmSgxBlockHeader {
    uint32_t block_type;
    int32_t block_id;
    uint32_t payload_bytes;
};

struct ObrmSgxMlpHeader {
    uint32_t input_dim;
    uint32_t hidden_dim;
    uint32_t output_dim;
};

struct ObrmSgxNodeHeader {
    uint32_t is_leaf;
    uint32_t dim;
    uint32_t pivot_count;
    uint32_t link_count;
    uint32_t encoding;
};

struct ObrmSgxDataHeaderV2 {
    uint32_t dim;
    uint32_t vector_count;
    uint32_t encoding;
    uint32_t reserved;
};

#define OBRM_DATA_ENCODING_FLOAT16 1u
#define OBRM_DATA_ENCODING_UINT8 2u

struct ObrmExportHeader {
    char magic[8];
    uint32_t version;
    uint32_t dim;
    uint32_t block_count;
    int32_t root_block_id;
    uint32_t encoding;
};

struct ObrmOptimizedExportHeader {
    char magic[8];
    uint32_t version;
    uint32_t dim;
    uint32_t index_block_count;
    uint32_t data_block_count;
    int32_t root_block_id;
    uint32_t encoding;
};


struct ObrmSgxOramStats {
    uint64_t path_accesses;
    uint64_t real_slots_seen;
    uint64_t dummy_slots_seen;
    uint64_t stash_max_size;
    uint64_t bucket_reads;
    uint64_t bucket_writes;
    uint64_t enclave_resident_bytes;
    uint64_t enclave_peak_bytes;
    uint64_t stash_hits;
    uint64_t query_cache_hits;
    uint64_t pending_stash_peak;
};


struct ObrmSgxPathSlotHeader {
    uint32_t real;
    uint32_t block_type;
    int32_t block_id;
    uint32_t position;
    uint32_t block_bytes;
};

#pragma pack(pop)

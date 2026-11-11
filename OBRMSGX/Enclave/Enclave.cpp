#include "Enclave_t.h"
#include "ObrmSgxTypes.h"

#include "sgx_tcrypto.h"
#include "sgx_trts.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <unordered_map>

int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return static_cast<int>(strnlen(buf, BUFSIZ - 1) + 1);
}

namespace {

struct LoadedBlock {
    int32_t id;
    uint32_t type;
    uint32_t position;
    std::vector<uint8_t> bytes;
};

struct OramSlot {
    int32_t id;
    uint32_t type;
    uint32_t position;
    uint8_t real;
    std::vector<uint8_t> bytes;
};


std::vector<OramSlot> g_buckets;
std::vector<LoadedBlock> g_stash;
size_t g_stash_payload_bytes = 0;
std::vector<uint32_t> g_position_map;
std::vector<std::vector<uint8_t>> g_index_cache;
size_t g_index_cache_bytes = 0;
size_t g_dim = 0;
uint32_t g_vector_encoding=0;
uint32_t g_oram_leaf_count = 1;
uint32_t g_oram_bucket_count = 1;
uint32_t g_oram_bucket_capacity = 4;
// Unplaced authoritative blocks survive queries; written-back copies are cached
// only until the end of the current read-only query (OBIR-tree, Section 5.3).
std::unordered_map<int32_t, LoadedBlock> g_query_cache;
size_t g_query_cache_bytes = 0;
bool g_query_active = false;
bool g_oram_failed = false;
uint32_t g_root_position = 0;
bool g_external_oram = false;
ObrmSgxOramStats g_oram_stats = {};
size_t g_path_buffer_limit = 1024u * 1024u;
size_t g_plain_block_capacity = 0;
size_t g_encrypted_slot_bytes = 0;
uint64_t g_slot_nonce_counter = 1;
std::vector<uint8_t> g_external_read_buffer;
std::vector<uint8_t> g_external_write_buffer;
const size_t kSlotNonceBytes = 12;
const size_t kSlotTagBytes = 16;
const size_t kMaxBulkPathBytes = 8u * 1024u * 1024u;
sgx_aes_gcm_128bit_key_t g_block_key = {};

struct OramState {
    std::vector<OramSlot> buckets;
    std::vector<LoadedBlock> stash;
    std::vector<uint32_t> position_map;
    uint32_t leaf_count;
    uint32_t bucket_count;
    uint32_t bucket_capacity;
    sgx_aes_gcm_128bit_key_t key;
    bool external_oram;
    ObrmSgxOramStats stats;
    size_t stash_payload_bytes;
    size_t path_buffer_limit;
    size_t plain_block_capacity;
    size_t encrypted_slot_bytes;
    uint64_t slot_nonce_counter;
};

OramState g_oram_states[2];
size_t g_active_oram_kind = 0;
bool g_initialization_phase = false;

bool valid_oram_kind(size_t kind)
{
    return kind < 2;
}

size_t estimate_enclave_bytes(size_t extra_bytes = 0)
{
    size_t total = extra_bytes + g_index_cache_bytes + g_query_cache_bytes
        + g_query_cache.size() * sizeof(LoadedBlock);
    total += g_position_map.size() * sizeof(uint32_t);
    total += g_stash.size() * sizeof(LoadedBlock);
    total += g_stash_payload_bytes;
    if (!g_external_oram) {
        total += g_buckets.size() * sizeof(OramSlot);
        for (size_t i = 0; i < g_buckets.size(); ++i) {
            total += g_buckets[i].bytes.size();
        }
    }
    // The cache payload is tracked incrementally in g_index_cache_bytes.  The
    // vector wrapper overhead is fixed once the cache is loaded; do not scan
    // every cached node from the ORAM hot path just to estimate it.
    total += g_index_cache.size() * sizeof(std::vector<uint8_t>);
    return total;
}

void recompute_stash_payload_bytes()
{
    g_stash_payload_bytes = 0;
    for (size_t i = 0; i < g_stash.size(); ++i) {
        g_stash_payload_bytes += g_stash[i].bytes.size();
    }
}

void clear_stash()
{
    g_stash.clear();
    g_stash_payload_bytes = 0;
}

void push_stash_block(LoadedBlock&& block)
{
    g_stash_payload_bytes += block.bytes.size();
    if (g_query_active && !g_query_cache.count(block.id)) {
        g_query_cache_bytes += block.bytes.size();
        g_query_cache.emplace(block.id, block);
    }
    g_stash.push_back(std::move(block));
}

void erase_stash_block(size_t index)
{
    if (index >= g_stash.size()) {
        return;
    }
    g_stash_payload_bytes -= g_stash[index].bytes.size();
    if (index + 1 != g_stash.size()) {
        g_stash[index] = std::move(g_stash.back());
    }
    g_stash.pop_back();
}

void update_enclave_peak(size_t extra_bytes = 0)
{
    const size_t estimate = estimate_enclave_bytes(extra_bytes);
    if (estimate > g_oram_stats.enclave_peak_bytes) {
        g_oram_stats.enclave_peak_bytes = estimate;
    }
}

void update_enclave_resident()
{
    if (!g_initialization_phase) {
        return;
    }
    const size_t resident = estimate_enclave_bytes();
    if (resident > g_oram_stats.enclave_resident_bytes) {
        g_oram_stats.enclave_resident_bytes = resident;
    }
}

void save_oram_state(size_t kind)
{
    if (!valid_oram_kind(kind)) {
        return;
    }
    update_enclave_peak();
    OramState& state = g_oram_states[kind];
    state.buckets = g_buckets;
    state.stash = g_stash;
    state.position_map = g_position_map;
    state.leaf_count = g_oram_leaf_count;
    state.bucket_count = g_oram_bucket_count;
    state.bucket_capacity = g_oram_bucket_capacity;
    memcpy(state.key, g_block_key, sizeof(g_block_key));
    state.external_oram = g_external_oram;
    state.stats = g_oram_stats;
    state.stash_payload_bytes = g_stash_payload_bytes;
    state.path_buffer_limit = g_path_buffer_limit;
    state.plain_block_capacity = g_plain_block_capacity;
    state.encrypted_slot_bytes = g_encrypted_slot_bytes;
    state.slot_nonce_counter = g_slot_nonce_counter;
}

void load_oram_state(size_t kind)
{
    if (!valid_oram_kind(kind)) {
        return;
    }
    const OramState& state = g_oram_states[kind];
    g_buckets = state.buckets;
    g_stash = state.stash;
    g_stash_payload_bytes = state.stash_payload_bytes;
    if (g_stash_payload_bytes == 0 && !g_stash.empty()) {
        recompute_stash_payload_bytes();
    }
    g_position_map = state.position_map;
    g_oram_leaf_count = state.leaf_count == 0 ? 1 : state.leaf_count;
    g_oram_bucket_count = state.bucket_count == 0 ? 1 : state.bucket_count;
    g_oram_bucket_capacity = state.bucket_capacity == 0 ? 4 : state.bucket_capacity;
    memcpy(g_block_key, state.key, sizeof(g_block_key));
    g_external_oram = state.external_oram;
    g_oram_stats = state.stats;
    g_path_buffer_limit = state.path_buffer_limit == 0 ? 1024u * 1024u : state.path_buffer_limit;
    g_plain_block_capacity = state.plain_block_capacity;
    g_encrypted_slot_bytes = state.encrypted_slot_bytes;
    g_slot_nonce_counter = state.slot_nonce_counter == 0 ? 1 : state.slot_nonce_counter;
    g_active_oram_kind = kind;
}

template <typename T>
bool read_pod(const uint8_t*& cursor, const uint8_t* end, T& out)
{
    if (static_cast<size_t>(end - cursor) < sizeof(T)) {
        return false;
    }
    memcpy(&out, cursor, sizeof(T));
    cursor += sizeof(T);
    return true;
}

template <typename T>
void append_pod(std::vector<uint8_t>& out, const T& value)
{
    const uint8_t* src = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), src, src + sizeof(T));
}

void make_slot_nonce(uint8_t nonce[kSlotNonceBytes])
{
    memset(nonce, 0, kSlotNonceBytes);
    const uint64_t value = g_slot_nonce_counter++;
    memcpy(nonce, &value, sizeof(value));
}

bool encrypt_external_slot_to(uint32_t real, uint32_t block_type, int32_t block_id, uint32_t position,
    const uint8_t* block, size_t block_len, uint8_t* out, size_t out_len)
{
    if (out == nullptr || out_len != g_encrypted_slot_bytes || g_plain_block_capacity == 0
        || g_encrypted_slot_bytes == 0 || block_len > g_plain_block_capacity
        || (real && (!block || !block_len)) || g_slot_nonce_counter > UINT64_MAX - 2) {
        return false;
    }
    const size_t header_len = sizeof(ObrmSgxPathSlotHeader);
    const size_t header_nonce_offset = 0;
    const size_t header_cipher_offset = header_nonce_offset + kSlotNonceBytes;
    const size_t header_tag_offset = header_cipher_offset + header_len;
    const size_t payload_nonce_offset = header_tag_offset + kSlotTagBytes;
    const size_t payload_cipher_offset = payload_nonce_offset + kSlotNonceBytes;
    const size_t payload_tag_offset = payload_cipher_offset + g_plain_block_capacity;
    const ObrmSgxPathSlotHeader header = {
        real,
        real ? block_type : OBRM_BLOCK_DUMMY,
        real ? block_id : -1,
        real ? position : 0u,
        real ? static_cast<uint32_t>(block_len) : 0u,
    };

    memset(out, 0, out_len);
    uint8_t* header_nonce = out + header_nonce_offset;
    uint8_t* header_cipher = out + header_cipher_offset;
    sgx_aes_gcm_128bit_tag_t* header_tag =
        reinterpret_cast<sgx_aes_gcm_128bit_tag_t*>(out + header_tag_offset);
    make_slot_nonce(header_nonce);
    if (sgx_rijndael128GCM_encrypt(&g_block_key, reinterpret_cast<const uint8_t*>(&header),
            static_cast<uint32_t>(header_len), header_cipher, header_nonce,
            static_cast<uint32_t>(kSlotNonceBytes), nullptr, 0, header_tag) != SGX_SUCCESS) {
        return false;
    }
    // Encrypt the entire padded payload, including dummy slots.
    std::vector<uint8_t> padded(g_plain_block_capacity, 0);
    if (real && block_len) memcpy(padded.data(), block, block_len);
    uint8_t* payload_nonce = out + payload_nonce_offset;
    uint8_t* payload_cipher = out + payload_cipher_offset;
    sgx_aes_gcm_128bit_tag_t* payload_tag =
        reinterpret_cast<sgx_aes_gcm_128bit_tag_t*>(out + payload_tag_offset);
    make_slot_nonce(payload_nonce);
    return sgx_rijndael128GCM_encrypt(&g_block_key, padded.data(), static_cast<uint32_t>(padded.size()),
        payload_cipher, payload_nonce, static_cast<uint32_t>(kSlotNonceBytes),
        reinterpret_cast<const uint8_t*>(&header), sizeof(header), payload_tag) == SGX_SUCCESS;
}

bool decrypt_external_slot(const uint8_t* slot, size_t slot_len, LoadedBlock& block, bool& real)
{
    real = false;
    if (slot == nullptr || slot_len != g_encrypted_slot_bytes || g_plain_block_capacity == 0) {
        return false;
    }
    const size_t header_len = sizeof(ObrmSgxPathSlotHeader);
    const size_t header_nonce_offset = 0;
    const size_t header_cipher_offset = header_nonce_offset + kSlotNonceBytes;
    const size_t header_tag_offset = header_cipher_offset + header_len;
    const size_t payload_nonce_offset = header_tag_offset + kSlotTagBytes;
    const size_t payload_cipher_offset = payload_nonce_offset + kSlotNonceBytes;
    const size_t payload_tag_offset = payload_cipher_offset + g_plain_block_capacity;

    ObrmSgxPathSlotHeader header;
    const uint8_t* header_nonce = slot + header_nonce_offset;
    const uint8_t* header_cipher = slot + header_cipher_offset;
    const sgx_aes_gcm_128bit_tag_t* header_tag =
        reinterpret_cast<const sgx_aes_gcm_128bit_tag_t*>(slot + header_tag_offset);
    if (sgx_rijndael128GCM_decrypt(&g_block_key, header_cipher, static_cast<uint32_t>(header_len),
            reinterpret_cast<uint8_t*>(&header), header_nonce, static_cast<uint32_t>(kSlotNonceBytes),
            nullptr, 0, header_tag) != SGX_SUCCESS) {
        return false;
    }
    if (!header.real) return true;
    if (header.real != 1 || header.block_id < 0 || header.position >= g_oram_leaf_count
        || !header.block_bytes || header.block_bytes > g_plain_block_capacity) {
        return false;
    }
    block.bytes.resize(g_plain_block_capacity);
    const uint8_t* payload_nonce = slot + payload_nonce_offset;
    const uint8_t* payload_cipher = slot + payload_cipher_offset;
    const sgx_aes_gcm_128bit_tag_t* payload_tag =
        reinterpret_cast<const sgx_aes_gcm_128bit_tag_t*>(slot + payload_tag_offset);
    if (sgx_rijndael128GCM_decrypt(&g_block_key, payload_cipher, static_cast<uint32_t>(g_plain_block_capacity),
            block.bytes.data(), payload_nonce, static_cast<uint32_t>(kSlotNonceBytes),
            reinterpret_cast<const uint8_t*>(&header), sizeof(header), payload_tag) != SGX_SUCCESS) {
        block.bytes.clear();
        return false;
    }
    block.bytes.resize(header.block_bytes);
    block.id = header.block_id;
    block.type = header.block_type;
    block.position = header.position;
    real = true;
    return true;
}

void top_k_logits(const float* logits, size_t output_dim, int* results, size_t result_len)
{
    for (size_t i = 0; i < result_len; ++i) {
        results[i] = -1;
    }
    for (size_t o = 0; o < output_dim; ++o) {
        size_t pos = result_len;
        for (size_t i = 0; i < result_len; ++i) {
            if (results[i] < 0 || logits[o] > logits[static_cast<size_t>(results[i])]) {
                pos = i;
                break;
            }
        }
        if (pos == result_len) {
            continue;
        }
        for (size_t i = result_len - 1; i > pos; --i) {
            results[i] = results[i - 1];
        }
        results[pos] = static_cast<int>(o);
    }
}

LoadedBlock* find_stash_block(int32_t id)
{
    for (size_t i = 0; i < g_stash.size(); ++i) {
        if (g_stash[i].id == id) {
            return &g_stash[i];
        }
    }
    return nullptr;
}

uint32_t next_position_label()
{
    uint32_t random = 0;
    if (sgx_read_rand(reinterpret_cast<uint8_t*>(&random), sizeof(random)) != SGX_SUCCESS)
        return UINT32_MAX;
    // The physical ORAM has a power-of-two leaf count; masking is unbiased.
    return random & (g_oram_leaf_count - 1);
}

void ensure_position_map_id(int32_t id)
{
    if (id < 0) {
        return;
    }
    const size_t index = static_cast<size_t>(id);
    if (g_position_map.size() <= index) {
        g_position_map.resize(index + 1, UINT32_MAX);
    }
}

void set_position_label(int32_t id, uint32_t position)
{
    if (id < 0 || g_oram_leaf_count == 0) {
        return;
    }
    ensure_position_map_id(id);
    g_position_map[static_cast<size_t>(id)] = position % g_oram_leaf_count;
}

bool get_position_label(int32_t id, uint32_t& position)
{
    if (id < 0) {
        return false;
    }
    const size_t index = static_cast<size_t>(id);
    if (index >= g_position_map.size() || g_position_map[index] == UINT32_MAX) {
        return false;
    }
    position = g_position_map[index] % g_oram_leaf_count;
    return true;
}

uint32_t leaf_to_node(uint32_t leaf)
{
    return g_oram_leaf_count + leaf;
}

uint32_t parent_node(uint32_t node)
{
    return node >> 1;
}

bool node_on_path(uint32_t node, uint32_t leaf)
{
    uint32_t current = leaf_to_node(leaf);
    while (current >= 1) {
        if (current == node) {
            return true;
        }
        if (current == 1) {
            break;
        }
        current = parent_node(current);
    }
    return false;
}

uint32_t bucket_offset(uint32_t node)
{
    return (node - 1) * g_oram_bucket_capacity;
}

void pull_path_to_stash(uint32_t leaf)
{
    ++g_oram_stats.path_accesses;
    uint32_t current = leaf_to_node(leaf);
    while (true) {
        const uint32_t offset = bucket_offset(current);
        for (uint32_t i = 0; i < g_oram_bucket_capacity; ++i) {
            OramSlot& slot = g_buckets[offset + i];
            if (slot.real) {
                ++g_oram_stats.real_slots_seen;
                LoadedBlock block;
                block.id = slot.id;
                block.type = slot.type;
                block.position = slot.position;
                block.bytes = slot.bytes;
                push_stash_block(std::move(block));
                slot.real = 0;
                slot.id = -1;
                slot.type = OBRM_BLOCK_DUMMY;
                slot.position = 0;
                slot.bytes.clear();
            } else {
                ++g_oram_stats.dummy_slots_seen;
            }
        }
        if (current == 1) {
            break;
        }
        current = parent_node(current);
    }
    if (g_stash.size() > g_oram_stats.stash_max_size) {
        g_oram_stats.stash_max_size = g_stash.size();
    }
}

bool pull_external_nodes_to_stash(const std::vector<uint32_t>& nodes);

bool pull_external_path_to_stash(uint32_t leaf)
{
    ++g_oram_stats.path_accesses;
    uint32_t read_node_count = 0;
    uint32_t current = leaf_to_node(leaf);
    while (true) {
        ++read_node_count;
        if (current == 1) {
            break;
        }
        current = parent_node(current);
    }
    const size_t path_bytes =
        static_cast<size_t>(read_node_count) * g_oram_bucket_capacity * g_encrypted_slot_bytes;
    if (path_bytes == 0 || path_bytes > g_path_buffer_limit) {
        return false;
    }
    if (path_bytes > kMaxBulkPathBytes) {
        current = leaf_to_node(leaf);
        while (true) {
            const std::vector<uint32_t> node = {current};
            if (!pull_external_nodes_to_stash(node)) {
                return false;
            }
            if (current == 1) {
                break;
            }
            current = parent_node(current);
        }
        return true;
    }
    g_oram_stats.bucket_reads += read_node_count;
    g_external_read_buffer.resize(path_bytes);
    update_enclave_peak(g_external_read_buffer.size());
    size_t written = 0;
    if (ocall_oram_read_path(static_cast<uint32_t>(g_active_oram_kind), leaf,
        g_external_read_buffer.data(), g_external_read_buffer.size(), &written) != SGX_SUCCESS) return false;
    if (written != g_external_read_buffer.size()) {
        return false;
    }
    const uint8_t* cursor = g_external_read_buffer.data();
    const uint8_t* end = g_external_read_buffer.data() + written;
    while (cursor < end) {
        if (g_encrypted_slot_bytes == 0 || static_cast<size_t>(end - cursor) < g_encrypted_slot_bytes) {
            return false;
        }
        LoadedBlock block;
        bool real = false;
        if (!decrypt_external_slot(cursor, g_encrypted_slot_bytes, block, real)) {
            return false;
        }
        if (real) {
            ++g_oram_stats.real_slots_seen;
            push_stash_block(std::move(block));
            update_enclave_peak();
        } else {
            ++g_oram_stats.dummy_slots_seen;
        }
        cursor += g_encrypted_slot_bytes;
    }
    if (g_stash.size() > g_oram_stats.stash_max_size) {
        g_oram_stats.stash_max_size = g_stash.size();
    }
    return true;
}

bool pull_external_nodes_to_stash(const std::vector<uint32_t>& nodes)
{
    if (nodes.empty()) {
        return true;
    }
    g_oram_stats.bucket_reads += nodes.size();
    const size_t bytes = nodes.size() * static_cast<size_t>(g_oram_bucket_capacity) * g_encrypted_slot_bytes;
    if (g_encrypted_slot_bytes == 0 || bytes == 0 || bytes > g_path_buffer_limit * nodes.size()) {
        return false;
    }
    g_external_read_buffer.resize(bytes);
    update_enclave_peak(g_external_read_buffer.size());
    size_t written = 0;
    if (ocall_oram_read_nodes(static_cast<uint32_t>(g_active_oram_kind),
        nodes.data(), nodes.size(), g_external_read_buffer.data(), g_external_read_buffer.size(), &written) != SGX_SUCCESS) return false;
    if (written != g_external_read_buffer.size()) {
        return false;
    }
    const uint8_t* cursor = g_external_read_buffer.data();
    const uint8_t* end = g_external_read_buffer.data() + written;
    while (cursor < end) {
        if (static_cast<size_t>(end - cursor) < g_encrypted_slot_bytes) {
            return false;
        }
        LoadedBlock block;
        bool real = false;
        if (!decrypt_external_slot(cursor, g_encrypted_slot_bytes, block, real)) {
            return false;
        }
        if (real) {
            ++g_oram_stats.real_slots_seen;
            push_stash_block(std::move(block));
        } else {
            ++g_oram_stats.dummy_slots_seen;
        }
        cursor += g_encrypted_slot_bytes;
    }
    if (g_stash.size() > g_oram_stats.stash_max_size) {
        g_oram_stats.stash_max_size = g_stash.size();
    }
    return true;
}

void evict_stash_to_path(uint32_t leaf)
{
    uint32_t path_nodes[64];
    uint32_t count = 0;
    uint32_t current = leaf_to_node(leaf);
    while (true) {
        path_nodes[count++] = current;
        if (current == 1) {
            break;
        }
        current = parent_node(current);
    }

    for (uint32_t pi = 0; pi < count; ++pi) {
        const uint32_t node = path_nodes[pi];
        const uint32_t offset = bucket_offset(node);
        for (uint32_t slot_i = 0; slot_i < g_oram_bucket_capacity; ++slot_i) {
            OramSlot& slot = g_buckets[offset + slot_i];
            if (slot.real) {
                continue;
            }
            for (size_t si = 0; si < g_stash.size(); ++si) {
                if (!node_on_path(node, g_stash[si].position)) {
                    continue;
                }
                slot.id = g_stash[si].id;
                slot.type = g_stash[si].type;
                slot.position = g_stash[si].position;
                slot.real = 1;
                slot.bytes = g_stash[si].bytes;
                erase_stash_block(si);
                if (g_stash.size() > g_oram_stats.stash_max_size) {
                    g_oram_stats.stash_max_size = g_stash.size();
                }
                break;
            }
        }
    }
}

bool evict_external_stash_to_nodes(const std::vector<uint32_t>& nodes);

bool evict_external_stash_to_path(uint32_t leaf)
{
    uint32_t path_nodes[64];
    uint32_t count = 0;
    uint32_t current = leaf_to_node(leaf);
    while (true) {
        path_nodes[count++] = current;
        if (current == 1) {
            break;
        }
        current = parent_node(current);
    }

    const size_t path_bytes = static_cast<size_t>(count) * g_oram_bucket_capacity * g_encrypted_slot_bytes;
    if (path_bytes > kMaxBulkPathBytes) {
        for (uint32_t pi = 0; pi < count; ++pi) {
            const std::vector<uint32_t> node = {path_nodes[pi]};
            if (!evict_external_stash_to_nodes(node)) {
                return false;
            }
        }
        return true;
    }

    const size_t out_bytes = static_cast<size_t>(count) * g_oram_bucket_capacity * g_encrypted_slot_bytes;
    g_external_write_buffer.resize(out_bytes);
    uint8_t* cursor = g_external_write_buffer.data();
    update_enclave_peak();
    for (uint32_t pi = 0; pi < count; ++pi) {
        const uint32_t node = path_nodes[pi];
        for (uint32_t slot_i = 0; slot_i < g_oram_bucket_capacity; ++slot_i) {
            size_t selected = g_stash.size();
            for (size_t si = 0; si < g_stash.size(); ++si) {
                if (node_on_path(node, g_stash[si].position)) {
                    selected = si;
                    break;
                }
            }
            if (selected == g_stash.size()) {
                if (!encrypt_external_slot_to(0, OBRM_BLOCK_DUMMY, -1, 0, nullptr, 0,
                        cursor, g_encrypted_slot_bytes)) {
                    return false;
                }
                cursor += g_encrypted_slot_bytes;
                continue;
            }
            const LoadedBlock& block = g_stash[selected];
            if (!encrypt_external_slot_to(1, block.type, block.id, block.position,
                    block.bytes.data(), block.bytes.size(), cursor, g_encrypted_slot_bytes)) {
                return false;
            }
            cursor += g_encrypted_slot_bytes;
            erase_stash_block(selected);
        }
    }
    if (g_stash.size() > g_oram_stats.stash_max_size) {
        g_oram_stats.stash_max_size = g_stash.size();
    }
    g_oram_stats.bucket_writes += count;
    size_t written = 0;
    const sgx_status_t status = ocall_oram_write_path(static_cast<uint32_t>(g_active_oram_kind), leaf,
        g_external_write_buffer.data(), out_bytes, &written);
    g_oram_stats.pending_stash_peak = std::max(g_oram_stats.pending_stash_peak, uint64_t(g_stash.size()));
    return status == SGX_SUCCESS && written == out_bytes;
}

bool evict_external_stash_to_nodes(const std::vector<uint32_t>& nodes)
{
    if (nodes.empty()) {
        return true;
    }
    g_oram_stats.bucket_writes += nodes.size();
    const size_t out_bytes = nodes.size() * static_cast<size_t>(g_oram_bucket_capacity) * g_encrypted_slot_bytes;
    g_external_write_buffer.resize(out_bytes);
    uint8_t* cursor = g_external_write_buffer.data();
    update_enclave_peak(out_bytes);
    for (size_t pi = 0; pi < nodes.size(); ++pi) {
        const uint32_t node = nodes[pi];
        for (uint32_t slot_i = 0; slot_i < g_oram_bucket_capacity; ++slot_i) {
            size_t selected = g_stash.size();
            for (size_t si = 0; si < g_stash.size(); ++si) {
                if (node_on_path(node, g_stash[si].position)) {
                    selected = si;
                    break;
                }
            }
            if (selected == g_stash.size()) {
                if (!encrypt_external_slot_to(0, OBRM_BLOCK_DUMMY, -1, 0, nullptr, 0,
                        cursor, g_encrypted_slot_bytes)) {
                    return false;
                }
                cursor += g_encrypted_slot_bytes;
                continue;
            }
            const LoadedBlock& block = g_stash[selected];
            if (!encrypt_external_slot_to(1, block.type, block.id, block.position,
                    block.bytes.data(), block.bytes.size(), cursor, g_encrypted_slot_bytes)) {
                return false;
            }
            cursor += g_encrypted_slot_bytes;
            erase_stash_block(selected);
        }
    }
    if (g_stash.size() > g_oram_stats.stash_max_size) {
        g_oram_stats.stash_max_size = g_stash.size();
    }
    size_t written = 0;
    const sgx_status_t status = ocall_oram_write_nodes(static_cast<uint32_t>(g_active_oram_kind),
        nodes.data(), nodes.size(), g_external_write_buffer.data(), out_bytes, &written);
    return status == SGX_SUCCESS && written == out_bytes;
}

bool oram_access_block(int32_t id, LoadedBlock& out)
{
    if (LoadedBlock* block = find_stash_block(id)) {
        ++g_oram_stats.stash_hits;
        out = *block;
        return true;
    }
    const auto cached = g_query_cache.find(id);
    if (cached != g_query_cache.end()) {
        ++g_oram_stats.query_cache_hits;
        out = cached->second;
        return true;
    }
    uint32_t old_position = 0;
    if (!get_position_label(id, old_position)) return false;
    if (g_external_oram) {
        if (!pull_external_path_to_stash(old_position)) return false;
    } else {
        pull_path_to_stash(old_position);
    }
    LoadedBlock* block = find_stash_block(id);
    if (!block) return false;
    const uint32_t new_position = next_position_label();
    if (new_position == UINT32_MAX) return false;
    block->position = new_position;
    set_position_label(id, new_position);
    out = *block;
    // Deepest eligible buckets first, on the OLD path; unplaced blocks stay.
    if (g_external_oram) {
        if (!evict_external_stash_to_path(old_position)) return false;
        g_oram_stats.pending_stash_peak = std::max(g_oram_stats.pending_stash_peak, uint64_t(g_stash.size()));
        return true;
    }
    evict_stash_to_path(old_position);
    return true;
}


int parse_node_layout(const uint8_t* block, size_t block_len, ObrmSgxBlockHeader& header,
    ObrmSgxNodeHeader& node, ObrmSgxMlpHeader& mlp, const uint8_t*& pivot_vectors,
    const int32_t*& link_ids, const uint32_t*& link_positions, const float*& mlp_params)
{
    const uint8_t* cursor = block;
    const uint8_t* end = block + block_len;
    if (!read_pod(cursor, end, header) || !read_pod(cursor, end, node) || !read_pod(cursor, end, mlp)) {
        return -1;
    }
    if (header.block_type != OBRM_BLOCK_INTERNAL_NODE && header.block_type != OBRM_BLOCK_LEAF_NODE) {
        return -2;
    }
    if (vector_format::element_bytes(node.encoding)==0 || (g_vector_encoding && node.encoding!=g_vector_encoding)) {
        return -3;
    }
    const size_t pivot_count = static_cast<size_t>(node.pivot_count) * node.dim;
    const size_t pivot_element_bytes=vector_format::element_bytes(node.encoding);
    if (static_cast<size_t>(end - cursor) < pivot_count * pivot_element_bytes) {
        return -3;
    }
    pivot_vectors = cursor;
    cursor += pivot_count * pivot_element_bytes;

    const size_t link_id_bytes = static_cast<size_t>(node.link_count) * sizeof(int32_t);
    if (static_cast<size_t>(end - cursor) < link_id_bytes) {
        return -4;
    }
    link_ids = reinterpret_cast<const int32_t*>(cursor);
    cursor += link_id_bytes;
    const size_t link_pos_bytes = static_cast<size_t>(node.link_count) * sizeof(uint32_t);
    if (static_cast<size_t>(end - cursor) < link_pos_bytes) {
        return -5;
    }
    link_positions = reinterpret_cast<const uint32_t*>(cursor);
    cursor += link_pos_bytes;

    const size_t param_count =
        static_cast<size_t>(mlp.input_dim) * mlp.hidden_dim
        + mlp.hidden_dim
        + static_cast<size_t>(mlp.hidden_dim) * mlp.output_dim
        + mlp.output_dim
        + static_cast<size_t>(mlp.input_dim) * 2;
    if (static_cast<size_t>(end - cursor) < param_count * sizeof(float)) {
        return -6;
    }
    mlp_params = reinterpret_cast<const float*>(cursor);
    cursor += param_count * sizeof(float);
    return cursor == end ? 0 : -7;
}

int route_node_to_link_indices(const uint8_t* block, size_t block_len, const float* query, size_t query_len,
    int* results, size_t result_len)
{
    ObrmSgxBlockHeader header;
    ObrmSgxNodeHeader node;
    ObrmSgxMlpHeader mlp;
    const uint8_t* pivot_vectors = nullptr;
    const int32_t* link_ids = nullptr;
    const uint32_t* link_positions = nullptr;
    const float* params = nullptr;
    const int parsed = parse_node_layout(block, block_len, header, node, mlp, pivot_vectors, link_ids, link_positions, params);
    if (parsed != 0) {
        return -10 + parsed;
    }
    if (query_len != node.dim || mlp.input_dim != node.pivot_count || mlp.output_dim != node.link_count) {
        return -20;
    }

    const size_t w1_count = static_cast<size_t>(mlp.input_dim) * mlp.hidden_dim;
    const size_t b1_count = mlp.hidden_dim;
    const size_t w2_count = static_cast<size_t>(mlp.hidden_dim) * mlp.output_dim;
    const size_t b2_count = mlp.output_dim;
    const float* w1 = params;
    const float* b1 = w1 + w1_count;
    const float* w2 = b1 + b1_count;
    const float* b2 = w2 + w2_count;
    const float* mean = b2 + b2_count;
    const float* inv_std = mean + mlp.input_dim;

    float feature[64];
    float hidden[128];
    float logits[512];
    if (mlp.input_dim > 64 || mlp.hidden_dim > 128 || mlp.output_dim > 512) {
        return -21;
    }

    for (size_t p = 0; p < node.pivot_count; ++p) {
        float dist = 0.0f;
        for (size_t d = 0; d < node.dim; ++d) {
            const size_t offset = (p * node.dim + d);
            const float pivot_value=vector_format::coordinate(pivot_vectors,offset,node.encoding);
            const float diff = query[d] - pivot_value;
            dist += diff * diff;
        }
        feature[p] = (sqrtf(dist) - mean[p]) * inv_std[p];
    }
    for (size_t h = 0; h < mlp.hidden_dim; ++h) {
        float z = b1[h];
        for (size_t d = 0; d < mlp.input_dim; ++d) {
            z += feature[d] * w1[d * mlp.hidden_dim + h];
        }
        hidden[h] = z > 0.0f ? z : 0.0f;
    }
    for (size_t o = 0; o < mlp.output_dim; ++o) {
        float z = b2[o];
        for (size_t h = 0; h < mlp.hidden_dim; ++h) {
            z += hidden[h] * w2[h * mlp.output_dim + o];
        }
        logits[o] = z;
    }
    top_k_logits(logits, mlp.output_dim, results, result_len);
    (void)link_ids;
    (void)link_positions;
    return 0;
}







void insert_top_result(int32_t id, float dist, int* results, float* dists, size_t result_len)
{
    if (dist >= dists[result_len - 1]) {
        return;
    }
    size_t pos = result_len - 1;
    while (pos > 0 && dist < dists[pos - 1]) {
        dists[pos] = dists[pos - 1];
        results[pos] = results[pos - 1];
        --pos;
    }
    dists[pos] = dist;
    results[pos] = id;
}

int parse_data_layout_v2(const uint8_t* block,size_t block_len,ObrmSgxDataHeaderV2& data,
    const int32_t*& ids,const uint8_t*& vectors)
{
    const uint8_t* cursor=block;const uint8_t* end=block+block_len;
    ObrmSgxBlockHeader header;
    if(!read_pod(cursor,end,header) || header.block_type!=OBRM_BLOCK_DATA
        || header.payload_bytes!=size_t(end-cursor) || !read_pod(cursor,end,data)) return -1;
    const size_t width=vector_format::element_bytes(data.encoding);
    if(!width || data.encoding!=g_vector_encoding || !data.dim || !data.vector_count || data.reserved) return -2;
    const size_t id_bytes=size_t(data.vector_count)*sizeof(int32_t);
    const size_t coordinate_bytes=size_t(data.vector_count)*data.dim*width;
    if(size_t(end-cursor)!=id_bytes+coordinate_bytes) return -3;
    ids=reinterpret_cast<const int32_t*>(cursor);cursor+=id_bytes;
    vectors=cursor;return 0;
}


bool add_candidates_from_data_block(const LoadedBlock& block,const float* query,size_t query_len,
    int* results,float* distances,size_t result_len)
{
    ObrmSgxDataHeaderV2 data;const int32_t* ids=nullptr;const uint8_t* vectors=nullptr;
    if(parse_data_layout_v2(block.bytes.data(),block.bytes.size(),data,ids,vectors)!=0 || data.dim!=query_len) return false;
    for(size_t i=0;i<data.vector_count;++i) {
        if(ids[i]<0) return false;
        float distance=0;
        const float worst=distances[result_len-1];
        for(size_t d=0;d<query_len;++d) {
            const float value=vector_format::coordinate(vectors,i*query_len+d,data.encoding);
            const float delta=query[d]-value;distance+=delta*delta;
            if(distance>=worst) break;
        }
        insert_top_result(ids[i],distance,results,distances,result_len);
    }
    return true;
}


bool oram_access_data_block_candidates(int32_t id, const float* query, size_t query_len,
    int* results, float* best_dists, size_t result_len)
{
    LoadedBlock block;
    return oram_access_block(id, block) && block.type == OBRM_BLOCK_DATA
        && add_candidates_from_data_block(block, query, query_len, results, best_dists, result_len);
}

bool process_node_recursive(int32_t node_id, const float* query, size_t query_len,
    size_t r, int* results, float* best_dists, size_t result_len, uint32_t depth)
{
    if (depth > 16 || r > 32) {
        return false;
    }

    LoadedBlock node_access;
    if (!oram_access_block(node_id, node_access)) {
        return false;
    }
    if (node_access.type != OBRM_BLOCK_INTERNAL_NODE && node_access.type != OBRM_BLOCK_LEAF_NODE) {
        return false;
    }

    std::vector<uint8_t> node_bytes = node_access.bytes;
    int choices[32];
    if (route_node_to_link_indices(node_bytes.data(), node_bytes.size(), query, query_len, choices, r) != 0) {
        return false;
    }

    ObrmSgxBlockHeader header;
    ObrmSgxNodeHeader node;
    ObrmSgxMlpHeader mlp;
    const uint8_t* pivots = nullptr;
    const int32_t* links = nullptr;
    const uint32_t* positions = nullptr;
    const float* params = nullptr;
    if (parse_node_layout(node_bytes.data(), node_bytes.size(), header, node, mlp, pivots, links, positions, params) != 0) {
        return false;
    }

    for (size_t i = 0; i < r; ++i) {
        const int choice = choices[i];
        if (choice < 0 || static_cast<size_t>(choice) >= node.link_count) {
            continue;
        }
        const int32_t child_id = links[choice];

        if (header.block_type == OBRM_BLOCK_LEAF_NODE) {
            LoadedBlock data_access;
            bool data_ok = false;
            data_ok = oram_access_block(child_id, data_access)
                && data_access.type == OBRM_BLOCK_DATA;
            if (!data_ok || !add_candidates_from_data_block(data_access, query, query_len,
                    results, best_dists, result_len)) return false;
        } else {
            if (!process_node_recursive(child_id, query, query_len, r,
                    results, best_dists, result_len, depth + 1)) {
                return false;
            }
        }
    }

    (void)positions;
    return true;
}

bool get_cached_index_block(int32_t node_id, const std::vector<uint8_t>*& out)
{
    if (node_id < 0) {
        return false;
    }
    const size_t index = static_cast<size_t>(node_id);
    if (index >= g_index_cache.size() || g_index_cache[index].empty()) {
        return false;
    }
    out = &g_index_cache[index];
    return true;
}

bool collect_cached_data_ids_recursive(int32_t node_id, const float* query, size_t query_len,
    size_t r, std::vector<int32_t>& data_ids, uint32_t depth)
{
    if (depth > 16 || r > 32) {
        return false;
    }

    const std::vector<uint8_t>* node_bytes = nullptr;
    if (!get_cached_index_block(node_id, node_bytes)) {
        return false;
    }

    int choices[32];
    if (route_node_to_link_indices(node_bytes->data(), node_bytes->size(), query, query_len, choices, r) != 0) {
        return false;
    }

    ObrmSgxBlockHeader header;
    ObrmSgxNodeHeader node;
    ObrmSgxMlpHeader mlp;
    const uint8_t* pivots = nullptr;
    const int32_t* links = nullptr;
    const uint32_t* positions = nullptr;
    const float* params = nullptr;
    if (parse_node_layout(node_bytes->data(), node_bytes->size(), header, node, mlp, pivots, links, positions, params) != 0) {
        return false;
    }

    for (size_t i = 0; i < r; ++i) {
        const int choice = choices[i];
        if (choice < 0 || static_cast<size_t>(choice) >= node.link_count) {
            continue;
        }
        const int32_t child_id = links[choice];
        if (header.block_type == OBRM_BLOCK_LEAF_NODE) {
            data_ids.push_back(child_id);
        } else {
            if (!collect_cached_data_ids_recursive(child_id, query, query_len, r,
                    data_ids, depth + 1)) {
                return false;
            }
        }
    }

    (void)positions;
    return true;
}

bool add_candidates_from_data_ids(const std::vector<int32_t>& data_ids, const float* query, size_t query_len,
    int* results, float* best_dists, size_t result_len)
{
    if (g_active_oram_kind != 1) {
        load_oram_state(1);
    }
    for (size_t i = 0; i < data_ids.size(); ++i) {
        if (!oram_access_data_block_candidates(data_ids[i], query, query_len, results, best_dists, result_len)) return false;
    }
    return true;
}

} // namespace

int ecall_obrm_load_index_cache(const uint8_t* blocks, size_t blocks_len, size_t block_count, size_t dim)
{
    if (blocks == nullptr || block_count == 0 || dim == 0) {
        return -1;
    }
    g_index_cache.clear();
    g_index_cache_bytes = 0;
    g_dim = dim;

    const uint8_t* cursor = blocks;
    const uint8_t* end = blocks + blocks_len;
    for (size_t i = 0; i < block_count; ++i) {
        uint32_t block_bytes = 0;
        if (!read_pod(cursor, end, block_bytes) || block_bytes == 0) {
            return -2;
        }
        if (static_cast<size_t>(end - cursor) < block_bytes) {
            return -3;
        }
        ObrmSgxBlockHeader header;
        const uint8_t* block_cursor = cursor;
        const uint8_t* block_end = cursor + block_bytes;
        if (!read_pod(block_cursor, block_end, header)) {
            return -4;
        }

        if (header.block_type != OBRM_BLOCK_INTERNAL_NODE && header.block_type != OBRM_BLOCK_LEAF_NODE) {
            return -5;
        }
        if (header.block_id < 0) {
            return -6;
        }
        ObrmSgxNodeHeader nh;
        if(!read_pod(block_cursor,block_end,nh) || !vector_format::element_bytes(nh.encoding)
            || (g_vector_encoding && nh.encoding!=g_vector_encoding)) return -8;
        g_vector_encoding=nh.encoding;
        const size_t id = static_cast<size_t>(header.block_id);
        if (g_index_cache.size() <= id) {
            g_index_cache.resize(id + 1);
        }
        g_index_cache[id].assign(cursor, cursor + block_bytes);
        g_index_cache_bytes += block_bytes;
        cursor += block_bytes;
    }
    if (cursor != end) {
        return -7;
    }
    return 0;
}

int ecall_obrm_init_external_oram(size_t oram_kind, size_t block_count, size_t dim, size_t plain_block_capacity)
{
    if (!valid_oram_kind(oram_kind)) {
        return -10;
    }
    if (block_count == 0 || dim == 0 || plain_block_capacity == 0) {
        return -1;
    }
    clear_stash();
    g_buckets.clear();
    g_position_map.clear();
    g_dim = dim;
    g_oram_leaf_count = 1;
    while (g_oram_leaf_count < block_count) {
        g_oram_leaf_count <<= 1;
    }
    g_oram_bucket_count = g_oram_leaf_count * 2 - 1;
    g_query_cache.clear();
    g_query_cache_bytes = 0;
    g_query_active = false;
    g_oram_failed = false;
    if (sgx_read_rand(g_block_key, sizeof(g_block_key)) != SGX_SUCCESS) return -11;
    if (oram_kind == 0) g_vector_encoding = 0;
    g_root_position = 0;
    g_external_oram = true;
    g_plain_block_capacity = plain_block_capacity;
    g_encrypted_slot_bytes = kSlotNonceBytes + sizeof(ObrmSgxPathSlotHeader) + kSlotTagBytes
        + kSlotNonceBytes + g_plain_block_capacity + kSlotTagBytes;
    g_path_buffer_limit = g_encrypted_slot_bytes * g_oram_bucket_capacity * 64u;
    g_slot_nonce_counter = 1;
    memset(&g_oram_stats, 0, sizeof(g_oram_stats));
    g_initialization_phase = true;
    g_active_oram_kind = oram_kind;
    save_oram_state(oram_kind);
    return 0;
}

int ecall_obrm_stage_initial_block(size_t oram_kind, const uint8_t* block, size_t block_len)
{
    if (!valid_oram_kind(oram_kind) || oram_kind != g_active_oram_kind || !g_initialization_phase
        || !block || block_len > g_plain_block_capacity) return -1;
    const uint8_t* cursor = block;
    const uint8_t* end = block + block_len;
    ObrmSgxBlockHeader header;
    if (!read_pod(cursor, end, header) || header.payload_bytes != static_cast<uint32_t>(end - cursor)) {
        return -2;
    }
    uint32_t encoding=0;
    if(header.block_type==OBRM_BLOCK_DATA) {
        ObrmSgxDataHeaderV2 dh;
        if(!read_pod(cursor,end,dh)) return -4;
        encoding=dh.encoding;
    } else if(header.block_type==OBRM_BLOCK_INTERNAL_NODE || header.block_type==OBRM_BLOCK_LEAF_NODE) {
        ObrmSgxNodeHeader nh;
        if(!read_pod(cursor,end,nh)) return -4;
        encoding=nh.encoding;
    } else return -4;
    if(!vector_format::element_bytes(encoding) || (g_vector_encoding && encoding!=g_vector_encoding)) return -5;
    g_vector_encoding=encoding;
    uint32_t previous;
    if (header.block_id < 0 || get_position_label(header.block_id, previous)) return -6;
    const uint32_t position = next_position_label();
    if (position == UINT32_MAX) return -7;
    LoadedBlock loaded{header.block_id, header.block_type, position,
        std::vector<uint8_t>(block, block + block_len)};
    set_position_label(header.block_id, position);
    push_stash_block(std::move(loaded));
    return 0;
}

int ecall_obrm_finish_initialization(size_t oram_kind)
{
    if (!valid_oram_kind(oram_kind) || oram_kind != g_active_oram_kind || !g_initialization_phase) return -1;
    // Bottom-up greedy placement. Unplaced blocks remain in the trusted stash.
    // Each slot, including every dummy slot, receives its own fresh ciphertext.
    for (uint32_t node = g_oram_bucket_count; node > 0; --node) {
        if (!evict_external_stash_to_nodes(std::vector<uint32_t>{node})) {
            g_oram_failed = true;
            return -2;
        }
    }
    g_oram_stats.pending_stash_peak = g_stash.size();
    update_enclave_resident();
    g_initialization_phase = false;
    // Initialization I/O is excluded from query counters.
    g_oram_stats.bucket_writes = 0;
    save_oram_state(oram_kind);
    return 0;
}

int ecall_obrm_get_oram_stats_by_kind(size_t oram_kind, uint8_t* stats, size_t stats_len)
{
    if (!valid_oram_kind(oram_kind)) {
        return -10;
    }
    if (stats == nullptr || stats_len < sizeof(ObrmSgxOramStats)) {
        return -1;
    }

    if (oram_kind == g_active_oram_kind) {
        update_enclave_peak();
        save_oram_state(oram_kind);
    }
    memcpy(stats, &g_oram_states[oram_kind].stats, sizeof(ObrmSgxOramStats));
    return 0;
}

int query_base_decoded(const float* query, size_t query_len, size_t r, int* results, size_t result_len)
{
    g_initialization_phase = false;
    if (g_active_oram_kind != 0) load_oram_state(0);
    if (query == nullptr || results == nullptr || query_len != g_dim || r == 0 || result_len == 0) {
        return -1;
    }
    for (size_t i = 0; i < result_len; ++i) {
        results[i] = -1;
    }
    float best_dists[16];
    if (result_len > 16 || r > 32) {
        return -3;
    }
    for (size_t i = 0; i < result_len; ++i) {
        best_dists[i] = FLT_MAX;
    }
    if (!process_node_recursive(0, query, query_len, r,
            results, best_dists, result_len, 0)) {
        return -2;
    }
    return 0;
}

int query_optimized_decoded(const float* query, size_t query_len, size_t r, int* results, size_t result_len)
{
    g_initialization_phase = false;
    if (query == nullptr || results == nullptr || query_len != g_dim || r == 0 || result_len == 0
        || g_index_cache.empty()) {
        return -1;
    }
    for (size_t i = 0; i < result_len; ++i) {
        results[i] = -1;
    }
    float best_dists[16];
    if (result_len > 16 || r > 32) {
        return -3;
    }
    for (size_t i = 0; i < result_len; ++i) {
        best_dists[i] = FLT_MAX;
    }
    std::vector<int32_t> data_ids;
    data_ids.reserve(r * r * r);
    if (!collect_cached_data_ids_recursive(0, query, query_len, r, data_ids, 0)) {
        return -2;
    }
    return add_candidates_from_data_ids(data_ids, query, query_len, results, best_dists, result_len) ? 0 : -2;
}



int query_encoded(bool optimized,const uint8_t* coordinates,size_t coordinate_bytes,uint32_t encoding,
    size_t dim,size_t r,int* results,size_t result_len)
{
    const size_t width=vector_format::element_bytes(encoding);
    if(!coordinates || !width || encoding!=g_vector_encoding || dim!=g_dim || coordinate_bytes!=dim*width) return -20;
    std::vector<float> computational_query(dim);
    for(size_t d=0;d<dim;++d) {
        computational_query[d]=vector_format::coordinate(coordinates,d,encoding);
        if(!std::isfinite(computational_query[d])) return -21;
    }
    if (g_oram_failed || g_initialization_phase) return -30;
    g_query_active = true;
    const int status = optimized ? query_optimized_decoded(computational_query.data(),dim,r,results,result_len)
        : query_base_decoded(computational_query.data(),dim,r,results,result_len);
    update_enclave_peak();
    g_query_active = false;
    g_query_cache.clear();
    g_query_cache_bytes = 0;
    // Only disposable read copies are cleared; pending blocks remain in g_stash.
    if (status != 0) {
        g_oram_failed = true;
        if (results) std::fill(results, results + result_len, -1);
    }
    return status;
}
int ecall_obrm_query_loaded(const uint8_t* coordinates,size_t coordinate_bytes,uint32_t encoding,
    size_t dim,size_t r,int* results,size_t result_len)
{
    return query_encoded(false,coordinates,coordinate_bytes,encoding,dim,r,results,result_len);
}
int ecall_obrm_query_optimized_loaded(const uint8_t* coordinates,size_t coordinate_bytes,uint32_t encoding,
    size_t dim,size_t r,int* results,size_t result_len)
{
    return query_encoded(true,coordinates,coordinate_bytes,encoding,dim,r,results,result_len);
}

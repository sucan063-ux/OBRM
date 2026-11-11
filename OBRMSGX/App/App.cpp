#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

#include "Enclave_u.h"
#include "ObrmSgxTypes.h"
#include "sgx_urts.h"

#define ENCLAVE_FILENAME "./enclave.signed.so"

namespace {

using ExportHeader=ObrmExportHeader;

using OptimizedExportHeader=ObrmOptimizedExportHeader;

struct AppOramContext {
    std::vector<uint8_t> storage;
    std::vector<uint8_t> occupied;
    uint32_t leaf_count = 1;
    uint32_t bucket_capacity = 4;
    size_t slot_bytes = 0;
};

AppOramContext g_app_orams[2];

struct EncodedQueries {
    size_t count,dim;
    uint32_t encoding;
    std::vector<uint8_t> values;
};

struct IntVectors {
    size_t count;
    size_t dim;
    std::vector<int32_t> values;
};

size_t parse_size(const char* value)
{
    size_t pos = 0;
    const std::string text(value);
    const size_t parsed = std::stoull(text, &pos);
    if (pos != text.size()) {
        throw std::runtime_error("invalid integer");
    }
    return parsed;
}

int32_t read_dim(std::ifstream& in)
{
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim <= 0) {
        throw std::runtime_error("failed to read vector dimension");
    }
    return dim;
}

size_t file_size(std::ifstream& in)
{
    const auto current = in.tellg();
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    in.seekg(current, std::ios::beg);
    return static_cast<size_t>(end);
}

EncodedQueries read_queries(const char* path,size_t limit,uint32_t encoding)
{
    const size_t width=vector_format::element_bytes(encoding);
    if(!width) throw std::runtime_error("invalid query coordinate encoding");
    const std::string filename(path),extension=encoding==vector_format::UInt8?".bvecs":".hvecs";
    if(filename.size()<extension.size() || filename.substr(filename.size()-extension.size())!=extension)
        throw std::runtime_error("query file format mismatch");
    std::ifstream in(path,std::ios::binary);
    if(!in) throw std::runtime_error("failed to open query file");
    const int32_t dim=read_dim(in);
    const size_t stride=4+size_t(dim)*width,bytes=file_size(in);
    if(bytes%stride) throw std::runtime_error("truncated encoded query file");
    EncodedQueries out;out.count=std::min(bytes/stride,limit);out.dim=size_t(dim);out.encoding=encoding;
    out.values.resize(out.count*out.dim*width);in.seekg(0);
    for(size_t i=0;i<out.count;++i) {
        if(read_dim(in)!=dim) throw std::runtime_error("mixed query dimensions");
        in.read(reinterpret_cast<char*>(out.values.data()+i*out.dim*width),out.dim*width);
        if(!in) throw std::runtime_error("truncated encoded query");
    }
    return out;
}


IntVectors read_ivecs(const char* path, size_t limit)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open ivecs");
    }
    const int32_t dim = read_dim(in);
    const size_t record_size = sizeof(int32_t) + sizeof(int32_t) * static_cast<size_t>(dim);
    const size_t bytes = file_size(in);
    if (bytes % record_size != 0) {
        throw std::runtime_error("invalid ivecs file size");
    }
    size_t count = bytes / record_size;
    if (limit != 0 && limit < count) {
        count = limit;
    }
    IntVectors out;
    out.count = count;
    out.dim = static_cast<size_t>(dim);
    out.values.resize(count * out.dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        const int32_t current_dim = read_dim(in);
        if (current_dim != dim) {
            throw std::runtime_error("mixed dimensions");
        }
        in.read(reinterpret_cast<char*>(out.values.data() + i * out.dim), sizeof(int32_t) * out.dim);
        if (!in) {
            throw std::runtime_error("failed to read ivecs payload");
        }
    }
    return out;
}

double recall_at_10(const std::vector<int>& results, size_t result_dim, const IntVectors& gt)
{
    if (gt.dim < 10 || result_dim < 10 || results.size() < gt.count * result_dim) {
        throw std::runtime_error("invalid recall inputs");
    }
    size_t hits = 0;
    for (size_t i = 0; i < gt.count; ++i) {
        const int* got = results.data() + i * result_dim;
        const int32_t* truth = gt.values.data() + i * gt.dim;
        for (size_t a = 0; a < 10; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                if (got[a] == truth[b]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(gt.count * 10);
}

sgx_enclave_id_t global_eid = 0;

void print_sgx_error(sgx_status_t status)
{
    printf("SGX error: 0x%X\n", status);
}

int initialize_enclave()
{
    sgx_launch_token_t token = {0};
    int updated = 0;
    const sgx_status_t status = sgx_create_enclave(
        ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &global_eid, nullptr);
    if (status != SGX_SUCCESS) {
        char cwd[1024] = {0};
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            printf("cwd: %s\n", cwd);
        }
        printf("enclave file access: %s\n", access(ENCLAVE_FILENAME, R_OK) == 0 ? "ok" : "failed");
        print_sgx_error(status);
        return -1;
    }
    return 0;
}

uint32_t app_leaf_to_node(const AppOramContext& ctx, uint32_t leaf);
uint32_t app_parent_node(uint32_t node);
uint32_t app_bucket_offset(const AppOramContext& ctx, uint32_t node);
AppOramContext* app_oram_by_kind(uint32_t kind);
uint8_t* app_slot_ptr(AppOramContext& ctx, size_t slot_index);
bool load_exported_blocks(const char* path, ExportHeader& header);
bool load_optimized_blocks(const char* path, OptimizedExportHeader& header);

} // namespace

int main(int argc,char** argv)
{
    if (argc!=9) {
        fprintf(stderr,"Usage: app base|optimized blocks query.bvecs|hvecs gt.ivecs query_count r per_query.csv results.ivecs\n");
        return 1;
    }
    if (initialize_enclave()!=0) return 1;
    try {
        const std::string scheme=argv[1];
        if (scheme!="base" && scheme!="optimized") throw std::runtime_error("invalid scheme");
        size_t count=parse_size(argv[5]),r=parse_size(argv[6]);
        if (!count || !r || r>32) throw std::runtime_error("invalid query count or branch budget");
        uint32_t dim=0,encoding=0;
        if (scheme=="base") {
            ExportHeader header;
            if (!load_exported_blocks(argv[2],header)) throw std::runtime_error("base loading failed");
            dim=header.dim;encoding=header.encoding;
        } else {
            OptimizedExportHeader header;
            if (!load_optimized_blocks(argv[2],header)) throw std::runtime_error("optimized loading failed");
            dim=header.dim;encoding=header.encoding;
        }
        const auto queries=read_queries(argv[3],count,encoding);
        const auto gt=read_ivecs(argv[4],count);
        if (queries.count!=count || queries.dim!=dim || gt.count!=count || gt.dim<10)
            throw std::runtime_error("query/groundtruth count or dimension mismatch");
        const size_t width=vector_format::element_bytes(encoding);
        for(size_t i=0;i<queries.count*queries.dim;++i)
            if(!std::isfinite(vector_format::coordinate(queries.values.data(),i,encoding)))
                throw std::runtime_error("non-finite query");
        std::vector<int> results(count*10,-1);
        std::vector<double> times(count);
        double seconds=0;
        for (size_t qi=0;qi<count;++qi) {
            int retval=-1;
            const auto begin=std::chrono::steady_clock::now();
            sgx_status_t status=scheme=="base"
                ?ecall_obrm_query_loaded(global_eid,&retval,&queries.values[qi*dim*width],dim*width,encoding,dim,r,&results[qi*10],10)
                :ecall_obrm_query_optimized_loaded(global_eid,&retval,&queries.values[qi*dim*width],dim*width,encoding,dim,r,&results[qi*10],10);
            const auto end=std::chrono::steady_clock::now();
            if (status!=SGX_SUCCESS || retval!=0) throw std::runtime_error("query failed: "+std::to_string(qi));
            times[qi]=std::chrono::duration<double>(end-begin).count();
            seconds+=times[qi];
        }
        std::ofstream per_query(argv[7]),ids_out(argv[8],std::ios::binary);
        per_query<<"query_index,time_ms,recall_at_10\n"<<std::setprecision(12);
        const int32_t k=10;
        for (size_t qi=0;qi<count;++qi) {
            size_t hits=0;
            for (size_t a=0;a<10;++a) {
                if (results[qi*10+a]<0) throw std::runtime_error("fewer than 10 candidates returned");
                for (size_t b=0;b<a;++b)
                    if (results[qi*10+a]==results[qi*10+b]) throw std::runtime_error("duplicate result identifier");
                for (size_t b=0;b<10;++b) if (results[qi*10+a]==gt.values[qi*gt.dim+b]) {++hits;break;}
            }
            per_query<<qi<<","<<times[qi]*1000<<","<<double(hits)/10<<"\n";
            ids_out.write(reinterpret_cast<const char*>(&k),sizeof(k));
            ids_out.write(reinterpret_cast<const char*>(&results[qi*10]),10*sizeof(int));
        }
        if (!per_query || !ids_out) throw std::runtime_error("failed to write query outputs");
        ObrmSgxOramStats stats[2]={};
        for (size_t kind=0;kind<2;++kind) {
            int retval=-1;
            sgx_status_t status=ecall_obrm_get_oram_stats_by_kind(global_eid,&retval,kind,
                reinterpret_cast<uint8_t*>(&stats[kind]),sizeof(stats[kind]));
            if (status!=SGX_SUCCESS || retval!=0) throw std::runtime_error("stats retrieval failed");
        }
#ifdef OBRM_SGX_SIMULATION
        printf("execution_mode=sgx_sdk_simulation\n");
#else
        printf("execution_mode=sgx_hardware\n");
#endif
        printf("coordinate_encoding=%s\n",encoding==vector_format::UInt8?"uint8":"float16");
        printf("query_count=%zu\nr=%zu\nquery_seconds_total=%.12f\nquery_seconds_avg=%.12f\nquery_ms_avg=%.9f\nrecall_at_10=%.12f\n",
            count,r,seconds,seconds/double(count),seconds*1000/double(count),recall_at_10(results,10,gt));
        printf("oram_path_accesses=%llu\n",static_cast<unsigned long long>(stats[0].path_accesses+stats[1].path_accesses));
        const auto& active_stats = stats[scheme == "base" ? 0 : 1];
        printf("stash_hits=%llu\nquery_cache_hits=%llu\npending_stash_peak=%llu\n",
            (unsigned long long)active_stats.stash_hits, (unsigned long long)active_stats.query_cache_hits,
            (unsigned long long)active_stats.pending_stash_peak);
        printf("enclave_resident_bytes_estimate=%llu\n",static_cast<unsigned long long>(
            std::max(stats[0].enclave_resident_bytes,stats[1].enclave_resident_bytes)));
        sgx_destroy_enclave(global_eid);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr,"error: %s\n",e.what());
        sgx_destroy_enclave(global_eid);
        return 1;
    }
}

void ocall_print_string(const char* str)
{
    printf("%s", str);
}

void ocall_oram_path_size(uint32_t oram_kind, uint32_t leaf, size_t* path_bytes)
{
    if (path_bytes == nullptr) {
        return;
    }
    *path_bytes = 0;
    AppOramContext* ctx = app_oram_by_kind(oram_kind);
    if (ctx == nullptr || ctx->storage.empty() || leaf >= ctx->leaf_count || ctx->slot_bytes == 0) {
        return;
    }
    uint32_t current = app_leaf_to_node(*ctx, leaf);
    while (true) {
        *path_bytes += static_cast<size_t>(ctx->bucket_capacity) * ctx->slot_bytes;
        if (current == 1) {
            break;
        }
        current = app_parent_node(current);
    }
}

void ocall_oram_read_path(uint32_t oram_kind, uint32_t leaf, uint8_t* buffer, size_t buffer_len, size_t* written)
{
    if (written == nullptr) {
        return;
    }
    *written = 0;
    AppOramContext* ctx = app_oram_by_kind(oram_kind);
    if (ctx == nullptr || buffer == nullptr || ctx->storage.empty() || leaf >= ctx->leaf_count || ctx->slot_bytes == 0) {
        printf("app: ocall read path invalid kind=%u leaf=%u\n", oram_kind, leaf);
        fflush(stdout);
        return;
    }
    uint8_t* cursor = buffer;
    uint8_t* end = buffer + buffer_len;
    uint32_t current = app_leaf_to_node(*ctx, leaf);
    while (true) {
        const uint32_t offset = app_bucket_offset(*ctx, current);
        for (uint32_t i = 0; i < ctx->bucket_capacity; ++i) {
            if (static_cast<size_t>(end - cursor) < ctx->slot_bytes) {
                printf("app: ocall read path overflow limit=%zu\n", buffer_len);
                fflush(stdout);
                return;
            }
            const size_t slot_index = static_cast<size_t>(offset + i);
            const uint8_t* slot = app_slot_ptr(*ctx, slot_index);
            memcpy(cursor, slot, ctx->slot_bytes);
            cursor += ctx->slot_bytes;
        }
        if (current == 1) {
            break;
        }
        current = app_parent_node(current);
    }
    *written = static_cast<size_t>(cursor - buffer);
}

void ocall_oram_write_path(uint32_t oram_kind, uint32_t leaf, const uint8_t* buffer, size_t buffer_len, size_t* written)
{
    if (!written) return;
    *written = 0;
    AppOramContext* ctx = app_oram_by_kind(oram_kind);
    if (ctx == nullptr || buffer == nullptr || ctx->storage.empty() || leaf >= ctx->leaf_count || ctx->slot_bytes == 0) {
        return;
    }
    const uint8_t* cursor = buffer;
    const uint8_t* end = buffer + buffer_len;
    uint32_t current = app_leaf_to_node(*ctx, leaf);
    while (true) {
        const uint32_t offset = app_bucket_offset(*ctx, current);
        for (uint32_t i = 0; i < ctx->bucket_capacity; ++i) {
            if (static_cast<size_t>(end - cursor) < ctx->slot_bytes) {
                return;
            }
            const size_t slot_index = static_cast<size_t>(offset + i);
            memcpy(app_slot_ptr(*ctx, slot_index), cursor, ctx->slot_bytes);
            ctx->occupied[slot_index] = 1;
            cursor += ctx->slot_bytes;
        }
        if (current == 1) {
            break;
        }
        current = app_parent_node(current);
    }
    *written = static_cast<size_t>(cursor - buffer);
}

void ocall_oram_read_nodes(uint32_t oram_kind, const uint32_t* nodes, size_t node_count,
    uint8_t* buffer, size_t buffer_len, size_t* written)
{
    if (written == nullptr) {
        return;
    }
    *written = 0;
    AppOramContext* ctx = app_oram_by_kind(oram_kind);
    if (ctx == nullptr || nodes == nullptr || buffer == nullptr || node_count == 0
        || ctx->storage.empty() || ctx->slot_bytes == 0) {
        return;
    }
    uint8_t* cursor = buffer;
    uint8_t* end = buffer + buffer_len;
    for (size_t ni = 0; ni < node_count; ++ni) {
        const uint32_t node = nodes[ni];
        const size_t bucket_count = ctx->occupied.size() / ctx->bucket_capacity;
        if (node == 0 || node > bucket_count) {
            continue;
        }
        const uint32_t offset = app_bucket_offset(*ctx, node);
        for (uint32_t i = 0; i < ctx->bucket_capacity; ++i) {
            if (static_cast<size_t>(end - cursor) < ctx->slot_bytes) {
                printf("app: ocall read nodes overflow limit=%zu\n", buffer_len);
                fflush(stdout);
                return;
            }
            const size_t slot_index = static_cast<size_t>(offset + i);
            const uint8_t* slot = app_slot_ptr(*ctx, slot_index);
            memcpy(cursor, slot, ctx->slot_bytes);
            cursor += ctx->slot_bytes;
        }
    }
    *written = static_cast<size_t>(cursor - buffer);
}

void ocall_oram_write_nodes(uint32_t oram_kind, const uint32_t* nodes, size_t node_count,
    const uint8_t* buffer, size_t buffer_len, size_t* written)
{
    if (!written) return;
    *written = 0;
    AppOramContext* ctx = app_oram_by_kind(oram_kind);
    if (ctx == nullptr || nodes == nullptr || buffer == nullptr || node_count == 0
        || ctx->storage.empty() || ctx->slot_bytes == 0) {
        return;
    }
    const uint8_t* cursor = buffer;
    const uint8_t* end = buffer + buffer_len;
    for (size_t ni = 0; ni < node_count; ++ni) {
        const uint32_t node = nodes[ni];
        const size_t bucket_count = ctx->occupied.size() / ctx->bucket_capacity;
        if (node == 0 || node > bucket_count) {
            continue;
        }
        const uint32_t offset = app_bucket_offset(*ctx, node);
        for (uint32_t i = 0; i < ctx->bucket_capacity; ++i) {
            if (static_cast<size_t>(end - cursor) < ctx->slot_bytes) {
                return;
            }
            const size_t slot_index = static_cast<size_t>(offset + i);
            memcpy(app_slot_ptr(*ctx, slot_index), cursor, ctx->slot_bytes);
            ctx->occupied[slot_index] = 1;
            cursor += ctx->slot_bytes;
        }
    }
    *written = static_cast<size_t>(cursor - buffer);
}

namespace {

template <typename T>
void append_pod(std::vector<uint8_t>& out, const T& value)
{
    const uint8_t* src = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), src, src + sizeof(T));
}

uint32_t app_leaf_to_node(const AppOramContext& ctx, uint32_t leaf)
{
    return ctx.leaf_count + leaf;
}

uint32_t app_parent_node(uint32_t node)
{
    return node >> 1;
}

uint32_t app_bucket_offset(const AppOramContext& ctx, uint32_t node)
{
    return (node - 1) * ctx.bucket_capacity;
}

AppOramContext* app_oram_by_kind(uint32_t kind)
{
    if (kind >= 2) {
        return nullptr;
    }
    return &g_app_orams[kind];
}

uint8_t* app_slot_ptr(AppOramContext& ctx, size_t slot_index)
{
    return ctx.storage.data() + slot_index * ctx.slot_bytes;
}

bool init_app_oram_from_blocks(uint32_t oram_kind, uint32_t dim, const std::vector<std::vector<uint8_t>>& blocks)
{
    if (blocks.empty() || oram_kind >= 2) return false;
    AppOramContext& ctx = g_app_orams[oram_kind];
    ctx.leaf_count = 1;
    while (ctx.leaf_count < blocks.size()) ctx.leaf_count <<= 1;
    const size_t slot_count = size_t(ctx.leaf_count * 2 - 1) * ctx.bucket_capacity;
    size_t max_block_bytes = 0;
    for (const auto& block : blocks) max_block_bytes = std::max(max_block_bytes, block.size());
    int retval = -1;
    sgx_status_t status = ecall_obrm_init_external_oram(global_eid, &retval,
        oram_kind, blocks.size(), dim, max_block_bytes);
    if (status != SGX_SUCCESS || retval) return false;
    ctx.slot_bytes = 12 + sizeof(ObrmSgxPathSlotHeader) + 16 + 12 + max_block_bytes + 16;
    ctx.storage.assign(slot_count * ctx.slot_bytes, 0);
    ctx.occupied.assign(slot_count, 0);
    for (const auto& block : blocks) {
        status = ecall_obrm_stage_initial_block(global_eid, &retval, oram_kind, block.data(), block.size());
        if (status != SGX_SUCCESS || retval) return false;
    }
    status = ecall_obrm_finish_initialization(global_eid, &retval, oram_kind);
    return status == SGX_SUCCESS && retval == 0;
}

bool load_exported_blocks(const char* path, ExportHeader& header)
{
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || strncmp(header.magic, "OBRMSGX", 7) != 0 || !header.dim || !header.block_count
        || header.version != 2 || !vector_format::element_bytes(header.encoding)) return false;
    std::vector<std::vector<uint8_t>> blocks;
    for (size_t i = 0; i < header.block_count; ++i) {
        uint32_t size = 0;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!in || !size) return false;
        blocks.emplace_back(size);
        in.read(reinterpret_cast<char*>(blocks.back().data()), size);
        if (!in) return false;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!init_app_oram_from_blocks(0, header.dim, blocks)) return false;
    printf("oram_build_seconds=%f\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    return true;
}

bool read_sized_blocks(std::ifstream& in, size_t count, std::vector<std::vector<uint8_t>>& blocks)
{
    blocks.clear();
    blocks.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        uint32_t block_bytes = 0;
        in.read(reinterpret_cast<char*>(&block_bytes), sizeof(block_bytes));
        if (!in || block_bytes == 0) {
            return false;
        }
        std::vector<uint8_t> block(block_bytes);
        in.read(reinterpret_cast<char*>(block.data()), block.size());
        if (!in) {
            return false;
        }
        blocks.push_back(std::move(block));
    }
    return true;
}

std::vector<uint8_t> pack_sized_blocks(const std::vector<std::vector<uint8_t>>& blocks)
{
    std::vector<uint8_t> out;
    size_t total = 0;
    for (const auto& block : blocks) {
        total += sizeof(uint32_t) + block.size();
    }
    out.reserve(total);
    for (const auto& block : blocks) {
        const uint32_t block_bytes = static_cast<uint32_t>(block.size());
        const uint8_t* size_bytes = reinterpret_cast<const uint8_t*>(&block_bytes);
        out.insert(out.end(), size_bytes, size_bytes + sizeof(block_bytes));
        out.insert(out.end(), block.begin(), block.end());
    }
    return out;
}

bool load_optimized_blocks(const char* path, OptimizedExportHeader& header)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        printf("app: failed to open optimized blocks %s\n", path);
        return false;
    }
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || strncmp(header.magic, "OBRMOPT", 7) != 0 || header.dim == 0
        || header.index_block_count == 0 || header.data_block_count == 0 || header.version!=2 || !vector_format::element_bytes(header.encoding)) {
        printf("app: invalid optimized block file\n");
        return false;
    }

    std::vector<std::vector<uint8_t>> index_blocks;
    std::vector<std::vector<uint8_t>> data_blocks;
    if (!read_sized_blocks(in, header.index_block_count, index_blocks)
        || !read_sized_blocks(in, header.data_block_count, data_blocks)) {
        printf("app: failed to read optimized block payloads\n");
        return false;
    }

    const auto packed_index = pack_sized_blocks(index_blocks);
    const auto cache_start = std::chrono::steady_clock::now();
    int retval = -1;
    sgx_status_t status = ecall_obrm_load_index_cache(
        global_eid, &retval, packed_index.data(), packed_index.size(), index_blocks.size(), header.dim);
    if (status != SGX_SUCCESS) {
        print_sgx_error(status);
        return false;
    }
    if (retval != 0) {
        printf("load optimized index cache failed with retval=%d\n", retval);
        return false;
    }

    const auto cache_end = std::chrono::steady_clock::now();
    printf("index_cache_load_seconds=%f\n", std::chrono::duration<double>(cache_end - cache_start).count());

    const auto oram_start = std::chrono::steady_clock::now();
    const bool ok = init_app_oram_from_blocks(1, header.dim, data_blocks);
    const auto oram_end = std::chrono::steady_clock::now();
    printf("oram_build_seconds=%f\n", std::chrono::duration<double>(oram_end - oram_start).count());
    return ok;
}

} // namespace


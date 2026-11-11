#include "vectors.h"
#include "learned_ann.h"
#include "ObrmSgxTypes.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ExportHeader=ObrmExportHeader;

using OptimizedExportHeader=ObrmOptimizedExportHeader;




uint32_t parse_data_encoding(const std::string& value)
{
    if(value=="uint8") return OBRM_DATA_ENCODING_UINT8;
    if(value=="float16") return OBRM_DATA_ENCODING_FLOAT16;
    throw std::runtime_error("coordinates must use uint8 or float16");
}
void append_coordinates(std::vector<uint8_t>& out,const float* values,size_t count,uint32_t encoding)
{
    for(size_t i=0;i<count;++i) {
        const float value=values[i];
        if(encoding==OBRM_DATA_ENCODING_UINT8) {
            if(!std::isfinite(value) || value<0 || value>255 || std::floor(value)!=value)
                throw std::runtime_error("non-uint8 coordinate in computational view");
            out.push_back(static_cast<uint8_t>(value));
        } else {
            const uint16_t half=vector_format::encode_half(value);
            if(!std::isfinite(value) || vector_format::decode_half(half)!=value)
                throw std::runtime_error("non-float16 coordinate in computational view");
            const auto* raw=reinterpret_cast<const uint8_t*>(&half);
            out.insert(out.end(),raw,raw+sizeof(half));
        }
    }
}


std::size_t parse_size(const std::string& value)
{
    std::size_t pos = 0;
    const auto parsed = std::stoull(value, &pos);
    if (pos != value.size()) {
        throw std::runtime_error("invalid integer: " + value);
    }
    return static_cast<std::size_t>(parsed);
}

float parse_float(const std::string& value)
{
    std::size_t pos = 0;
    const auto parsed = std::stof(value, &pos);
    if (pos != value.size()) {
        throw std::runtime_error("invalid float: " + value);
    }
    return parsed;
}

template <typename T>
void append_pod(std::vector<uint8_t>& out, const T& value)
{
    const auto* src = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), src, src + sizeof(T));
}

template <typename T>
void append_vector(std::vector<uint8_t>& out, const std::vector<T>& values)
{
    if (values.empty()) {
        return;
    }
    const auto* src = reinterpret_cast<const uint8_t*>(values.data());
    out.insert(out.end(), src, src + sizeof(T) * values.size());
}

std::vector<uint8_t> wrap_block(uint32_t type, int32_t block_id, const std::vector<uint8_t>& payload)
{
    std::vector<uint8_t> out;
    const ObrmSgxBlockHeader header = {type, block_id, static_cast<uint32_t>(payload.size())};
    append_pod(out, header);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::vector<uint8_t> serialize_node(const LearnedNodeBlock& node,const FloatVectors& base,uint32_t encoding)
{
    std::vector<uint8_t> payload;
    const auto& links=node.is_leaf?node.data_block_ids:node.child_node_ids;
    const ObrmSgxNodeHeader nh={node.is_leaf?1u:0u,uint32_t(base.dim),uint32_t(node.pivot_ids.size()),uint32_t(links.size()),encoding};
    const ObrmSgxMlpHeader mh={uint32_t(node.model.input_dim()),uint32_t(node.model.hidden_dim()),uint32_t(node.model.output_dim())};
    append_pod(payload,nh);append_pod(payload,mh);
    for(int32_t id:node.pivot_ids) {
        if(id<0 || size_t(id)>=base.count) throw std::runtime_error("invalid pivot identifier");
        append_coordinates(payload,&base.values[size_t(id)*base.dim],base.dim,encoding);
    }
    append_vector(payload,links);
    std::vector<uint32_t> positions;
    for(int32_t id:links) positions.push_back(uint32_t(id<0?0:id));
    append_vector(payload,positions);
    std::vector<float> parameters;node.model.append_parameters(parameters);append_vector(payload,parameters);
    return wrap_block(node.is_leaf?OBRM_BLOCK_LEAF_NODE:OBRM_BLOCK_INTERNAL_NODE,node.logical_id,payload);
}


std::vector<uint8_t> serialize_data_block(const LearnedDataBlock& block,const FloatVectors& base,uint32_t encoding)
{
    std::vector<uint8_t> payload;
    const ObrmSgxDataHeaderV2 h={uint32_t(base.dim),uint32_t(block.point_ids.size()),encoding,0};
    append_pod(payload,h);append_vector(payload,block.point_ids);
    for(int32_t id:block.point_ids) {
        if(id<0 || size_t(id)>=base.count) throw std::runtime_error("invalid data identifier");
        append_coordinates(payload,&base.values[size_t(id)*base.dim],base.dim,encoding);
    }
    return wrap_block(OBRM_BLOCK_DATA,block.logical_id,payload);
}


} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 4) throw std::runtime_error(
            "Usage: obrm_export base.bvecs|hvecs output_prefix count [--K n --N n --B n --m n --r n "
            "--hidden n --epochs n --lr x --seed n --data-encoding uint8|float16]");
        LearnedAnnParams params;
        uint32_t encoding=0;
        const size_t count=parse_size(argv[3]);
        for (int i=4;i<argc;++i) {
            const std::string option=argv[i];
            if (++i>=argc) throw std::runtime_error("missing value for "+option);
            const std::string value=argv[i];
            if (option=="--K") params.K=parse_size(value);
            else if (option=="--N") params.leaf_threshold=parse_size(value);
            else if (option=="--B") params.block_size=parse_size(value);
            else if (option=="--m") params.pivot_count=parse_size(value);
            else if (option=="--r") params.r=parse_size(value);
            else if (option=="--hidden") params.hidden_dim=parse_size(value);
            else if (option=="--epochs") params.epochs=parse_size(value);
            else if (option=="--lr") params.learning_rate=parse_float(value);
            else if (option=="--seed") {
                const auto seed=parse_size(value);
                if (seed>UINT32_MAX) throw std::runtime_error("seed exceeds uint32 range");
                params.seed=static_cast<uint32_t>(seed);
            }
            else if (option=="--data-encoding") encoding=parse_data_encoding(value);
            else throw std::runtime_error("unknown option: "+option);
        }
        if (!count || count>INT32_MAX || params.K<2 || !params.leaf_threshold
            || !params.block_size || !params.pivot_count || !params.r || params.r>32
            || !params.epochs || !params.hidden_dim || !std::isfinite(params.learning_rate)
            || params.learning_rate<=0) throw std::runtime_error("invalid main-experiment parameters");
        auto base=read_vectors(argv[1],encoding,count);
        if (base.count!=count) throw std::runtime_error("insufficient base records");
        LearnedAnnIndex index;
        index.build(base,params);
        const auto snapshot=index.snapshot();
        std::vector<std::vector<uint8_t>> nodes,data;
        for (const auto& node:snapshot.nodes) nodes.push_back(serialize_node(node,base,encoding));
        for (const auto& block:snapshot.data_blocks) data.push_back(serialize_data_block(block,base,encoding));
        auto write_blocks=[](std::ofstream& out,const std::vector<std::vector<uint8_t>>& blocks) {
            for (const auto& block:blocks) {
                if (block.size()>UINT32_MAX) throw std::runtime_error("block exceeds uint32 length");
                uint32_t bytes=static_cast<uint32_t>(block.size());
                out.write(reinterpret_cast<const char*>(&bytes),sizeof(bytes));
                out.write(reinterpret_cast<const char*>(block.data()),block.size());
            }
        };
        const std::string prefix=argv[2];
        {
            std::ofstream out(prefix+".base.blocks",std::ios::binary);
            ExportHeader h={{'O','B','R','M','S','G','X','\0'},2,uint32_t(base.dim),uint32_t(nodes.size()+data.size()),0,encoding};
            out.write(reinterpret_cast<const char*>(&h),sizeof(h));
            write_blocks(out,nodes);write_blocks(out,data);
            if (!out) throw std::runtime_error("failed to write base export");
        }
        {
            std::ofstream out(prefix+".optimized.blocks",std::ios::binary);
            OptimizedExportHeader h={{'O','B','R','M','O','P','T','\0'},2,uint32_t(base.dim),uint32_t(nodes.size()),uint32_t(data.size()),0,encoding};
            out.write(reinterpret_cast<const char*>(&h),sizeof(h));
            write_blocks(out,nodes);write_blocks(out,data);
            if (!out) throw std::runtime_error("failed to write optimized export");
        }
        const auto& stats=index.stats();
        std::cout<<"dataset_vectors="<<base.count<<"\ndim="<<base.dim
            <<"\nindex_blocks="<<nodes.size()<<"\ndata_blocks="<<data.size()
            <<"\ncoordinate_encoding="<<(encoding==vector_format::UInt8?"uint8":"float16")<<"\nseed="<<params.seed<<"\nindex_construction_seconds="<<stats.index_construction_seconds
            <<"\nmodel_training_seconds="<<stats.model_training_seconds<<"\n";
        return 0;
    } catch (const std::exception& e) {std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}


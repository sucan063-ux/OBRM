#include "vectors.h"
#include "VectorFormat.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <utility>

int main(int argc,char** argv)
{
    try {
        if (argc!=7) throw std::runtime_error(
            "Usage: obrm_groundtruth base.bvecs|hvecs query.bvecs|hvecs output.ivecs base_count query_count uint8|float16");
        auto positive=[](const std::string& s) {
            size_t end=0;
            if (s.empty() || s[0]=='-') throw std::runtime_error("invalid count");
            size_t n=std::stoull(s,&end);
            if (!n || end!=s.size()) throw std::runtime_error("invalid count");
            return n;
        };
        size_t n=positive(argv[4]),nq=positive(argv[5]);
        const std::string format=argv[6];
        const uint32_t encoding=format=="uint8"?vector_format::UInt8:format=="float16"?vector_format::Float16:0;
        const auto base=read_vectors(argv[1],encoding,n);
        const auto queries=read_vectors(argv[2],encoding,nq);
        if (base.count!=n || queries.count!=nq || n<10 || base.dim!=queries.dim)
            throw std::runtime_error("insufficient records or dimension mismatch");
        for (float v:base.values) if (!std::isfinite(v)) throw std::runtime_error("non-finite base");
        for (float v:queries.values) if (!std::isfinite(v)) throw std::runtime_error("non-finite query");
        IntVectors gt;gt.count=nq;gt.dim=10;gt.values.resize(nq*10);
        using Hit=std::pair<float,int32_t>;
        for (size_t qi=0;qi<nq;++qi) {
            std::priority_queue<Hit> heap;
            const float* q=&queries.values[qi*base.dim];
            for (size_t id=0;id<n;++id) {
                Hit hit{squared_l2(q,&base.values[id*base.dim],base.dim),static_cast<int32_t>(id)};
                if (heap.size()<10) heap.push(hit);
                else if (hit<heap.top()) {heap.pop();heap.push(hit);}
            }
            for (size_t j=10;j>0;--j) {gt.values[qi*10+j-1]=heap.top().second;heap.pop();}
        }
        write_ivecs(argv[3],gt);
        std::cout<<"base_count="<<n<<"\nquery_count="<<nq<<"\ngroundtruth_k=10\n";
        return 0;
    } catch (const std::exception& e) {std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}

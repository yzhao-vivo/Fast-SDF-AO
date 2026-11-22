// bake_sdf: clipped signed distance volume for SDF ambient occlusion, from closed triangle meshes.
//
// Distances are exact (closest point on the nearest triangle, not on surface samples).
// Sign comes from the winding number along a z ray through every voxel column: each crossing
// counts +1 entering and -1 leaving (from the triangle's orientation), and a voxel is inside where
// the total is nonzero. Overlapping closed shells (posts through shelves) are handled without
// needing to know which triangles belong to which object. Meshes must be closed and consistently wound.
// Threads: the distance pass runs on --threads + 1 workers (at least 2, if the machine has the cores), and
// the sign pass runs on its own thread alongside the BVH build and distance pass (inside voxels negated at the end).
// Far space is skipped exactly: an 8^3 block whose centre is farther than clip + half-diagonal
// from every triangle is filled with +/-clip without per-voxel queries (distance is 1-Lipschitz).
//
// Input  --boxes scene.json   JSON array of [cx,cy,cz,hx,hy,hz] boxes
//        --obj   scene.obj    v / f lines (polygons are fanned)
//        --points cloud.ply   oriented point cloud (scan, photogrammetry, splat centres): x y z [nx ny nz],
//                             ascii or binary little-endian. Distance to the nearest point, signed by that
//                             point's normal (unsigned when there are no normals); same culling as meshes.
//        --sample N           with --boxes/--obj: bake from N area-uniform samples of the mesh instead,
//                             dropping samples on surfaces covered by other geometry (what a scan would see);
//                             --write-points f.ply saves them.
// Output float32, x fastest, then y, then z; voxel i at lo + (i + 0.5) * span / res. Negative inside.
//
//   bake_sdf --boxes archive.json --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --out archive.f32
// Check:   --validate K compares K random voxels with brute force (every triangle or point, and the
//          winding number of every triangle); any mismatch exits 1. --seed S picks the voxels and samples.
// Unsigned: --unsigned skips the sign pass (distances only) for meshes that are not closed, e.g. scanned or
//          architectural models whose holes would make the winding number wrong in whole voxel columns.
// Vote:    --sign vote takes the sign by majority of the winding numbers along +x, +y and +z rays, so a hole
//          that corrupts the columns of one axis is outvoted by the other two (nearly closed meshes such as
//          architecture). Default --sign z is the single +z winding number.
// Options: --threads N  --block B (0 disables culling)  --reach r (AO reach; distances past r + 1 voxel diagonal are not resolved)  --search bvh-neighbour|bvh|grid  --cell metres  --subdiv n (boxes)  --stats file.json
// Build:   c++ -O3 -std=c++17 -pthread bake_sdf.cpp -o bake_sdf    (no dependencies)
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <memory>
#include <cstddef>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

using V3 = std::array<double,3>;
static V3 operator-(const V3& a,const V3& b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static V3 operator+(const V3& a,const V3& b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
static V3 operator*(const V3& a,double k){return {a[0]*k,a[1]*k,a[2]*k};}
static double dot3(const V3& a,const V3& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
using Clk = std::chrono::steady_clock;
static double since(Clk::time_point t){return std::chrono::duration<double,std::milli>(Clk::now()-t).count();}

struct Tri { V3 a,b,c; int comp; Tri(){} Tri(const V3& a_,const V3& b_,const V3& c_,int k):a(a_),b(b_),c(c_),comp(k){} };

// Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
template<class T> static inline __attribute__((always_inline)) double triDist2(const V3& p,const T& t){
    const V3 &a=t.a,&b=t.b,&c=t.c; V3 ab=b-a,ac=c-a,ap=p-a; double d1=dot3(ab,ap),d2=dot3(ac,ap);
    auto at=[&](const V3& q){V3 d=p-q;return dot3(d,d);};
    if(d1<=0&&d2<=0)return at(a);
    V3 bp=p-b; double d3=dot3(ab,bp),d4=dot3(ac,bp); if(d3>=0&&d4<=d3)return at(b);
    double vc=d1*d4-d3*d2; if(vc<=0&&d1>=0&&d3<=0){double v=d1/(d1-d3);return at(a+ab*v);}
    V3 cp=p-c; double d5=dot3(ab,cp),d6=dot3(ac,cp); if(d6>=0&&d5<=d6)return at(c);
    double vb=d5*d2-d1*d6; if(vb<=0&&d2>=0&&d6<=0){double w=d2/(d2-d6);return at(a+ac*w);}
    double va=d3*d6-d5*d4; if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){double w=(d4-d3)/((d4-d3)+(d5-d6));return at(b+(c-b)*w);}
    double den=1/(va+vb+vc),v=vb*den,w=vc*den; return at(a+ab*v+ac*w);
}

// ------------------------------------------------------------------ input
static std::vector<double> numbers(const std::string& s){
    std::vector<double> v; const char* p=s.c_str();
    while(*p){ if((*p>='0'&&*p<='9')||*p=='-'||*p=='+'||*p=='.'){char* e;double x=std::strtod(p,&e);if(e!=p){v.push_back(x);p=e;continue;}} ++p; }
    return v;
}
static std::vector<Tri> loadBoxes(const std::string& path,int subdiv=1,int threads=1){
    std::ifstream in(path); if(!in)throw std::runtime_error("cannot read "+path);
    std::stringstream ss; ss<<in.rdbuf(); auto v=numbers(ss.str());
    if(v.empty()||v.size()%6)throw std::runtime_error("expected a JSON array of [cx,cy,cz,hx,hy,hz]");
    // Presized (each box makes exactly 12*subdiv^2 triangles) and filled in parallel, box by box.
    const size_t per=size_t(12)*subdiv*subdiv,nb=v.size()/6; std::vector<Tri> t(nb*per);
    for(size_t i=0;i<v.size();i+=6)if(!(v[i+3]>0&&v[i+4]>0&&v[i+5]>0))throw std::runtime_error("box half-sizes must be positive");
    // 12 outward-wound triangles per box.
    static const int faces[6][4]={{0,2,3,1},{4,5,7,6},{0,1,5,4},{2,6,7,3},{0,4,6,2},{1,3,7,5}};
    std::atomic<size_t> nextBox{0};
    auto fill=[&]{for(size_t bi;(bi=nextBox++)<nb;){const size_t i=bi*6; Tri* o=t.data()+bi*per;
        V3 c{v[i],v[i+1],v[i+2]},h{v[i+3],v[i+4],v[i+5]};
        V3 p[8]; for(int k=0;k<8;++k)p[k]={c[0]+((k&1)?h[0]:-h[0]),c[1]+((k&2)?h[1]:-h[1]),c[2]+((k&4)?h[2]:-h[2])};
        // Each face as an n x n grid of quads (n = subdiv), to test dense meshes with a known exact answer.
        for(auto& f:faces){const V3 &q0=p[f[0]],&q1=p[f[1]],&q2=p[f[2]],&q3=p[f[3]];
            auto at=[&](double u,double v){return q0*((1-u)*(1-v))+q1*(u*(1-v))+q2*(u*v)+q3*((1-u)*v);};
            for(int a=0;a<subdiv;++a)for(int b=0;b<subdiv;++b){double u0=double(a)/subdiv,u1=double(a+1)/subdiv,v0=double(b)/subdiv,v1=double(b+1)/subdiv;
                *o++=Tri(at(u0,v0),at(u1,v0),at(u1,v1),int(i/6));*o++=Tri(at(u0,v0),at(u1,v1),at(u0,v1),int(i/6));}}
    }};
    const int nt=int(std::min<size_t>(size_t(threads),std::max<size_t>(1,nb*per/20000)));
    std::vector<std::thread> pool; for(int k=1;k<nt;++k)pool.emplace_back(fill); fill(); for(auto& p:pool)p.join();
    return t;
}
static std::vector<Tri> loadObj(const std::string& path){
    std::ifstream in(path); if(!in)throw std::runtime_error("cannot read "+path);
    std::vector<V3> v; std::vector<Tri> t; std::string line;
    while(std::getline(in,line)){
        if(line.rfind("v ",0)==0){auto n=numbers(line.substr(2));if(n.size()<3)throw std::runtime_error("bad vertex");v.push_back({n[0],n[1],n[2]});}
        else if(line.rfind("f ",0)==0){
            std::vector<int> f; std::istringstream ls(line.substr(2)); std::string tok;
            while(ls>>tok){int k=std::atoi(tok.c_str());if(k<0)k=int(v.size())+k+1;if(k<1||k>int(v.size()))throw std::runtime_error("bad face index");f.push_back(k-1);}
            for(size_t k=2;k<f.size();++k)t.push_back(Tri(v[f[0]],v[f[k-1]],v[f[k]],-1));
        }
    }
    return t;
}

// ------------------------------------------------------------------ oriented point clouds
// A scan, photogrammetry or splat export: positions, plus normals when the source has them.
struct PF { float x,y,z; };
struct PointCloud { std::vector<PF> p,n; bool oriented=false; size_t hidden=0; };
static inline double pd2(const V3& q,const PF& p){const double dx=q[0]-p.x,dy=q[1]-p.y,dz=q[2]-p.z;return dx*dx+dy*dy+dz*dz;}
// PLY (ascii or binary_little_endian), vertex element first, scalar properties x y z [nx ny nz].
// Normals that are missing or all zero (e.g. 3D Gaussian splat exports) give an unsigned field.
static PointCloud loadPly(const std::string& path){
    std::ifstream in(path,std::ios::binary); if(!in)throw std::runtime_error("cannot read "+path);
    std::string line; std::getline(in,line); if(line.rfind("ply",0)!=0)throw std::runtime_error(path+" is not a PLY file");
    bool ascii=false,inVertex=false; size_t nv=0; int elem=0;
    struct Prop { std::string name; int size; char kind; }; std::vector<Prop> props;   // kind: i(signed) u f
    auto typeOf=[&](const std::string& t,int& size,char& kind){
        if(t=="char"||t=="int8"){size=1;kind='i';}else if(t=="uchar"||t=="uint8"){size=1;kind='u';}
        else if(t=="short"||t=="int16"){size=2;kind='i';}else if(t=="ushort"||t=="uint16"){size=2;kind='u';}
        else if(t=="int"||t=="int32"){size=4;kind='i';}else if(t=="uint"||t=="uint32"){size=4;kind='u';}
        else if(t=="float"||t=="float32"){size=4;kind='f';}else if(t=="double"||t=="float64"){size=8;kind='f';}
        else throw std::runtime_error("unsupported PLY property type "+t);};
    while(std::getline(in,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream ls(line); std::string w; ls>>w;
        if(w=="format"){std::string f;ls>>f;ascii=f=="ascii";if(!ascii&&f!="binary_little_endian")throw std::runtime_error("unsupported PLY format "+f);}
        else if(w=="element"){std::string e;size_t c=0;ls>>e>>c;inVertex=e=="vertex";if(inVertex){if(elem)throw std::runtime_error("PLY vertex element must come first");nv=c;}++elem;}
        else if(w=="property"&&inVertex){std::string t,nm;ls>>t;if(t=="list")throw std::runtime_error("PLY vertex lists are not supported");ls>>nm;Prop p{nm,0,0};typeOf(t,p.size,p.kind);props.push_back(p);}
        else if(w=="end_header")break;
    }
    auto find=[&](const char* nm){for(size_t k=0;k<props.size();++k)if(props[k].name==nm)return int(k);return -1;};
    const int ix=find("x"),iy=find("y"),iz=find("z"),inx=find("nx"),iny=find("ny"),inz=find("nz");
    if(ix<0||iy<0||iz<0)throw std::runtime_error("PLY vertices need x y z");
    const bool hasN=inx>=0&&iny>=0&&inz>=0;
    PointCloud pc; pc.p.resize(nv); if(hasN)pc.n.resize(nv);
    size_t rec=0; std::vector<size_t> off(props.size()); for(size_t k=0;k<props.size();++k){off[k]=rec;rec+=size_t(props[k].size);}
    std::vector<double> v(props.size()); std::vector<char> buf(rec);
    for(size_t i=0;i<nv;++i){
        if(ascii){for(auto& x:v)if(!(in>>x))throw std::runtime_error("PLY ended early");}
        else{
            if(!in.read(buf.data(),std::streamsize(rec)))throw std::runtime_error("PLY ended early");
            for(size_t k=0;k<props.size();++k){const char* b=buf.data()+off[k]; const Prop& p=props[k];
                if(p.kind=='f'){if(p.size==4){float f;std::memcpy(&f,b,4);v[k]=f;}else{double d;std::memcpy(&d,b,8);v[k]=d;}}
                else if(p.kind=='i'){int64_t x=0;std::memcpy(&x,b,size_t(p.size));const int sh=64-8*p.size;v[k]=double((x<<sh)>>sh);}
                else{uint64_t x=0;std::memcpy(&x,b,size_t(p.size));v[k]=double(x);}}
        }
        pc.p[i]={float(v[ix]),float(v[iy]),float(v[iz])};
        if(hasN){double nx=v[inx],ny=v[iny],nz=v[inz],l=std::sqrt(nx*nx+ny*ny+nz*nz);
            if(l>0){pc.oriented=true;nx/=l;ny/=l;nz/=l;} pc.n[i]={float(nx),float(ny),float(nz)};}
    }
    if(nv==0)throw std::runtime_error("PLY has no vertices");
    if(!pc.oriented)pc.n.clear();
    return pc;
}
static void writePly(const std::string& path,const PointCloud& pc){
    std::ofstream o(path,std::ios::binary); if(!o)throw std::runtime_error("cannot write "+path);
    o<<"ply\nformat binary_little_endian 1.0\nelement vertex "<<pc.p.size()<<"\nproperty float x\nproperty float y\nproperty float z\n";
    if(pc.oriented)o<<"property float nx\nproperty float ny\nproperty float nz\n";
    o<<"end_header\n";
    for(size_t i=0;i<pc.p.size();++i){o.write((const char*)&pc.p[i],12);if(pc.oriented)o.write((const char*)&pc.n[i],12);}
}
// Inside = nonzero winding number of the closed mesh along a +z ray through each point (as in insideMask,
// but for arbitrary points: triangles binned by xy footprint on a coarse grid, every crossing below counted).
static std::vector<uint8_t> insidePoints(const std::vector<Tri>& t,const std::vector<V3>& q,int threads){
    double x0=INFINITY,x1=-INFINITY,y0=INFINITY,y1=-INFINITY;
    for(auto& tr:t)for(const V3* p:{&tr.a,&tr.b,&tr.c}){x0=std::min(x0,(*p)[0]);x1=std::max(x1,(*p)[0]);y0=std::min(y0,(*p)[1]);y1=std::max(y1,(*p)[1]);}
    const int G=std::clamp(int(2*std::sqrt(double(t.size()))),1,1024);
    const double sx=std::max(x1-x0,1e-12)/G,sy=std::max(y1-y0,1e-12)/G;
    auto cx=[&](double x){return std::clamp(int((x-x0)/sx),0,G-1);}; auto cy=[&](double y){return std::clamp(int((y-y0)/sy),0,G-1);};
    std::vector<uint32_t> start(size_t(G)*G+1,0),items;
    for(int pass=0;pass<2;++pass){
        std::vector<uint32_t> fill; if(pass){for(size_t c=1;c<start.size();++c)start[c]+=start[c-1];items.resize(start.back());fill.assign(start.begin(),start.end()-1);}
        for(uint32_t i=0;i<t.size();++i){const Tri& tr=t[i];
            // Constant-x or constant-y triangles can't pass the edge test (see insideMask).
            if((tr.a[0]==tr.b[0]&&tr.a[0]==tr.c[0])||(tr.a[1]==tr.b[1]&&tr.a[1]==tr.c[1]))continue;
            const int i0=cx(std::min({tr.a[0],tr.b[0],tr.c[0]})),i1=cx(std::max({tr.a[0],tr.b[0],tr.c[0]}));
            const int j0=cy(std::min({tr.a[1],tr.b[1],tr.c[1]})),j1=cy(std::max({tr.a[1],tr.b[1],tr.c[1]}));
            for(int j=j0;j<=j1;++j)for(int k=i0;k<=i1;++k){const size_t c=size_t(j)*G+k;if(pass)items[fill[c]++]=i;else ++start[c+1];}}
    }
    std::vector<uint8_t> in(q.size(),0); std::atomic<size_t> next{0};
    auto work=[&]{
        for(size_t b;(b=next.fetch_add(4096))<q.size();)for(size_t k=b;k<std::min(q.size(),b+4096);++k){
            const double x=q[k][0],y=q[k][1],z=q[k][2]; if(x<x0||x>x1||y<y0||y>y1)continue;
            const size_t c=size_t(cy(y))*G+cx(x); int winding=0;
            for(uint32_t e=start[c];e<start[c+1];++e){const Tri& tr=t[items[e]];
                double ax=tr.a[0]-x,ay=tr.a[1]-y,bx=tr.b[0]-x,by=tr.b[1]-y,ccx=tr.c[0]-x,ccy=tr.c[1]-y;
                double w0=bx*ccy-by*ccx,w1=ccx*ay-ccy*ax,w2=ax*by-ay*bx;
                if(!((w0>0&&w1>0&&w2>0)||(w0<0&&w1<0&&w2<0)))continue;
                const double s=w0+w1+w2; if((w0*tr.a[2]+w1*tr.b[2]+w2*tr.c[2])/s<z)winding+=s>0?-1:1;}
            in[k]=winding!=0;}
    };
    std::vector<std::thread> pool; for(int k=1;k<threads;++k)pool.emplace_back(work); work(); for(auto& p:pool)p.join();
    return in;
}
// Area-uniform samples of a closed mesh with outward normals. A sample is dropped as hidden when a point a
// hair outside it, along its normal, is inside the mesh: that surface is covered by other geometry (a box
// standing on the floor), so a scanner would not see it.
static PointCloud sampleMesh(const std::vector<Tri>& t,size_t n,uint64_t seed,int threads){
    std::vector<double> cdf(t.size()); double tot=0,ext=0;
    for(size_t i=0;i<t.size();++i){const V3 ab=t[i].b-t[i].a,ac=t[i].c-t[i].a;const V3 c{ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]};
        tot+=.5*std::sqrt(dot3(c,c));cdf[i]=tot;for(int k=0;k<3;++k)ext=std::max(ext,std::fabs(t[i].a[k]));}
    std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0,1);
    std::vector<V3> pos(n),nrm(n),probe(n);
    for(size_t k=0;k<n;++k){
        const size_t i=std::min(size_t(std::lower_bound(cdf.begin(),cdf.end(),U(rng)*tot)-cdf.begin()),t.size()-1);
        const double s=std::sqrt(U(rng)),r=U(rng); const Tri& tr=t[i];
        pos[k]=tr.a*(1-s)+tr.b*(s*(1-r))+tr.c*(s*r);
        const V3 ab=tr.b-tr.a,ac=tr.c-tr.a; V3 c{ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]};
        const double l=std::sqrt(dot3(c,c)); nrm[k]=l>0?c*(1/l):V3{0,0,1}; probe[k]=pos[k]+nrm[k]*(1e-6*std::max(ext,1.));
    }
    const auto hid=insidePoints(t,probe,threads);
    PointCloud pc; pc.oriented=true;
    for(size_t k=0;k<n;++k){if(hid[k]){++pc.hidden;continue;}
        pc.p.push_back({float(pos[k][0]),float(pos[k][1]),float(pos[k][2])});pc.n.push_back({float(nrm[k][0]),float(nrm[k][1]),float(nrm[k][2])});}
    return pc;
}

// ------------------------------------------------------------------ triangle grid
struct TriGrid {
    V3 lo; double cell; std::array<int,3> dim;
    std::vector<uint32_t> start,items;
    TriGrid(const std::vector<Tri>& t,double cellSize){
        cell=cellSize; V3 hi; lo=t[0].a; hi=lo;
        for(auto& x:t)for(const V3* p:{&x.a,&x.b,&x.c})for(int k=0;k<3;++k){lo[k]=std::min(lo[k],(*p)[k]);hi[k]=std::max(hi[k],(*p)[k]);}
        for(int k=0;k<3;++k)dim[k]=std::max(1,int(std::ceil((hi[k]-lo[k])/cell))+1);
        start.assign(size_t(dim[0])*dim[1]*dim[2]+1,0);
        auto range=[&](const Tri& x,std::array<int,3>& a,std::array<int,3>& b){for(int k=0;k<3;++k){double mn=std::min({x.a[k],x.b[k],x.c[k]}),mx=std::max({x.a[k],x.b[k],x.c[k]});a[k]=cellOf(mn,k);b[k]=cellOf(mx,k);}};
        for(int pass=0;pass<2;++pass){
            std::vector<uint32_t> fill; if(pass){for(size_t i=1;i<start.size();++i)start[i]+=start[i-1];items.resize(start.back());fill.assign(start.begin(),start.end()-1);}
            for(uint32_t i=0;i<t.size();++i){std::array<int,3> a,b;range(t[i],a,b);
                for(int z=a[2];z<=b[2];++z)for(int y=a[1];y<=b[1];++y)for(int x=a[0];x<=b[0];++x){size_t c=(size_t(z)*dim[1]+y)*dim[0]+x;if(pass)items[fill[c]++]=i;else ++start[c+1];}}
        }
    }
    int cellOf(double v,int k) const {return std::clamp(int(std::floor((v-lo[k])/cell)),0,dim[k]-1);}
};
// Exact nearest triangle distance, searched shell by shell; returns +inf if none within `limit`.
// `bound` (if finite) is a known upper bound on the answer, e.g. from a seed triangle.
struct Searcher {
    const std::vector<Tri>& t; const TriGrid& g; std::vector<uint32_t> stamp; uint32_t q=0; uint64_t tested=0;
    Searcher(const std::vector<Tri>& tris,const TriGrid& grid):t(tris),g(grid),stamp(tris.size(),0){}
    double nearest(const V3& p,double limit,double bound=INFINITY){
        if(++q==0){std::fill(stamp.begin(),stamp.end(),0);q=1;}
        double best2=bound*bound; std::array<int,3> c; for(int k=0;k<3;++k)c[k]=g.cellOf(p[k],k);
        const int maxR=std::max({g.dim[0],g.dim[1],g.dim[2]});
        for(int r=0;r<=maxR;++r){
            // Lower bound on anything outside the visited (2r+1)^3 block.
            double outside=INFINITY;
            for(int k=0;k<3;++k){double a=p[k]-(g.lo[k]+(c[k]-r)*g.cell),b=g.lo[k]+(c[k]+r+1)*g.cell-p[k];
                if(c[k]-r>0)outside=std::min(outside,a); if(c[k]+r<g.dim[k]-1)outside=std::min(outside,b);}
            for(int z=c[2]-r;z<=c[2]+r;++z){if(z<0||z>=g.dim[2])continue;
             for(int y=c[1]-r;y<=c[1]+r;++y){if(y<0||y>=g.dim[1])continue;
              const bool edge=std::abs(z-c[2])==r||std::abs(y-c[1])==r;
              for(int x=c[0]-r;x<=c[0]+r;x+=(edge||r==0)?1:2*r){if(x<0||x>=g.dim[0])continue;
                const size_t cc=(size_t(z)*g.dim[1]+y)*g.dim[0]+x;
                for(uint32_t k=g.start[cc];k<g.start[cc+1];++k){uint32_t i=g.items[k];if(stamp[i]==q)continue;stamp[i]=q;++tested;best2=std::min(best2,triDist2(p,t[i]));}
              }}}
            const double best=std::sqrt(best2);
            if(best<=outside)return best<=limit?best:INFINITY;
            if(outside>limit)return INFINITY;      // nothing unvisited can be within the limit, and nothing visited was
        }
        const double best=std::sqrt(best2); return best<=limit?best:INFINITY;
    }
};

// ------------------------------------------------------------------ triangle BVH
// Binary BVH, binned-SAH split on centroids, leaves of up to BVH_LEAF triangles.
// Closest-triangle queries visit the nearer child first and skip boxes beyond the best distance,
// so a good starting bound (e.g. the previous voxel's triangle) prunes most of the tree.
#ifndef BVH_BINS
#define BVH_BINS 16
#endif
#ifndef BVH_SAH_MIN
#define BVH_SAH_MIN 0   // ranges of at most this many triangles use the median split (SAH all the way down is fastest to query)
#endif
#ifndef BVH_LEAF
#define BVH_LEAF 1   // one triangle per leaf: Archive -21% vs 4, dense mesh unchanged, output identical
#endif
static_assert(BVH_LEAF==1,"compact BVH nodes inline one-triangle leaves");
struct BVH {
    // Build primitive: triangle bounds, centroid and index, partitioned in place (contiguous, cache friendly).
    // float32 halves the build's memory traffic. Bounds are rounded outward, so node boxes are
    // conservative (at most one float ulp larger) and queries stay exact.
    using F3 = std::array<float,3>;
    struct Prim { F3 lo,hi,c; uint32_t id; };
    static float down(double x){float f=float(x);return double(f)>x?std::nextafter(f,-INFINITY):f;}
    static float up(double x){float f=float(x);return double(f)<x?std::nextafter(f,INFINITY):f;}
    const std::vector<Tri>& t; std::vector<Prim> prim;
    // Query layout: one 64-byte node per internal node, holding both children's boxes (float32,
    // structure of arrays: [axis][child]) and child references. A reference >= 0 is an internal
    // node; < 0 is a one-triangle leaf, ~ref the leaf's slot in `pt` (leaves get no node of their own).
    // The float boxes are the build's boxes exactly (those were already float values, rounded
    // outward), and box distances are still computed in double. Query distances are exact for any
    // tree shape; the shape only changes how much gets pruned.
    struct alignas(64) CNode { float lo[3][2],hi[3][2]; int child[2]; };
    // Triangles in leaf order: leaf ref ~k is pt[k], next to its siblings in memory.
    struct PT { V3 a,b,c; }; std::vector<PT> pt; std::vector<uint32_t> leafId;   // leafId[k]: index into the input triangles
    std::vector<CNode> cn; int rootRef=0; V3 rootLo,rootHi;
    // Box and centroid bounds of a range.
    struct BB { F3 lo{INFINITY,INFINITY,INFINITY},hi{-INFINITY,-INFINITY,-INFINITY},clo{INFINITY,INFINITY,INFINITY},chi{-INFINITY,-INFINITY,-INFINITY};
        void add(const Prim& q){for(int k=0;k<3;++k){lo[k]=std::min(lo[k],q.lo[k]);hi[k]=std::max(hi[k],q.hi[k]);clo[k]=std::min(clo[k],q.c[k]);chi[k]=std::max(chi[k],q.c[k]);}}
        void add(const BB& b){for(int k=0;k<3;++k){lo[k]=std::min(lo[k],b.lo[k]);hi[k]=std::max(hi[k],b.hi[k]);clo[k]=std::min(clo[k],b.clo[k]);chi[k]=std::max(chi[k],b.chi[k]);}} };
    // Binned SAH, built straight into the final pre-order layout: a subtree of n triangles has
    // n-1 internal nodes, so node cid's left child is cid+1 and its right child cid+(m-s).
    // Child bounds are unions of the chosen axis' bins (min/max are exact), so no per-node bounds pass.
    // threads > 1 builds the top subtrees in parallel (disjoint prim ranges and node slots).
    explicit BVH(const std::vector<Tri>& tris,int threads=1):t(tris){
        const int n=int(t.size()); prim.resize(n); BB all;
        for(int i=0;i<n;++i){Prim& q=prim[i];const V3 c=(t[i].a+t[i].b+t[i].c)*(1./3);q.id=uint32_t(i);
            for(int k=0;k<3;++k){q.c[k]=float(c[k]);q.lo[k]=down(std::min({t[i].a[k],t[i].b[k],t[i].c[k]}));q.hi[k]=up(std::max({t[i].a[k],t[i].b[k],t[i].c[k]}));}
            all.add(q);}
        for(int k=0;k<3;++k){rootLo[k]=all.lo[k];rootHi[k]=all.hi[k];}
        int depth=0; while((1<<depth)<2*threads&&depth<7)++depth;   // up to 2x as many subtrees as threads (SAH splits are uneven)
        if(threads<=1)depth=0;
        if(n>=2){cn.resize(size_t(n)-1);rootRef=0;build(0,n,0,all,depth);}else rootRef=~0;
        pt.resize(n); leafId.resize(n);
        for(int k=0;k<n;++k){const Tri& x=t[prim[k].id];pt[k]={x.a,x.b,x.c};leafId[k]=prim[k].id;}
        std::vector<Prim>().swap(prim);
    }
    void build(int s,int e,int cid,const BB& bb,int depth){
        CNode& cnode=cn[cid];
        auto setChild=[&](int j,const F3& lo,const F3& hi,int r){for(int k=0;k<3;++k){cnode.lo[k][j]=lo[k];cnode.hi[k][j]=hi[k];}cnode.child[j]=r;};
        if(e-s==2){setChild(0,prim[s].lo,prim[s].hi,~s);setChild(1,prim[s+1].lo,prim[s+1].hi,~(s+1));return;}
        int m=s; BB L,R; bool done=false;
        {
            // Small ranges use fewer bins (at most one per triangle).
            const int NB=std::min(BVH_BINS,e-s);
            // Bin bounds live in separate lo[] and hi[] arrays (4 floats each, lane 3 unused): a per-prim
            // read-modify-write of a {lo,hi} pair compiles to ldp/stp q-pairs that miss store-to-load
            // forwarding. Only the NB bins in use are initialised.
            alignas(16) float bLo[3][BVH_BINS][4],bHi[3][BVH_BINS][4]; int bN[3][BVH_BINS]; float sc[3];
            for(int ax=0;ax<3;++ax){const float ext=bb.chi[ax]-bb.clo[ax];sc[ax]=ext>0?float(NB)/ext:0;}
            for(int ax=0;ax<3;++ax)for(int b=0;b<NB;++b){for(int k=0;k<4;++k){bLo[ax][b][k]=INFINITY;bHi[ax][b][k]=-INFINITY;}bN[ax][b]=0;}
            auto binOf=[&](const Prim& q,int ax){return std::min(NB-1,int((q.c[ax]-bb.clo[ax])*sc[ax]));};
            const bool use0=sc[0]>0,use1=sc[1]>0,use2=sc[2]>0;
#if defined(__aarch64__)
            {static_assert(sizeof(Prim)==40&&offsetof(Prim,c)==24,"4-wide prim loads");
             const float32x4_t clo4={bb.clo[0],bb.clo[1],bb.clo[2],0.f},sc4={sc[0],sc[1],sc[2],0.f};
             const int32x4_t nb1=vdupq_n_s32(NB-1);
             auto put=[&](int ax,int b,float32x4_t ql,float32x4_t qh){++bN[ax][b];
                 vst1q_f32(bLo[ax][b],vminq_f32(vld1q_f32(bLo[ax][b]),ql)); vst1q_f32(bHi[ax][b],vmaxq_f32(vld1q_f32(bHi[ax][b]),qh));};
             for(int i=s;i<e;++i){const float* q=&prim[i].lo[0];
                const float32x4_t ql=vld1q_f32(q),qh=vld1q_f32(q+3),qc=vld1q_f32(q+6);
                // Same float ops as binOf: (c - clo) * sc, truncated toward zero, clamped to NB-1.
                const int32x4_t bi=vminq_s32(vcvtq_s32_f32(vmulq_f32(vsubq_f32(qc,clo4),sc4)),nb1);
                if(use0)put(0,vgetq_lane_s32(bi,0),ql,qh); if(use1)put(1,vgetq_lane_s32(bi,1),ql,qh); if(use2)put(2,vgetq_lane_s32(bi,2),ql,qh);}}
#else
            for(int i=s;i<e;++i){const Prim& q=prim[i];
                for(int ax=0;ax<3;++ax){if(!(sc[ax]>0))continue;const int b=binOf(q,ax);++bN[ax][b];
                    for(int k=0;k<3;++k){bLo[ax][b][k]=std::min(bLo[ax][b][k],q.lo[k]);bHi[ax][b][k]=std::max(bHi[ax][b][k],q.hi[k]);}}}
#endif
            (void)use0;(void)use1;(void)use2;
            auto area=[](const F3& a,const F3& b){float x=b[0]-a[0],y=b[1]-a[1],z=b[2]-a[2];return x*y+y*z+z*x;};
            double bestCost=INFINITY; int bestAx=-1,bestB=0;
            for(int ax=0;ax<3;++ax){
                if(!(sc[ax]>0))continue;
                float rArea[BVH_BINS]; int rCnt[BVH_BINS]; F3 lo={INFINITY,INFINITY,INFINITY},hi={-INFINITY,-INFINITY,-INFINITY}; int c=0;
                for(int b=NB-1;b>0;--b){c+=bN[ax][b];for(int k=0;k<3;++k){lo[k]=std::min(lo[k],bLo[ax][b][k]);hi[k]=std::max(hi[k],bHi[ax][b][k]);}rCnt[b]=c;rArea[b]=c?area(lo,hi):0;}
                lo={INFINITY,INFINITY,INFINITY};hi={-INFINITY,-INFINITY,-INFINITY}; c=0;
                for(int b=0;b<NB-1;++b){c+=bN[ax][b];for(int k=0;k<3;++k){lo[k]=std::min(lo[k],bLo[ax][b][k]);hi[k]=std::max(hi[k],bHi[ax][b][k]);}
                    if(!c||!rCnt[b+1])continue;
                    const double cost=double(area(lo,hi))*c+double(rArea[b+1])*rCnt[b+1];
                    if(cost<bestCost){bestCost=cost;bestAx=ax;bestB=b+1;}}
            }
            if(bestAx>=0){
                const int ax=bestAx;
                // Hoare partition that also takes each side's centroid bounds as it goes; box bounds
                // are the unions of the chosen axis' bins.
                F3 cl0=L.clo,ch0=L.chi,cl1=R.clo,ch1=R.chi;
                auto addC=[](F3& cl,F3& ch,const Prim& q){for(int k=0;k<3;++k){cl[k]=std::min(cl[k],q.c[k]);ch[k]=std::max(ch[k],q.c[k]);}};
                int i=s,j=e;
                for(;;){
                    while(i<j&&binOf(prim[i],ax)<bestB){addC(cl0,ch0,prim[i]);++i;}
                    while(i<j&&!(binOf(prim[j-1],ax)<bestB)){addC(cl1,ch1,prim[j-1]);--j;}
                    if(i>=j)break;
                    std::swap(prim[i],prim[j-1]); addC(cl0,ch0,prim[i]); addC(cl1,ch1,prim[j-1]); ++i; --j;
                }
                m=i;
                if(m>s&&m<e){
                    L.clo=cl0;L.chi=ch0;R.clo=cl1;R.chi=ch1;
                    for(int b=0;b<NB;++b){BB& o=b<bestB?L:R;for(int k=0;k<3;++k){o.lo[k]=std::min(o.lo[k],bLo[ax][b][k]);o.hi[k]=std::max(o.hi[k],bHi[ax][b][k]);}}
                    done=true;}
            }
        }
        if(!done){
            // No bin boundary separates the centroids: median of the longest centroid axis.
            int ax=0; for(int k=1;k<3;++k)if(bb.chi[k]-bb.clo[k]>bb.chi[ax]-bb.clo[ax])ax=k;
            m=(s+e)/2;
            std::nth_element(prim.begin()+s,prim.begin()+m,prim.begin()+e,[&](const Prim& a,const Prim& b){return a.c[ax]<b.c[ax];});
            for(int i=s;i<m;++i)L.add(prim[i]); for(int i=m;i<e;++i)R.add(prim[i]);
        }
        const int lc=cid+1,rc=cid+(m-s);
        setChild(0,L.lo,L.hi,m-s==1?~s:lc); setChild(1,R.lo,R.hi,e-m==1?~m:rc);
        auto right=[&]{if(e-m>1)build(m,e,rc,R,depth-1);};
        if(depth>0&&e-s>8192&&m-s>1){std::thread th([&]{build(s,m,lc,L,depth-1);}); right(); th.join();}
        else{if(m-s>1)build(s,m,lc,L,depth-1); right();}
    }
    // Returns the squared distance and sets `tri`; keeps `best2`/`tri` if nothing is closer.
    // Same visit order as a binary BVH walk: nearer child first, skip boxes at or beyond best2.
    // `skip` is a leaf already tested by the caller (its distance is in best2), so the walk skips it.
    inline __attribute__((always_inline)) double closest(const V3& p,double best2,int& tri,uint64_t& tested,int skip=-1) const {
        // Separate arrays for refs and distances: a {int,double} entry gets read back with a wider
        // load than its int store, which defeats store-to-load forwarding on every push/pop.
        int sr[128]; double sd[128]; int sp=0;
        {double d=0;for(int k=0;k<3;++k){double v=std::max({rootLo[k]-p[k],0.,p[k]-rootHi[k]});d+=v*v;}sr[0]=rootRef;sd[0]=d;sp=1;}
        const CNode* const nd=cn.data();
        while(sp){
            --sp; const int er=sr[sp];
            if(sd[sp]>=best2)continue;
            if(er<0){const int k=~er;if(k==skip)continue;++tested;double d=triDist2(p,pt[k]);if(d<best2){best2=d;tri=k;}continue;}
            const CNode& n=nd[er];
            double dl,dr;
#if defined(__aarch64__)
            // Both children at once: lane j = child j. max(lo-p, p-hi, 0) per axis, branch free.
            {float64x2_t acc=vdupq_n_f64(0);
             for(int k=0;k<3;++k){const float64x2_t pk=vdupq_n_f64(p[k]);
                const float64x2_t v=vmaxq_f64(vmaxq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(n.lo[k])),pk),vsubq_f64(pk,vcvt_f64_f32(vld1_f32(n.hi[k])))),vdupq_n_f64(0));
                acc=k?vaddq_f64(acc,vmulq_f64(v,v)):vmulq_f64(v,v);}
             dl=vgetq_lane_f64(acc,0); dr=vgetq_lane_f64(acc,1);}
#else
            dl=0; dr=0;
            for(int k=0;k<3;++k){
                const double l0=double(n.lo[k][0])-p[k],l1=double(n.lo[k][1])-p[k],h0=p[k]-double(n.hi[k][0]),h1=p[k]-double(n.hi[k][1]);
                const double v0=std::fmax(std::fmax(l0,h0),0.),v1=std::fmax(std::fmax(l1,h1),0.);
                dl+=v0*v0; dr+=v1*v1;}
#endif
            if(dl<dr){if(dr<best2){sr[sp]=n.child[1];sd[sp++]=dr;}if(dl<best2){sr[sp]=n.child[0];sd[sp++]=dl;}}
            else{if(dl<best2){sr[sp]=n.child[0];sd[sp++]=dl;}if(dr<best2){sr[sp]=n.child[1];sd[sp++]=dr;}}
        }
        return best2;
    }
};

// ------------------------------------------------------------------ BVH over points
// Median split on the longest axis, leaves of up to 4 points, pre-order layout (a subtree of n points has
// nodesFor(n) nodes, so both halves are built in parallel straight into place). Points are stored in leaf
// order. Float boxes are the min/max of float points (exact), distances are computed in double.
struct PointBVH {
    struct Node { float lo[3],hi[3]; uint32_t right,begin,count; };   // count > 0: leaf over P[begin, begin+count)
    static constexpr uint32_t L=4;
    std::vector<Node> nodes; std::vector<uint32_t> idx; std::vector<PF> P,N;
    static size_t nodesFor(size_t n){return n<=L?1:1+nodesFor(n/2)+nodesFor(n-n/2);}
    PointBVH(const PointCloud& pc,int threads){
        P=pc.p; idx.resize(P.size()); std::iota(idx.begin(),idx.end(),0u); nodes.resize(nodesFor(P.size()));
        std::vector<std::thread> pool; build(0,0,P.size(),std::max(1,threads),pool); for(auto& t:pool)t.join();
        std::vector<PF> Q(P.size()); for(size_t i=0;i<idx.size();++i)Q[i]=P[idx[i]]; P.swap(Q);
        if(pc.oriented){N.resize(P.size());for(size_t i=0;i<idx.size();++i)N[i]=pc.n[idx[i]];}
    }
    void build(size_t ni,size_t b,size_t e,int budget,std::vector<std::thread>& pool){
        Node& nd=nodes[ni]; for(int k=0;k<3;++k){nd.lo[k]=INFINITY;nd.hi[k]=-INFINITY;}
        for(size_t i=b;i<e;++i){const float* p=&P[idx[i]].x;for(int k=0;k<3;++k){nd.lo[k]=std::min(nd.lo[k],p[k]);nd.hi[k]=std::max(nd.hi[k],p[k]);}}
        if(e-b<=L){nd.count=uint32_t(e-b);nd.begin=uint32_t(b);nd.right=0;return;}
        int ax=0; for(int k=1;k<3;++k)if(nd.hi[k]-nd.lo[k]>nd.hi[ax]-nd.lo[ax])ax=k;
        const size_t n=e-b,m=b+n/2;
        std::nth_element(idx.begin()+b,idx.begin()+m,idx.begin()+e,[&](uint32_t x,uint32_t y){return (&P[x].x)[ax]<(&P[y].x)[ax];});
        nd.count=0; nd.right=uint32_t(ni+1+nodesFor(n/2)); const size_t r=nd.right;
        // Threads are handed down the tree: the left half gets its own thread while the budget lasts.
        if(budget>1&&n>50000){const int lb=budget/2; pool.emplace_back([this,ni,b,m,lb]{std::vector<std::thread> sub;build(ni+1,b,m,lb,sub);for(auto& t:sub)t.join();}); build(r,m,e,budget-lb,pool);}
        else{build(ni+1,b,m,1,pool);build(r,m,e,1,pool);}
    }
    static inline double boxD2(const Node& n,const V3& q){
        double s=0; for(int k=0;k<3;++k){const double d=std::max({double(n.lo[k])-q[k],q[k]-double(n.hi[k]),0.});s+=d*d;} return s;
    }
    // Nearest point strictly closer than sqrt(best2); updates best2 and bi (leaf-order index), or leaves them.
    double closest(const V3& q,double best2,int& bi,uint64_t& tested) const {
        uint32_t stack[64]; int sp=0; uint32_t ni=0;
        if(boxD2(nodes[0],q)>=best2)return best2;
        for(;;){
            const Node& n=nodes[ni];
            if(n.count){for(uint32_t i=n.begin;i<n.begin+n.count;++i){const double d=pd2(q,P[i]);++tested;if(d<best2){best2=d;bi=int(i);}}}
            else{
                uint32_t a=ni+1,b=n.right; double da=boxD2(nodes[a],q),db=boxD2(nodes[b],q);
                if(db<da){std::swap(a,b);std::swap(da,db);}
                if(db<best2)stack[sp++]=b;
                if(da<best2){ni=a;continue;}
            }
            for(;;){if(!sp)return best2; ni=stack[--sp]; if(boxD2(nodes[ni],q)<best2)break;}
        }
    }
    // +1 / -1 from the side of the point's tangent plane q is on (+1 without normals: unsigned field).
    inline float side(const V3& q,int i) const {
        if(N.empty())return 1.f; const PF& p=P[i]; const PF& n=N[i];
        return (q[0]-p.x)*n.x+(q[1]-p.y)*n.y+(q[2]-p.z)*n.z<0?-1.f:1.f;
    }
};

// ------------------------------------------------------------------ sign by winding number
// Inside = nonzero winding number along a +z ray through the voxel column centre.
static std::vector<uint8_t> insideMask(const std::vector<Tri>& t,const V3& lo,const V3& span,int RX,int RY,int RZ,int threads){
    std::vector<uint8_t> in(size_t(RX)*RY*RZ,0);
    const double sx=span[0]/RX,sy=span[1]/RY;
    struct FP { uint32_t id; int i0,i1,j0,j1; }; std::vector<FP> fp; fp.reserve(t.size());
    std::vector<uint32_t> start(size_t(RX)*RY+1,0);
    for(uint32_t i=0;i<t.size();++i){
        const Tri& tr=t[i];
        if((tr.a[0]==tr.b[0]&&tr.a[0]==tr.c[0])||(tr.a[1]==tr.b[1]&&tr.a[1]==tr.c[1]))continue;
        double x0=std::min({tr.a[0],tr.b[0],tr.c[0]}),x1=std::max({tr.a[0],tr.b[0],tr.c[0]});
        double y0=std::min({tr.a[1],tr.b[1],tr.c[1]}),y1=std::max({tr.a[1],tr.b[1],tr.c[1]});
        int i0=std::max(0,int(std::floor((x0-lo[0])/sx-.5))),i1=std::min(RX-1,int(std::ceil((x1-lo[0])/sx-.5)));
        int j0=std::max(0,int(std::floor((y0-lo[1])/sy-.5))),j1=std::min(RY-1,int(std::ceil((y1-lo[1])/sy-.5)));
        if(i0>i1||j0>j1)continue;
        fp.push_back({i,i0,i1,j0,j1});
        for(int j=j0;j<=j1;++j)for(int k=i0;k<=i1;++k)++start[size_t(j)*RX+k+1];
    }
    for(size_t c=1;c<start.size();++c)start[c]+=start[c-1];
    std::vector<uint32_t> items(start.back());
    {std::vector<uint32_t> fill(start.begin(),start.end()-1);
     for(const FP& f:fp)for(int j=f.j0;j<=f.j1;++j)for(int k=f.i0;k<=f.i1;++k)items[fill[size_t(j)*RX+k]++]=f.id;}
    std::atomic<int> next{0};
    auto work=[&]{
        std::vector<std::pair<double,int>> hits;
        for(int col;(col=next++)<RX*RY;){
            const int i=col%RX,j=col/RX;
            // Tiny irrational offset keeps the ray off shared edges and vertices of axis-aligned input.
            const double x=lo[0]+(i+.5)*sx+1.2345e-9*sx,y=lo[1]+(j+.5)*sy+2.7182e-9*sy;
            hits.clear();
            for(uint32_t e=start[col];e<start[col+1];++e){
                const Tri& tr=t[items[e]];
                double ax=tr.a[0]-x,ay=tr.a[1]-y,bx=tr.b[0]-x,by=tr.b[1]-y,cx=tr.c[0]-x,cy=tr.c[1]-y;
                double w0=bx*cy-by*cx,w1=cx*ay-cy*ax,w2=ax*by-ay*bx;
                if(!((w0>0&&w1>0&&w2>0)||(w0<0&&w1<0&&w2<0)))continue;
                // s > 0: triangle faces +z in projection, so a +z ray leaves through it (outward winding).
                const double s=w0+w1+w2; hits.push_back({(w0*tr.a[2]+w1*tr.b[2]+w2*tr.c[2])/s,s>0?-1:1});
            }
            if(hits.empty())continue;
            std::sort(hits.begin(),hits.end());
            int winding=0; size_t h=0;
            for(int z=0;z<RZ;++z){
                const double zc=lo[2]+(z+.5)*span[2]/RZ;
                while(h<hits.size()&&hits[h].first<zc)winding+=hits[h++].second;
                if(winding!=0)in[(size_t(z)*RY+j)*RX+i]=1;
            }
        }
    };
    std::vector<std::thread> pool; for(int k=0;k<threads;++k)pool.emplace_back(work); for(auto& p:pool)p.join();
    return in;
}

// Majority of the +z, +x and +y winding signs. The x and y passes reuse insideMask on cyclically rotated
// coordinates ((x,y,z) -> (y,z,x) and (z,x,y) are rotations, so triangle orientation is preserved).
static std::vector<uint8_t> insideMaskVote(const std::vector<Tri>& t,const V3& lo,const V3& span,int RX,int RY,int RZ,int threads){
    auto rot=[](const V3& v,int r){return r==1?V3{v[1],v[2],v[0]}:V3{v[2],v[0],v[1]};};
    std::vector<uint8_t> vz,vx,vy;
    auto pass=[&](int r,std::vector<uint8_t>& out){
        std::vector<Tri> tr; tr.reserve(t.size());
        for(const Tri& x:t)tr.emplace_back(rot(x.a,r),rot(x.b,r),rot(x.c,r),x.comp);
        const int R[3]={RX,RY,RZ};
        const int A=r==1?1:2,B=r==1?2:0,C=r==1?0:1; // rotated axis order
        out=insideMask(tr,rot(lo,r),rot(span,r),R[A],R[B],R[C],threads);
    };
    std::thread tz([&]{vz=insideMask(t,lo,span,RX,RY,RZ,threads);});
    std::thread tx([&]{pass(1,vx);});   // ray along x: rotated (y,z,x), dims (RY,RZ,RX)
    pass(2,vy);                          // ray along y: rotated (z,x,y), dims (RZ,RX,RY)
    tz.join(); tx.join();
    std::vector<uint8_t> in(size_t(RX)*RY*RZ,0);
    for(int k=0;k<RZ;++k)for(int j=0;j<RY;++j)for(int i=0;i<RX;++i){
        const size_t id=(size_t(k)*RY+j)*RX+i;
        const int votes=vz[id]+vx[(size_t(i)*RZ+k)*RY+j]+vy[(size_t(j)*RX+i)*RZ+k];
        in[id]=votes>=2;
    }
    return in;
}

static uint16_t floatToHalf(float f){
    uint32_t u; std::memcpy(&u,&f,4);
    uint32_t sign=(u>>16)&0x8000u;
    int32_t exp=int32_t((u>>23)&0xFFu)-127+15;
    uint32_t frac=u&0x7FFFFFu;
    if(exp<=0){
        if(exp<-10)return uint16_t(sign);
        frac=(frac|0x800000u)>>(1-exp);
        return uint16_t(sign|((frac+0x1000u)>>13));
    }
    if(exp>=31)return uint16_t(sign|0x7C00u);
    uint32_t h=sign|(uint32_t(exp)<<10)|((frac+0x1000u)>>13);
    return uint16_t(h);
}

int main(int argc,char** argv){
    std::string boxes,obj,points,writePoints,out,brickOut,statsPath,search="bvh-neighbour"; int subdiv=1; size_t sample=0,validate=0; uint64_t seed=1; double cellOverride=0; V3 lo{0,0,0},span{0,0,0}; int RX=128,RY=128,RZ=128,block=8,threads=int(std::max(1u,std::thread::hardware_concurrency())); double clip=1.5,reach=0; bool writeF16=false,unsignedField=false,signVote=false;
    for(int i=1;i<argc;++i){
        std::string a=argv[i]; auto need=[&](int n){if(i+n>=argc)throw std::runtime_error("missing value for "+a);};
        if(a=="--boxes"){need(1);boxes=argv[++i];}
        else if(a=="--obj"){need(1);obj=argv[++i];}
        else if(a=="--points"){need(1);points=argv[++i];}
        else if(a=="--sample"){need(1);sample=std::strtoull(argv[++i],nullptr,10);}
        else if(a=="--write-points"){need(1);writePoints=argv[++i];}
        else if(a=="--validate"){need(1);validate=std::strtoull(argv[++i],nullptr,10);}
        else if(a=="--seed"){need(1);seed=std::strtoull(argv[++i],nullptr,10);}
        else if(a=="--out"){need(1);out=argv[++i];}
        else if(a=="--brick-out"){need(1);brickOut=argv[++i];}
        else if(a=="--f16"){writeF16=true;}
        else if(a=="--unsigned"){unsignedField=true;}
        else if(a=="--sign"){need(1);const std::string m=argv[++i];if(m=="vote")signVote=true;else if(m!="z")throw std::runtime_error("--sign must be z or vote");}
        else if(a=="--stats"){need(1);statsPath=argv[++i];}
        else if(a=="--lo"){need(3);for(int k=0;k<3;++k)lo[k]=std::atof(argv[++i]);}
        else if(a=="--span"){need(3);for(int k=0;k<3;++k)span[k]=std::atof(argv[++i]);}
        else if(a=="--res"){need(1);RX=RY=RZ=std::atoi(argv[++i]);}
        else if(a=="--res-xyz"){need(3);RX=std::atoi(argv[++i]);RY=std::atoi(argv[++i]);RZ=std::atoi(argv[++i]);}
        else if(a=="--clip"){need(1);clip=std::atof(argv[++i]);}
        else if(a=="--reach"){need(1);reach=std::atof(argv[++i]);}
        else if(a=="--threads"){need(1);threads=std::max(1,std::atoi(argv[++i]));}
        else if(a=="--block"){need(1);block=std::max(0,std::atoi(argv[++i]));}
        else if(a=="--search"){need(1);search=argv[++i];}
        else if(a=="--subdiv"){need(1);subdiv=std::max(1,std::atoi(argv[++i]));}
        else if(a=="--cell"){need(1);cellOverride=std::atof(argv[++i]);}
        else throw std::runtime_error("unknown option "+a);
    }
    if(int(!boxes.empty())+int(!obj.empty())+int(!points.empty())!=1||(sample&&!points.empty())||(!writePoints.empty()&&!sample)||out.empty()||RX<2||RY<2||RZ<2||RX>1024||RY>1024||RZ>1024||!(span[0]>0&&span[1]>0&&span[2]>0)||!(clip>0))
        {std::cerr<<"Usage: bake_sdf (--boxes f.json | --obj f.obj | --points f.ply) --lo x y z --span x y z --out field.f32 [--res 128 | --res-xyz nx ny nz] [--f16] [--brick-out b.bin] [--clip 1.5] [--unsigned | --sign z|vote] [--threads N] [--block 8] [--validate K]\n";return 2;}
    if(block&&(RX%block||RY%block||RZ%block)){std::cerr<<"--res-xyz must be multiples of --block\n";return 2;}

    // One more worker than --threads (at least 2) for the parallel phases, when the machine has the core.
    const int hw=int(std::max(1u,std::thread::hardware_concurrency()));
    const int work2=std::max(2,std::min(threads+1,std::max(hw,2)));
    auto total=Clk::now(),t0=Clk::now();
    const bool pointMode=!points.empty()||sample>0;
    const auto tris=!points.empty()?std::vector<Tri>{}:boxes.empty()?loadObj(obj):loadBoxes(boxes,subdiv,work2);
    // Point clouds: read, or sampled from the mesh (hidden samples dropped); the mesh is not used after that.
    PointCloud pc; if(!points.empty())pc=loadPly(points); else if(sample)pc=sampleMesh(tris,sample,seed,work2);
    if(pointMode&&pc.p.empty())throw std::runtime_error("no points to bake from");
    if(!writePoints.empty())writePly(writePoints,pc);
    const double tLoad=since(t0);
    // Grid cell: about two mean edge lengths, but never finer than 1/256 of the extent or coarser than the clip.
    double edge=0; for(auto& x:tris)edge+=std::sqrt(dot3(x.b-x.a,x.b-x.a));
    edge/=std::max<size_t>(1,tris.size());
    const double ext=std::max({span[0],span[1],span[2]});
    // --search: bvh-neighbour (previous voxel's triangle as the starting bound, default) | bvh | grid
    const bool useGrid=search=="grid",neighbour=search=="bvh-neighbour";
    if(!(useGrid||neighbour||search=="bvh"))throw std::runtime_error("unknown --search "+search);
    // Only the search that is used gets built.
    t0=Clk::now(); std::vector<Tri> noTris{tris.empty()?Tri(V3{0,0,0},V3{1,0,0},V3{0,1,0},-1):tris[0]}; TriGrid grid(useGrid?tris:noTris,cellOverride>0?cellOverride:std::clamp(2*edge,ext/256,clip/2)); const double tGrid=since(t0);
    // The sign pass only needs the triangles: it runs on its own thread alongside the BVH build and
    // the distance pass, which write unsigned distances; inside voxels are negated at the end.
    std::vector<uint8_t> inside; double tSign=0;
    std::thread signThread([&]{if(pointMode||unsignedField)return;auto ts=Clk::now();inside=signVote?insideMaskVote(tris,lo,span,RX,RY,RZ,1):insideMask(tris,lo,span,RX,RY,RZ,1);tSign=since(ts);});
    t0=Clk::now(); BVH bvh(useGrid||pointMode?noTris:tris,std::max(threads,2));
    const std::vector<PF> onePoint{PF{0,0,0}}; PointCloud none; none.p=onePoint;
    PointBVH pbvh(pointMode?pc:none,work2); const double tBvh=since(t0);

    const auto seedBound=[](const V3&){return double(INFINITY);};


    // Left uninitialised: the workers write every voxel (culled ones as clip).
    const size_t NV=size_t(RX)*RY*RZ; std::unique_ptr<float[]> fieldBuf(new float[NV]); float* const field=fieldBuf.get();
    const float fclip=float(clip);
    const V3 vox{span[0]/RX,span[1]/RY,span[2]/RZ};
    const int B=block?block:1,NBX=RX/B,NBY=RY/B,NBZ=RZ/B;
    const double halfDiag=.5*std::sqrt(dot3(vox,vox))*B;
    // Distances above `limit` are written as clip without being resolved. With --reach r (the
    // farthest distance the AO shader can use, 1.12-.045 for the viewer), limit = r + one voxel
    // diagonal: any trilinear sample below r has all 8 corners within a diagonal of each other,
    // so every corner it reads stays exact, and samples above r stay above r.
    const double limit=reach>0?std::min(clip,reach+std::sqrt(dot3(vox,vox))):clip;
    // Points are addressed in half-voxel units within the grid: half coordinate h is lo + .5*h*vox
    // (h = 2i+1 is voxel i's centre, bit-identical to lo + (i+.5)*vox). Every offset between two points
    // of one block is in [-B,B] half-voxels per axis, so one (2B+1)^3 table holds all the distances the
    // culling needs, rounded up (1e-12 relative and absolute) so every bound stays conservative.
    const int TW=2*B+1; std::vector<double> dtab(size_t(TW)*TW*TW);
    for(int z=-B;z<=B;++z)for(int y=-B;y<=B;++y)for(int x=-B;x<=B;++x){
        const double ox=.5*x*vox[0],oy=.5*y*vox[1],oz=.5*z*vox[2];
        dtab[(size_t(z+B)*TW+(y+B))*TW+(x+B)]=std::sqrt(ox*ox+oy*oy+oz*oz)*(1+1e-12)+1e-12;}
    const double* const T0=dtab.data()+(size_t(B)*TW+B)*TW+B;   // T0[(z*TW+y)*TW+x] for offsets in [-B,B]
    std::atomic<int> next{0}; std::atomic<uint64_t> queries{0},tested{0},culled{0},voxCulled{0};
    t0=Clk::now();
    // PM (point clouds) is a separate instantiation, so the mesh path compiles exactly as before.
    // Point clouds have no winding number: a voxel takes the side of its nearest point's tangent plane.
    // Queried voxels always resolve their nearest point (bounded by the reference point's distance), so each
    // has its own sign. A culled voxel takes its reference point's sign: the segment between them stays
    // farther than `limit` from every point (1-Lipschitz bound), so it crosses no sampled surface.
    auto work=[&](auto PT){
        constexpr bool PM=decltype(PT)::value;
        Searcher s(tris,grid); uint64_t q=0,bt=0,vc=0; int prevTri=-1,prevPt=-1,lastPt=-1;
        // Exact distance to the nearest triangle, or +inf if it is farther than `lim`.
        // `upper` is a known upper bound on the answer (from a reference point).
        auto nearest=[&](const V3& p,double lim,double upper=INFINITY)->double{
            if constexpr(PM){
                // Points: always resolved (lim is ignored); prevPt seeds the bound like prevTri does.
                double best2=upper*(1+1e-9)+1e-12; best2*=best2; int pi=-1;
                if(prevPt>=0){const double dp2=pd2(p,pbvh.P[prevPt]);if(dp2<=best2){best2=dp2;pi=prevPt;}}
                best2=pbvh.closest(p,best2,pi,bt); if(pi>=0)prevPt=pi; lastPt=pi;
                return std::sqrt(best2);
            }
            if(useGrid)return s.nearest(p,lim,std::min(seedBound(p),lim+1e-9));
            // The derived bound gets a hair of slack so rounding can't hide the true nearest triangle.
            // The seed triangle's own distance gets none: equal-distance triangles are common in box
            // scenes, and strict pruning at the seed distance is what keeps them from being tested.
            double start=std::min(lim+1e-9,upper*(1+1e-9)+1e-12); int tri=-1;
            if(!neighbour)start=std::min(start,seedBound(p));
            double best2=start*start; int skip=-1;
            if(neighbour&&prevTri>=0){const double dp2=triDist2(p,bvh.pt[prevTri]);if(dp2<=best2){best2=dp2;tri=prevTri;skip=prevTri;}}
            const double d2=bvh.closest(p,best2,tri,bt,skip);
            if(tri>=0)prevTri=tri;
            const double d=std::sqrt(d2); return tri>=0&&d<=lim?d:INFINITY;
        };
        // Sign of the last resolved query point (mesh mode: +1, signed later by the winding mask).
        auto sideOf=[&](const V3& p)->float{if constexpr(PM)return pbvh.side(p,lastPt); else return 1.f;};
        // Points: a distance past `lim` counts as unresolved, as in mesh mode.
        auto cut=[&](double d,double lim){if constexpr(PM)return d<=lim?d:INFINITY; else return d;};
        auto pt=[&](int hx,int hy,int hz)->V3{return {lo[0]+.5*hx*vox[0],lo[1]+.5*hy*vox[1],lo[2]+.5*hz*vox[2]};};
        auto T=[&](int x,int y,int z){return T0[(ptrdiff_t(z)*TW+y)*TW+x];};
        auto fillCube=[&](int ox,int oy,int oz,int S,float sg){
            const float v=PM?sg*fclip:fclip;
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){float* f=field+(size_t(oz+z)*RY+(oy+y))*RX+ox;for(int x=0;x<S;++x)f[x]=v;}};
        // Cube of S^3 voxels at voxel origin o, with reference distance dref at half coordinate r.
        // 1-Lipschitz: every point p has d(p) >= dref - |p-r|. Even cubes of 4+ split into 8 subcubes;
        // a subcube is culled from the reference bound, or from its own centre's exact distance, which
        // is only queried when it could cull (its upper bound dref+|c-r| is past limit + half-diagonal).
        // rs: the reference point's side (point clouds only).
        auto cube=[&](auto& self,int ox,int oy,int oz,int S,double dref,int rx,int ry,int rz,float rs)->void{
            if(S>=4&&!(S&1)){
                const int h=S/2; const double hd=T(h-1,h-1,h-1);   // subcube centre to its voxel centres
                for(int cz=0;cz<2;++cz)for(int cy=0;cy<2;++cy)for(int cx=0;cx<2;++cx){
                    const int x0=ox+cx*h,y0=oy+cy*h,z0=oz+cz*h,hx=2*x0+h,hy=2*y0+h,hz=2*z0+h;
                    const double rc=T(hx-rx,hy-ry,hz-rz);
                    if(dref-rc-hd>limit){fillCube(x0,y0,z0,h,rs);vc+=uint64_t(h)*h*h;continue;}
                    if(dref+rc-hd>limit){
                        ++q; const V3 c=pt(hx,hy,hz); const double ds=cut(nearest(c,limit+hd,dref+rc),limit+hd); const float cs=sideOf(c);
                        if(std::isinf(ds)){fillCube(x0,y0,z0,h,cs);vc+=uint64_t(h)*h*h;continue;}
                        self(self,x0,y0,z0,h,ds,hx,hy,hz,cs);
                    }else self(self,x0,y0,z0,h,dref,rx,ry,rz,rs);
                }
                return;
            }
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){
                const int k=oz+z,j=oy+y; float* const f=field+(size_t(k)*RY+j)*RX;
                for(int x=0;x<S;++x){const int i=ox+x;
                    const double r=T(2*i+1-rx,2*j+1-ry,2*k+1-rz);
                    if(dref-r>limit){f[i]=PM?rs*fclip:fclip;++vc;continue;}
                    ++q; const V3 v=pt(2*i+1,2*j+1,2*k+1); double d=cut(nearest(v,limit,dref+r),limit);
                    if(std::isinf(d))d=clip;
                    f[i]=float(std::min(d,clip))*sideOf(v);
                }
            }
        };
        for(int b;(b=next++)<NBX*NBY*NBZ;){
            const int bx=b%NBX,by=(b/NBX)%NBY,bz=b/(NBX*NBY);
            if(block){
                const int hx=(2*bx+1)*B,hy=(2*by+1)*B,hz=(2*bz+1)*B;
                ++q; const V3 c=pt(hx,hy,hz); const double dc=cut(nearest(c,limit+halfDiag),limit+halfDiag); const float cs=sideOf(c);
                // Every voxel is more than `limit` from the surface; no surface crossing inside the block.
                if(std::isinf(dc)){++culled;fillCube(bx*B,by*B,bz*B,B,cs);continue;}
                cube(cube,bx*B,by*B,bz*B,B,dc,hx,hy,hz,cs);
            }else{
                const int i=bx,j=by,k=bz;
                ++q; const V3 v=pt(2*i+1,2*j+1,2*k+1); double d=cut(nearest(v,limit),limit);
                if(std::isinf(d))d=clip;
                field[(size_t(k)*RY+j)*RX+i]=float(std::min(d,clip))*sideOf(v);
            }
        }
        queries+=q; tested+=s.tested+bt; voxCulled+=vc;
    };
    auto run=[&](auto PT){std::vector<std::thread> pool; for(int k=1;k<work2;++k)pool.emplace_back([&]{work(PT);}); work(PT); for(auto& p:pool)p.join();};
    if(pointMode)run(std::true_type{}); else run(std::false_type{});
    const double tField=since(t0);
    // -float(d) is float(d) with the sign bit set, so this is bit-identical to signing per voxel.
    t0=Clk::now(); signThread.join(); const double tSignWait=since(t0);
    if(!pointMode&&!unsignedField)for(size_t k=0;k<NV;++k)if(inside[k])field[k]=-field[k];
    const double tTotal=since(total);

    // --validate K: K random voxels against brute force, after the timing. Meshes: distance to every
    // triangle and the winding number of every triangle along the voxel's column (the bake bins both).
    // Points: distance to every point, and the nearest point's side where the voxel is within `limit`
    // (farther voxels take a reference point's side, see the distance pass). Expected values follow the
    // bake's own rule: distances past `limit` are written as clip. Any mismatch fails the run (exit 1).
    std::atomic<size_t> vMis{0},vSign{0}; double tValidate=0;
    if(validate){
        const auto tv=Clk::now(); std::mt19937_64 rng(seed^0x9e3779b97f4a7c15ULL); std::uniform_int_distribution<size_t> U(0,NV-1);
        std::vector<size_t> pick(validate); for(auto& k:pick)k=U(rng);
        std::atomic<size_t> nx{0};
        auto chk=[&]{
            for(size_t t;(t=nx++)<pick.size();){
                const size_t k=pick[t]; const int i=int(k%RX),j=int((k/RX)%RY),z=int(k/(size_t(RX)*RY));
                const V3 p{lo[0]+.5*(2*i+1)*vox[0],lo[1]+.5*(2*j+1)*vox[1],lo[2]+.5*(2*z+1)*vox[2]};
                double best=INFINITY; bool neg=false;
                if(pointMode){int bi=0; for(size_t m=0;m<pbvh.P.size();++m){const double d=pd2(p,pbvh.P[m]);if(d<best){best=d;bi=int(m);}} neg=pbvh.side(p,bi)<0;}
                else{
                    for(const Tri& tr:tris)best=std::min(best,triDist2(p,tr));
                    // Brute-force winding along one axis: ax0/ax1 span the column, ax2 is the ray axis.
                    const int idx[3]={i,j,z}; const int R3[3]={RX,RY,RZ};
                    auto wind=[&](int ax0,int ax1,int ax2){
                        const double s0=span[ax0]/R3[ax0],s1=span[ax1]/R3[ax1];
                        const double x=lo[ax0]+(idx[ax0]+.5)*s0+1.2345e-9*s0,y=lo[ax1]+(idx[ax1]+.5)*s1+2.7182e-9*s1,zc=lo[ax2]+(idx[ax2]+.5)*span[ax2]/R3[ax2];
                        int w=0;
                        for(const Tri& tr:tris){
                            const double ax=tr.a[ax0]-x,ay=tr.a[ax1]-y,bx=tr.b[ax0]-x,by=tr.b[ax1]-y,cx=tr.c[ax0]-x,cy=tr.c[ax1]-y;
                            const double w0=bx*cy-by*cx,w1=cx*ay-cy*ax,w2=ax*by-ay*bx;
                            if(!((w0>0&&w1>0&&w2>0)||(w0<0&&w1<0&&w2<0)))continue;
                            const double sw=w0+w1+w2; if((w0*tr.a[ax2]+w1*tr.b[ax2]+w2*tr.c[ax2])/sw<zc)w+=sw>0?-1:1;}
                        return w!=0;
                    };
                    if(unsignedField)neg=false;
                    else if(signVote)neg=int(wind(0,1,2))+int(wind(1,2,0))+int(wind(2,0,1))>=2;
                    else neg=wind(0,1,2);
                }
                const double d=std::sqrt(best); const float mag=d<=limit?float(std::min(d,clip)):fclip, got=field[k];
                if(std::fabs(got)!=mag)++vMis;
                else if(mag!=0&&(!pointMode||d<=limit)&&std::signbit(got)!=neg)++vSign;
            }
        };
        std::vector<std::thread> pool; for(int k=1;k<work2;++k)pool.emplace_back(chk); chk(); for(auto& t:pool)t.join();
        tValidate=since(tv);
    }
    const bool vFail=vMis||vSign;

    std::ostringstream js; js<<std::fixed<<std::setprecision(3)
      <<"{\"triangles\":"<<tris.size()<<",\"points\":"<<(pointMode?pc.p.size():0)<<",\"points_hidden\":"<<pc.hidden<<",\"signed\":"<<((pointMode&&!pc.oriented)||unsignedField?"false":"true")<<",\"res\":["<<RX<<","<<RY<<","<<RZ<<"],\"f16\":"<<(writeF16?"true":"false")<<",\"clip\":"<<clip<<",\"threads\":"<<threads<<",\"block\":"<<block<<",\"search\":\""<<search<<"\""
      <<",\"grid_cell\":"<<grid.cell<<",\"ms\":{\"load\":"<<tLoad<<",\"grid\":"<<tGrid<<",\"bvh\":"<<tBvh<<",\"sign\":"<<tSign<<",\"sign_wait\":"<<tSignWait<<",\"distance\":"<<tField<<",\"total\":"<<tTotal<<"}"
      <<",\"limit\":"<<limit<<",\"blocks\":"<<NBX*NBY*NBZ<<",\"blocks_culled\":"<<culled<<",\"voxels_culled\":"<<voxCulled<<",\"distance_queries\":"<<queries<<",\"triangle_tests\":"<<tested;
    if(validate)js<<",\"validate\":{\"voxels\":"<<validate<<",\"mismatches\":"<<vMis<<",\"sign_errors\":"<<vSign<<",\"ms\":"<<tValidate<<"}";
    js<<"}";
    std::cout<<js.str()<<"\n";
    if(!statsPath.empty())std::ofstream(statsPath)<<js.str()<<"\n";
    // Optional 2-level sparse narrow-band 8^3 brick map export (indirection table + active bricks in f16).
    if(!brickOut.empty()&&block==8){
        std::vector<uint32_t> indir(size_t(NBX)*NBY*NBZ,0xFFFFFFFFu);
        std::vector<uint16_t> bricks;
        uint32_t activeBricks=0;
        for(int bz=0;bz<NBZ;++bz)for(int by=0;by<NBY;++by)for(int bx=0;bx<NBX;++bx){
            bool active=false;
            for(int z=0;z<8&&!active;++z)for(int y=0;y<8&&!active;++y){
                const float* row=field+(size_t(bz*8+z)*RY+(by*8+y))*RX+bx*8;
                for(int x=0;x<8;++x)if(std::fabs(row[x])<fclip){active=true;break;}
            }
            if(!active)continue;
            indir[(size_t(bz)*NBY+by)*NBX+bx]=activeBricks++;
            for(int z=0;z<8;++z)for(int y=0;y<8;++y){
                const float* row=field+(size_t(bz*8+z)*RY+(by*8+y))*RX+bx*8;
                for(int x=0;x<8;++x)bricks.push_back(floatToHalf(row[x]));
            }
        }
        std::ofstream bo(brickOut,std::ios::binary);
        uint32_t hdr[6]={uint32_t(RX),uint32_t(RY),uint32_t(RZ),uint32_t(NBX),uint32_t(NBY),uint32_t(NBZ)};
        bo.write((const char*)hdr,sizeof(hdr));
        bo.write((const char*)&activeBricks,4);
        bo.write((const char*)indir.data(),indir.size()*4);
        bo.write((const char*)bricks.data(),bricks.size()*2);
        std::cerr<<"brick-map: "<<activeBricks<<"/"<<(NBX*NBY*NBZ)<<" active 8^3 bricks ("<<(indir.size()*4+bricks.size()*2)/1024<<" KiB)\n";
    }
    // Raw write() loop, then flush and _exit.
    {const int fd=::open(out.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0644);
     if(fd<0){std::cerr<<"cannot write "<<out<<": "<<std::strerror(errno)<<"\n";return 1;}
     if(writeF16){
      std::vector<uint16_t> f16(NV);
      for(size_t k=0;k<NV;++k)f16[k]=floatToHalf(field[k]);
      const char* w=(const char*)f16.data(); size_t left=NV*2;
      while(left){const ssize_t n=::write(fd,w,left);if(n<0){if(errno==EINTR)continue;std::cerr<<"write failed: "<<std::strerror(errno)<<"\n";::close(fd);return 1;}w+=n;left-=size_t(n);}
     }else{
      const char* w=(const char*)field; size_t left=NV*4;
      while(left){const ssize_t n=::write(fd,w,left);if(n<0){if(errno==EINTR)continue;std::cerr<<"write failed: "<<std::strerror(errno)<<"\n";::close(fd);return 1;}w+=n;left-=size_t(n);}
     }
     if(::close(fd)!=0){std::cerr<<"close failed: "<<std::strerror(errno)<<"\n";return 1;}}
    if(vFail)std::cerr<<"validation FAILED: "<<vMis<<" distance mismatches, "<<vSign<<" sign errors in "<<validate<<" voxels\n";
    std::cout.flush(); std::fflush(nullptr); std::_Exit(vFail?1:0);
    return 0;
}

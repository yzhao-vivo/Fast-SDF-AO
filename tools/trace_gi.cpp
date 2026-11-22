// trace_gi: independent 512-ray 1-bounce diffuse GI + AO reference ray tracer.
//   trace_gi gi_tris.f32 samples.f32 out_ref.f32 sx sy sz [out_sun.f32]
// out_sun.f32 (optional): 1 float per pixel, soft sun visibility from 64 stratified shadow rays over a
// 1.6-degree solar disk (cone radius 0.028 rad, as the WebGPU renderer jitters its sun shadow rays).
// gi_tris.f32: 15 floats per triangle (v0, v1, v2, unitNormal, avgAlbedo).
// samples.f32: 9 floats per pixel (pos, nor, albedo).
// out_ref.f32: 10 floats per pixel (ao, giR, giG, giB, aoOnlyR, aoOnlyG, aoOnlyB, giRgbR, giRgbG, giRgbB).
// Build: c++ -O3 -std=c++17 -pthread tools/trace_gi.cpp -o trace_gi
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct V { float x, y, z; };
static inline V add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V mul(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline V mul3(V a, V b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
static inline V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V norm(V a) { float l = std::sqrt(dot(a, a)); return l > 1e-8f ? mul(a, 1.f / l) : V{0, 0, 1}; }

struct GiTri { V a, b, c, n, alb; };
struct Node { float lo[3], hi[3]; int left = -1, right = -1, start = 0, count = 0; };

template <class T> static std::vector<T> load(const std::string& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + p);
    size_t n = size_t(in.tellg()); in.seekg(0);
    std::vector<T> v(n / sizeof(T));
    in.read((char*)v.data(), n);
    return v;
}

struct BVH {
    const std::vector<GiTri>& t;
    std::vector<uint32_t> order;
    std::vector<Node> nodes;
    explicit BVH(const std::vector<GiTri>& tris) : t(tris) {
        order.resize(t.size());
        std::iota(order.begin(), order.end(), 0);
        build(0, int(t.size()));
    }
    float cen(uint32_t i, int ax) const {
        const GiTri& x = t[i];
        const float* a = &x.a.x; const float* b = &x.b.x; const float* c = &x.c.x;
        return (a[ax] + b[ax] + c[ax]) / 3.f;
    }
    int build(int s, int e) {
        Node n;
        for (int k = 0; k < 3; ++k) { n.lo[k] = INFINITY; n.hi[k] = -INFINITY; }
        for (int i = s; i < e; ++i)
            for (const V* p : {&t[order[i]].a, &t[order[i]].b, &t[order[i]].c}) {
                const float* q = &p->x;
                for (int k = 0; k < 3; ++k) { n.lo[k] = std::min(n.lo[k], q[k]); n.hi[k] = std::max(n.hi[k], q[k]); }
            }
        const int id = int(nodes.size()); nodes.push_back(n);
        if (e - s <= 4) { nodes[id].start = s; nodes[id].count = e - s; return id; }
        int ax = 0; float ext = -1.f;
        for (int k = 0; k < 3; ++k) if (n.hi[k] - n.lo[k] > ext) { ext = n.hi[k] - n.lo[k]; ax = k; }
        const int m = (s + e) / 2;
        std::nth_element(order.begin() + s, order.begin() + m, order.begin() + e,
            [&](uint32_t a, uint32_t b) { return cen(a, ax) < cen(b, ax); });
        const int l = build(s, m), r = build(m, e);
        nodes[id].left = l; nodes[id].right = r;
        return id;
    }
    static inline bool box(const Node& n, V o, V inv, float tmax) {
        float t0 = 0.f, t1 = tmax;
        const float* oo = &o.x; const float* ii = &inv.x;
        for (int k = 0; k < 3; ++k) {
            float a = (n.lo[k] - oo[k]) * ii[k], b = (n.hi[k] - oo[k]) * ii[k];
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a); t1 = std::min(t1, b);
        }
        return t0 <= t1;
    }
    bool any(V o, V d, float tmax) const {
        V inv{1.f / (std::fabs(d.x) < 1e-12f ? 1e-12f : d.x),
              1.f / (std::fabs(d.y) < 1e-12f ? 1e-12f : d.y),
              1.f / (std::fabs(d.z) < 1e-12f ? 1e-12f : d.z)};
        int stack[128], sp = 0; stack[sp++] = 0;
        while (sp) {
            const Node& n = nodes[stack[--sp]];
            if (!box(n, o, inv, tmax)) continue;
            if (n.count) {
                for (int i = 0; i < n.count; ++i) {
                    const GiTri& x = t[order[n.start + i]];
                    V e1 = sub(x.b, x.a), e2 = sub(x.c, x.a), p = cross(d, e2);
                    float det = dot(e1, p);
                    if (std::fabs(det) < 1e-12f) continue;
                    float id = 1.f / det; V s = sub(o, x.a);
                    float u = dot(s, p) * id; if (u < 0.f || u > 1.f) continue;
                    V q = cross(s, e1);
                    float v = dot(d, q) * id; if (v < 0.f || u + v > 1.f) continue;
                    float tt = dot(e2, q) * id;
                    if (tt > 0.f && tt <= tmax) return true;
                }
                continue;
            }
            stack[sp++] = n.left; stack[sp++] = n.right;
        }
        return false;
    }
    int closest(V o, V d, float tmax, float& outT) const {
        V inv{1.f / (std::fabs(d.x) < 1e-12f ? 1e-12f : d.x),
              1.f / (std::fabs(d.y) < 1e-12f ? 1e-12f : d.y),
              1.f / (std::fabs(d.z) < 1e-12f ? 1e-12f : d.z)};
        int stack[128], sp = 0; stack[sp++] = 0;
        float bestT = tmax;
        int bestTri = -1;
        while (sp) {
            const Node& n = nodes[stack[--sp]];
            if (!box(n, o, inv, bestT)) continue;
            if (n.count) {
                for (int i = 0; i < n.count; ++i) {
                    uint32_t ti = order[n.start + i];
                    const GiTri& x = t[ti];
                    V e1 = sub(x.b, x.a), e2 = sub(x.c, x.a), p = cross(d, e2);
                    float det = dot(e1, p);
                    if (std::fabs(det) < 1e-12f) continue;
                    float id = 1.f / det; V s = sub(o, x.a);
                    float u = dot(s, p) * id; if (u < 0.f || u > 1.f) continue;
                    V q = cross(s, e1);
                    float v = dot(d, q) * id; if (v < 0.f || u + v > 1.f) continue;
                    float tt = dot(e2, q) * id;
                    if (tt > 0.f && tt < bestT) { bestT = tt; bestTri = int(ti); }
                }
                continue;
            }
            stack[sp++] = n.left; stack[sp++] = n.right;
        }
        outT = bestT;
        return bestTri;
    }
};

static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static inline float rand01(uint32_t& s) { s = hash32(s + 1u); return (s & 0xFFFFFFu) / 16777216.f; }

static inline float aces1(float x) {
    float y = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    return y < 0.f ? 0.f : (y > 1.f ? 1.f : y);
}
static inline float linToSrgb1(float c) {
    c = c < 0.f ? 0.f : (c > 1.f ? 1.f : c);
    return c < 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}
static inline V toneSrgb(V c) {
    return {linToSrgb1(aces1(c.x)), linToSrgb1(aces1(c.y)), linToSrgb1(aces1(c.z))};
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr, "usage: trace_gi gi_tris.f32 samples.f32 out_ref.f32 sx sy sz [out_sun.f32]\n");
        return 2;
    }
    const bool wantSun = argc == 8;
    auto tris = load<GiTri>(argv[1]);
    auto samples = load<float>(argv[2]);
    const size_t n = samples.size() / 9;
    V sunDir = norm({std::stof(argv[4]), std::stof(argv[5]), std::stof(argv[6])});
    const V sunCol{2.25f, 2.10f, 1.85f};
    const V bounceSunCol{3.10f, 2.90f, 2.55f};
    const V skyCol{0.22f, 0.26f, 0.34f};
    const V bounceSkyCol{0.56f, 0.64f, 0.76f};
    const float AO_R = 1.12f;
    const float GI_R = 8.5f;
    const int SU = 16, SV = 32, RAYS = SU * SV;

    BVH bvh(tris);
    std::vector<float> out(n * 10, 0.f);
    std::vector<float> sunOut(wantSun ? n : 0, 0.f);
    V sunT = norm(cross(sunDir, std::fabs(sunDir.x) < 0.9f ? V{1, 0, 0} : V{0, 1, 0}));
    V sunB = cross(sunDir, sunT);
    std::atomic<size_t> next{0};

    auto work = [&] {
        for (size_t b; (b = next.fetch_add(256)) < n;) {
            for (size_t i = b; i < std::min(n, b + 256); ++i) {
                const float* sp = &samples[i * 9];
                V p{sp[0], sp[1], sp[2]};
                V nor = norm({sp[3], sp[4], sp[5]});
                V alb{sp[6], sp[7], sp[8]};
                V a = std::fabs(nor.x) < 0.9f ? V{1, 0, 0} : V{0, 1, 0};
                V t = norm(cross(nor, a));
                V bv = cross(nor, t);
                V orig = add(p, mul(nor, 0.001f));

                uint32_t rng = hash32(uint32_t(i) * 747796405u + 2891336453u);
                int aoHits = 0;
                V giSum{0, 0, 0};
                for (int iu = 0; iu < SU; ++iu) {
                    for (int iv = 0; iv < SV; ++iv) {
                        float u = (iu + rand01(rng)) / float(SU);
                        float v = (iv + rand01(rng)) / float(SV);
                        float r = std::sqrt(std::max(0.f, std::min(1.f, u)));
                        float phi = 6.283185307f * v;
                        float lx = r * std::cos(phi), ly = r * std::sin(phi);
                        float lz = std::sqrt(std::max(0.f, 1.f - u));
                        V w = add(add(mul(t, lx), mul(bv, ly)), mul(nor, lz));
                        float hitT = GI_R;
                        int ti = bvh.closest(orig, w, GI_R, hitT);
                        if (ti >= 0) {
                            if (hitT <= AO_R) aoHits++;
                            const GiTri& ht = tris[ti];
                            V hitP = add(orig, mul(w, hitT));
                            V hn = ht.n;
                            if (dot(hn, w) > 0.f) hn = mul(hn, -1.f);
                            float ndl = std::max(0.f, dot(hn, sunDir));
                            float sh = 0.f;
                            if (ndl > 0.001f) {
                                V shO = add(hitP, mul(hn, 0.004f));
                                sh = bvh.any(shO, sunDir, 12.f) ? 0.f : 1.f;
                            }
                            float skyF = 0.42f + 0.32f * std::max(0.f, hn.z);
                            V irr = add(mul(bounceSunCol, ndl * sh), mul(bounceSkyCol, skyF));
                            giSum = add(giSum, mul3(ht.alb, irr));
                        }
                    }
                }
                float ao = 1.f - float(aoHits) / float(RAYS);
                V giIrr = mul(giSum, 1.f / float(RAYS));

                // Direct sun + shadow + sky at primary point
                float pNdl = std::max(0.f, dot(nor, sunDir));
                float pSh = (pNdl > 0.001f && !bvh.any(add(p, mul(nor, 0.004f)), sunDir, 12.f)) ? 1.f : 0.f;
                float pSkyF = 0.16f + 0.24f * std::max(0.f, nor.z);
                V direct = mul(sunCol, pNdl * pSh);
                V skyInd = mul(skyCol, pSkyF * ao);
                V colAO = toneSrgb(mul(mul3(alb, add(direct, skyInd)), 0.88f));
                V colGI = toneSrgb(mul(mul3(alb, add(add(direct, skyInd), giIrr)), 0.88f));

                if (wantSun) {
                    // Soft sun visibility: 8 x 8 stratified samples over the solar disk
                    int vis = 0;
                    V so = add(p, mul(nor, 0.004f));
                    for (int a = 0; a < 8; ++a) {
                        for (int b = 0; b < 8; ++b) {
                            float r = 0.028f * std::sqrt((a + rand01(rng)) / 8.f);
                            float ph = 6.283185307f * (b + rand01(rng)) / 8.f;
                            V d = norm(add(sunDir, add(mul(sunT, r * std::cos(ph)), mul(sunB, r * std::sin(ph)))));
                            if (dot(d, nor) > 0.f && !bvh.any(so, d, 12.f)) vis++;
                        }
                    }
                    sunOut[i] = float(vis) / 64.f;
                }

                float* o = &out[i * 10];
                o[0] = ao;
                o[1] = giIrr.x; o[2] = giIrr.y; o[3] = giIrr.z;
                o[4] = colAO.x; o[5] = colAO.y; o[6] = colAO.z;
                o[7] = colGI.x; o[8] = colGI.y; o[9] = colGI.z;
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned k = 0; k < std::max(1u, std::thread::hardware_concurrency()); ++k) pool.emplace_back(work);
    for (auto& th : pool) th.join();

    std::ofstream(argv[3], std::ios::binary).write((const char*)out.data(), out.size() * sizeof(float));
    if (wantSun) std::ofstream(argv[7], std::ios::binary).write((const char*)sunOut.data(), sunOut.size() * sizeof(float));
    std::printf("trace_gi: %zu tris, %zu pixels x %d rays\n", tris.size(), n, RAYS);
}

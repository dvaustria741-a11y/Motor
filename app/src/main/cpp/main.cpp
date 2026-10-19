// NEON RIDER - highway motorbike dodger. C++17, OpenGL ES 3.0, Android NativeActivity.
// Everything is drawn procedurally / from compact .mdl meshes (see tools/convert_models.py).
#include <android/log.h>
#include <android/sensor.h>
#include <android/asset_manager.h>
#include <android_native_app_glue.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NeonRider", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NeonRider", __VA_ARGS__)

static const float PI = 3.14159265f;
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float sgn(float v) { return v < 0 ? -1.f : 1.f; }

// ------------------------------------------------------------------ math
struct V3 { float x, y, z; };
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline V3 norm(V3 a) { float l = sqrtf(dot(a, a)); return l > 1e-8f ? a * (1.f / l) : V3{0, 1, 0}; }

struct M4 { float m[16]; };  // column-major
static M4 mIdent() { M4 r; memset(r.m, 0, sizeof r.m); r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
static M4 mMul(const M4& a, const M4& b) {
    M4 r;
    for (int c = 0; c < 4; c++) for (int rw = 0; rw < 4; rw++) {
        float s = 0; for (int k = 0; k < 4; k++) s += a.m[k * 4 + rw] * b.m[c * 4 + k];
        r.m[c * 4 + rw] = s;
    }
    return r;
}
static M4 mTrans(float x, float y, float z) { M4 r = mIdent(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r; }
static M4 mScale(float x, float y, float z) { M4 r = mIdent(); r.m[0] = x; r.m[5] = y; r.m[10] = z; return r; }
static M4 mRotX(float a) { M4 r = mIdent(); float c = cosf(a), s = sinf(a); r.m[5] = c; r.m[9] = -s; r.m[6] = s; r.m[10] = c; return r; }
static M4 mRotY(float a) { M4 r = mIdent(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[8] = s; r.m[2] = -s; r.m[10] = c; return r; }
static M4 mRotZ(float a) { M4 r = mIdent(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[4] = -s; r.m[1] = s; r.m[5] = c; return r; }
static M4 mPersp(float fovy, float asp, float n, float f) {
    M4 r; memset(r.m, 0, sizeof r.m);
    float t = 1.f / tanf(fovy * 0.5f);
    r.m[0] = t / asp; r.m[5] = t; r.m[10] = (f + n) / (n - f); r.m[11] = -1; r.m[14] = 2 * f * n / (n - f);
    return r;
}
static M4 mLookAt(V3 e, V3 c, V3 up) {
    V3 f = norm(c - e), s = norm(cross(f, up)), u = cross(s, f);
    M4 r = mIdent();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, e); r.m[13] = -dot(u, e); r.m[14] = dot(f, e);
    return r;
}

static uint32_t rngState = 2463534242u;
static float rnd() { rngState ^= rngState << 13; rngState ^= rngState >> 17; rngState ^= rngState << 5; return (rngState & 0xFFFFFF) / 16777216.f; }
static float rndr(float a, float b) { return a + (b - a) * rnd(); }

static double nowSec() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

// ------------------------------------------------------------------ shaders
static const char* MAIN_VS = R"GLSL(#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aN;
layout(location=2) in vec4 aCol;
uniform mat4 uVP;
uniform mat4 uModel;
out vec3 vW; out vec3 vN; out vec3 vL; out vec3 vLN; out vec4 vC; out vec2 vUV;
void main(){
  vec4 w = uModel * vec4(aPos,1.0);
  vW = w.xyz; vN = mat3(uModel) * aN.xyz; vL = aPos; vLN = aN.xyz; vC = aCol; vUV = aN.xy;
  gl_Position = uVP * w;
}
)GLSL";

static const char* MAIN_FS = R"GLSL(#version 300 es
precision highp float;
in vec3 vW; in vec3 vN; in vec3 vL; in vec3 vLN; in vec4 vC; in vec2 vUV;
uniform int uKind;
uniform vec4 uColor;
uniform vec3 uTint;
uniform vec3 uCam;
uniform vec3 uFog;
uniform float uFogK;
uniform float uDist;   // distance travelled mod 96
uniform float uSeed;
uniform float uBikeX;
uniform float uTime;
out vec4 outC;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
void main(){
  vec3 V = uCam - vW; float dist = length(V); V /= max(dist, 1e-4);
  vec3 N = normalize(vN);
  vec3 base = uColor.rgb * vC.rgb;
  float alpha = uColor.a * vC.a;
  vec3 col = base;
  float fog = 1.0 - exp(-dist*dist*uFogK);
  vec3 Ld = normalize(vec3(-0.35, 0.75, 0.55));
  float nl = max(dot(N, Ld), 0.0);
  vec3 amb = mix(vec3(0.10,0.09,0.16), vec3(0.20,0.22,0.36), N.y*0.5+0.5);
  vec3 lit = amb + vec3(0.42,0.45,0.70) * nl;
  float zAbs = uDist - vW.z;

  if(uKind==0){                         // painted body (tinted)
    vec3 c = uTint * vC.rgb;
    vec3 H = normalize(Ld + V);
    float sp = pow(max(dot(N,H),0.0), 48.0);
    float rim = pow(1.0 - max(dot(N,V),0.0), 3.0);
    col = c * lit + vec3(0.7,0.75,1.0)*sp*0.6 + vec3(0.25,0.15,0.45)*rim*0.5;
  } else if(uKind==1){                  // glass
    float fr = pow(1.0 - max(dot(N,V),0.0), 2.0);
    col = base * lit + vec3(0.25,0.18,0.40) * fr;
    alpha = clamp(alpha*0.9 + fr*0.35, 0.0, 1.0);
  } else if(uKind==2 || uKind==4){      // dark parts / generic
    vec3 H = normalize(Ld + V);
    float sp = pow(max(dot(N,H),0.0), 32.0);
    col = base * lit + vec3(0.6,0.65,0.9)*sp*0.25;
  } else if(uKind==3 || uKind==5){      // emissive
    col = base * 1.5;
  } else if(uKind==6){                  // buildings with procedural lit windows
    vec3 ln = normalize(vLN);
    vec3 wall = vec3(0.05,0.055,0.09) + vec3(0.05,0.04,0.08) * hash(floor(vL.xz*0.05)+uSeed);
    col = wall * lit;
    if(abs(ln.y) < 0.6){
      float u = abs(ln.x) > abs(ln.z) ? vL.z : vL.x;
      vec2 cell = vec2(floor(u/6.0), floor(vL.y/9.0));
      vec2 f = vec2(fract(u/6.0), fract(vL.y/9.0));
      float r = hash(cell + uSeed*13.7 + (abs(ln.x)>abs(ln.z)?7.0:0.0));
      float win = step(0.18,f.x)*step(f.x,0.82)*step(0.28,f.y)*step(f.y,0.78);
      float on = step(0.52, r);
      vec3 wc = r > 0.93 ? vec3(1.0,0.2,0.65) : (r > 0.80 ? vec3(0.2,0.9,1.0) : (r > 0.68 ? vec3(1.0,0.8,0.4) : vec3(1.0,0.9,0.7)));
      col += wc * win * on * (0.65 + 0.35*hash(cell+3.3));
    }
    if(ln.y > 0.7){ col += vec3(0.08,0.06,0.12); }
  } else if(uKind==7){                  // barrier
    float stripe = step(0.5, fract(zAbs/4.0));
    vec3 c = mix(vec3(0.30,0.30,0.34), vec3(0.42,0.42,0.46), stripe);
    float refl = step(0.97, fract(zAbs/8.0)) * step(0.55, vW.y) ;
    col = c * lit + vec3(1.0,0.25,0.1)*refl*0.8;
    float pool = exp(-pow((fract(zAbs/32.0)-0.5)*32.0/10.0, 2.0));
    col += vec3(1.0,0.6,0.3)*pool*0.25;
  } else if(uKind==8){                  // road
    float ax = abs(vW.x);
    vec3 c = vec3(0.055,0.055,0.07);
    c += (hash(floor(vec2(vW.x*2.5, zAbs*2.5))) - 0.5) * 0.03;
    float dash = step(fract(zAbs/12.0), 0.4);
    float l1 = (1.0 - smoothstep(0.05,0.10,abs(ax-3.6))) * dash;
    float e  = 1.0 - smoothstep(0.07,0.12,abs(ax-7.2));
    float yc = 1.0 - smoothstep(0.04,0.09,abs(ax-0.16));
    c = mix(c, vec3(0.75), max(l1, e));
    c = mix(c, vec3(0.95,0.70,0.10), yc);
    if(ax > 7.3) c *= 0.8;
    float pool = exp(-pow((fract(zAbs/32.0)-0.5)*32.0/9.0, 2.0));
    float fwd = max(-vW.z, 0.0);
    float lat = vW.x - uBikeX;
    float cone = exp(-lat*lat/(1.6+0.05*fwd)) * exp(-fwd/38.0) * step(-4.0, -fwd);
    float wet = pow(1.0 - clamp(V.y, 0.0, 1.0), 2.0);
    col = c * (0.55 + pool*1.8) + vec3(1.0,0.62,0.30)*pool*0.14 + vec3(0.95,0.88,0.72)*cone*0.55 + uFog*wet*0.35 + vec3(0.6,0.15,0.5)*wet*0.05*(1.0+sin(zAbs*0.2));
  } else if(uKind==9){                  // outer ground
    col = vec3(0.025,0.028,0.045) + vec3(0.02,0.015,0.03) * hash(floor(vec2(vW.x, zAbs)*0.5));
  } else if(uKind==12){                 // soft blob shadow
    float r = length(vUV);
    alpha *= smoothstep(1.0, 0.1, r);
    col = vec3(0.0);
  } else if(uKind==11){                 // additive glow
    float r = length(vUV);
    float g = pow(clamp(1.0 - r, 0.0, 1.0), 2.2);
    col = base * g * (1.0 - fog);
    outC = vec4(col, 1.0); return;
  }
  if(uKind != 12) col = mix(col, uFog, fog);
  col = pow(col, vec3(0.92));
  outC = vec4(col, alpha);
}
)GLSL";

static const char* FS_VS = R"GLSL(#version 300 es
out vec2 vUV;
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); vUV = p; gl_Position = vec4(p*2.0-1.0, 0.0, 1.0); }
)GLSL";

static const char* SKY_FS = R"GLSL(#version 300 es
precision highp float;
in vec2 vUV; uniform float uHorizon; uniform float uTime; uniform vec3 uFog; out vec4 outC;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
void main(){
  float h = vUV.y - uHorizon;
  float t = clamp(h*1.6, 0.0, 1.0);
  vec3 top = vec3(0.015,0.015,0.07), mid = vec3(0.12,0.05,0.22);
  vec3 c = mix(uFog*1.5, mid, smoothstep(0.0,0.18,h));
  c = mix(c, top, smoothstep(0.1,0.8,h));
  c += vec3(0.8,0.2,0.5) * exp(-abs(h)*14.0) * 0.18;
  if(h > 0.05){
    vec2 g = floor(vUV*vec2(260.0,140.0));
    float s = step(0.9975, hash(g));
    c += vec3(s) * (0.6 + 0.4*sin(uTime*2.0+hash(g+1.0)*30.0));
  }
  outC = vec4(c, 1.0);
}
)GLSL";

static const char* BLIT_FS = R"GLSL(#version 300 es
precision highp float;
in vec2 vUV; uniform sampler2D uTex; uniform float uBlur; uniform float uVig; out vec4 outC;
void main(){
  vec2 d = vUV - vec2(0.5,0.46);
  vec3 c = texture(uTex, vUV).rgb;
  if(uBlur > 0.001){
    c += texture(uTex, vUV - d*uBlur*0.5).rgb;
    c += texture(uTex, vUV - d*uBlur).rgb;
    c += texture(uTex, vUV - d*uBlur*1.6).rgb;
    c *= 0.25;
  }
  c *= 1.0 - dot(d,d)*uVig;
  outC = vec4(c, 1.0);
}
)GLSL";

static const char* UI_VS = R"GLSL(#version 300 es
layout(location=0) in vec2 aP; layout(location=1) in vec4 aC;
uniform vec2 uRes; out vec4 vC;
void main(){ gl_Position = vec4(aP.x/uRes.x*2.0-1.0, 1.0-aP.y/uRes.y*2.0, 0.0, 1.0); vC = aC; }
)GLSL";
static const char* UI_FS = R"GLSL(#version 300 es
precision mediump float; in vec4 vC; out vec4 outC; void main(){ outC = vC; }
)GLSL";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type); glShaderSource(s, 1, &src, nullptr); glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char buf[1024]; glGetShaderInfoLog(s, 1024, nullptr, buf); LOGE("shader error: %s", buf); }
    return s;
}
static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint p = glCreateProgram(); GLuint a = compile(GL_VERTEX_SHADER, vs), b = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, a); glAttachShader(p, b); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char buf[1024]; glGetProgramInfoLog(p, 1024, nullptr, buf); LOGE("link error: %s", buf); }
    glDeleteShader(a); glDeleteShader(b); return p;
}

// ------------------------------------------------------------------ geometry containers
struct DV { float x, y, z; int8_t nx, ny, nz, nw; uint8_t r, g, b, a; };  // 20 bytes
struct SV { float x, y, z; int8_t nx, ny, nz, nw; };                       // 16 bytes (static .mdl)

struct Part {
    uint8_t kind = 4; uint8_t col[4] = {255, 255, 255, 255};
    std::vector<SV> v; std::vector<uint16_t> idx;
    GLuint vao = 0, vbo = 0, ibo = 0;
};
struct Model {
    std::vector<Part> parts;
    V3 mn{1e9f, 1e9f, 1e9f}, mx{-1e9f, -1e9f, -1e9f};
    bool loaded = false;
};

static AAssetManager* gAssets = nullptr;

static bool loadModel(const char* name, Model& m) {
    AAsset* a = AAssetManager_open(gAssets, name, AASSET_MODE_BUFFER);
    if (!a) { LOGE("missing asset %s", name); return false; }
    size_t len = AAsset_getLength(a); const uint8_t* d = (const uint8_t*)AAsset_getBuffer(a);
    if (len < 8 || memcmp(d, "MDL1", 4) != 0) { AAsset_close(a); return false; }
    uint32_t n; memcpy(&n, d + 4, 4); size_t o = 8;
    m.parts.clear(); m.parts.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        Part& p = m.parts[i];
        p.kind = d[o]; memcpy(p.col, d + o + 1, 4);
        uint32_t nv, ni; memcpy(&nv, d + o + 5, 4); memcpy(&ni, d + o + 9, 4); o += 13;
        p.v.resize(nv); memcpy(p.v.data(), d + o, nv * 16); o += nv * 16;
        p.idx.resize(ni); memcpy(p.idx.data(), d + o, ni * 2); o += ni * 2;
        for (auto& v : p.v) {
            m.mn.x = std::min(m.mn.x, v.x); m.mn.y = std::min(m.mn.y, v.y); m.mn.z = std::min(m.mn.z, v.z);
            m.mx.x = std::max(m.mx.x, v.x); m.mx.y = std::max(m.mx.y, v.y); m.mx.z = std::max(m.mx.z, v.z);
        }
    }
    AAsset_close(a); m.loaded = true; return true;
}
static void uploadModel(Model& m) {
    for (auto& p : m.parts) {
        glGenVertexArrays(1, &p.vao); glGenBuffers(1, &p.vbo); glGenBuffers(1, &p.ibo);
        glBindVertexArray(p.vao);
        glBindBuffer(GL_ARRAY_BUFFER, p.vbo); glBufferData(GL_ARRAY_BUFFER, p.v.size() * sizeof(SV), p.v.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, p.ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, p.idx.size() * 2, p.idx.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SV), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_BYTE, GL_TRUE, sizeof(SV), (void*)12);
        glDisableVertexAttribArray(2);
        glBindVertexArray(0);
    }
}
static void freeModel(Model& m) {
    for (auto& p : m.parts) {
        if (p.vao) glDeleteVertexArrays(1, &p.vao);
        if (p.vbo) glDeleteBuffers(1, &p.vbo);
        if (p.ibo) glDeleteBuffers(1, &p.ibo);
        p.vao = p.vbo = p.ibo = 0;
    }
}

// dynamic batch (per-vertex colour)
struct Batch {
    GLuint vao = 0, vbo = 0, ibo = 0; std::vector<DV> v; std::vector<uint16_t> i;
    void init() {
        glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo); glGenBuffers(1, &ibo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DV), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_BYTE, GL_TRUE, sizeof(DV), (void*)12);
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(DV), (void*)16);
        glBindVertexArray(0);
    }
    void destroy() { if (vao) { glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &vbo); glDeleteBuffers(1, &ibo); vao = vbo = ibo = 0; } }
    void clear() { v.clear(); i.clear(); }
    bool room(size_t n) const { return v.size() + n < 65000; }
    static int8_t b(float f) { return (int8_t)lrintf(clampf(f, -1.f, 1.f) * 127.f); }
    void vert(V3 p, V3 n, const uint8_t* c) { v.push_back({p.x, p.y, p.z, b(n.x), b(n.y), b(n.z), 0, c[0], c[1], c[2], c[3]}); }
    void quad(V3 a, V3 bb, V3 c, V3 d, V3 n, const uint8_t* col) {
        if (!room(4)) return; uint16_t o = (uint16_t)v.size();
        vert(a, n, col); vert(bb, n, col); vert(c, n, col); vert(d, n, col);
        i.push_back(o); i.push_back(o + 1); i.push_back(o + 2); i.push_back(o); i.push_back(o + 2); i.push_back(o + 3);
    }
    // oriented box: centre + half-extent vectors
    void obox(V3 c, V3 ax, V3 ay, V3 az, const uint8_t* col) {
        V3 p[8];
        for (int k = 0; k < 8; k++) p[k] = c + ax * ((k & 1) ? 1.f : -1.f) + ay * ((k & 2) ? 1.f : -1.f) + az * ((k & 4) ? 1.f : -1.f);
        auto face = [&](int a, int bb, int cc, int d) {
            V3 n = norm(cross(p[bb] - p[a], p[cc] - p[a]));
            V3 mid = (p[a] + p[cc]) * 0.5f; if (dot(n, mid - c) < 0) n = n * -1.f;
            quad(p[a], p[bb], p[cc], p[d], n, col);
        };
        face(0, 1, 3, 2); face(4, 5, 7, 6); face(0, 1, 5, 4); face(2, 3, 7, 6); face(0, 2, 6, 4); face(1, 3, 7, 5);
    }
    void box(V3 c, V3 h, const uint8_t* col) { obox(c, {h.x, 0, 0}, {0, h.y, 0}, {0, 0, h.z}, col); }
    void limb(V3 a, V3 bb, float th, const uint8_t* col) {
        V3 d = (bb - a) * 0.5f; V3 dn = norm(d); V3 up = fabsf(dn.y) > 0.9f ? V3{1, 0, 0} : V3{0, 1, 0};
        V3 s = norm(cross(dn, up)) * th; V3 t = norm(cross(s, dn)) * th;
        obox((a + bb) * 0.5f, s, t, d, col);
    }
    void sphere(V3 c, float r, const uint8_t* col, int seg = 10) {
        if (!room((seg + 1) * (seg + 1))) return;
        uint16_t base = (uint16_t)v.size();
        for (int y = 0; y <= seg; y++) for (int x = 0; x <= seg; x++) {
            float th = PI * y / seg, ph = 2 * PI * x / seg;
            V3 n = {sinf(th) * cosf(ph), cosf(th), sinf(th) * sinf(ph)};
            vert(c + n * r, n, col);
        }
        for (int y = 0; y < seg; y++) for (int x = 0; x < seg; x++) {
            uint16_t a = base + y * (seg + 1) + x, b2 = a + 1, c2 = a + (seg + 1), d = c2 + 1;
            i.push_back(a); i.push_back(c2); i.push_back(b2); i.push_back(b2); i.push_back(c2); i.push_back(d);
        }
    }
    // camera-facing sprite; uv stored in normal.xy for radial shaders
    void sprite(V3 c, V3 right, V3 up, float size, const uint8_t* col) {
        if (!room(4)) return; uint16_t o = (uint16_t)v.size();
        const float sx[4] = {-1, 1, 1, -1}, sy[4] = {-1, -1, 1, 1};
        for (int k = 0; k < 4; k++) {
            V3 p = c + right * (sx[k] * size) + up * (sy[k] * size);
            v.push_back({p.x, p.y, p.z, (int8_t)(sx[k] * 127), (int8_t)(sy[k] * 127), 0, 0, col[0], col[1], col[2], col[3]});
        }
        i.push_back(o); i.push_back(o + 1); i.push_back(o + 2); i.push_back(o); i.push_back(o + 2); i.push_back(o + 3);
    }
    // ground blob (flat, uv in normal.xy)
    void blob(float cx, float cz, float hx, float hz, float y, const uint8_t* col) {
        if (!room(4)) return; uint16_t o = (uint16_t)v.size();
        const float sx[4] = {-1, 1, 1, -1}, sz[4] = {-1, -1, 1, 1};
        for (int k = 0; k < 4; k++)
            v.push_back({cx + sx[k] * hx, y, cz + sz[k] * hz, (int8_t)(sx[k] * 127), (int8_t)(sz[k] * 127), 0, 0, col[0], col[1], col[2], col[3]});
        i.push_back(o); i.push_back(o + 2); i.push_back(o + 1); i.push_back(o); i.push_back(o + 3); i.push_back(o + 2);
    }
    void draw() {
        if (i.empty()) return;
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(DV), v.data(), GL_STREAM_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, i.size() * 2, i.data(), GL_STREAM_DRAW);
        glDrawElements(GL_TRIANGLES, (GLsizei)i.size(), GL_UNSIGNED_SHORT, 0);
        glBindVertexArray(0);
    }
};

// ------------------------------------------------------------------ 5x7 font
struct Glyph { char c; uint8_t r[7]; };
static const Glyph FONT[] = {
    {'0', {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}}, {'1', {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
    {'2', {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}}, {'3', {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}},
    {'4', {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}}, {'5', {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
    {'6', {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}}, {'7', {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
    {'8', {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}}, {'9', {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
    {'A', {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}}, {'B', {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
    {'C', {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}}, {'D', {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
    {'E', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}}, {'F', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
    {'G', {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}}, {'H', {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
    {'I', {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}}, {'J', {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
    {'K', {0x11,0x12,0x14,0x18,0x14,0x12,0x11}}, {'L', {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
    {'M', {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}}, {'N', {0x11,0x11,0x19,0x15,0x13,0x11,0x11}},
    {'O', {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'P', {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
    {'Q', {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}}, {'R', {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
    {'S', {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}}, {'T', {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
    {'U', {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'V', {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
    {'W', {0x11,0x11,0x11,0x15,0x15,0x15,0x0A}}, {'X', {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
    {'Y', {0x11,0x11,0x11,0x0A,0x04,0x04,0x04}}, {'Z', {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
    {'-', {0,0,0,0x1F,0,0,0}}, {'.', {0,0,0,0,0,0x0C,0x0C}}, {':', {0,0x0C,0x0C,0,0x0C,0x0C,0}},
    {'!', {0x04,0x04,0x04,0x04,0x04,0,0x04}}, {'/', {0x01,0x02,0x02,0x04,0x08,0x08,0x10}},
    {'%', {0x18,0x19,0x02,0x04,0x08,0x13,0x03}}, {'+', {0,0x04,0x04,0x1F,0x04,0x04,0}},
};
static const uint8_t* glyphFor(char c) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 32);
    for (auto& g : FONT) if (g.c == c) return g.r;
    return nullptr;
}

struct UVert { float x, y; uint8_t c[4]; };
struct UIBatch {
    std::vector<UVert> v; GLuint vao = 0, vbo = 0;
    void init() {
        glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo); glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(UVert), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(UVert), (void*)8);
        glBindVertexArray(0);
    }
    void destroy() { if (vao) { glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &vbo); vao = vbo = 0; } }
    void rect(float x, float y, float w, float h, const uint8_t* c) {
        UVert a{x, y, {c[0], c[1], c[2], c[3]}}, b{x + w, y, {c[0], c[1], c[2], c[3]}}, d{x + w, y + h, {c[0], c[1], c[2], c[3]}}, e{x, y + h, {c[0], c[1], c[2], c[3]}};
        v.push_back(a); v.push_back(b); v.push_back(d); v.push_back(a); v.push_back(d); v.push_back(e);
    }
    void line(float x1, float y1, float x2, float y2, float th, const uint8_t* c) {
        float dx = x2 - x1, dy = y2 - y1, l = sqrtf(dx * dx + dy * dy) + 1e-5f; float nx = -dy / l * th * 0.5f, ny = dx / l * th * 0.5f;
        UVert a{x1 + nx, y1 + ny, {c[0], c[1], c[2], c[3]}}, b{x2 + nx, y2 + ny, {c[0], c[1], c[2], c[3]}}, d{x2 - nx, y2 - ny, {c[0], c[1], c[2], c[3]}}, e{x1 - nx, y1 - ny, {c[0], c[1], c[2], c[3]}};
        v.push_back(a); v.push_back(b); v.push_back(d); v.push_back(a); v.push_back(d); v.push_back(e);
    }
    float textW(const char* s, float px) { return strlen(s) * 6 * px - px; }
    void text(const char* s, float x, float y, float px, const uint8_t* c, int align = 0) {
        float w = textW(s, px); if (align == 1) x -= w * 0.5f; else if (align == 2) x -= w;
        for (; *s; s++, x += 6 * px) {
            const uint8_t* g = glyphFor(*s); if (!g) continue;
            for (int r = 0; r < 7; r++) {
                int run = -1;
                for (int col = 0; col <= 5; col++) {
                    bool on = col < 5 && (g[r] & (1 << (4 - col)));
                    if (on && run < 0) run = col;
                    if (!on && run >= 0) { rect(x + run * px, y + r * px, (col - run) * px, px, c); run = -1; }
                }
            }
        }
    }
    void draw() {
        if (v.empty()) return;
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(UVert), v.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)v.size()); glBindVertexArray(0); v.clear();
    }
};

// ------------------------------------------------------------------ game data
static const float LANE_W = 3.6f;
static float laneX(int l) { return -5.4f + LANE_W * l; }   // lanes 0,1 oncoming (left); 2,3 same direction (right)
static const float VMAX = 88.f;            // m/s  (~316 km/h)
static const float VBOOST = 104.f;

struct Tier { float scaleMax, scaleMin; int maxCars; int bikeLod; bool carHiNear; float drawDist; int fps; bool blur; int lamps; const char* name; };
static const Tier TIERS[3] = {
    {0.62f, 0.45f, 9, 0, false, 170.f, 30, false, 4, "LOW"},
    {0.85f, 0.55f, 13, 1, true, 220.f, 60, false, 6, "MID"},
    {1.00f, 0.70f, 18, 2, true, 270.f, 60, true, 8, "HIGH"},
};

struct Settings { int steer = 0; int quality = 0; int paint = 0; unsigned best = 0; };
static const uint8_t PAINTS[8][3] = {{225, 25, 40}, {20, 120, 255}, {20, 220, 120}, {255, 140, 0}, {170, 60, 255}, {240, 240, 245}, {255, 40, 160}, {255, 225, 40}};
static const uint8_t CAR_COLS[10][3] = {{200, 30, 30}, {230, 230, 235}, {30, 60, 160}, {20, 20, 24}, {170, 175, 185}, {20, 120, 80}, {220, 140, 20}, {120, 30, 140}, {60, 70, 80}, {200, 190, 60}};

struct Car { int type, lane; float x, dz, v, prevDz; float tint[3]; bool onc; float hw, hl; };
struct Bld { int model; float x, dz, scale, yaw, depth, seed; };

enum State { ST_MENU, ST_PLAY, ST_CRASH, ST_PAUSE, ST_OVER };

static const char* CAR_NAMES[10] = {"wagonbody", "suvbody", "sportbody", "sedanbody", "pickupbody", "offroadbody", "minivanbody", "hatchbackbody", "coupebody", "compactbody"};
static const int NUM_BLD = 19;

struct Btn { float x, y, w, h; int id; };
enum { B_BRAKE = 1, B_NITRO, B_PAUSE, B_PLAY, B_STEER, B_QUAL, B_PAINT, B_RETRY, B_MENU, B_RESUME };

struct Touch { int id = -1; float x = 0, y = 0, sx = 0, sy = 0; int role = 0; };

// ------------------------------------------------------------------ engine
struct Engine {
    android_app* app = nullptr;
    EGLDisplay dpy = EGL_NO_DISPLAY; EGLSurface surf = EGL_NO_SURFACE; EGLContext ctx = EGL_NO_CONTEXT;
    int W = 1, H = 1; bool hasGL = false; bool active = false;
    // GL objects
    GLuint progMain = 0, progSky = 0, progBlit = 0, progUI = 0, emptyVao = 0, quadVao = 0, quadVbo = 0, quadIbo = 0;
    GLuint fbo = 0, fboTex = 0, fboDepth = 0; int fw = 0, fh = 0; float scale = 1.f;
    struct { GLint vp, model, kind, color, tint, cam, fog, fogk, dist, seed, bikex, time; } u;
    Batch bBarrier, bPole, bLamp, bShadow, bGlow, bRider;
    UIBatch ui;
    Model bike[3]; Model cars[10][2]; Model blds[NUM_BLD];
    bool modelsLoaded = false;
    // settings / tier
    Settings st; int tier = 1; bool riderBuilt = false;
    // game
    State state = ST_MENU;
    float px = 5.4f * 0.f + 1.8f, vx = 0, pv = 12.f, lean = 0, pitch = 0, yawv = 0, bob = 0, bobT = 0;
    double dist = 0; float distMod = 0;
    float nitro = 1.f; bool nitroOn = false, braking = false;
    float score = 0; int combo = 1; float comboT = 0; float popupT = 0; int popupVal = 0; int popupCombo = 1;
    float crashT = 0, crashVx = 0, crashY = 0, crashYv = 0, crashRoll = 0, crashSpin = 0, crashPitch = 0, shake = 0, menuT = 0;
    float timeScale = 1.f; float tAcc = 0; bool newBest = false;
    std::vector<Car> traffic; std::vector<Bld> bl[2]; float front[2] = {0, 0};
    // input
    Touch touches[10]; float steerTouch = 0, steerTilt = 0, tiltRaw = 0, steerCmd = 0; bool tiltOk = false;
    std::vector<Btn> btns; double lastUiTap = 0;
    ASensorManager* sm = nullptr; const ASensor* accel = nullptr; ASensorEventQueue* sq = nullptr;
    // timing
    double lastT = 0; float fpsAvg = 60; int fpsFrames = 0; double fpsT0 = 0; double lastScaleUp = 0;
    M4 view, proj, vp; V3 camPos{0, 2, 6}, camRight{1, 0, 0}, camUp{0, 1, 0}; float horizon = 0.5f; float camRoll = 0;

    const Tier& T() const { return TIERS[tier]; }

    // ---------- persistence
    std::string path(const char* f) { return std::string(app->activity->internalDataPath) + "/" + f; }
    void loadSettings() { FILE* f = fopen(path("settings.dat").c_str(), "rb"); if (f) { Settings s; if (fread(&s, sizeof s, 1, f) == 1) st = s; fclose(f); } if (st.paint < 0 || st.paint > 7) st.paint = 0; if (st.steer < 0 || st.steer > 2) st.steer = 0; if (st.quality < 0 || st.quality > 3) st.quality = 0; }
    void saveSettings() { FILE* f = fopen(path("settings.dat").c_str(), "wb"); if (f) { fwrite(&st, sizeof st, 1, f); fclose(f); } }

    void pickTier() {
        if (st.quality > 0) { tier = st.quality - 1; return; }
        double ram = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0 * 1024.0);
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        tier = (ram >= 5.5 && cores >= 8) ? 2 : (ram >= 3.0 ? 1 : 0);
        LOGI("auto tier %d (ram %.1fGB cores %ld)", tier, ram, cores);
    }

    // ---------- EGL
    bool initGL(ANativeWindow* win) {
        dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY); eglInitialize(dpy, nullptr, nullptr);
        const EGLint ca[] = {EGL_RENDERABLE_TYPE, 0x40 /*ES3*/, EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE, 0, EGL_NONE};
        EGLConfig cfg; EGLint n = 0; if (!eglChooseConfig(dpy, ca, &cfg, 1, &n) || n < 1) { LOGE("no EGL config"); return false; }
        EGLint fmt; eglGetConfigAttrib(dpy, cfg, EGL_NATIVE_VISUAL_ID, &fmt); ANativeWindow_setBuffersGeometry(win, 0, 0, fmt);
        surf = eglCreateWindowSurface(dpy, cfg, win, nullptr);
        const EGLint cx[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, cx);
        if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, surf, surf, ctx)) { LOGE("EGL make current failed"); return false; }
        eglQuerySurface(dpy, surf, EGL_WIDTH, &W); eglQuerySurface(dpy, surf, EGL_HEIGHT, &H);
        eglSwapInterval(dpy, 1);
        LOGI("GL %s | %s | %dx%d", glGetString(GL_RENDERER), glGetString(GL_VERSION), W, H);
        return true;
    }
    void termGL() {
        if (hasGL) freeGLResources();
        if (dpy != EGL_NO_DISPLAY) {
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (ctx != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctx);
            if (surf != EGL_NO_SURFACE) eglDestroySurface(dpy, surf);
            eglTerminate(dpy);
        }
        dpy = EGL_NO_DISPLAY; ctx = EGL_NO_CONTEXT; surf = EGL_NO_SURFACE; hasGL = false;
    }

    void setupResources() {
        progMain = makeProgram(MAIN_VS, MAIN_FS); progSky = makeProgram(FS_VS, SKY_FS);
        progBlit = makeProgram(FS_VS, BLIT_FS); progUI = makeProgram(UI_VS, UI_FS);
        u.vp = glGetUniformLocation(progMain, "uVP"); u.model = glGetUniformLocation(progMain, "uModel");
        u.kind = glGetUniformLocation(progMain, "uKind"); u.color = glGetUniformLocation(progMain, "uColor");
        u.tint = glGetUniformLocation(progMain, "uTint"); u.cam = glGetUniformLocation(progMain, "uCam");
        u.fog = glGetUniformLocation(progMain, "uFog"); u.fogk = glGetUniformLocation(progMain, "uFogK");
        u.dist = glGetUniformLocation(progMain, "uDist"); u.seed = glGetUniformLocation(progMain, "uSeed");
        u.bikex = glGetUniformLocation(progMain, "uBikeX"); u.time = glGetUniformLocation(progMain, "uTime");
        glGenVertexArrays(1, &emptyVao);
        // unit quad (y = 0 plane)
        SV qv[4] = {{-1, 0, -1, 0, 127, 0, 0}, {1, 0, -1, 0, 127, 0, 0}, {1, 0, 1, 0, 127, 0, 0}, {-1, 0, 1, 0, 127, 0, 0}};
        uint16_t qi[6] = {0, 2, 1, 0, 3, 2};
        glGenVertexArrays(1, &quadVao); glGenBuffers(1, &quadVbo); glGenBuffers(1, &quadIbo);
        glBindVertexArray(quadVao); glBindBuffer(GL_ARRAY_BUFFER, quadVbo); glBufferData(GL_ARRAY_BUFFER, sizeof qv, qv, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadIbo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof qi, qi, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SV), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_BYTE, GL_TRUE, sizeof(SV), (void*)12);
        glBindVertexArray(0);
        glVertexAttrib4f(2, 1.f, 1.f, 1.f, 1.f);
        bBarrier.init(); bPole.init(); bLamp.init(); bShadow.init(); bGlow.init(); bRider.init(); ui.init();
        if (!modelsLoaded) loadAllModels();
        for (auto& m : bike) uploadModel(m);
        for (auto& c : cars) for (auto& m : c) uploadModel(m);
        for (auto& b : blds) uploadModel(b);
        buildRider();
        makeFbo();
        hasGL = true;
    }
    void freeGLResources() {
        for (auto& m : bike) freeModel(m);
        for (auto& c : cars) for (auto& m : c) freeModel(m);
        for (auto& b : blds) freeModel(b);
        bBarrier.destroy(); bPole.destroy(); bLamp.destroy(); bShadow.destroy(); bGlow.destroy(); bRider.destroy(); ui.destroy();
        killFbo();
        if (quadVao) { glDeleteVertexArrays(1, &quadVao); glDeleteBuffers(1, &quadVbo); glDeleteBuffers(1, &quadIbo); quadVao = 0; }
        if (emptyVao) { glDeleteVertexArrays(1, &emptyVao); emptyVao = 0; }
        glDeleteProgram(progMain); glDeleteProgram(progSky); glDeleteProgram(progBlit); glDeleteProgram(progUI);
    }
    void loadAllModels() {
        loadModel("bike_low.mdl", bike[0]); loadModel("bike_mid.mdl", bike[1]); loadModel("bike_hi.mdl", bike[2]);
        char nm[64];
        for (int i = 0; i < 10; i++) {
            snprintf(nm, sizeof nm, "car_%s.mdl", CAR_NAMES[i]); loadModel(nm, cars[i][0]);
            snprintf(nm, sizeof nm, "car_%s_low.mdl", CAR_NAMES[i]); loadModel(nm, cars[i][1]);
        }
        for (int i = 0; i < NUM_BLD; i++) { snprintf(nm, sizeof nm, "bld_%d.mdl", i); loadModel(nm, blds[i]); }
        modelsLoaded = true;
    }
    void makeFbo() {
        killFbo();
        fw = std::max(64, (int)(W * scale)); fh = std::max(64, (int)(H * scale));
        glGenTextures(1, &fboTex); glBindTexture(GL_TEXTURE_2D, fboTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fw, fh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glGenRenderbuffers(1, &fboDepth); glBindRenderbuffer(GL_RENDERBUFFER, fboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, tier == 2 ? GL_DEPTH_COMPONENT24 : GL_DEPTH_COMPONENT16, fw, fh);
        glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fboDepth);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) LOGE("FBO incomplete");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void killFbo() {
        if (fbo) { glDeleteFramebuffers(1, &fbo); fbo = 0; }
        if (fboTex) { glDeleteTextures(1, &fboTex); fboTex = 0; }
        if (fboDepth) { glDeleteRenderbuffers(1, &fboDepth); fboDepth = 0; }
    }

    // rider built once in bike-local space (bike faces -Z)
    void buildRider() {
        bRider.clear();
        const uint8_t jacket[4] = {20, 20, 26, 255}, accent[4] = {230, 40, 60, 255}, glove[4] = {15, 15, 18, 255}, helmet[4] = {235, 235, 240, 255}, visor[4] = {20, 30, 50, 255}, boot[4] = {25, 25, 28, 255}, pants[4] = {28, 32, 48, 255};
        V3 hip{0, 0.86f, 0.42f}, sh{0, 1.27f, -0.04f}, head{0, 1.46f, -0.14f};
        bRider.limb(hip, sh, 0.17f, jacket);
        bRider.box({0, 1.2f, 0.02f}, {0.2f, 0.04f, 0.05f}, accent);
        bRider.sphere(head, 0.15f, helmet, 10);
        bRider.box({0, 1.47f, -0.27f}, {0.1f, 0.04f, 0.04f}, visor);
        for (int s = -1; s <= 1; s += 2) {
            V3 shoulder{0.2f * s, 1.25f, -0.03f}, elbow{0.3f * s, 1.12f, -0.27f}, hand{0.27f * s, 1.02f, -0.5f};
            bRider.limb(shoulder, elbow, 0.055f, jacket); bRider.limb(elbow, hand, 0.05f, jacket); bRider.sphere(hand, 0.05f, glove, 6);
            V3 hipS{0.12f * s, 0.86f, 0.42f}, knee{0.3f * s, 0.74f, -0.02f}, foot{0.24f * s, 0.42f, 0.22f};
            bRider.limb(hipS, knee, 0.085f, pants); bRider.limb(knee, foot, 0.07f, pants); bRider.box(foot + V3{0, -0.03f, -0.05f}, {0.05f, 0.04f, 0.11f}, boot);
        }
    }

    // ---------- world setup
    float laneFree(int lane, float dz, float gap) { for (auto& c : traffic) if (c.lane == lane && fabsf(c.dz - dz) < gap) return 0; return 1; }
    void spawnCar(float dmin, float dmax) {
        for (int tries = 0; tries < 8; tries++) {
            float r = rnd(); int lane = r < 0.18f ? 0 : r < 0.42f ? 1 : r < 0.72f ? 2 : 3;
            float dz = rndr(dmin, dmax);
            if (!laneFree(lane, dz, 24.f)) continue;
            Car c{}; c.type = (int)(rnd() * 10) % 10; c.lane = lane; c.x = laneX(lane); c.dz = dz; c.prevDz = dz; c.onc = lane < 2;
            c.v = c.onc ? -rndr(14.f, 28.f) : rndr(10.f, 26.f);
            int col = (int)(rnd() * 10) % 10; for (int k = 0; k < 3; k++) c.tint[k] = CAR_COLS[col][k] / 255.f;
            Model& m = cars[c.type][0]; c.hw = (m.mx.x - m.mn.x) * 0.5f; c.hl = (m.mx.z - m.mn.z) * 0.5f;
            traffic.push_back(c); return;
        }
    }
    void spawnBuilding(int side) {
        Bld b{}; b.model = (int)(rnd() * NUM_BLD) % NUM_BLD; Model& m = blds[b.model];
        float ex = m.mx.x - m.mn.x, ey = m.mx.y - m.mn.y, ez = m.mx.z - m.mn.z;
        float hWorld = clampf(ey * 0.2f, 16.f, 60.f); b.scale = hWorld / std::max(ey, 1.f);
        int q = (int)(rnd() * 4); b.yaw = q * PI * 0.5f; bool odd = q & 1;
        float wx = (odd ? ez : ex) * b.scale, wz = (odd ? ex : ez) * b.scale;
        b.depth = std::max(wz, 8.f); b.x = (side ? 1.f : -1.f) * (13.5f + wx * 0.5f + rnd() * 4.f);
        b.dz = front[side] + b.depth * 0.5f; front[side] += b.depth + 2.f + rnd() * 6.f; b.seed = rnd() * 50.f;
        bl[side].push_back(b);
    }
    void resetWorld() {
        traffic.clear(); bl[0].clear(); bl[1].clear(); front[0] = front[1] = -60.f;
        for (int s = 0; s < 2; s++) while (front[s] < T().drawDist + 40) spawnBuilding(s);
    }
    void startGame() {
        resetWorld(); px = laneX(3); vx = 0; pv = 18.f; lean = pitch = 0; dist = 0; nitro = 1.f; score = 0; combo = 1; comboT = 0;
        crashT = 0; shake = 0; timeScale = 1.f; newBest = false; popupT = 0;
        for (int i = 0; i < 8; i++) spawnCar(40.f + i * 26.f, 60.f + i * 26.f);
        state = ST_PLAY;
    }
    void toMenu() { resetWorld(); px = laneX(2) + 1.8f; vx = 0; pv = 10.f; lean = pitch = 0; state = ST_MENU; menuT = 0; crashRoll = crashPitch = crashY = 0; shake = 0; timeScale = 1.f; }

    // ---------- game update
    void doCrash(float impact) {
        state = ST_CRASH; crashT = 0; crashVx = impact; crashYv = 3.f + pv * 0.06f; crashY = 0; crashRoll = 0; crashPitch = 0;
        crashSpin = (rnd() < 0.5f ? -1.f : 1.f) * rndr(5.f, 9.f); shake = 1.f; timeScale = 0.35f; nitroOn = false;
        if ((unsigned)score > st.best) { st.best = (unsigned)score; newBest = true; saveSettings(); }
    }

    void update(float dtReal) {
        float dt = dtReal * (state == ST_CRASH ? timeScale : 1.f);
        if (state == ST_PAUSE) return;
        steerCmd = 0;
        if (st.steer == 0 || !tiltOk) steerCmd = steerTouch; else steerCmd = steerTilt;
        steerCmd = clampf(steerCmd, -1.f, 1.f);
        menuT += dtReal; shake = std::max(0.f, shake - dtReal * (state == ST_CRASH ? 0.7f : 2.5f));

        if (state == ST_MENU) { pv = 10.f; }
        else if (state == ST_PLAY) {
            nitroOn = false;
            bool wantNitro = false, wantBrake = false;
            for (auto& t : touches) if (t.id >= 0) { if (t.role == B_NITRO) wantNitro = true; if (t.role == B_BRAKE) wantBrake = true; }
            braking = wantBrake;
            if (wantNitro && nitro > 0.02f && !wantBrake) { nitroOn = true; nitro = std::max(0.f, nitro - dt * 0.22f); }
            else if (!wantNitro) nitro = std::min(1.f, nitro + dt * 0.06f);
            float vlim = nitroOn ? VBOOST : VMAX;
            float a = 9.5f * (nitroOn ? 1.8f : 1.f) * (1.f - powf(clampf(pv / vlim, 0.f, 1.2f), 2.f));
            if (pv > vlim) a = -6.f;
            float drag = 0.5f + 0.0006f * pv * pv;
            float accel = wantBrake ? -26.f : (a - drag * 0.35f);
            pv = std::max(0.f, pv + accel * dt);
            // steering (needs speed, gets heavier when fast)
            float gate = clampf(pv / 7.f, 0.f, 1.f), latMax = lerpf(9.5f, 5.6f, clampf(pv / VMAX, 0.f, 1.f));
            float vxT = steerCmd * latMax * gate;
            vx += (vxT - vx) * std::min(1.f, dt * 6.5f);
            px += vx * dt;
            float pt = clampf(accel * 0.0125f, -0.12f, 0.17f) + (nitroOn && pv < 55.f ? 0.14f : 0.f);
            pitch += (pt - pitch) * std::min(1.f, dt * 4.5f);
            bobT += dt * (18.f + pv * 0.25f); bob = sinf(bobT) * 0.012f * clampf(pv / 50.f, 0.f, 1.5f);
            dist += pv * dt;
            bool onc = px < 0; float mult = onc ? 2.f : 1.f;
            score += pv * dt * 0.1f * mult * (float)combo;
            comboT -= dt; if (comboT <= 0 && combo > 1) { combo = 1; }
            if (popupT > 0) popupT -= dt;
            if (fabsf(px) > 7.75f) { doCrash(-sgn(px) * 4.f - vx * 0.3f); }
        } else if (state == ST_CRASH) {
            crashT += dtReal;
            pv = std::max(0.f, pv - 24.f * dt * (crashY < 0.05f ? 1.f : 0.3f));
            px += (crashVx + vx * 0.4f) * dt; crashVx *= (1.f - dt * 0.8f);
            crashYv -= 22.f * dt; crashY += crashYv * dt;
            if (crashY < 0) { crashY = 0; if (fabsf(crashYv) > 2.5f) crashYv = -crashYv * 0.38f; else crashYv = 0; }
            crashRoll += crashSpin * dt * (crashY > 0.01f ? 1.f : 0.25f); crashPitch += crashSpin * 0.4f * dt;
            dist += pv * dt;
            if (crashT > 2.4f) state = ST_OVER;
            if (crashT > 0.6f) timeScale = lerpf(timeScale, 1.f, dtReal * 3.f);
        } else if (state == ST_OVER) {
            pv = std::max(0.f, pv - 24.f * dt); dist += pv * dt; crashSpin *= (1.f - dt * 2.f); crashRoll += crashSpin * dt * 0.1f;
        }
        distMod = (float)fmod(dist, 96.0);
        lean += ((state == ST_PLAY ? -clampf(vx * 0.055f * (0.5f + pv / 60.f), -0.62f, 0.62f) : 0.f) - lean) * std::min(1.f, dt * 7.f);

        // ---- buildings
        for (int s = 0; s < 2; s++) {
            front[s] -= pv * dt;
            for (auto& b : bl[s]) b.dz -= pv * dt;
            bl[s].erase(std::remove_if(bl[s].begin(), bl[s].end(), [](const Bld& b) { return b.dz + b.depth < -70.f; }), bl[s].end());
            while (front[s] < T().drawDist + 40.f) spawnBuilding(s);
        }
        // ---- traffic
        if (state != ST_MENU) {
            for (auto& c : traffic) { c.prevDz = c.dz; }
            // simple following behaviour per lane
            for (auto& a : traffic) for (auto& b : traffic) {
                if (&a == &b || a.lane != b.lane) continue;
                if (!a.onc) { if (b.dz > a.dz && b.dz - a.dz < 15.f && a.v > b.v) a.v = b.v; }
                else { if (b.dz < a.dz && a.dz - b.dz < 15.f && a.v < b.v) a.v = b.v; }
            }
            for (auto& c : traffic) c.dz += (c.v - pv) * dt;
            traffic.erase(std::remove_if(traffic.begin(), traffic.end(), [](const Car& c) { return c.dz < -90.f || c.dz > 420.f; }), traffic.end());
            if (state == ST_PLAY) {
                float density = 1.f + std::min(0.6f, (float)(dist / 20000.0));
                tAcc += dt * (0.8f + pv / 30.f) * density;
                while (tAcc > 1.f) { tAcc -= 1.f; if ((int)traffic.size() < T().maxCars) spawnCar(280.f, 330.f); }
                // collisions + near misses
                for (auto& c : traffic) {
                    float L = 1.05f + c.hl * 0.92f, Wd = 0.36f + c.hw * 0.9f;
                    float lo = std::min(c.prevDz, c.dz), hi = std::max(c.prevDz, c.dz);
                    float gap = fabsf(px - c.x) - (0.4f + c.hw);
                    if (hi > -L && lo < L && fabsf(px - c.x) < Wd) { doCrash(sgn(px - c.x) * (3.f + pv * 0.08f)); break; }
                    bool crossed = (c.prevDz > 0 && c.dz <= 0) || (c.prevDz < 0 && c.dz >= 0);
                    if (crossed && gap < 0.75f && pv > 12.f) {
                        combo = std::min(8, combo + 1); comboT = 3.2f; int pts = 100 * combo; score += (float)pts; popupT = 1.2f; popupVal = pts; popupCombo = combo;
                    }
                }
            }
        }
    }

    // ---------- rendering
    void setKind(int k, float r, float g, float b, float a = 1.f) { glUniform1i(u.kind, k); glUniform4f(u.color, r, g, b, a); }
    void drawModel(Model& m, const M4& model, const float* tint, int pass) {
        glUniformMatrix4fv(u.model, 1, GL_FALSE, model.m);
        for (auto& p : m.parts) {
            bool glass = p.kind == 1; if ((pass == 1) != glass) continue;
            if (p.idx.empty() || !p.vao) continue;
            glUniform1i(u.kind, p.kind);
            glUniform4f(u.color, p.col[0] / 255.f, p.col[1] / 255.f, p.col[2] / 255.f, p.col[3] / 255.f);
            if (p.kind == 0) glUniform3f(u.tint, tint[0], tint[1], tint[2]);
            glBindVertexArray(p.vao); glDrawElements(GL_TRIANGLES, (GLsizei)p.idx.size(), GL_UNSIGNED_SHORT, 0);
        }
    }
    M4 bikeMatrix() {
        M4 m = mTrans(px, 0.f, 0.f);
        float yaw = state == ST_PLAY ? -clampf(atan2f(vx, std::max(pv, 4.f)) * 1.4f, -0.5f, 0.5f) : 0.f;
        if (state == ST_CRASH || state == ST_OVER) {
            m = mMul(m, mTrans(0, crashY, 0)); m = mMul(m, mRotY(crashRoll * 0.5f));
            m = mMul(m, mRotZ(crashRoll)); m = mMul(m, mRotX(crashPitch * 0.5f)); return m;
        }
        if (state == ST_MENU) m = mMul(m, mRotY(0.f));
        m = mMul(m, mTrans(0, bob, 0)); m = mMul(m, mRotY(yaw)); m = mMul(m, mRotZ(lean));
        m = mMul(m, mTrans(0, 0, 0.75f)); m = mMul(m, mRotX(pitch)); m = mMul(m, mTrans(0, 0, -0.75f));
        return m;
    }

    void render(double tNow) {
        float speedF = clampf(pv / VMAX, 0.f, 1.2f);
        // camera
        float fov = (58.f + speedF * 20.f + (nitroOn ? 7.f : 0.f)) * PI / 180.f;
        float sh = shake * 0.12f + speedF * 0.015f;
        float ox = (rnd() - 0.5f) * sh, oy = (rnd() - 0.5f) * sh;
        V3 cp, tg;
        if (state == ST_MENU) {
            float a = 0.9f + sinf(menuT * 0.35f) * 0.55f; cp = {px + sinf(a) * 4.6f, 1.5f, cosf(a) * 4.6f}; tg = {px, 0.75f, 0.2f};
        } else {
            float h = 2.35f - speedF * 0.25f, d = 5.6f - speedF * 0.4f - (nitroOn ? 0.4f : 0.f);
            cp = {px * 0.6f + ox, h + oy + (state == ST_CRASH ? 0.6f : 0.f), d}; tg = {px * 0.8f, 0.95f, -13.f};
        }
        camPos = cp;
        float roll = (state == ST_PLAY ? lean * 0.35f : 0.f);
        V3 up = {sinf(-roll), cosf(roll), 0};
        view = mLookAt(cp, tg, up); proj = mPersp(fov, (float)W / H, 0.4f, T().drawDist + 120.f); vp = mMul(proj, view);
        camRight = {view.m[0], view.m[4], view.m[8]}; camUp = {view.m[1], view.m[5], view.m[9]};
        // horizon for sky
        { V3 f = norm(tg - cp); float hy = f.y; (void)hy; float vy = view.m[1] * 0 ; (void)vy;
          float cx = proj.m[5] * (view.m[1] * 0 + 0); (void)cx;
          // project far point straight ahead on the ground plane direction (0,0,-1)
          V3 far = {cp.x, 0.f, cp.z - 2000.f}; float cy = vp.m[1] * far.x + vp.m[5] * far.y + vp.m[9] * far.z + vp.m[13]; float cw = vp.m[3] * far.x + vp.m[7] * far.y + vp.m[11] * far.z + vp.m[15];
          horizon = clampf(0.5f + 0.5f * (cy / std::max(cw, 1e-3f)), 0.f, 1.f); }

        const V3 fogc = {0.13f, 0.06f, 0.20f};
        glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, fw, fh);
        glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDepthMask(GL_TRUE); glDisable(GL_CULL_FACE);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(progSky); glUniform1f(glGetUniformLocation(progSky, "uHorizon"), horizon); glUniform1f(glGetUniformLocation(progSky, "uTime"), (float)tNow);
        glUniform3f(glGetUniformLocation(progSky, "uFog"), fogc.x, fogc.y, fogc.z);
        glBindVertexArray(emptyVao); glDrawArrays(GL_TRIANGLES, 0, 3);

        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL);
        glUseProgram(progMain);
        glUniformMatrix4fv(u.vp, 1, GL_FALSE, vp.m); glUniform3f(u.cam, cp.x, cp.y, cp.z);
        glUniform3f(u.fog, fogc.x, fogc.y, fogc.z);
        float fk = powf(2.2f / T().drawDist, 2.f); glUniform1f(u.fogk, fk);
        glUniform1f(u.dist, distMod); glUniform1f(u.bikex, px); glUniform1f(u.time, (float)tNow);
        glUniform3f(u.tint, 1, 1, 1);
        glVertexAttrib4f(2, 1.f, 1.f, 1.f, 1.f);

        // ground + road
        glBindVertexArray(quadVao);
        setKind(8, 1, 1, 1);
        M4 mm = mMul(mTrans(0, 0, -150.f), mScale(8.2f, 1.f, 180.f)); glUniformMatrix4fv(u.model, 1, GL_FALSE, mm.m); glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        setKind(9, 1, 1, 1);
        for (int s = -1; s <= 1; s += 2) { mm = mMul(mTrans(s * 129.2f, 0, -150.f), mScale(121.f, 1.f, 180.f)); glUniformMatrix4fv(u.model, 1, GL_FALSE, mm.m); glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0); }

        // barriers, lamp poles (dynamic batches)
        const uint8_t white[4] = {255, 255, 255, 255}, poleC[4] = {55, 58, 70, 255}, lampC[4] = {255, 200, 130, 255};
        bBarrier.clear(); bPole.clear(); bLamp.clear(); bShadow.clear(); bGlow.clear();
        for (int s = -1; s <= 1; s += 2) bBarrier.box({s * 8.35f, 0.45f, -140.f}, {0.32f, 0.45f, 180.f}, white);
        float spacing = 32.f; float firstK = floorf((distMod - 60.f) / spacing);
        int nL = T().lamps;
        for (int k = 0; k < nL + 3; k++) {
            float zAbsLamp = (firstK + k + 1) * spacing; float zw = -(zAbsLamp - distMod) ;
            // zAbs = dist - z  =>  z = dist - zAbs ; lamps sit on multiples of 32 of zAbs
            float zWorld = distMod - zAbsLamp; zWorld = -( (zAbsLamp - distMod) );
            (void)zw;
            if (zWorld > 40.f || zWorld < -T().drawDist) continue;
            for (int s = -1; s <= 1; s += 2) {
                float x = s * 9.3f;
                bPole.box({x, 4.0f, zWorld}, {0.1f, 4.0f, 0.1f}, poleC);
                bPole.box({x - s * 1.4f, 7.9f, zWorld}, {1.5f, 0.07f, 0.07f}, poleC);
                bLamp.box({x - s * 2.6f, 7.85f, zWorld}, {0.45f, 0.06f, 0.18f}, lampC);
            }
        }
        setKind(7, 1, 1, 1); bBarrier.draw();
        setKind(4, 1, 1, 1); bPole.draw();
        setKind(3, 1, 1, 1); bLamp.draw();

        // buildings
        setKind(6, 1, 1, 1);
        for (int s = 0; s < 2; s++) for (auto& b : bl[s]) {
            if (b.dz - b.depth > T().drawDist + 20.f) continue;
            M4 bm = mMul(mTrans(b.x, 0, -b.dz), mMul(mRotY(b.yaw), mScale(b.scale, b.scale, b.scale)));
            glUniform1f(u.seed, b.seed); glUniform1i(u.kind, 6); glUniformMatrix4fv(u.model, 1, GL_FALSE, bm.m);
            for (auto& p : blds[b.model].parts) { if (!p.vao || p.idx.empty()) continue; glBindVertexArray(p.vao); glDrawElements(GL_TRIANGLES, (GLsizei)p.idx.size(), GL_UNSIGNED_SHORT, 0); }
        }

        // cars (opaque)
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1) { glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE); }
            if (state != ST_MENU) for (auto& c : traffic) {
                if (c.dz > T().drawDist + 10.f) continue;
                int lod = (T().carHiNear && c.dz < 48.f && c.dz > -20.f) ? 0 : 1;
                M4 cm = mMul(mTrans(c.x, 0, -c.dz), mRotY(c.onc ? PI : 0.f));
                drawModel(cars[c.type][lod], cm, c.tint, pass);
            }
            // player bike + rider
            Model& bm = bike[T().bikeLod]; M4 bmat = bikeMatrix();
            float tint[3] = {PAINTS[st.paint][0] / 255.f, PAINTS[st.paint][1] / 255.f, PAINTS[st.paint][2] / 255.f};
            drawModel(bm, bmat, tint, pass);
            if (pass == 0) {
                glUniformMatrix4fv(u.model, 1, GL_FALSE, bmat.m); setKind(4, 1, 1, 1); glUniform3f(u.tint, 1, 1, 1);
                bRider.draw();
            }
        }
        // shadows + glows
        const uint8_t shC[4] = {0, 0, 0, 150};
        bShadow.blob(px, 0.1f, 0.55f + fabsf(lean) * 0.3f, 1.5f, 0.03f, shC);
        for (auto& c : traffic) if (c.dz < T().drawDist && c.dz > -30.f) bShadow.blob(c.x, -c.dz, c.hw * 1.15f, c.hl * 1.2f, 0.03f, shC);
        float sgnCam = 1.f; (void)sgnCam;
        for (auto& c : traffic) {
            if (c.dz > T().drawDist + 20.f || c.dz < -25.f) continue;
            float d = std::max(c.dz, 1.f); float sz = 0.35f + d * 0.0105f;
            float a = clampf(1.f - d / (T().drawDist + 20.f), 0.f, 1.f);
            if (c.onc) {   // headlights face the player (+z)
                uint8_t col[4] = {(uint8_t)(255 * a + 40), (uint8_t)(245 * a + 30), (uint8_t)(210 * a + 20), 255};
                for (int s = -1; s <= 1; s += 2) bGlow.sprite({c.x + s * c.hw * 0.62f, 0.72f, -c.dz + c.hl * 0.95f}, camRight, camUp, sz, col);
            } else {       // tail-lights face the player (+z)
                uint8_t col[4] = {(uint8_t)(255 * a + 20), (uint8_t)(20 * a), (uint8_t)(30 * a), 255};
                for (int s = -1; s <= 1; s += 2) bGlow.sprite({c.x + s * c.hw * 0.66f, 0.78f, -c.dz + c.hl * 0.97f}, camRight, camUp, sz * 0.8f, col);
            }
        }
        for (int k = 0; k < nL + 3; k++) {
            float zAbsLamp = (firstK + k + 1) * spacing; float zWorld = -(zAbsLamp - distMod);
            if (zWorld > 40.f || zWorld < -T().drawDist) continue;
            float d = std::max(-zWorld, 3.f); float a = clampf(1.f - d / T().drawDist, 0.f, 1.f);
            uint8_t col[4] = {(uint8_t)(255 * a), (uint8_t)(170 * a), (uint8_t)(90 * a), 255};
            for (int s = -1; s <= 1; s += 2) bGlow.sprite({s * 6.7f, 7.7f, zWorld}, camRight, camUp, 0.9f + d * 0.012f, col);
        }
        // bike lights
        { M4 bmat = bikeMatrix(); (void)bmat; uint8_t tl[4] = {255, 30, 40, 255}; V3 p = {px, 0.95f + bob, 1.18f}; bGlow.sprite(p, camRight, camUp, 0.28f, tl); }
        setKind(12, 1, 1, 1, 1.f); bShadow.draw();
        glBlendFunc(GL_ONE, GL_ONE); setKind(11, 1, 1, 1, 1.f); bGlow.draw();
        glDepthMask(GL_TRUE); glDisable(GL_BLEND);

        // ---- present (upscale + radial blur)
        glBindFramebuffer(GL_FRAMEBUFFER, 0); glViewport(0, 0, W, H); glDisable(GL_DEPTH_TEST);
        glUseProgram(progBlit); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, fboTex);
        glUniform1i(glGetUniformLocation(progBlit, "uTex"), 0);
        float blur = T().blur ? clampf((pv - 30.f) / 70.f, 0.f, 1.f) * 0.045f + (nitroOn ? 0.02f : 0.f) : 0.f;
        glUniform1f(glGetUniformLocation(progBlit, "uBlur"), blur); glUniform1f(glGetUniformLocation(progBlit, "uVig"), 0.55f);
        glBindVertexArray(emptyVao); glDrawArrays(GL_TRIANGLES, 0, 3);

        drawUI(tNow);
    }

    // ---------- UI
    float U() const { return H / 360.f; }
    void button(float x, float y, float w, float h, int id, const char* label, bool active = false, float textPx = 0) {
        float u = U(); btns.push_back({x, y, w, h, id});
        uint8_t edge[4] = {120, 230, 255, 220}, fill[4] = {10, 14, 30, 190}, fa[4] = {30, 90, 150, 220};
        ui.rect(x - u, y - u, w + 2 * u, h + 2 * u, edge); ui.rect(x, y, w, h, active ? fa : fill);
        uint8_t tc[4] = {255, 255, 255, 255}; float px2 = textPx > 0 ? textPx : u * 2.2f;
        ui.text(label, x + w * 0.5f, y + h * 0.5f - 3.5f * px2, px2, tc, 1);
    }
    void drawUI(double tNow) {
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(progUI); glUniform2f(glGetUniformLocation(progUI, "uRes"), (float)W, (float)H);
        float u = U(); btns.clear(); char buf[96];
        const uint8_t white[4] = {255, 255, 255, 255}, cyan[4] = {120, 235, 255, 255}, pink[4] = {255, 70, 170, 255}, yel[4] = {255, 220, 60, 255}, dim[4] = {255, 255, 255, 140}, shade[4] = {0, 0, 10, 150};
        float kmh = pv * 3.6f;
        // speed lines
        if (state == ST_PLAY || state == ST_CRASH) {
            float sp = clampf((pv - 38.f) / 60.f, 0.f, 1.f);
            if (sp > 0.02f) {
                uint8_t lc[4] = {255, 255, 255, (uint8_t)(sp * 70.f)};
                int n = tier == 0 ? 10 : 22;
                for (int i = 0; i < n; i++) {
                    float a = rnd() * 2 * PI, r0 = rndr(0.55f, 0.8f) * H, len = rndr(0.12f, 0.35f) * H * sp;
                    float cx = W * 0.5f, cy = H * 0.46f;
                    ui.line(cx + cosf(a) * r0 * 1.5f, cy + sinf(a) * r0, cx + cosf(a) * (r0 + len) * 1.5f, cy + sinf(a) * (r0 + len), u * 1.5f, lc);
                }
            }
            if (nitroOn) { uint8_t nc[4] = {80, 160, 255, 40}; ui.rect(0, 0, (float)W, (float)H, nc); }
        }
        if (state == ST_PLAY) {
            // score / best
            snprintf(buf, sizeof buf, "%u", (unsigned)score); ui.text("SCORE", 18 * u, 12 * u, 2 * u, dim); ui.text(buf, 18 * u, 24 * u, 4 * u, white);
            snprintf(buf, sizeof buf, "BEST %u", st.best); ui.text(buf, 18 * u, 56 * u, 2 * u, dim);
            snprintf(buf, sizeof buf, "%.2f KM", dist / 1000.0); ui.text(buf, W * 0.5f, 12 * u, 2.4f * u, cyan, 1);
            if (combo > 1) { snprintf(buf, sizeof buf, "COMBO X%d", combo); ui.text(buf, W * 0.5f, 28 * u, 3 * u, yel, 1); float bw = 120 * u * clampf(comboT / 3.2f, 0.f, 1.f); ui.rect(W * 0.5f - 60 * u, 50 * u, bw, 3 * u, yel); }
            if (px < 0) ui.text("ONCOMING X2", W * 0.5f, 60 * u, 2.4f * u, pink, 1);
            if (popupT > 0) { uint8_t pc[4] = {255, 220, 60, (uint8_t)(clampf(popupT, 0.f, 1.f) * 255)}; snprintf(buf, sizeof buf, "NEAR MISS +%d", popupVal); ui.text(buf, W * 0.5f, H * 0.28f - (1.2f - popupT) * 14 * u, 4 * u, pc, 1); }
            // speedometer
            float bx = W * 0.5f, by = H - 48 * u; float segW = 6.2f * u; int segs = 32; float tot = segs * segW;
            snprintf(buf, sizeof buf, "%d", (int)kmh); ui.text(buf, bx, by - 36 * u, 6 * u, white, 1); ui.text("KM/H", bx + 54 * u, by - 12 * u, 2.2f * u, cyan, 0);
            float fillF = clampf(pv / VBOOST, 0.f, 1.f);
            for (int i = 0; i < segs; i++) {
                float f = (float)i / segs; bool on = f < fillF;
                uint8_t c[4] = {(uint8_t)(f < 0.5f ? 60 + f * 300 : 255), (uint8_t)(f < 0.5f ? 230 : 230 - (f - 0.5f) * 400), 60, (uint8_t)(on ? 235 : 60)};
                ui.rect(bx - tot * 0.5f + i * segW, by + (i % 2 ? 0 : 0), segW - 1.5f * u, 8 * u, c);
            }
            // buttons
            float bw = 120 * u, bh = 78 * u;
            button(18 * u, H - bh - 18 * u, bw, bh, B_BRAKE, "BRAKE");
            button(W - bw - 18 * u, H - bh - 18 * u, bw, bh, B_NITRO, "NITRO", nitroOn);
            { float nf = nitro; uint8_t nc[4] = {70, 170, 255, 230}; ui.rect(W - bw - 18 * u, H - 18 * u - 7 * u, bw * nf, 7 * u, nc); }
            button(W - 44 * u, 12 * u, 32 * u, 32 * u, B_PAUSE, "II", false, 2.6f * u);
            if (st.steer != 0 && tiltOk) { /* tilt hint bar */ float cx = W * 0.5f, cy = H - 14 * u; uint8_t tc[4] = {255, 255, 255, 90}; ui.rect(cx - 50 * u, cy, 100 * u, 2 * u, tc); uint8_t pc[4] = {120, 235, 255, 255}; ui.rect(cx + steerCmd * 48 * u - 2 * u, cy - 3 * u, 4 * u, 8 * u, pc); }
        } else if (state == ST_MENU) {
            ui.rect(0, 0, (float)W, (float)H, shade);
            uint8_t glow[4] = {255, 40, 160, 200};
            ui.text("NEON", 40 * u + 2 * u, 40 * u + 2 * u, 14 * u, glow); ui.text("NEON", 40 * u, 40 * u, 14 * u, white);
            ui.text("RIDER", 40 * u + 2 * u, 40 * u + 108 * u + 2 * u, 14 * u, glow); ui.text("RIDER", 40 * u, 40 * u + 108 * u, 14 * u, cyan);
            snprintf(buf, sizeof buf, "BEST %u", st.best); ui.text(buf, 44 * u, 250 * u, 3.2f * u, yel);
            float mx = W - 190 * u, bw = 170 * u, bh = 46 * u;
            button(mx, 40 * u, bw, 64 * u, B_PLAY, "PLAY", true, 4.2f * u);
            const char* sm[3] = {"STEER: TOUCH", "STEER: TILT", "TILT: INVERT"};
            button(mx, 120 * u, bw, bh, B_STEER, sm[st.steer], false, 1.9f * u);
            snprintf(buf, sizeof buf, "GFX: %s", st.quality == 0 ? "AUTO" : TIERS[st.quality - 1].name); button(mx, 120 * u + (bh + 10 * u), bw, bh, B_QUAL, buf, false, 2.2f * u);
            button(mx, 120 * u + 2 * (bh + 10 * u), bw, bh, B_PAINT, "PAINT", false, 2.2f * u);
            uint8_t pcol[4] = {PAINTS[st.paint][0], PAINTS[st.paint][1], PAINTS[st.paint][2], 255}; ui.rect(mx + bw - 34 * u, 120 * u + 2 * (bh + 10 * u) + 10 * u, 24 * u, bh - 20 * u, pcol);
            snprintf(buf, sizeof buf, "%s  %.0f%%", TIERS[tier].name, scale * 100); ui.text(buf, W - 20 * u, H - 22 * u, 1.8f * u, dim, 2);
        } else if (state == ST_PAUSE) {
            ui.rect(0, 0, (float)W, (float)H, shade);
            ui.text("PAUSED", W * 0.5f, H * 0.2f, 9 * u, white, 1);
            button(W * 0.5f - 90 * u, H * 0.5f - 10 * u, 180 * u, 52 * u, B_RESUME, "RESUME", true, 3.2f * u);
            button(W * 0.5f - 90 * u, H * 0.5f + 55 * u, 180 * u, 46 * u, B_MENU, "MENU", false, 2.6f * u);
        } else if (state == ST_CRASH || state == ST_OVER) {
            if (state == ST_OVER) {
                ui.rect(0, 0, (float)W, (float)H, shade);
                ui.text("CRASHED", W * 0.5f, H * 0.1f, 10 * u, pink, 1);
                snprintf(buf, sizeof buf, "SCORE %u", (unsigned)score); ui.text(buf, W * 0.5f, H * 0.36f, 5 * u, white, 1);
                snprintf(buf, sizeof buf, "%.2f KM", dist / 1000.0); ui.text(buf, W * 0.5f, H * 0.36f + 44 * u, 3 * u, cyan, 1);
                if (newBest) ui.text("NEW BEST!", W * 0.5f, H * 0.36f + 74 * u, 4 * u, yel, 1); else { snprintf(buf, sizeof buf, "BEST %u", st.best); ui.text(buf, W * 0.5f, H * 0.36f + 74 * u, 3 * u, dim, 1); }
                button(W * 0.5f - 190 * u, H - 100 * u, 180 * u, 56 * u, B_RETRY, "RETRY", true, 3.4f * u);
                button(W * 0.5f + 10 * u, H - 100 * u, 180 * u, 56 * u, B_MENU, "MENU", false, 3.4f * u);
            }
        }
        ui.draw(); glDisable(GL_BLEND);
    }

    // ---------- input
    void uiPress(int id) {
        double t = nowSec(); if (t - lastUiTap < 0.18) return; lastUiTap = t;
        switch (id) {
            case B_PLAY: case B_RETRY: startGame(); break;
            case B_PAUSE: if (state == ST_PLAY) state = ST_PAUSE; break;
            case B_RESUME: state = ST_PLAY; break;
            case B_MENU: toMenu(); break;
            case B_STEER: st.steer = (st.steer + 1) % 3; saveSettings(); break;
            case B_PAINT: st.paint = (st.paint + 1) % 8; saveSettings(); break;
            case B_QUAL: st.quality = (st.quality + 1) % 4; saveSettings(); pickTier(); scale = T().scaleMax; if (hasGL) makeFbo(); resetWorld(); break;
        }
    }
    int hit(float x, float y) { for (auto& b : btns) if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) return b.id; return 0; }
    void touchEvent(AInputEvent* ev) {
        int act = AMotionEvent_getAction(ev), a = act & AMOTION_EVENT_ACTION_MASK, idx = (act & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
        auto find = [&](int id) -> Touch* { for (auto& t : touches) if (t.id == id) return &t; return nullptr; };
        if (a == AMOTION_EVENT_ACTION_DOWN || a == AMOTION_EVENT_ACTION_POINTER_DOWN) {
            int id = AMotionEvent_getPointerId(ev, idx); float x = AMotionEvent_getX(ev, idx), y = AMotionEvent_getY(ev, idx);
            for (auto& t : touches) if (t.id < 0) {
                t.id = id; t.x = t.sx = x; t.y = t.sy = y; int h = hit(x, y); t.role = 0;
                if (h == B_BRAKE || h == B_NITRO) t.role = h;
                else if (h) { uiPress(h); t.role = -1; }
                else if (state == ST_PLAY) t.role = 100;   // steering drag
                break;
            }
        } else if (a == AMOTION_EVENT_ACTION_MOVE) {
            int n = (int)AMotionEvent_getPointerCount(ev);
            for (int i = 0; i < n; i++) { Touch* t = find(AMotionEvent_getPointerId(ev, i)); if (t) { t->x = AMotionEvent_getX(ev, i); t->y = AMotionEvent_getY(ev, i); } }
        } else if (a == AMOTION_EVENT_ACTION_UP || a == AMOTION_EVENT_ACTION_POINTER_UP || a == AMOTION_EVENT_ACTION_CANCEL) {
            if (a == AMOTION_EVENT_ACTION_CANCEL) for (auto& t : touches) t.id = -1;
            else { Touch* t = find(AMotionEvent_getPointerId(ev, idx)); if (t) t->id = -1; }
        }
        steerTouch = 0;
        for (auto& t : touches) if (t.id >= 0 && t.role == 100) steerTouch = clampf((t.x - t.sx) / (0.11f * W), -1.f, 1.f);
    }

    void initSensors() {
        sm = ASensorManager_getInstance(); if (!sm) return;
        accel = ASensorManager_getDefaultSensor(sm, ASENSOR_TYPE_ACCELEROMETER);
        if (accel) { sq = ASensorManager_createEventQueue(sm, ALooper_forThread(), LOOPER_ID_USER, nullptr, nullptr); }
    }
    void sensorsOn(bool on) {
        if (!sq || !accel) return;
        if (on) { ASensorEventQueue_enableSensor(sq, accel); ASensorEventQueue_setEventRate(sq, accel, 16000); }
        else ASensorEventQueue_disableSensor(sq, accel);
    }
    void pumpSensors() {
        if (!sq) return; ASensorEvent e;
        while (ASensorEventQueue_getEvents(sq, &e, 1) > 0) {
            if (e.type == ASENSOR_TYPE_ACCELEROMETER) {
                tiltOk = true; float raw = e.acceleration.y;   // landscape: screen-x == device-y
                tiltRaw += (raw - tiltRaw) * 0.25f;
                float v = -tiltRaw / 4.2f; if (st.steer == 2) v = -v;
                float dz = 0.06f; v = fabsf(v) < dz ? 0.f : v - sgn(v) * dz;
                steerTilt = clampf(v * 1.15f, -1.f, 1.f);
            }
        }
    }

    // ---------- frame
    void frame() {
        double t = nowSec(); float dt = (float)clampf((float)(t - lastT), 0.f, 0.05f); double rawDt = t - lastT; lastT = t;
        pumpSensors();
        update(dt);
        render(t);
        eglSwapBuffers(dpy, surf);
        // adaptive resolution
        fpsFrames++;
        if (t - fpsT0 > 1.0) {
            float fps = fpsFrames / (float)(t - fpsT0); fpsFrames = 0; fpsT0 = t;
            float target = (float)T().fps; bool changed = false;
            if (fps < target * 0.88f && scale > T().scaleMin + 0.01f) { scale = std::max(T().scaleMin, scale - 0.07f); changed = true; lastScaleUp = t; }
            else if (fps > target * 0.97f && scale < T().scaleMax - 0.01f && t - lastScaleUp > 8.0) { scale = std::min(T().scaleMax, scale + 0.04f); changed = true; lastScaleUp = t; }
            if (changed) makeFbo();
        }
        // frame cap
        double spent = nowSec() - t, want = 1.0 / T().fps - 0.0015;
        if (spent < want) { timespec ts{0, (long)((want - spent) * 1e9)}; nanosleep(&ts, nullptr); }
        (void)rawDt;
    }
};

static Engine E;

static void onCmd(android_app* app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (app->window) {
                if (E.initGL(app->window)) { E.pickTier(); E.scale = E.T().scaleMax; E.setupResources(); E.resetWorld(); E.lastT = nowSec(); E.fpsT0 = E.lastT; E.lastScaleUp = E.lastT; E.active = true; }
            } break;
        case APP_CMD_TERM_WINDOW: E.active = false; E.termGL(); break;
        case APP_CMD_GAINED_FOCUS: E.sensorsOn(true); E.lastT = nowSec(); break;
        case APP_CMD_LOST_FOCUS: E.sensorsOn(false); if (E.state == ST_PLAY) E.state = ST_PAUSE; for (auto& t : E.touches) t.id = -1; E.steerTouch = 0; break;
        case APP_CMD_PAUSE: if (E.state == ST_PLAY) E.state = ST_PAUSE; break;
        default: break;
    }
}
static int32_t onInput(android_app* app, AInputEvent* ev) {
    (void)app;
    if (AInputEvent_getType(ev) == AINPUT_EVENT_TYPE_MOTION) { E.touchEvent(ev); return 1; }
    if (AInputEvent_getType(ev) == AINPUT_EVENT_TYPE_KEY && AKeyEvent_getKeyCode(ev) == AKEYCODE_BACK) {
        if (AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_UP) {
            if (E.state == ST_PLAY) E.state = ST_PAUSE; else if (E.state == ST_PAUSE || E.state == ST_OVER) E.toMenu(); else ANativeActivity_finish(app->activity);
        }
        return 1;
    }
    return 0;
}

void android_main(android_app* app) {
    E.app = app; gAssets = app->activity->assetManager;
    app->userData = &E; app->onAppCmd = onCmd; app->onInputEvent = onInput;
    E.loadSettings(); E.initSensors();
    while (true) {
        int events; android_poll_source* src;
        while (true) {
            int ident = ALooper_pollOnce(E.active ? 0 : -1, nullptr, &events, (void**)&src);
            if (ident < 0) break;
            if (src) src->process(app, src);
            if (ident == LOOPER_ID_USER) E.pumpSensors();
            if (app->destroyRequested) { E.termGL(); return; }
        }
        if (E.active && E.hasGL) E.frame();
    }
}

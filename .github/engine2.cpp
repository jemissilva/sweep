#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <random>
#include <time.h>
#include <pthread.h>
#include <secp256k1.h>

static const uint64_t P[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                              0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
static const uint64_t CC = 0x1000003D1ULL;

static int f_cmp(const uint64_t a[4], const uint64_t b[4]) {
  for (int i = 3; i >= 0; i--) { if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1; }
  return 0;
}

static void f_subraw(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  unsigned __int128 borrow = 0;
  for (int i = 0; i < 4; i++) { unsigned __int128 t = (unsigned __int128)a[i] - b[i] - borrow; r[i] = (uint64_t)t; borrow = (t >> 64) ? 1 : 0; }
}

static void f_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  unsigned __int128 carry = 0;
  for (int i = 0; i < 4; i++) { carry += (unsigned __int128)a[i] + b[i]; r[i] = (uint64_t)carry; carry >>= 64; }
  if (carry) {
    unsigned __int128 c2 = (unsigned __int128)r[0] + CC;
    r[0] = (uint64_t)c2;
    c2 >>= 64;
    for (int i = 1; i < 4 && c2; i++) { c2 += r[i]; r[i] = (uint64_t)c2; c2 >>= 64; }
    return;
  }
  if (f_cmp(r, P) >= 0) f_subraw(r, r, P);
}

static void f_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  unsigned __int128 borrow = 0;
  for (int i = 0; i < 4; i++) { unsigned __int128 t = (unsigned __int128)a[i] - b[i] - borrow; r[i] = (uint64_t)t; borrow = (t >> 64) ? 1 : 0; }
  if (!borrow) return;
  unsigned __int128 carry = 0;
  for (int i = 0; i < 4; i++) { carry += (unsigned __int128)r[i] + P[i]; r[i] = (uint64_t)carry; carry >>= 64; }
}

static void f_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t t[8] = {0,0,0,0,0,0,0,0};
  for (int i = 0; i < 4; i++) {
    unsigned __int128 carry = 0;
    for (int j = 0; j < 4; j++) {
      unsigned __int128 acc = (unsigned __int128)t[i+j] + (unsigned __int128)a[i] * b[j] + carry;
      t[i+j] = (uint64_t)acc; carry = acc >> 64;
    }
    t[i+4] = (uint64_t)carry;
  }
  int guard = 0;
  while ((t[4] | t[5] | t[6] | t[7]) && guard++ < 16) {
    uint64_t m[5] = {0,0,0,0,0};
    unsigned __int128 c = 0;
    for (int i = 0; i < 4; i++) { c += (unsigned __int128)t[4+i] * CC; m[i] = (uint64_t)c; c >>= 64; }
    m[4] = (uint64_t)c;
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) { carry += (unsigned __int128)t[i] + m[i]; t[i] = (uint64_t)carry; carry >>= 64; }
    t[4] = (uint64_t)carry + m[4];
    t[5] = t[6] = t[7] = 0;
  }
  while (f_cmp(t, P) >= 0) f_subraw(t, t, P);
  memcpy(r, t, 32);
}

static void f_sqr(uint64_t r[4], const uint64_t a[4]) { f_mul(r, a, a); }

static void f_from_bytes(uint64_t r[4], const uint8_t* b) {
  for (int i = 0; i < 4; i++) {
    uint64_t v = 0;
    for (int j = 0; j < 8; j++) v = (v << 8) | b[24 - 8*i + j];
    r[i] = v;
  }
}
static const uint64_t GX[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
static const uint64_t GY[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};

typedef struct { uint64_t X[4], Y[4], Z[4]; } jac;

static secp256k1_context* g_ctx = NULL;
static void rev32(uint8_t*);
static uint8_t g_root[32];
static uint64_t g_id = 0;
static uint64_t g_tx[4], g_ty[4];
static volatile int g_found = 0;
static volatile int g_stop = 0;
static uint64_t g_count = 0;
static uint64_t g_total = 0;
static uint64_t g_t0 = 0;
static uint64_t g_seedlo = 262144; static uint64_t g_seedhi = 524288;

static void key_words(uint8_t key[32], const uint8_t root[32], uint64_t round, uint64_t id) {
  memcpy(key, root, 32);
  unsigned __int128 c = round;
  for (int i = 31; i >= 24 && c; i--) { c += key[i]; key[i] = (uint8_t)c; c >>= 8; }
  c = id;
  for (int i = 14; i >= 7 && c; i--) { c += key[i]; key[i] = (uint8_t)c; c >>= 8; }
}

static void jac_double(jac* q) {
  uint64_t A[4], B[4], C[4], D[4], E[4], F[4], t[4], x3[4], y3[4], z3[4];
  f_sqr(A, q->X);
  f_sqr(B, q->Y);
  f_sqr(C, B);
  f_add(t, q->X, B);
  f_sqr(t, t);
  f_sub(t, t, A);
  f_sub(t, t, C);
  f_add(D, t, t);
  f_add(E, A, A);
  f_add(E, E, A);
  f_sqr(F, E);
  f_add(x3, D, D);
  f_sub(x3, F, x3);
  f_sub(t, D, x3);
  f_mul(y3, E, t);
  f_add(t, C, C);
  f_add(t, t, t);
  f_add(t, t, t);
  f_sub(y3, y3, t);
  f_mul(z3, q->Y, q->Z);
  f_add(z3, z3, z3);
  memcpy(q->X, x3, 32); memcpy(q->Y, y3, 32); memcpy(q->Z, z3, 32);
}

static int g_dbg = 0;


static int on_curve(const uint64_t X[4], const uint64_t Y[4]) {
  uint64_t l[4], rr[4];
  const uint64_t seven[4] = {7,0,0,0};
  f_sqr(l, Y);
  f_sqr(rr, X);
  f_mul(rr, rr, X);
  f_add(rr, rr, seven);
  return f_cmp(l, rr) == 0;
}

static void jac_add_affine(jac* q, const uint64_t ax[4], const uint64_t ay[4]) {
  uint64_t Z1Z1[4], U2[4], S2[4], H[4], HH[4], I[4], J[4], r[4], V[4], t[4], x3[4], y3[4], z3[4];
  if (!(q->Z[0] | q->Z[1] | q->Z[2] | q->Z[3])) {
    memcpy(q->X, ax, 32); memcpy(q->Y, ay, 32);
    q->Z[0] = 1; q->Z[1] = q->Z[2] = q->Z[3] = 0;
    return;
  }
  f_sqr(Z1Z1, q->Z);
  f_mul(U2, ax, Z1Z1);
  f_mul(t, q->Z, Z1Z1);
  f_mul(S2, ay, t);
  f_sub(H, U2, q->X);
  f_sub(r, S2, q->Y);
  f_add(r, r, r);
  if (!(H[0] | H[1] | H[2] | H[3])) {
    if (!(r[0] | r[1] | r[2] | r[3])) { jac_double(q); return; }
    memset(q->X, 0, 32); memset(q->Y, 0, 32); memset(q->Z, 0, 32);
    return;
  }
  f_sqr(HH, H);
  f_add(I, HH, HH);
  f_add(I, I, I);
  f_mul(J, H, I);
  f_mul(V, q->X, I);
  f_sqr(x3, r);
  f_sub(x3, x3, J);
  f_sub(x3, x3, V);
  f_sub(x3, x3, V);
  f_sub(t, V, x3);
  f_mul(y3, r, t);
  f_mul(t, q->Y, J);
  f_add(t, t, t);
  f_sub(y3, y3, t);
  f_mul(z3, q->Z, H);
  f_add(z3, z3, z3);
  memcpy(q->X, x3, 32); memcpy(q->Y, y3, 32); memcpy(q->Z, z3, 32);
}

static int jac_eq_affine(const jac* q, const uint64_t ax[4], const uint64_t ay[4]) {
  if (!(q->Z[0] | q->Z[1] | q->Z[2] | q->Z[3])) return 0;
  uint64_t Z2[4], Z3[4], t[4];
  f_sqr(Z2, q->Z);
  f_mul(Z3, Z2, q->Z);
  f_mul(t, ax, Z2);
  if (f_cmp(q->X, t) != 0) return 0;
  f_mul(t, ay, Z3);
  return f_cmp(q->Y, t) == 0;
}

static int point_of(const uint8_t key[32], jac* J) {
  secp256k1_pubkey pk;
  uint8_t ser[65];
  size_t len = 65;
  if (!secp256k1_ec_pubkey_create(g_ctx, &pk, key)) return 0;
  if (!secp256k1_ec_pubkey_serialize(g_ctx, ser, &len, &pk, SECP256K1_EC_UNCOMPRESSED)) return 0;
  f_from_bytes(J->X, ser + 1);
  f_from_bytes(J->Y, ser + 33);
  memset(J->Z, 0, 32);
  J->Z[0] = 1;
  return 1;
}

static uint64_t g_rounds = 4096;
static int g_fixed = 0;
static void derive_root(uint8_t root[32], uint64_t seed);
typedef struct { int tid; uint64_t from, to; } warg;

static void* worker(void* p) {
  warg* a = (warg*)p;
  uint8_t root[32];
  uint8_t key[32];
  uint64_t local = 0;
  for (uint64_t sd = a->from; sd < a->to && !g_found && !g_stop; sd++) {
    if (g_fixed) memcpy(root, g_root, 32); else derive_root(root, sd); rev32(root);
    uint64_t r = 0;
    jac J;
    int ok = 0;
    while (r < g_rounds && !g_found && !g_stop) {
      key_words(key, root, r, g_id);
      if (point_of(key, &J)) { ok = 1; break; }
      r++;
    }
    if (!ok) continue;
    while (r < g_rounds && !g_found && !g_stop) {
      if (jac_eq_affine(&J, g_tx, g_ty)) {
        g_found = 1;
        FILE* f = fopen("ACHOU.txt", "w");
        if (f) {
          fprintf(f, "seed=%llu\nround=%llu\nid=%llu\nkey=", (unsigned long long)sd, (unsigned long long)r, (unsigned long long)g_id);
          for (int i = 0; i < 32; i++) fprintf(f, "%02x", key[i]);
          fprintf(f, "\n");
          fclose(f);
        }
        break;
      }
      jac_add_affine(&J, GX, GY);
      r++;
      key_words(key, root, r, g_id);
      local++;
      if ((local & 0xFFFF) == 0) { g_count += local; local = 0; }
    }
  }
  g_count += local;
  return NULL;
}
static void* progress(void* p) {
  (void)p;
  uint64_t last = 0;
  while (!g_found && !g_stop) {
    struct timespec ts = {2, 0};
    nanosleep(&ts, NULL);
    uint64_t c = g_count;
    double dt = (double)(time(NULL) - g_t0);
    double rate = dt > 0 ? (double)c / dt : 0;
    double rest = g_total > c ? (double)(g_total - c) / (rate > 1 ? rate : 1) : 0;
    fprintf(stderr, "%.2f M/s | %llu/%llu | ETA %.0f h\n",
            rate / 1e6, (unsigned long long)c, (unsigned long long)g_total, rest / 3600.0);
    last = c;
  }
  (void)last;
  return NULL;
}

static void hex_to_bytes(uint8_t out[32], const char* h) {
  if (h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) h += 2;
  size_t n = strlen(h);
  memset(out, 0, 32);
  for (size_t i = 0; i < n && i < 64; i++) {
    char c = h[n - 1 - i];
    int v = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
    out[31 - i / 2] |= (uint8_t)((i & 1) ? (v << 4) : v);
  }
}

static void derive_root(uint8_t root[32], uint64_t seed) {
  std::mt19937_64 rd(seed);
  for (int i = 0; i < 4; i++) { uint64_t v = rd(); memcpy(root + 8 * i, &v, 8); }
}

static int field_selftest(void) {
  uint64_t x[4], y[4];
  const uint64_t zero[4] = {0,0,0,0};
  const uint64_t one[4] = {1,0,0,0};
  const uint64_t two[4] = {2,0,0,0};
  const uint64_t three[4] = {3,0,0,0};
  const uint64_t six[4] = {6,0,0,0};
  const uint64_t pm1[4] = {0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
  f_add(x, pm1, one);
  if (f_cmp(x, zero) != 0) { printf("FALHA F1 (P-1)+1\n"); return 1; }
  f_sub(x, zero, one);
  if (f_cmp(x, pm1) != 0) { printf("FALHA F2 0-1\n"); return 1; }
  f_mul(x, two, three);
  if (f_cmp(x, six) != 0) { printf("FALHA F3 2*3\n"); return 1; }
  f_mul(x, pm1, pm1);
  if (f_cmp(x, one) != 0) { printf("FALHA F4 (P-1)^2\n"); return 1; }
  f_mul(x, pm1, two);
  f_add(y, x, two);
  if (f_cmp(y, zero) != 0) { printf("FALHA F5\n"); return 1; }
  printf("TESTE F OK\n");
  return 0;
}

static int fuzz(void) {
  uint64_t s = 0x123456789abcdefULL;
  uint64_t A[4], B[4], C[4], X[4], Y[4], Z[4];
  for (int it = 0; it < 2000; it++) {
    for (int i = 0; i < 4; i++) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL; A[i] = s;
      s = s * 6364136223846793005ULL + 1442695040888963407ULL; B[i] = s;
      s = s * 6364136223846793005ULL + 1442695040888963407ULL; C[i] = s;
    }
    while (f_cmp(A, P) >= 0) f_subraw(A, A, P);
    while (f_cmp(B, P) >= 0) f_subraw(B, B, P);
    while (f_cmp(C, P) >= 0) f_subraw(C, C, P);
    f_mul(X, A, B); f_mul(Y, B, A);
    if (f_cmp(X, Y) != 0) { printf("FUZZ comutativa mul it %d\n", it); return 1; }
    f_add(X, A, B); f_add(Y, B, A);
    if (f_cmp(X, Y) != 0) { printf("FUZZ comutativa add it %d\n", it); return 1; }
    f_sub(X, X, B);
    if (f_cmp(X, A) != 0) { printf("FUZZ (a+b)-b it %d\n", it); return 1; }
    f_mul(X, A, B); f_mul(Y, A, C); f_add(X, X, Y);
    f_add(Z, B, C); f_mul(Y, A, Z);
    if (f_cmp(X, Y) != 0) { printf("FUZZ distributiva it %d\n", it); return 1; }
    f_mul(X, A, B);
    if (f_cmp(X, P) >= 0) { printf("FUZZ resultado >= P it %d\n", it); return 1; }
  }
  printf("TESTE FUZZ OK\n");
  return 0;
}

static int point_selftest(void) {
  uint8_t one[32];
  jac P;
  memset(one, 0, 32);
  one[31] = 1;
  if (!point_of(one, &P)) { printf("FALHA create 1\n"); return 1; }
  printf("IGUAL_X %d IGUAL_Y %d CURVA %d\n", f_cmp(P.X, GX) == 0, f_cmp(P.Y, GY) == 0, on_curve(P.X, P.Y));
  printf("P.X %016llx%016llx%016llx%016llx\n", (unsigned long long)P.X[3],(unsigned long long)P.X[2],(unsigned long long)P.X[1],(unsigned long long)P.X[0]);
  printf("P.Y %016llx%016llx%016llx%016llx\n", (unsigned long long)P.Y[3],(unsigned long long)P.Y[2],(unsigned long long)P.Y[1],(unsigned long long)P.Y[0]);
  return 0;
}

static int run_tests(void) {
  if (field_selftest()) return 1;
  if (fuzz()) return 1;
  point_selftest();
  uint8_t root[32];
  derive_root(root, 12345);
  uint64_t base = 0x6696;
  uint8_t k[32];
  jac J;
  key_words(k, root, base, 0);
  if (!point_of(k, &J)) { printf("FALHA base A\n"); return 1; }
  for (int i = 0; i <= 777; i++) {
    uint8_t kk[32];
    jac R;
    key_words(kk, root, base + (uint64_t)i, 0);
    if (!point_of(kk, &R)) { printf("FALHA create A %d\n", i); return 1; }
    if (!jac_eq_affine(&J, R.X, R.Y)) {
      uint64_t z2[4], z3[4], cx[4], cy[4];
      f_sqr(z2, J.Z); f_mul(z3, z2, J.Z);
      f_mul(cx, R.X, z2); f_mul(cy, R.Y, z3);
      printf("FALHA A passo %d\n", i);
      printf("JX %016llx%016llx%016llx%016llx\n", (unsigned long long)J.X[3], (unsigned long long)J.X[2], (unsigned long long)J.X[1], (unsigned long long)J.X[0]);
      printf("CX %016llx%016llx%016llx%016llx\n", (unsigned long long)cx[3], (unsigned long long)cx[2], (unsigned long long)cx[1], (unsigned long long)cx[0]);
      printf("JY %016llx%016llx%016llx%016llx\n", (unsigned long long)J.Y[3], (unsigned long long)J.Y[2], (unsigned long long)J.Y[1], (unsigned long long)J.Y[0]);
      printf("CY %016llx%016llx%016llx%016llx\n", (unsigned long long)cy[3], (unsigned long long)cy[2], (unsigned long long)cy[1], (unsigned long long)cy[0]);
      return 1;
    }
    jac_add_affine(&J, GX, GY);
  }
  printf("TESTE A OK\n");
  uint8_t a[32], b[32];
  key_words(a, root, 0, 0x699f);
  key_words(b, root, 0, 0x6696);
  uint64_t va = 0, vb = 0;
  for (int i = 7; i <= 14; i++) { va = (va << 8) | a[i]; vb = (vb << 8) | b[i]; }
  if (va - vb != 777) { printf("FALHA B delta %llu\n", (unsigned long long)(va - vb)); return 1; }
  for (int i = 0; i < 7; i++) if (a[i] != b[i]) { printf("FALHA B byte %d\n", i); return 1; }
  printf("TESTE B OK\n");
  key_words(k, root, base, 0x699f);
  if (!point_of(k, &J)) { printf("FALHA base C\n"); return 1; }
  for (int i = 0; i <= 777; i++) {
    uint8_t kk[32];
    jac R;
    key_words(kk, root, base + (uint64_t)i, 0x699f);
    if (!point_of(kk, &R)) { printf("FALHA create C %d\n", i); return 1; }
    if (!jac_eq_affine(&J, R.X, R.Y)) { printf("FALHA C passo %d\n", i); return 1; }
    jac_add_affine(&J, GX, GY);
  }
  printf("TESTE C OK\n");
  printf("TODOS OS TESTES PASSARAM\n");
  return 0;
}
static void rev32(uint8_t*k){for(int i=0;i<16;i++){uint8_t t=k[i];k[i]=k[31-i];k[31-i]=t;}}
int main(int argc, char** argv) {
  uint64_t from = 0, to = 0, seed = 262144;
  int threads = 8;
  const char* roothex = NULL;
  int do_test = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--test")) do_test = 1;
    else if (!strcmp(argv[i], "--from") && i + 1 < argc) from = strtoull(argv[++i], NULL, 0);
    else if (!strcmp(argv[i], "--to") && i + 1 < argc) to = strtoull(argv[++i], NULL, 0);
    else if (!strcmp(argv[i], "--id") && i + 1 < argc) g_id = strtoull(argv[++i], NULL, 0);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
    else if (!strcmp(argv[i], "--seedlo") && i + 1 < argc) g_seedlo = strtoull(argv[++i], NULL, 0);
    else if (!strcmp(argv[i], "--seedhi") && i + 1 < argc) g_seedhi = strtoull(argv[++i], NULL, 0);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--root") && i + 1 < argc) roothex = argv[++i];
  }
  g_ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
  if (do_test) { int r = run_tests(); secp256k1_context_destroy(g_ctx); return r; }
  hex_to_bytes(g_root, "dc97975262ebe4a5b991f777b8070152b7c52da58337983faa15cb75230c61f5");
  uint8_t tmp[32];
  f_from_bytes(g_tx, g_root);
  hex_to_bytes(tmp, "a07859310dc6e4eabef9d2ef4e163d38751c8f14ebedb59e2161faae84317591");
  f_from_bytes(g_ty, tmp);
  if (roothex) hex_to_bytes(g_root, roothex);
  else derive_root(g_root, seed);
  if (to == 0) to = from + 1000000;
  if (to <= from) { printf("range invalido\n"); return 1; }
  if (threads < 1) threads = 1;
  if (seed != 262144) { g_seedlo = seed; g_seedhi = seed + 1; }
  if (roothex) { g_fixed = 1; g_seedlo = 0; g_seedhi = 1; }
  if (g_seedhi <= g_seedlo) g_seedhi = g_seedlo + 1;
  if (to > 0) g_rounds = to;
  if (g_rounds < 1) g_rounds = 1;
  g_total = (g_seedhi - g_seedlo) * g_rounds;
  printf("rounds=%llu seeds=%llu..%llu threads=%d\n",
         (unsigned long long)g_rounds, (unsigned long long)g_seedlo,
         (unsigned long long)g_seedhi, threads);
  g_t0 = (uint64_t)time(NULL);
  pthread_t pg;
  pthread_create(&pg, NULL, progress, NULL);
  pthread_t th[64];
  warg wa[64];
  uint64_t nseed = g_seedhi - g_seedlo;
  uint64_t schunk = nseed / (uint64_t)threads;
  if (schunk < 1) schunk = 1;
  for (int i = 0; i < threads; i++) {
    wa[i].tid = i;
    wa[i].from = g_seedlo + schunk * (uint64_t)i;
    wa[i].to = (i == threads - 1) ? g_seedhi : g_seedlo + schunk * (uint64_t)(i + 1);
    if (wa[i].from > g_seedhi) wa[i].from = wa[i].to = g_seedhi;
    pthread_create(&th[i], NULL, worker, &wa[i]);
  }
  for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
  g_stop = 1;
  pthread_join(pg, NULL);
  double dt = (double)(time(NULL) - g_t0);
  printf("%s | %llu chaves em %.0f s (%.2f M/s)\n", g_found ? "ACHOU" : "nada encontrado",
         (unsigned long long)g_count, dt, dt > 0 ? (double)g_count / dt / 1e6 : 0);
  if (g_found) {
    FILE* f = fopen("ACHOU.txt", "r");
    if (f) { char line[256]; while (fgets(line, sizeof line, f)) printf("%s", line); fclose(f); }
  }
  secp256k1_context_destroy(g_ctx);
  return 0;
}

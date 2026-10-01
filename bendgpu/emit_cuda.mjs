// Bend -> CUDA, first slice.
//
// Front end: the real Bend parser/checker (bendlang/bend, bend2/bend.ts) used as a library, untouched.
// Back end (this file): a small emitter for a first-order numeric subset of the checked, lowered
// terms, so the SAME Bend source that today runs on the interaction-net runtime can be compiled to a
// plain CUDA kernel with registers, loops and no heap:
//
//   scalars        U32, F32, Bool, Nat (as a counter)
//   records        single-constructor ADTs whose fields are scalars / records  -> C structs
//   control        `match` on Bool / Nat / records (tail position), tail self-calls -> `for(;;)` loops
//   calls          other subset defs (device functions), Base primitives mapped by name
//   entry          def kern(i: U32) -> scalar|record, run as a parallel map over i in [0, N)
//
// Anything outside the subset is rejected with the name of the construct (no silent fallback).
//
//   node --experimental-transform-types emit_cuda.mjs prog.bend kern > prog.cu
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";

const BEND_SRC = process.env.BEND_SRC || path.resolve(process.cwd(), "vendor/bend");
const Bend = await import(pathToFileURL(path.join(BEND_SRC, "bend2/bend.ts")).href);

class Unsupported extends Error {}
const unsup = (what, where) => { throw new Unsupported(`unsupported in ${where}: ${what}`); };

// ------------------------------------------------------------------------------------------ book
export async function readBook(file, base) {
  const book = Bend.book_nil();
  const seen = new Map();
  if (base) {
    for (const k of Object.keys(base.tlds)) book.tlds[k] = { ...base.tlds[k] };
    Object.assign(book.ctrs, base.ctrs);
    for (const k of Object.keys(base.tmps)) book.tmps[k] = { ...base.tmps[k] };
    book.order.push(...base.order);
    seen.set(Bend.BASE_BEND, "");
  }
  await Bend.book_load(book, file, "", seen);
  Bend.book_valid(book, base ? base.order.length : 0);
  return book;
}

// ------------------------------------------------------------------------------------ primitives
// Base implements U32.* with bit-level adders; the real compiler swaps in native ops by name, so do we.
// [arity, template]: $0.. are the (non-erased) arguments.
const PRIM = {
  "U32.add": [2, "($0 + $1)"], "U32.sub": [2, "($0 - $1)"], "U32.mul": [2, "($0 * $1)"],
  "U32.div": [2, "($1 ? $0 / $1 : 0u)"], "U32.mod": [2, "($1 ? $0 % $1 : 0u)"],
  "U32.and": [2, "($0 & $1)"], "U32.or": [2, "($0 | $1)"], "U32.xor": [2, "($0 ^ $1)"], "U32.not": [1, "(~$0)"],
  "U32.inc": [1, "($0 + 1u)"], "U32.shl": [1, "($0 << 1)"], "U32.shr": [1, "($0 >> 1)"],
  "U32.is_eq": [2, "($0 == $1)"], "U32.is_ne": [2, "($0 != $1)"], "U32.is_lt": [2, "($0 < $1)"],
  "U32.is_le": [2, "($0 <= $1)"], "U32.is_gt": [2, "($0 > $1)"], "U32.is_ge": [2, "($0 >= $1)"],
  "U32.is_zero": [1, "($0 == 0u)"], "U32.min": [2, "($0 < $1 ? $0 : $1)"], "U32.max": [2, "($0 > $1 ? $0 : $1)"],
  "U32.to_nat": [1, "$0"], "U32.from_nat": [1, "$0"], "U32.to_f32": [1, "((float)$0)"], "F32.to_u32": [1, "((uint32_t)$0)"],
  "Nat.add": [2, "($0 + $1)"], "Nat.sub": [2, "($0 > $1 ? $0 - $1 : 0u)"], "Nat.mul": [2, "($0 * $1)"],
  "F32.add": [2, "($0 + $1)"], "F32.sub": [2, "($0 - $1)"], "F32.mul": [2, "($0 * $1)"], "F32.div": [2, "($0 / $1)"],
  "F32.neg": [1, "(-$0)"], "F32.abs": [1, "fabsf($0)"], "F32.sqrt": [1, "sqrtf($0)"], "F32.exp": [1, "expf($0)"],
  "F32.log": [1, "logf($0)"], "F32.log2": [1, "log2f($0)"], "F32.log10": [1, "log10f($0)"], "F32.sin": [1, "sinf($0)"],
  "F32.cos": [1, "cosf($0)"], "F32.tan": [1, "tanf($0)"], "F32.asin": [1, "asinf($0)"], "F32.acos": [1, "acosf($0)"],
  "F32.atan": [1, "atanf($0)"], "F32.sinh": [1, "sinhf($0)"], "F32.cosh": [1, "coshf($0)"], "F32.tanh": [1, "tanhf($0)"],
  "F32.floor": [1, "floorf($0)"], "F32.ceil": [1, "ceilf($0)"], "F32.trunc": [1, "truncf($0)"],
  "F32.pow": [2, "powf($0, $1)"], "F32.atan2": [2, "atan2f($0, $1)"], "F32.mod": [2, "fmodf($0, $1)"],
  "F32.is_eq": [2, "($0 == $1)"], "F32.is_ne": [2, "($0 != $1)"], "F32.is_lt": [2, "($0 < $1)"],
  "F32.is_le": [2, "($0 <= $1)"], "F32.is_gt": [2, "($0 > $1)"], "F32.is_ge": [2, "($0 >= $1)"],
  "Bool.pick": [3, "($0 ? $1 : $2)"],  // (type arg erased) pick(c, a, b): True -> a
  "Bool.not": [1, "(!$0)"], "Bool.and": [2, "($0 && $1)"], "Bool.or": [2, "($0 || $1)"],
};

function f32lit(bits) {
  const f = new Float32Array(new Uint32Array([bits >>> 0]).buffer)[0];
  if (Number.isNaN(f) || !Number.isFinite(f)) return `__uint_as_float(0x${(bits >>> 0).toString(16)}u)`;
  let s = f.toPrecision(9);
  if (!/[.e]/.test(s)) s += ".0";
  return s + "f";
}

// ------------------------------------------------------------------------------------------ types
// returns the C type of a (lowered) type term; registers records as structs
function mkTypes(book) {
  const structs = new Map();    // adt name -> {name, fields:[{n, T}]}
  const order = [];
  function ctype(T, where) {
    if (T.$ === "Ref" || T.$ === "ADT") {
      const k = T.k;
      if (k === "U32" || k === "Nat") return "uint32_t";
      if (k === "F32") return "float";
      if (k === "Bool") return "bool";
      const adt = book.tlds[k];
      if (adt && adt.$ === "ADT") {
        if (adt.c.length !== 1) unsup(`type ${k} has ${adt.c.length} constructors (only Bool/Nat and single-constructor records)`, where);
        if (!structs.has(k)) {
          const ctr = adt.c[0];
          const fs = fieldTypes(ctr, where).map((ft, i) => ({ n: `f${i}`, T: ctype(ft, where) }));
          structs.set(k, { name: cname(k), ctr: ctr.k, fields: fs });
          order.push(k);
        }
        return cname(k);
      }
    }
    return unsup(`type ${Bend.term_show(T)}`, where);
  }
  // constructor field types: the constructor's type is @f0:A0 -> ... -> ADT
  function fieldTypes(ctr, where) {
    const out = [];
    let T = Bend.term_lower(ctr.T);
    while (T.$ === "All") { out.push(T.A); T = T.B; }
    return out;
  }
  return { ctype, structs, order, fieldTypes };
}

const cname = (k) => "bg_" + k.replace(/[^A-Za-z0-9_]/g, "_");

// the (q, name, type) chain of a def's type
function sig(def) {
  const ps = [];
  let T = Bend.term_lower(def.T);
  while (T.$ === "All") { ps.push({ q: T.q.$, k: T.k, A: T.A }); T = T.B; }
  return { ps, ret: T };
}

// ------------------------------------------------------------------------------------- emitter
export function emit(book, entry) {
  const T = mkTypes(book);
  const defs = new Map();   // compiled defs, in dependency order
  const compiling = new Set();

  function collect(name, where) {
    if (defs.has(name) || PRIM[name]) return;
    const d = book.tlds[name];
    if (!d || d.$ !== "Def") unsup(`reference to ${name}`, where);
    if (d.v === null) unsup(`primitive ${name} has no native mapping`, where);
    if (compiling.has(name)) return;
    compiling.add(name);
    defs.set(name, genDef(name, d));   // callees are collected while generating
  }

  function genDef(name, d) {
    const { ps, ret } = sig(d);
    const keep = ps.filter((p) => p.q !== "None");
    const params = keep.map((p, i) => `${T.ctype(p.A, name)} a${i}`);
    const retC = T.ctype(ret, name);
    let body = "";
    const env = [];
    let rest = ps.map((p, i) => ({ ...p, c: p.q === "None" ? null : `a${keep.indexOf(p)}` }));
    const self = { name, keep };
    body = stmts(Bend.term_lower(d.v), env, rest, self, "  ");
    const cn = cname(name);
    return `__host__ __device__ __forceinline__ ${retC} ${cn}(${params.join(", ")}) {\n  for (;;) {\n${body}  }\n}\n`;
  }

  // a term in "function position": consume Lam / Mat against the remaining params, then the body
  function stmts(t, env, rest, self, ind) {
    const where = self.name;
    switch (t.$) {
      case "Lam": {
        const p = rest.shift();
        if (!p) unsup("lambda beyond the declared parameters (a closure)", where);
        env[t.i] = p.c === null ? { erased: true } : p.c;
        return stmts(t.f, env, rest, self, ind);
      }
      case "Mat": {
        const p = rest.shift();
        if (!p || p.c === null) unsup("match on an erased value", where);
        return genMat(t, p, env, rest, self, ind);
      }
      case "Let": {
        let out = "";
        for (let j = 0; j < t.k.length; j++) {
          const v = `v${t.i[j]}_${self.name.length}`;
          out += `${ind}auto ${v} = ${expr(t.v[j], env, where)};\n`;
          env[t.i[j]] = v;
        }
        return out + stmts(t.f, env, rest, self, ind);
      }
      default:
        return ret(t, env, rest, self, ind);
    }
  }

  function genMat(t, p, env, rest, self, ind) {
    const where = self.name;
    const A = p.A;
    const k = A.k;
    // collect the handlers (Mat chain ends with Efq)
    const hs = [];
    let m = t;
    while (m.$ === "Mat") { hs.push([m.k, m.h]); m = m.m; }
    const branch = (h, bindFields, ind2) => {
      const env2 = env.slice();
      const rest2 = rest.map((r) => ({ ...r }));
      let body = h;
      let pre = "";
      for (const fx of bindFields) {
        if (body.$ !== "Lam") unsup("pattern handler without a binder", where);
        pre += `${ind2}${fx.decl(body.i)}\n`;
        env2[body.i] = fx.ref;
        body = body.f;
      }
      return pre + stmts(body, env2, rest2, self, ind2);
    };
    if (k === "Bool") {
      const f = hs.find((x) => x[0] === "False"), tr = hs.find((x) => x[0] === "True");
      return `${ind}if (${p.c}) {\n${branch(tr[1], [], ind + "  ")}${ind}} else {\n${branch(f[1], [], ind + "  ")}${ind}}\n`;
    }
    if (k === "Nat") {
      const z = hs.find((x) => x[0] === "Zero"), s = hs.find((x) => x[0] === "Succ");
      const pv = `p${self.name.length}_${hs.length}_${ind.length}`;
      return `${ind}if (${p.c} == 0u) {\n${branch(z[1], [], ind + "  ")}${ind}} else {\n`
        + branch(s[1], [{ decl: (i) => `uint32_t ${pv}_${i} = ${p.c} - 1u;`, ref: `${pv}_${i0(s[1])}` }], ind + "  ") + `${ind}}\n`;
    }
    const adt = book.tlds[k];
    if (adt && adt.$ === "ADT" && adt.c.length === 1) {
      T.ctype(A, where);
      const ctr = adt.c[0];
      const fields = [];
      for (let i = 0; i < ctr.n; i++) {
        const nm = `${p.c}.f${i}`;
        fields.push({ decl: (b) => `auto fld${b}_${ind.length} = ${nm};`, ref: null });
      }
      // bind each field to a fresh local named after the binder index
      const binders = [];
      let h = hs[0][1];
      for (let i = 0; i < ctr.n; i++) { binders.push(h.i); h = h.f; }
      const env2 = env.slice();
      let pre = "";
      h = hs[0][1];
      for (let i = 0; i < ctr.n; i++) {
        const v = `r${h.i}_${ind.length}_${self.name.length}`;
        pre += `${ind}auto ${v} = ${p.c}.f${i};\n`;
        env2[h.i] = v;
        h = h.f;
      }
      return pre + stmts(h, env2, rest.map((r) => ({ ...r })), self, ind);
    }
    return unsup(`match on type ${k}`, where);
  }
  const i0 = (h) => h.i;

  // body in value position of a function: a tail self call becomes `continue`
  function ret(t, env, rest, self, ind) {
    const where = self.name;
    const { head, args } = spine(t);
    if (head.$ === "Ref" && head.k === self.name && args.length === self.keep.length + (sig(book.tlds[self.name]).ps.length - self.keep.length)) {
      const sg = sig(book.tlds[self.name]);
      const vals = [];
      sg.ps.forEach((p, i) => { if (p.q !== "None") vals.push(expr(args[i], env, where)); });
      let out = "";
      vals.forEach((v, i) => { out += `${ind}auto n${i} = ${v};\n`; });
      vals.forEach((_, i) => { out += `${ind}a${i} = n${i};\n`; });
      return out + `${ind}continue;\n`;
    }
    return `${ind}return ${expr(t, env, where)};\n`;
  }

  function spine(t) {
    const args = [];
    while (t.$ === "App") { args.unshift(t.x); t = t.f; }
    return { head: t, args };
  }

  function expr(t, env, where) {
    switch (t.$) {
      case "Var": {
        const v = env[t.i];
        if (v === undefined || (typeof v === "object")) unsup(`variable ${t.k} (erased or out of scope)`, where);
        return v;
      }
      case "Lit":
        if (t.k === "F32") return f32lit(t.v);
        if (t.k === "U32" || t.k === "Nat") return `${t.v >>> 0}u`;
        return unsup(`literal ${t.k}`, where);
      case "Ctr": {
        if (t.k === "True") return "true";
        if (t.k === "False") return "false";
        const c = book.ctrs[t.k];
        const adtName = Bend.book_fam(book, t.k);
        T.ctype({ $: "Ref", k: adtName }, where);
        return `(${cname(adtName)}){${t.x.map((x) => expr(x, env, where)).join(", ")}}`;
      }
      case "Ref": return call(t.k, [], env, where);
      case "App": {
        const { head, args } = spine(t);
        if (head.$ !== "Ref") unsup("application of a non-global function (a closure)", where);
        return call(head.k, args, env, where);
      }
      case "Ann": return expr(t.x, env, where);   // {x : T}: the annotation is only for the checker
      case "Mat": return unsup("match in non-tail position", where);
      default: return unsup(`term ${t.$}`, where);
    }
  }

  function call(name, args, env, where) {
    const d = book.tlds[name];
    // drop erased (type-level) arguments according to the callee's type
    let kept = args;
    if (d && d.$ === "Def") {
      const { ps } = sig(d);
      kept = args.filter((_, i) => ps[i] && ps[i].q !== "None");
    }
    if (PRIM[name]) {
      const [ar, tpl] = PRIM[name];
      if (kept.length !== ar) unsup(`partial application of ${name}`, where);
      const es = kept.map((a) => expr(a, env, where));
      return tpl.replace(/\$(\d)/g, (_, n) => es[+n]);
    }
    collect(name, where);
    return `${cname(name)}(${kept.map((a) => expr(a, env, where)).join(", ")})`;
  }

  collect(entry, "<entry>");
  const ed = book.tlds[entry];
  const esig = sig(ed);
  if (esig.ps.length !== 1 || esig.ps[0].A.k !== "U32") unsup("entry must be (i: U32) -> value", entry);
  const retT = T.ctype(esig.ret, entry);
  // the C source
  let out = `// generated by bendgpu/emit_cuda.mjs from the Bend def \`${entry}\`\n#include <cstdio>\n#include <cstdint>\n#include <cstdlib>\n#include <cmath>\n#include <vector>\n#include <cuda_runtime.h>\n\n`;
  for (const k of T.order) {
    const s = T.structs.get(k);
    out += `struct ${s.name} { ${s.fields.map((f) => `${f.T} ${f.n};`).join(" ")} };\n`;
  }
  out += "\n" + [...defs.values()].join("\n");
  out += kernelHost(entry, retT, esig.ret, T);
  return out;
}

function kernelHost(entry, retT, retTerm, T) {
  const cn = cname(entry);
  const isStruct = T.structs.has(retTerm.k);
  const show = isStruct
    ? T.structs.get(retTerm.k).fields.map((f, i) => `printf(" %.7g", (double)v.f${i});`).join(" ")
    : `printf(" %.7g", (double)v);`;
  const acc = isStruct ? `T.f0` : `v`;
  return `
__global__ void k_${cn}(${retT}* out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = ${cn}(i);
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\\n", cudaGetErrorString(e)); return 1; } } while (0)
int main(int argc, char** argv) {
  uint32_t n = argc > 1 ? (uint32_t)atoll(argv[1]) : 1024;
  ${retT}* d; CK(cudaMalloc(&d, (size_t)n * sizeof(${retT})));
  int blk = 256, grd = (int)((n + blk - 1) / blk);
  k_${cn}<<<grd, blk>>>(d, n); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int reps = 5; cudaEventRecord(e0);
  for (int r = 0; r < reps; r++) k_${cn}<<<grd, blk>>>(d, n);
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1);
  std::vector<${retT}> h(n); CK(cudaMemcpy(h.data(), d, (size_t)n * sizeof(${retT}), cudaMemcpyDeviceToHost));
  double sum = 0;
  for (uint32_t i = 0; i < n; i++) { ${retT} v = h[i]; sum += (double)${isStruct ? "v.f0" : "v"}; }
  const uint32_t probe[] = {0, 1, 2, 3, 1000, 1500, 2000, 2047, 3000, 4095};
  for (uint32_t pi = 0; pi < 10; pi++) { uint32_t i = probe[pi]; if (i >= n) continue; ${retT} v = h[i]; printf("kern(%u) =", i); ${show} printf("\\n"); }
  printf("sum %.9g   %.4f ms per launch (n=%u)\\n", sum, ms / reps, n);
  return 0;
}
`;
}

// ------------------------------------------------------------------------------------------ CLI
if (process.argv[1] && process.argv[1].endsWith("emit_cuda.mjs")) {
  const [file, entry = "kern"] = process.argv.slice(2);
  const base = await readBook(Bend.BASE_BEND);
  const book = await readBook(path.resolve(file), base);
  try {
    process.stdout.write(emit(book, entry));
  } catch (e) {
    if (e instanceof Unsupported) { process.stderr.write(e.message + "\n" + (process.env.BG_DEBUG ? e.stack : "") + "\n"); process.exit(2); }
    throw e;
  }
}

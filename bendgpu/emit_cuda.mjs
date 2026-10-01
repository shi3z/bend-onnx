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

// the read-only buffer of prelude/buf.bend (names carry the importing path as a prefix)
const isBufTy = (k) => /(^|\/)buf\.Buf$/.test(k) || k === "Buf";
const bufPrim = (name) => (/(^|\/)buf\.Buf\.get$/.test(name) ? [3, "$1[$2 & ((1u << $0) - 1u)]"] : null);
const mapArity = (name) => { const m = /(^|\/)buf\.Buf\.map([1-4])$/.exec(name); return m ? +m[2] : 0; };
const isArrTy = (k) => /(^|\/)arr\.Arr$/.test(k);
const arrPrim = (name) => (/(^|\/)arr\.Arr\.get$/.test(name) ? [4, "$1[$2 & ((1u << $0) - 1u)]"] : null);
const arrMap = (name) => { const m = /(^|\/)arr\.Arr\.map([1-4])$/.exec(name); return m ? +m[2] : 0; };
const cname = (k) => "bg_" + k.replace(/[^A-Za-z0-9_]/g, "_");

// ------------------------------------------------------------------------------------------ types
// A type is one of: scalar (U32, F32, Nat, Bool), Buf (a device pointer), a record (single-constructor,
// non-recursive ADT: a C struct held by value) or a heap ADT (several constructors or recursive: a tagged node in
// the thread's arena, held by pointer). Type arguments are substituted into the constructor types, so
// `List<&2, P>` is a list of P.
function mkTypes(book) {
  const reg = new Map();     // key -> {kind, name, adt, ctors:[{k, fields:[{T, c}]}]}
  const order = [];
  const busy = new Set();

  const key = (T) => {
    if (T.$ === "Ref") return T.k;
    if (T.$ === "ADT") return T.k + "<" + T.x.filter((x) => x.$ !== "Qua" && x.$ !== "Qnt").map(key).join(",") + ">";
    if (T.$ === "Qua") return "&" + T.q.$;
    return JSON.stringify(T);
  };
  const subst = (T, env) => {
    if (T.$ === "Var") return env[T.i] !== undefined ? env[T.i] : T;
    if (T.$ === "ADT") return { ...T, x: T.x.map((x) => subst(x, env)) };
    return T;
  };
  const mentions = (T, k) => (T.$ === "ADT" ? (T.k === k || T.x.some((x) => mentions(x, k))) : T.$ === "Ref" && T.k === k);

  function ctorFields(adt, ctr, args) {
    let Tm = Bend.term_lower(ctr.T);
    const env = {};
    for (let j = 0; j < args.length; j++) { if (Tm.$ !== "All") break; env[Tm.i] = args[j]; Tm = Tm.B; }
    const out = [];
    while (Tm.$ === "All") { out.push(subst(Tm.A, env)); Tm = Tm.B; }
    return out;
  }

  function cn(T) {
    const k = key(T);
    return "bg_" + k.replace(/[^A-Za-z0-9]+/g, "_").replace(/_+$/, "");
  }

  function ctype(T, where) {
    if (T.$ === "Ref" || T.$ === "ADT") {
      const k = T.k;
      if (k === "U32" || k === "Nat") return "uint32_t";
      if (k === "F32") return "float";
      if (k === "Bool") return "bool";
      if (isBufTy(k)) return "const float*";
      if (isArrTy(k)) return "const " + ctype(T.x[1], where) + "*";
      const adt = book.tlds[k];
      if (!adt || adt.$ !== "ADT") return unsup(`type ${Bend.term_show(T)}`, where);
      const kk = key(T);
      const name = cn(T);
      const info = reg.get(kk);
      if (info) return info.kind === "heap" ? name + "*" : name;
      const args = T.$ === "ADT" ? T.x : [];
      const recursive = adt.c.some((c) => ctorFields(adt, c, args).some((f) => mentions(f, k)));
      const kind = adt.c.length > 1 || recursive ? "heap" : "struct";
      if (busy.has(kk)) return kind === "heap" ? name + "*" : unsup(`recursive record ${k}`, where);
      busy.add(kk);
      const ctors = adt.c.map((c) => ({ k: c.k, fields: ctorFields(adt, c, args).map((f) => ({ T: f })) }));
      // compute field C types (registers inner types first)
      for (const c of ctors) c.fields.forEach((f, i) => { f.c = ctype(f.T, where); f.n = `f${i}`; });
      busy.delete(kk);
      reg.set(kk, { kind, name, adt: k, ctors });
      order.push(kk);
      return kind === "heap" ? name + "*" : name;
    }
    return unsup(`type ${Bend.term_show(T)}`, where);
  }
  const info = (T) => { ctype(T, "<type>"); return reg.get(key(T)); };
  return { ctype, info, reg, order, key };
}

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
  const defs = new Map();       // compiled defs, in dependency order
  const compiling = new Set();
  const mks = new Map();        // heap constructor helpers already requested: "type:ctor" -> name
  const mkText = [];
  let uid = 0;
  const fresh = (p) => `${p}${uid++}`;
  const strip = (x) => (x.$ === "Ann" ? strip(x.x) : x);
  // the C element type of a Buf / Arr parameter (null when the type is neither)
  const bufElem = (A) => (isBufTy(A.k || "") ? "float" : isArrTy(A.k || "") ? T.ctype(A.x[1], "<buffer>") : null);

  function collect(name, where) {
    if (defs.has(name) || PRIM[name] || bufPrim(name) || arrPrim(name)) return;
    const d = book.tlds[name];
    if (!d || d.$ !== "Def") unsup(`reference to ${name}`, where);
    if (d.v === null) unsup(`primitive ${name} has no native mapping`, where);
    if (compiling.has(name)) return;
    compiling.add(name);
    defs.set(name, genDef(name, d));   // callees are collected while generating
  }

  // ---- data layout helpers
  const famType = (ctrName) => Bend.book_fam(book, ctrName);
  const inst = (T0) => T.info(T0);
  const tagOf = (inf, k) => inf.ctors.findIndex((c) => c.k === k);
  // a heap ADT with exactly one nullary constructor represents it as nullptr
  const nullCtor = (inf) => {
    const z = inf.ctors.map((c, i) => [c, i]).filter(([c]) => c.fields.length === 0);
    return z.length === 1 ? z[0][1] : -1;
  };

  function mkHelper(inf, ci) {
    const k = `${inf.name}:${ci}`;
    if (mks.has(k)) return mks.get(k);
    const c = inf.ctors[ci];
    const fn = `mk_${inf.name}_${ci}`;
    mks.set(k, fn);
    const ps = c.fields.map((f, i) => `${f.c} x${i}`).join(", ");
    const sets = c.fields.map((f, i) => `p->u.c${ci}.${f.n} = x${i};`).join(" ");
    mkText.push(`__host__ __device__ __forceinline__ ${inf.name}* ${fn}(bg_arena* ar${c.fields.length ? ", " : ""}${ps}) {\n  ${inf.name}* p = (${inf.name}*)bg_alloc(ar, sizeof(${inf.name})); p->tag = ${ci}; ${sets} return p;\n}\n`);
    return fn;
  }

  function genDef(name, d) {
    const { ps, ret } = sig(d);
    const keep = ps.filter((p) => p.q !== "None");
    const params = ["bg_arena* ar", ...keep.map((p, i) => `${T.ctype(p.A, name)} a${i}`)];
    const retC = T.ctype(ret, name);
    let j = 0;
    const pend = ps.map((p) => ({ c: p.q === "None" ? null : `a${j++}`, A: p.A }));
    const self = { name, ps, keep, ret };
    const body = stmts(Bend.term_lower(d.v), [], pend, self, "  ");
    return `__host__ __device__ ${retC} ${cname(name)}(${params.join(", ")}) {\n  for (;;) {\n${body}  }\n}\n`;
  }

  // a term in "function position": Lam binds / Mat consumes the next pending value
  function stmts(t, env, pend, self, ind) {
    const where = self.name;
    switch (t.$) {
      case "Lam": {
        const e = pend.shift();
        if (!e) unsup("lambda beyond the available values (a closure)", where);
        env[t.i] = e.c === null ? { erased: true } : e.c;
        return stmts(t.f, env, pend, self, ind);
      }
      case "Mat": {
        const e = pend.shift();
        if (!e || e.c === null) unsup("match on an erased value", where);
        return genMat(t, e, env, pend, self, ind);
      }
      case "Let": {
        let out = "";
        for (let j = 0; j < t.k.length; j++) {
          const v = fresh("v");
          const val = t.v[j];
          const exp = val.$ === "Ann" ? val.T : undefined;
          out += `${ind}auto ${v} = ${expr(val, env, where, exp)};\n`;
          env[t.i[j]] = v;
        }
        return out + stmts(t.f, env, pend, self, ind);
      }
      default:
        return ret(t, env, pend, self, ind);
    }
  }

  function genMat(t, e, env, pend, self, ind) {
    const where = self.name;
    const hs = [];
    let m = t;
    while (m.$ === "Mat") { hs.push([m.k, m.h]); m = m.m; }
    const A = e.A;
    const kn = A.k;
    const branch = (h, fields, ind2) => {
      let pre = "";
      const pend2 = [];
      for (const f of fields) {
        const v = fresh("f");
        pre += `${ind2}auto ${v} = ${f.expr};\n`;
        pend2.push({ c: v, A: f.T });
      }
      for (const r of pend) pend2.push({ ...r });
      return pre + stmts(h, env.slice(), pend2, self, ind2);
    };
    if (kn === "Bool") {
      const f = hs.find((x) => x[0] === "False"), tr = hs.find((x) => x[0] === "True");
      if (!f || !tr) unsup("incomplete match on Bool", where);
      return `${ind}if (${e.c}) {\n${branch(tr[1], [], ind + "  ")}${ind}} else {\n${branch(f[1], [], ind + "  ")}${ind}}\n`;
    }
    if (kn === "Nat") {
      const z = hs.find((x) => x[0] === "Zero"), s2 = hs.find((x) => x[0] === "Succ");
      if (!z || !s2) unsup("incomplete match on Nat", where);
      return `${ind}if (${e.c} == 0u) {\n${branch(z[1], [], ind + "  ")}${ind}} else {\n`
        + branch(s2[1], [{ expr: `${e.c} - 1u`, T: { $: "Ref", k: "Nat" } }], ind + "  ") + `${ind}}\n`;
    }
    const inf = inst(A);
    if (inf.kind === "struct") {
      const c = inf.ctors[0];
      const h = hs.find((x) => x[0] === c.k);
      if (!h) unsup(`match on ${kn} without its constructor`, where);
      return branch(h[1], c.fields.map((f) => ({ expr: `${e.c}.${f.n}`, T: f.T })), ind);
    }
    // heap ADT: switch on the tag
    const nc = nullCtor(inf);
    const tagExpr = nc >= 0 ? `(${e.c} ? ${e.c}->tag : ${nc}u)` : `${e.c}->tag`;
    let out = `${ind}switch (${tagExpr}) {\n`;
    for (const [ck, h] of hs) {
      const ci = tagOf(inf, ck);
      if (ci < 0) unsup(`constructor ${ck} is not of ${kn}`, where);
      const c = inf.ctors[ci];
      out += `${ind}  case ${ci}: {\n${branch(h, c.fields.map((f) => ({ expr: `${e.c}->u.c${ci}.${f.n}`, T: f.T })), ind + "    ")}${ind}  }\n`;
    }
    return out + `${ind}  default: __builtin_unreachable();\n${ind}}\n`;
  }

  // body in value position: a tail self call becomes `continue`
  function ret(t, env, pend, self, ind) {
    const where = self.name;
    const { head, args } = spine(t);
    if (head.$ === "Ref" && head.k === self.name && args.length === self.ps.length) {
      const vals = [];
      self.ps.forEach((p, i) => { if (p.q !== "None") vals.push(expr(args[i], env, where, p.A)); });
      let out = "";
      vals.forEach((v, i) => { out += `${ind}auto n${i} = ${v};\n`; });
      vals.forEach((_, i) => { out += `${ind}a${i} = n${i};\n`; });
      return out + `${ind}continue;\n`;
    }
    return `${ind}return ${expr(t, env, where, self.ret)};\n`;
  }

  function spine(t) {
    const args = [];
    while (t.$ === "App") { args.unshift(t.x); t = t.f; }
    return { head: t, args };
  }

  function expr(t, env, where, exp) {
    switch (t.$) {
      case "Var": {
        const v = env[t.i];
        if (v === undefined || typeof v === "object") unsup(`variable ${t.k} (erased or out of scope)`, where);
        return v;
      }
      case "Lit":
        if (t.k === "F32") return f32lit(t.v);
        if (t.k === "U32" || t.k === "Nat") return `${t.v >>> 0}u`;
        return unsup(`literal ${t.k}`, where);
      case "Ctr": {
        if (t.k === "True") return "true";
        if (t.k === "False") return "false";
        const fam = famType(t.k);
        // the type to build: the expected one when it is of this family, else the bare family (non-parametric only)
        let T0 = exp && exp.k === fam ? exp : null;
        if (!T0) {
          const famAdt = book.tlds[fam];
          const tl = Bend.term_lower(famAdt.c[0].T);
          if (tl.$ === "All" && (tl.q.$ === "None")) unsup(`cannot infer the type arguments of ${t.k}; annotate with {x : T}`, where);
          T0 = { $: "Ref", k: fam };
        }
        const inf = inst(T0);
        const ci = tagOf(inf, t.k);
        const c = inf.ctors[ci];
        const xs = t.x.map((x, i) => expr(x, env, where, c.fields[i].T));
        if (inf.kind === "struct") return `(${inf.name}){${xs.join(", ")}}`;
        if (xs.length === 0 && nullCtor(inf) === ci) return "nullptr";
        return `${mkHelper(inf, ci)}(ar${xs.length ? ", " : ""}${xs.join(", ")})`;
      }
      case "Ref": return call(t.k, [], env, where);
      case "App": {
        const { head, args } = spine(t);
        if (head.$ !== "Ref") unsup("application of a non-global function (a closure)", where);
        return call(head.k, args, env, where);
      }
      case "Ann": return expr(t.x, env, where, t.T);   // {x : T}: the annotation also gives constructors their type
      case "Mat": return unsup("match in non-tail position", where);
      default: return unsup(`term ${t.$}`, where);
    }
  }

  function call(name, args, env, where) {
    const d = book.tlds[name];
    let kept = args, kt = [];
    if (d && d.$ === "Def") {
      const { ps } = sig(d);
      kept = []; kt = [];
      args.forEach((a, i) => { if (ps[i] && ps[i].q !== "None") { kept.push(a); kt.push(ps[i].A); } });
    }
    const prim = PRIM[name] || bufPrim(name) || arrPrim(name);
    if (prim) {
      const [ar, tpl] = prim;
      if (kept.length !== ar) unsup(`partial application of ${name}`, where);
      const es = kept.map((a, i) => expr(a, env, where, kt[i]));
      return tpl.replace(/\$(\d)/g, (_, n) => es[+n]);
    }
    collect(name, where);
    return `${cname(name)}(ar${kept.length ? ", " : ""}${kept.map((a, i) => expr(a, env, where, kt[i])).join(", ")})`;
  }

  // ------------------------------------------------------------------------ whole programs
  const ed0 = book.tlds[entry];
  if (!ed0 || ed0.$ !== "Def") unsup(`no def named ${entry}`, "<entry>");
  const esig0 = sig(ed0);
  const launchers = isBufTy(esig0.ret.k || "") || isArrTy(esig0.ret.k || "") ? pipeline() : mapEntry();

  function structsText() {
    let out = "";
    for (const kk of T.order) out += `struct ${T.reg.get(kk).name};\n`;
    for (const kk of T.order) {
      const inf = T.reg.get(kk);
      if (inf.kind === "struct") {
        out += `struct ${inf.name} { ${inf.ctors[0].fields.map((f) => `${f.c} ${f.n};`).join(" ")} };\n`;
      } else {
        const us = inf.ctors.map((c, i) => `struct { ${c.fields.map((f) => `${f.c} ${f.n};`).join(" ")} } c${i};`).join(" ");
        out += `struct ${inf.name} { uint32_t tag; union { ${us} } u; };\n`;
      }
    }
    return out;
  }

  const prologue = `// generated by bendgpu/emit_cuda.mjs from the Bend def \`${entry}\`
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#ifndef BG_ARENA_KB
#define BG_ARENA_KB 64      // per-thread arena for heap values (lists, trees): bump allocated, reset for each item
#endif
#ifndef BG_STACK_KB
#define BG_STACK_KB 16      // per-thread call stack (non-tail recursion)
#endif
#define BG_ARENA_BYTES ((size_t)BG_ARENA_KB * 1024)
struct bg_arena { char* cur; };
static __host__ __device__ __forceinline__ void* bg_alloc(bg_arena* ar, size_t n) { void* p = ar->cur; ar->cur += (n + 15) & ~(size_t)15; return p; }

`;
  return prologue + structsText() + "\n" + mkText.join("\n") + "\n" + [...defs.values()].join("\n") + launchers;

  // ---------------------------------------------------------------------- entry: parallel map
  function mapEntry() {
    collect(entry, "<entry>");
    const esig = sig(ed0);
    if (esig.ps.length < 1 || esig.ps[0].A.k !== "U32") unsup("entry must be (i: U32, buffers...) -> value", entry);
    const bufs = esig.ps.slice(1).map((b) => {
      const elem = bufElem(b.A);
      if (elem === null) unsup(`entry parameter ${b.k} is not a Buf / Arr (scalars: not yet)`, entry);
      return { k: b.k, elem };
    });
    for (const b of bufs) { collect(`${b.k}_init`, entry); collect(`${b.k}_depth`, entry); }
    const retT = T.ctype(esig.ret, entry);
    const scalar = ["U32", "F32", "Nat", "Bool"].includes(esig.ret.k);
    const retInf = scalar ? null : inst(esig.ret);
    if (retInf && retInf.kind !== "struct") unsup("a kernel result must be a scalar or a record", entry);
    return kernelHost(entry, retT, retInf, bufs);
  }

  // ---------------------------------------------------------------------- entry: pipeline of maps
  function pipeline() {
    const where = entry;
    const ps = esig0.ps.map((b) => {
      const elem = bufElem(b.A);
      if (elem === null) unsup(`pipeline parameter ${b.k} is not a Buf / Arr`, where);
      return { k: b.k, elem };
    });
    for (const b of ps) { collect(`${b.k}_init`, where); collect(`${b.k}_depth`, where); }
    const outElem = bufElem(esig0.ret);
    const env = [], elems = [];             // device buffer name of each variable, and its C element type
    const setup = [], launch = [], kernels = new Map();
    let tmp = 0;
    let t = Bend.term_lower(ed0.v);
    let idx = 0;
    while (t.$ === "Lam") { env[t.i] = `d_${ps[idx].k}`; elems[t.i] = ps[idx].elem; idx++; t = t.f; }
    const mapCall = (v) => {
      const { head, args } = spine(strip(v));
      const nb = head.$ === "Ref" ? mapArity(head.k) : 0, na = head.$ === "Ref" ? arrMap(head.k) : 0;
      const n = nb || na;
      if (!n) unsup("pipeline stages must be Buf.map1..4 / Arr.map1..4 calls", where);
      const lead = na ? n + 1 : 0;           // Arr.mapN: n input element types and the output type come first
      const f = strip(args[lead]);
      if (f.$ !== "Ref") unsup("the stage function of a map must be a top-level def", where);
      if (args.length !== lead + n + 3) unsup(`map${n} called with ${args.length} arguments`, where);
      const fs = sig(book.tlds[f.k]);
      if (fs.ps.length !== n + 1 || fs.ps[0].A.k !== "U32" || fs.ps.slice(1).some((p) => bufElem(p.A) === null))
        unsup(`stage ${f.k} must be (i: U32, ${n} arrays) -> value`, where);
      const outC = na ? T.ctype(args[n], where) : "float";
      if (!na && fs.ret.k !== "F32") unsup(`stage ${f.k} of a Buf.map must return F32`, where);
      if (T.ctype(fs.ret, where) !== outC) unsup(`stage ${f.k} returns ${T.ctype(fs.ret, where)}, the map declares ${outC}`, where);
      collect(f.k, where);
      const off = strip(args[lead + n + 2]);
      if (!(off.$ === "Lit" && off.v === 0)) unsup("map offsets other than 0", where);
      const dRaw = strip(args[lead + 1]);
      const dExpr = dRaw.$ === "Lit" ? `${dRaw.v >>> 0}u` : dRaw.$ === "Ref" ? (collect(dRaw.k, where), `${cname(dRaw.k)}(nullptr)`)
        : unsup("the depth of a map must be a literal or a constant def", where);
      const bufsOf = args.slice(lead + 2, lead + 2 + n).map((a) => expr(strip(a), env, where, undefined));
      const inElems = args.slice(lead + 2, lead + 2 + n).map((a) => { const e = strip(a); return e.$ === "Var" ? elems[e.i] : "float"; });
      const kk = f.k;
      if (!kernels.has(kk)) kernels.set(kk, { n, outC, inElems: fs.ps.slice(1).map((p) => bufElem(p.A)) });
      return { f: f.k, n, dExpr, bufs: bufsOf, outC };
    };
    const emitMap = (m, outName) => {
      setup.push(`uint32_t n_${outName} = 1u << (${m.dExpr}); ${m.outC}* ${outName}; CK(cudaMalloc(&${outName}, (size_t)n_${outName} * sizeof(${m.outC})));`);
      launch.push(`km_${cname(m.f)}<<<(bg_res(n_${outName}) + 255) / 256, 256>>>(${outName}, n_${outName}, arena${m.bufs.map((b) => ", " + b).join("")});`);
    };
    let outName = null, outC = outElem;
    while (true) {
      if (t.$ === "Let") {
        for (let j = 0; j < t.k.length; j++) {
          const m = mapCall(t.v[j]);
          const name = `t${tmp++}`;
          emitMap(m, name);
          env[t.i[j]] = name;
          elems[t.i[j]] = m.outC;
        }
        t = t.f;
        continue;
      }
      const e = strip(t);
      if (e.$ === "Var") { outName = env[e.i]; outC = elems[e.i]; break; }
      const m = mapCall(e);
      outName = `t${tmp++}`;
      emitMap(m, outName);
      outC = m.outC;
      break;
    }
    const outInf = ["float", "uint32_t", "bool"].includes(outC) ? null : [...T.reg.values()].find((i) => i.name === outC);
    let out = commonKernels();
    for (const [f, kd] of kernels) {
      const bs = kd.inElems.map((e, i) => `, const ${e}* b${i}`).join("");
      const as = kd.inElems.map((_, i) => `, b${i}`).join("");
      out += `__global__ void km_${cname(f)}(${kd.outC}* out, uint32_t n, char* arena${bs}) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t i = tid; i < n; i += stride) { bg_arena ar; ar.cur = arena + (size_t)tid * BG_ARENA_BYTES; out[i] = ${cname(f)}(&ar, i${as}); }
}\n`;
    }
    for (const b of ps) {
      out += `__global__ void init_${b.k}(${b.elem}* p, uint32_t n) { uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; if (j < n) p[j] = ${cname(b.k + "_init")}(nullptr, j); }\n`;
    }
    const inSetup = ps.map((b) => `
  uint32_t n_${b.k} = 1u << ${cname(b.k + "_depth")}(nullptr); ${b.elem}* d_${b.k}; CK(cudaMalloc(&d_${b.k}, (size_t)n_${b.k} * sizeof(${b.elem})));
  init_${b.k}<<<(n_${b.k} + 255) / 256, 256>>>(d_${b.k}, n_${b.k});`).join("");
    const first = outInf ? "h[i].f0" : "h[i]";
    const showProbe = outInf
      ? outInf.ctors[0].fields.map((f, i) => `printf(" %.7g", (double)h[i].f${i});`).join(" ")
      : `printf(" %.7g", (double)h[i]);`;
    out += `
int main() {
  CK(cudaDeviceSetLimit(cudaLimitStackSize, (size_t)BG_STACK_KB * 1024));
  char* arena; CK(cudaMalloc(&arena, (size_t)BG_RESIDENT * BG_ARENA_BYTES));${inSetup}
  ${setup.join("\n  ")}
  CK(cudaDeviceSynchronize());
  ${launch.join("\n  ")}
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int reps = 20; cudaEventRecord(e0);
  for (int r = 0; r < reps; r++) {
    ${launch.join("\n    ")}
  }
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1);
  uint32_t n = n_${outName};
  std::vector<${outC}> h(n); CK(cudaMemcpy(h.data(), ${outName}, (size_t)n * sizeof(${outC}), cudaMemcpyDeviceToHost));
  double sum = 0; for (uint32_t i = 0; i < n; i++) sum += (double)${first};
  const uint32_t probe[] = {0, 1, 2, 3, 1000, 1500, 2000, 2047, 3000, 4095};
  for (uint32_t pi = 0; pi < 10; pi++) { uint32_t i = probe[pi]; if (i >= n) continue; printf("kern(%u) =", i); ${showProbe} printf("\\n"); }
  printf("sum %.9g   %.4f ms per pipeline run (%zu launches, out n=%u)\\n", sum, ms / reps, (size_t)${launch.length}, n);
  return 0;
}
`;
    return out;
  }

  function commonKernels() {
    return `
#define BG_RESIDENT 16384
static inline uint32_t bg_res(uint32_t n) { return n < BG_RESIDENT ? n : BG_RESIDENT; }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\\n", cudaGetErrorString(e)); return 1; } } while (0)
`;
  }

  function kernelHost(entry, retT, retInf, bufs) {
    const cn = cname(entry);
    const isStruct = !!retInf;
    const show = isStruct
      ? retInf.ctors[0].fields.map((f, i) => `printf(" %.7g", (double)v.f${i});`).join(" ")
      : `printf(" %.7g", (double)v);`;
    const bp = bufs.map((b) => `, const ${b.elem}* b_${b.k}`).join("");
    const ba = bufs.map((b) => `, d_${b.k}`).join("");
    const inits = bufs.map((b) => `
__global__ void init_${b.k}(${b.elem}* p, uint32_t n) { uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; if (j < n) p[j] = ${cname(b.k + "_init")}(nullptr, j); }`).join("");
    const alloc = bufs.map((b) => `
  uint32_t n_${b.k} = 1u << ${cname(b.k + "_depth")}(nullptr); ${b.elem}* d_${b.k}; CK(cudaMalloc(&d_${b.k}, (size_t)n_${b.k} * sizeof(${b.elem})));
  init_${b.k}<<<(n_${b.k} + 255) / 256, 256>>>(d_${b.k}, n_${b.k});`).join("");
    return `${commonKernels()}${inits}
__global__ void k_${cn}(${retT}* out, uint32_t n, char* arena${bp}) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t i = tid; i < n; i += stride) {
    bg_arena ar; ar.cur = arena + (size_t)tid * BG_ARENA_BYTES;
    out[i] = ${cn}(&ar, i${bufs.map((b) => `, b_${b.k}`).join("")});
  }
}

int main(int argc, char** argv) {
  uint32_t n = argc > 1 ? (uint32_t)atoll(argv[1]) : 1024;
  CK(cudaDeviceSetLimit(cudaLimitStackSize, (size_t)BG_STACK_KB * 1024));
  char* arena; CK(cudaMalloc(&arena, (size_t)bg_res(n) * BG_ARENA_BYTES));
  ${retT}* d; CK(cudaMalloc(&d, (size_t)n * sizeof(${retT})));${alloc}
  CK(cudaDeviceSynchronize());
  int blk = 256, grd = (int)((bg_res(n) + blk - 1) / blk);
  k_${cn}<<<grd, blk>>>(d, n, arena${ba}); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int reps = 5; cudaEventRecord(e0);
  for (int r = 0; r < reps; r++) k_${cn}<<<grd, blk>>>(d, n, arena${ba});
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

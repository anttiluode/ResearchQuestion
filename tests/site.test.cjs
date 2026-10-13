const {test}=require('node:test');
const assert=require('node:assert/strict');
const {CORE_DIM,NUISANCE_DIM,MAX_STEPS,READS,makeFull,makeCode,stepFull,stepCode,query,sameCurrentAnswers}=require('../site.js');

const ACTIONS=[0,1,3,2,1,0,3,2,0,2,1,3];
function run(leak,keepBridge,initial=1.3){
  let full=makeFull(initial),code=makeCode(full,keepBridge);
  const diffs=[];
  for(const a of ACTIONS){
    full=stepFull(full,a,leak);
    code=stepCode(code,a,leak,keepBridge);
    diffs.push(...READS.map((_,q)=>Math.abs(query(full,q)-query(code,q))));
  }
  return Math.max(...diffs);
}

test('website example really has 32 changing coordinates',()=>{
  const x=makeFull();
  assert.equal(x.s.length,CORE_DIM);
  assert.equal(x.n.length,NUISANCE_DIM);
  assert.equal(CORE_DIM+1+NUISANCE_DIM,32);
  assert.equal(MAX_STEPS,ACTIONS.length);
  const next=stepFull(x,0,.28);
  assert.notDeepEqual(next.n,x.n);
});
test('two histories with opposite bridge have all six identical present answers',()=>{
  assert.ok(sameCurrentAnswers(makeFull(1.3),makeFull(-1.3)));
});
test('the same action reveals the hidden bridge in a leaky world',()=>{
  const x=stepFull(makeFull(1.3),0,.28);
  const y=stepFull(makeFull(-1.3),0,.28);
  assert.ok(Math.max(...READS.map((_,q)=>Math.abs(query(x,q)-query(y,q))))>.1);
});
test('four-number code remains closed in world without bridge-to-core feedback',()=>{
  assert.ok(run(0,false)<1e-12);
});
test('four-number code fails after permitted actions when bridge feeds back',()=>{
  assert.ok(run(.28,false)>.05);
});
test('five-number code remains closed with causal bridge retained',()=>{
  assert.ok(run(.28,true)<1e-12);
});
test('nuisance coordinates can be changed without changing future answers',()=>{
  let x=makeFull(1.3),y=makeFull(1.3);
  y.n=y.n.map((v,i)=>v+Math.sin(i)*1.7);
  for(const a of ACTIONS){
    x=stepFull(x,a,.28);y=stepFull(y,a,.28);
    for(let q=0;q<READS.length;q++) assert.ok(Math.abs(query(x,q)-query(y,q))<1e-12);
  }
});
test('changing which query is asked does not require reconstructing nuisance state',()=>{
  const state=makeCode(makeFull(.8),true);
  const all=READS.map((_,q)=>query(state,q));
  assert.equal(all.length,6);
  assert.ok(new Set(all).size>3);
});
test('unknown actions and queries are refused',()=>{
  assert.throws(()=>stepFull(makeFull(),-1,.28),RangeError);
  assert.throws(()=>query(makeFull(),999),RangeError);
});

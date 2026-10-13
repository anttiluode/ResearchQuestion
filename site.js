/* ResearchQuestion website — illustrative dynamics only.
   This is an independently chosen deterministic 32-variable example of the
   same mechanism class as the frozen eight-seed Python experiment.
   Never present website values as the original Python receipt.
*/
(function (root) {
  'use strict';

  const CORE_DIM = 4;
  const NUISANCE_DIM = 27;
  const MAX_STEPS = 12;
  const A = [
    [[.56,.07,-.03,0],[.02,.51,.06,-.03],[-.02,.02,.57,.04],[.01,-.03,.07,.52]],
    [[.48,-.10,.06,.03],[.05,.59,-.01,.04],[.07,.01,.49,-.07],[-.03,.04,.05,.60]],
    [[.51,.02,.09,-.04],[.04,.53,-.06,.06],[-.08,.08,.54,.02],[.06,-.02,.02,.51]],
    [[.58,-.06,.02,.08],[-.05,.52,.06,.02],[.02,.07,.52,-.04],[.03,-.06,-.01,.56]]
  ];
  const COUPLING = [[1,.36,-.22,.18],[-.7,.78,.23,-.34],[.58,-.35,.96,.18],[-.28,.54,-.52,1.06]];
  const RHO = [.74,.67,.81,.58];
  const READS = [
    [.9,.12,-.15,.22],
    [.18,.95,.22,-.09],
    [-.23,.16,.9,.13],
    [.3,-.2,.27,.88],
    [.64,-.47,.38,-.3],
    [-.18,.41,.65,.52]
  ];
  const INITIAL_CORE = [.38,-.22,.35,-.14];
  const DEFAULT_BRIDGE = 1.30;

  function dot(a,b) { return a.reduce((sum,x,i)=>sum+x*b[i],0); }
  function stepCore(s,b,action,leak) {
    return A[action].map((row,i)=>Math.tanh(dot(row,s)+.13*Math.sin(s[i])+leak*COUPLING[action][i]*b));
  }
  function makeFull(bridge=DEFAULT_BRIDGE) {
    return {s:INITIAL_CORE.slice(),b:bridge,n:Array.from({length:NUISANCE_DIM},(_,i)=>.35*Math.sin(i*1.19+.3))};
  }
  function makeCode(full,retainBridge) {
    return {s:full.s.slice(),b:retainBridge?full.b:0};
  }
  function stepFull(full,action,leak) {
    checkAction(action);
    const nextS=stepCore(full.s,full.b,action,leak);
    const nextB=RHO[action]*full.b+.11*Math.tanh(full.s[0]);
    const nextN=full.n.map((v,i)=>Math.tanh((.43+.02*Math.sin(i+action))*v+.06*Math.sin(i*.45+action)*full.b));
    return {s:nextS,b:nextB,n:nextN};
  }
  function stepCode(code,action,leak,retainBridge) {
    checkAction(action);
    return {
      s:stepCore(code.s,retainBridge?code.b:0,action,leak),
      b:retainBridge?RHO[action]*code.b+.11*Math.tanh(code.s[0]):0
    };
  }
  function checkAction(action) {
    if (!Number.isInteger(action)||action<0||action>=A.length) throw new RangeError('Unknown action');
  }
  function query(state,index) {
    if(!Number.isInteger(index)||index<0||index>=READS.length) throw new RangeError('Unknown query');
    return dot(READS[index],state.s);
  }
  function sameCurrentAnswers(left,right) {
    return READS.every((_,i)=>Math.abs(query(left,i)-query(right,i))<1e-12);
  }

  const maths={CORE_DIM,NUISANCE_DIM,MAX_STEPS,READS,DEFAULT_BRIDGE,makeFull,makeCode,stepFull,stepCode,query,sameCurrentAnswers};
  if(typeof module!=='undefined'&&module.exports) module.exports=maths;
  root.RQLabMath=maths;

  if (typeof document==='undefined') return;

  function get(id){return document.getElementById(id);}
  const state={world:'bridge',code:4,bridge:DEFAULT_BRIDGE,queryIndex:0,action:0,history:[]};
  const worldButtons=Array.from(document.querySelectorAll('[data-world]'));
  const codeButtons=Array.from(document.querySelectorAll('[data-code]'));
  const actionButtons=Array.from(document.querySelectorAll('[data-action]'));
  const bridgeSlider=get('bridge-value');
  const bridgeOut=get('bridge-readout');
  const select=get('query-select');
  const stepButton=get('step-button');
  const resetButton=get('reset-button');

  function currentLeak(){return state.world==='bridge'?.28:0;}
  function reset() {
    const full=makeFull(state.bridge);
    const code=makeCode(full,state.code===5);
    const twin=makeFull(-state.bridge);
    state.history=[{full,code,twin,action:null}];
    render();
  }
  function applyAction() {
    if(state.history.length>MAX_STEPS) return;
    const prev=state.history[state.history.length-1];
    const action=state.action;
    const leak=currentLeak();
    state.history.push({
      full:stepFull(prev.full,action,leak),
      code:stepCode(prev.code,action,leak,state.code===5),
      twin:stepFull(prev.twin,action,leak),
      action
    });
    render();
  }
  function updatePressed(buttons,key,value){
    buttons.forEach(el=>{
      const yes=String(el.dataset[key])===String(value);
      el.classList.toggle(key==='action'?'selected':'active',yes);
      el.setAttribute('aria-pressed',String(yes));
    });
  }
  function nformat(n,digits=4){
    const safe=Math.abs(n)<Math.pow(10,-digits)*.5?0:n;
    return (safe>=0?'+':'')+safe.toFixed(digits);
  }
  function getValues() {
    return state.history.map(h=>({
      trueValue:query(h.full,state.queryIndex),
      codeValue:query(h.code,state.queryIndex),
      twinValue:query(h.twin,state.queryIndex)
    }));
  }
  function drawChart(vals) {
    const xmin=45,xmax=730,ybase=108,scale=82;
    const xf=i=>xmin+(xmax-xmin)*i/MAX_STEPS;
    const yf=value=>Math.max(17,Math.min(202,ybase-value*scale));
    const draw=key=>vals.map((v,i)=>(i?'L':'M')+' '+xf(i).toFixed(2)+' '+yf(v[key]).toFixed(2)).join(' ');
    get('true-path').setAttribute('d',draw('trueValue'));
    get('code-path').setAttribute('d',draw('codeValue'));
    const labels=[];
    for(const val of [1,0,-1]) {
      labels.push('<line x1="45" x2="730" y1="'+yf(val)+'" y2="'+yf(val)+'" stroke="#273848" stroke-width="1" stroke-dasharray="2 7"></line>');
      labels.push('<text class="chart-axis-text" x="8" y="'+(yf(val)+4)+'">'+(val===1?'+1.0':val===-1?'−1.0':'0.0')+'</text>');
    }
    for(const step of [0,3,6,9,12]) labels.push('<text class="chart-axis-text" text-anchor="middle" x="'+xf(step)+'" y="216">'+step+'</text>');
    if(vals.length===1) {
      for(const [key,color] of [['trueValue','#67e4da'],['codeValue','#ffbb84']])
        labels.push('<circle cx="'+xf(0)+'" cy="'+yf(vals[0][key])+'" r="4" fill="'+color+'"></circle>');
    } else {
      const pos=vals.length-1;
      for(const [key,color] of [['trueValue','#67e4da'],['codeValue','#ffbb84']])
        labels.push('<circle cx="'+xf(pos)+'" cy="'+yf(vals[pos][key])+'" r="4" fill="'+color+'" stroke="#0b1420" stroke-width="2"></circle>');
    }
    get('chart-labels').innerHTML=labels.join('');
  }
  function render() {
    const values=getValues();
    const last=values[values.length-1];
    const step=values.length-1;
    const d=Math.abs(last.trueValue-last.codeValue);
    get('step-count').textContent='STEP '+String(step).padStart(2,'0')+' / '+MAX_STEPS;
    get('true-value').textContent=nformat(last.trueValue);
    get('code-value').textContent=nformat(last.codeValue);
    get('code-dim-label').textContent=state.code+' DIMENSIONS';
    get('difference-value').textContent=d.toFixed(6);
    get('true-orb').style.left=(50+Math.max(-42,Math.min(42,last.trueValue*35)))+'%';
    get('code-orb').style.left=(50+Math.max(-42,Math.min(42,last.codeValue*35)))+'%';
    const broken=d>1e-6;
    const pill=get('difference-state');
    pill.classList.toggle('broken',broken);
    pill.textContent=broken?'DIVERGENCE':'AGREEMENT';
    get('twin-now').textContent='0.000000';
    get('twin-future').textContent=Math.abs(last.trueValue-last.twinValue).toFixed(6);
    get('world-help').textContent=state.world==='bridge'
      ?'The bridge can influence the next core state.'
      :'The bridge cannot feed back into the core: four coordinates are enough.';
    get('code-help').textContent=state.code===4
      ?'The compact model discards the bridge residue.'
      :'The compact model retains the hidden bridge and updates it.';
    bridgeOut.textContent=nformat(state.bridge,2);
    bridgeSlider.setAttribute('aria-valuetext',nformat(state.bridge,2));
    stepButton.disabled=step>=MAX_STEPS;
    stepButton.innerHTML=step>=MAX_STEPS?'12 steps complete':'Apply action <span aria-hidden="true">→</span>';
    const message=get('lab-message');
    let text;
    if(step===0) text='Every present answer matches. The bridge is invisible until an interaction lets it influence the future.';
    else if(state.world==='closed') text='The bridge changes internally, but never feeds back into the core. Both codes preserve all permitted future answers.';
    else if(state.code===5) text='The fifth number is retained and updated. The compact model stays aligned with the full system.';
    else if(broken) text='A discarded residue has entered the future core. Matching today’s answers was not enough to predict the next response.';
    else text='The missing bridge remains causal, although this particular query currently shows little error. Switch queries or actions.';
    message.textContent=text;
    message.classList.toggle('warning',broken);
    drawChart(values);
    updatePressed(worldButtons,'world',state.world);
    updatePressed(codeButtons,'code',state.code);
    updatePressed(actionButtons,'action',state.action);
  }

  worldButtons.forEach(button=>button.addEventListener('click',()=>{
    state.world=button.dataset.world;reset();
  }));
  codeButtons.forEach(button=>button.addEventListener('click',()=>{
    state.code=Number(button.dataset.code);reset();
  }));
  actionButtons.forEach(button=>button.addEventListener('click',()=>{
    state.action=Number(button.dataset.action);
    updatePressed(actionButtons,'action',state.action);
  }));
  bridgeSlider.addEventListener('input',()=>{
    state.bridge=Number(bridgeSlider.value);reset();
  });
  select.addEventListener('change',()=>{
    state.queryIndex=Number(select.value);render();
  });
  stepButton.addEventListener('click',applyAction);
  resetButton.addEventListener('click',reset);
  reset();
})(typeof globalThis!=='undefined'?globalThis:this);

/* Original browser adapter for the exported trained Q table. */
"use strict";
const M = window.PRIVACYGUARD;
const C = M.config;
const names = ["Deny", "Coarse", "Ask consent", "Full"];
const $ = id => document.getElementById(id);
const ids = ["h", "b", "f", "u", "s", "p"];
const encode = ([h,b,f,u,s,p]) => (((((h*(C.budget+1)+b)*3+f)*3+u)*3+s)*2+p);
function mask([h,b,f,u,s,p]) {const cost=[1,2,4][s];return [true,b>=Math.ceil(.35*cost),b>=cost,!!p&&b>=cost];}
function decision(state) {const allowed=mask(state), q=M.q[encode(state)];let best=0;for(let a=1;a<4;a++)if(allowed[a]&&q[a]>q[best])best=a;return {best,q,allowed};}
function inspect() {
  const state=ids.map(id=>Number($(id).value));const [h,b,f,u,s,p]=state;const {best,q,allowed}=decision(state);const visits=M.visits[encode(state)];
  $("hv").textContent=h;$("bv").textContent=b;$("choice").textContent=names[best];$("context").textContent=`${h} requests remaining · ${b} exposure units available`;
  const acceptance=Math.max(.1,Math.min(.95,C.accept_base-C.accept_sensitivity*s-C.accept_fatigue*f));
  $("guard").textContent=(p?"Full permission exists":"Full sharing is blocked")+` · Consent acceptance ${(100*acceptance).toFixed(0)}%`;
  const valid=q.filter((v,a)=>allowed[a]), min=Math.min(...valid), max=Math.max(...valid);
  $("bars").innerHTML=names.map((name,a)=>`<div class="barrow"><span>${name}</span><div class="barbg"><div class="bar ${best===a?"chosen":""}" style="width:${allowed[a]?20+80*(q[a]-min)/Math.max(.01,max-min):0}%"></div></div><span>${allowed[a]?q[a].toFixed(2):"Blocked"}</span></div>`).join("");
  const expectedImmediate=immediate(state,best);
  const rule=best===0?"The learned table prefers preserving resources over serving this request.":best===1?"A coarse response supplies 60% of task utility and spends fewer exposure units for medium/high sensitivity.":best===2?"The agent will ask for one-request permission. A rejection shares no data; an acceptance still spends the full exposure cost.":"The agent uses existing permission and spends full exposure to serve this request.";
  $("reason").textContent=`${rule} Expected immediate reward: ${expectedImmediate.toFixed(2)}. The action is selected by the largest feasible learned Q value; this description reports model mechanics, not a causal explanation of training.`;
  $("uncertainty").textContent=`Selected action updated ${visits[best]} times in training.${visits[best]<20?" Sparse coverage: treat this Q estimate cautiously.":""} Bar lengths are rescaled Q values, not probabilities.`;
}
function immediate(state,a) {let [h,b,f,u,s,p]=state,U=[1,3,7][u],E=[1,2,4][s],P=Math.max(.1,Math.min(.95,C.accept_base-C.accept_sensitivity*s-C.accept_fatigue*f));let full=U-C.privacy_weight*E,deny=-C.missed_weight*U;return a===0?deny:a===1?C.coarse_fraction*U-C.privacy_weight*Math.ceil(.35*E):a===3?full:P*full+(1-P)*deny-C.prompt_base-C.prompt_fatigue*f;}
let rngState=314159, session;
function random(){rngState=(Math.imul(1664525,rngState)+1013904223)>>>0;return rngState/4294967296;}
function draw(probs){let x=random();for(let i=0;i<probs.length;i++){x-=probs[i];if(x<0)return i;}return probs.length-1;}
function request(h,b,f){return [h,b,f,draw(C.utility_probs),draw(C.sensitivity_probs),draw(C.scope_probs)];}
function reset(){rngState=314159;session={state:request(C.horizon,C.budget,0),ret:0,used:0,asks:0,steps:0};$("trace").innerHTML="";$("event").textContent="Seeded session ready. Press Next request to execute the learned policy.";displaySession();}
function displaySession(){$("ret").textContent=session.ret.toFixed(2);$("used").textContent=`${session.used} / ${C.budget}`;$("asks").textContent=session.asks;for(let i=0;i<6;i++)$(ids[i]).value=session.state[i]||0;if(session.state[0]===0)$("h").value=1;inspect();$("step").disabled=session.state[0]===0;$("run").disabled=session.state[0]===0;}
function step(){if(!session||session.state[0]===0)return;const state=session.state,[h,b,f,u,s,p]=state,{best:a}=decision(state);let U=[1,3,7][u],E=[1,2,4][s],P=Math.max(.1,Math.min(.95,C.accept_base-C.accept_sensitivity*s-C.accept_fatigue*f)),accepted=a===2&&random()<P,full=a===3||accepted,exposure=full?E:a===1?Math.ceil(.35*E):0,utility=full?U:a===1?C.coarse_fraction*U:0,prompt=a===2?C.prompt_base+C.prompt_fatigue*f:0,reward=utility-C.privacy_weight*exposure-prompt-(utility===0?C.missed_weight*U:0);
  if(exposure>b||(a===3&&!p))throw Error("Guard violation");session.steps++;session.ret+=reward;session.used+=exposure;session.asks+=a===2?1:0;
  const outcome=a===2?(accepted?"Accepted":"Rejected"):a===0?"No disclosure":a===1?"Coarse disclosed":"Full disclosed";
  $("trace").insertAdjacentHTML("beforeend",`<tr><td>${session.steps}</td><td>${["Low","Med","High"][u]} / ${["Low","Med","High"][s]}</td><td>${names[a]}</td><td>${outcome}</td><td>${reward.toFixed(2)}</td><td>${b-exposure}</td></tr>`);
  session.state=request(h-1,b-exposure,a===2?Math.min(2,f+1):Math.max(0,f-1));$("event").textContent=h===1?"Session complete. Every action passed the permission and budget guard.":`Step ${session.steps}: ${names[a]}. ${outcome}. The next request is ready.`;displaySession();
}
for(const id of ids)$(id).addEventListener("input",inspect);
document.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{const preset={reserve:[12,3,0,0,1,1],consent:[2,8,0,2,0,0],empty:[6,0,2,2,2,0]}[button.dataset.preset];ids.forEach((id,i)=>$(id).value=preset[i]);inspect();}));
$("new").addEventListener("click",reset);$("step").addEventListener("click",step);$("run").addEventListener("click",()=>{while(session.state[0]>0)step();});
const show=["Q-learning","SARSA","Myopic Q (gamma=0)","Greedy immediate","Always coarse","Exact DP ceiling"];
$("metrics").innerHTML=show.map(name=>M.summary.find(r=>r.policy===name)).filter(Boolean).map(r=>`<tr><td>${r.policy}</td><td>${r.return.toFixed(2)} ± ${r.return_ci95.toFixed(2)}</td><td>${r.utility.toFixed(2)}</td><td>${r.exposure.toFixed(2)}</td><td>${r.prompts.toFixed(2)}</td></tr>`).join("");
$("resultnote").textContent="Final experiment results. Confidence intervals use independent seed means; see results/manifest.json for exact run counts. DP is an analytical diagnostic, never used to train the agent.";
reset();

import React from 'react';
import {AbsoluteFill} from 'remotion';

// LinkedIn 4:5 portrait architecture image, 1080x1350. Light palette, system fonts (Ubuntu).
const C = {bg: '#f5f7fa', card: '#ffffff', ink: '#14202b', muted: '#55626f', line: '#aab6c3',
  accent: '#1d5fd0', accentSoft: '#e3ecfc', warn: '#b25e00', warnSoft: '#fff1d6', cache: '#6b7686'};
const FONT = 'Ubuntu, "Liberation Sans", system-ui, Arial, sans-serif';

const CX = 350, CW = 380, CH = 78, GAP = 108, Y0 = 320;
const stages = [
  {t: 'Sanitize', s: 'strips injected instructions', dot: true},
  {t: 'Rewrite', s: 'turns follow-ups into questions', dot: true},
  {t: 'Access + version filter', s: 'applied before ranking', key: true},
  {t: 'Hybrid retrieve', s: 'dense + BM25, fused with RRF', dot: true},
  {t: 'Rerank', s: 'cross-encoder keeps the best', dot: true},
  {t: 'Grounded answer', s: 'every claim cites [n]', dot: true},
  {t: 'Output guard', s: 'drops citations that point nowhere'},
];
const y = (i: number) => Y0 + GAP * i;

const Arrow: React.FC<{d: string; color: string; dash?: string; marker?: boolean}> = ({d, color, dash, marker = true}) => (
  <path d={d} fill="none" stroke={color} strokeWidth={3} strokeDasharray={dash} strokeLinecap="round" strokeLinejoin="round"
    markerEnd={marker ? `url(#m-${color.slice(1)})` : undefined} />
);

export const Diagram: React.FC = () => (
  <AbsoluteFill style={{background: C.bg, fontFamily: FONT}}>
    <svg width={1080} height={1350} viewBox="0 0 1080 1350" xmlns="http://www.w3.org/2000/svg" fontFamily={FONT}>
      <defs>
        {[C.ink, C.accent, C.warn].map((c) => (
          <marker key={c} id={`m-${c.slice(1)}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0 0 L10 5 L0 10 z" fill={c} />
          </marker>
        ))}
      </defs>

      {/* header */}
      <text x={40} y={70} fontSize={19} fill={C.muted} letterSpacing={2}>ASKWELL · RAG KNOWLEDGE ASSISTANT</text>
      <text x={40} y={146} fontSize={64} fontWeight={700} fill={C.ink}>Filter first. Answer last.</text>
      <text x={40} y={190} fontSize={24} fill={C.muted}>How restricted documents stay out of the model's reach</text>

      {/* question pill */}
      <rect x={CX} y={226} width={CW} height={52} rx={26} fill={C.ink} />
      <text x={CX + CW / 2} y={260} fontSize={22} fontWeight={500} fill="#fff" textAnchor="middle">Question + user id</text>
      <Arrow d={`M${CX + CW / 2} 278 V ${Y0 - 3}`} color={C.ink} />

      {/* pipeline cards */}
      {stages.map((st, i) => (
        <g key={st.t}>
          <rect x={CX} y={y(i)} width={CW} height={CH} rx={12} fill={st.key ? C.accentSoft : C.card} stroke={st.key ? C.accent : C.line} strokeWidth={st.key ? 3.5 : 2} />
          <text x={CX + 20} y={y(i) + 35} fontSize={27} fontWeight={700} fill={st.key ? C.accent : C.ink}>{st.t}</text>
          <text x={CX + 20} y={y(i) + 62} fontSize={19} fill={C.muted}>{st.s}</text>
          {st.dot && <circle cx={CX + CW - 22} cy={y(i) + 22} r={6} fill={C.accent} />}
          {i < stages.length - 1 && <Arrow d={`M${CX + CW / 2} ${y(i) + CH} V ${y(i + 1) - 3}`} color={i === 2 ? C.accent : C.ink} />}
        </g>
      ))}

      {/* reply pill */}
      <Arrow d={`M${CX + CW / 2} ${y(6) + CH} V 1075`} color={C.ink} />
      <rect x={CX} y={1080} width={CW} height={52} rx={26} fill={C.ink} />
      <text x={CX + CW / 2} y={1114} fontSize={22} fontWeight={500} fill="#fff" textAnchor="middle">Cited answer + trace id</text>

      {/* left column: ingestion, chroma, cache */}
      <rect x={40} y={y(2)} width={260} height={CH} rx={12} fill={C.card} stroke={C.line} strokeWidth={2} />
      <text x={58} y={y(2) + 35} fontSize={25} fontWeight={700} fill={C.ink}>Ingestion</text>
      <text x={58} y={y(2) + 62} fontSize={17} fill={C.muted}>PDF, DOCX, MD, HTML, CSV</text>
      <Arrow d={`M170 ${y(2) + CH} V ${y(3) - 3}`} color={C.ink} />
      <rect x={40} y={y(3)} width={260} height={CH} rx={12} fill={C.card} stroke={C.ink} strokeWidth={2.5} />
      <text x={58} y={y(3) + 35} fontSize={25} fontWeight={700} fill={C.ink}>Chroma</text>
      <text x={58} y={y(3) + 62} fontSize={17} fill={C.muted}>chunks tagged by access</text>
      <Arrow d={`M300 ${y(3) + 39} H ${CX - 3}`} color={C.ink} />

      <rect x={40} y={790} width={260} height={92} rx={12} fill={C.card} stroke={C.cache} strokeWidth={2.5} strokeDasharray="3 6" strokeLinecap="round" />
      <text x={58} y={824} fontSize={25} fontWeight={700} fill={C.ink}>Cache</text>
      <text x={58} y={850} fontSize={17} fill={C.muted}>memory or Redis</text>
      <text x={58} y={871} fontSize={17} fill={C.muted}>a hit skips the LLM</text>
      <path d={`M300 806 L ${CX} ${y(3) + 66}`} stroke={C.cache} strokeWidth={3} strokeDasharray="2 7" strokeLinecap="round" fill="none" />
      <path d={`M300 866 L ${CX} ${y(5) + 39}`} stroke={C.cache} strokeWidth={3} strokeDasharray="2 7" strokeLinecap="round" fill="none" />

      {/* right column: refusal + LLM */}
      <rect x={790} y={560} width={250} height={270} rx={12} fill={C.warnSoft} stroke={C.warn} strokeWidth={2.5} />
      <text x={810} y={602} fontSize={27} fontWeight={700} fill={C.warn}>Refuse</text>
      <text x={810} y={630} fontSize={18} fill={C.warn}>no model call when:</text>
      <text x={810} y={672} fontSize={19} fill={C.ink}>• only an attack</text>
      <text x={810} y={702} fontSize={19} fill={C.ink}>• no chunks left</text>
      <text x={810} y={732} fontSize={19} fill={C.ink}>• top score too low</text>
      <text x={810} y={792} fontSize={21} fontStyle="italic" fill={C.warn}>“I don't know.”</text>
      <Arrow d={`M${CX + CW} ${y(0) + 39} H 915 V 556`} color={C.warn} dash="9 7" />
      <Arrow d={`M${CX + CW} ${y(3) + 39} H 786`} color={C.warn} dash="9 7" />
      <Arrow d={`M${CX + CW} ${y(4) + 39} H 786`} color={C.warn} dash="9 7" />

      <rect x={790} y={y(5)} width={250} height={CH} rx={12} fill={C.card} stroke={C.ink} strokeWidth={2.5} />
      <text x={808} y={y(5) + 35} fontSize={25} fontWeight={700} fill={C.ink}>LLM</text>
      <text x={808} y={y(5) + 62} fontSize={17} fill={C.muted}>sees numbered sources</text>
      <Arrow d={`M${CX + CW} ${y(5) + 39} H 786`} color={C.ink} />

      {/* legend */}
      <g transform="translate(40 1200)">
        <line x1={0} y1={0} x2={34} y2={0} stroke={C.ink} strokeWidth={3} strokeLinecap="round" />
        <text x={44} y={6} fontSize={17} fill={C.muted}>request flow</text>
        <line x1={190} y1={0} x2={224} y2={0} stroke={C.accent} strokeWidth={3} strokeLinecap="round" />
        <text x={234} y={6} fontSize={17} fill={C.muted}>the guarantee</text>
        <line x1={390} y1={0} x2={424} y2={0} stroke={C.warn} strokeWidth={3} strokeDasharray="9 7" />
        <text x={434} y={6} fontSize={17} fill={C.muted}>refusal, no LLM call</text>
        <line x1={650} y1={0} x2={684} y2={0} stroke={C.cache} strokeWidth={3} strokeDasharray="2 7" strokeLinecap="round" />
        <text x={694} y={6} fontSize={17} fill={C.muted}>cache lookup</text>
        <circle cx={850} cy={0} r={6} fill={C.accent} />
        <text x={864} y={6} fontSize={17} fill={C.muted}>traced span</text>
      </g>
      <line x1={40} y1={1240} x2={1040} y2={1240} stroke={C.line} strokeWidth={1.5} />
      <text x={40} y={1284} fontSize={22} fontWeight={500} fill={C.ink}>Python · FastAPI · Chroma · Redis · Docker · OpenTelemetry</text>
      <text x={40} y={1318} fontSize={19} fill={C.muted}>Demo corpus: 52 documents, 10 user roles, 80-question eval set</text>
    </svg>
  </AbsoluteFill>
);

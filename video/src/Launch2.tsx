import React from 'react';
import {AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';

/**
 * Launch video 2: rebuilt from the screencast "Screencast from 08-11-2026 13:13:18.webm".
 * Every label, answer excerpt, document field and trace id below is copied from that recording.
 * Colors come from app/ui/style.css (dark theme). 30 fps, 900 frames = 30 s.
 */
export const DURATION2 = 900;

const C = {
  bg: '#0b1312', surface: '#111c1b', surface2: '#172524', line: '#233231', lineStrong: '#324544',
  fg: '#e3eeed', muted: '#94a9a7', faint: '#667b79',
  accent: '#3cc6b6', accentStrong: '#7de0d3', accentFg: '#052420', accentSoft: '#123430',
  warn: '#f0c36a', warnBg: '#352a0e', err: '#ff9d9d', errBg: '#3a1919', ok: '#7ee0a3', okBg: '#12301f',
};
const FONT = 'system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const S = 1.5; // UI scale: the real UI at 1280x720 CSS px fills the 1920x1080 frame
const DRAWER_W = 400 * S;
const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

type User = {ini: string; label: string; role: string; level: string; depts: string[]};
const GUEST: User = {ini: 'G', label: 'Guest', role: 'Guest', level: 'public', depts: []};
const MARCUS: User = {ini: 'ML', label: 'Marcus Lee (u02, IT Servic…', role: 'IT Service Manager', level: 'internal', depts: []};
const ELENA: User = {ini: 'ER', label: 'Elena Ruiz (u03, Shift Supe…', role: 'Shift Supervisor, WH-DAL-02', level: 'internal', depts: []};
const TOM: User = {ini: 'TB', label: 'Tom Becker (u04, Finance …', role: 'Finance Analyst', level: 'confidential', depts: ['Finance']};

const CARDS = [
  {q: 'What is the salary range for grade G5?', tag: 'RESTRICTED', kind: 'red'},
  {q: 'How much parental leave do employees get?', tag: 'PUBLIC', kind: 'green'},
  {q: 'What are the first steps in the security incident response runbook?', tag: 'CONFIDENTIAL', kind: 'amber'},
  {q: 'How does the privileged access break-glass procedure work?', tag: 'RESTRICTED', kind: 'red'},
  {q: 'What is the current travel and expense policy for meals?', tag: 'MULTIPLE VERSIONS', kind: 'teal'},
  {q: 'How do visitors get on the guest Wi-Fi?', tag: 'PUBLIC', kind: 'green'},
];
const REFUSAL = "I don't know. The company documents I can access do not answer this question. Please check with the relevant department.";

type Seg = {t: string; cite?: boolean};
const MARCUS_ANSWER: Seg[] = [
  {t: 'Primary caregivers receive 16 weeks of paid parental leave at 100% of base pay', cite: true},
  {t: '. Secondary caregivers receive 6 weeks of paid parental leave at 100% of base pay', cite: true},
  {t: '.'},
];
const TOM_ANSWER: Seg[] = [
  {t: 'The meal allowance is $75 per day when traveling overnight, covering breakfast, lunch, and dinner', cite: true},
  {t: '. On the first and last day of a trip, the allowance is 75% of the daily amount', cite: true},
  {t: '.'},
];

type Drawer = {title: string; rows: [string, string][]; chunk: string[]};
const MARCUS_DRAWER: Drawer = {
  title: 'Parental Leave Policy',
  rows: [['Section', 'Paid Leave Amounts'], ['Document', 'hr-pol-005-v1'], ['Version', '1'], ['Status', 'active'], ['Effective', '2024-07-01']],
  chunk: [
    'Parental Leave Policy > Paid Leave Amounts', 'Paid Leave Amounts',
    'Primary caregivers receive 16 weeks of paid leave at 100% of base pay.',
    'Secondary caregivers receive 6 weeks of paid leave at 100% of base pay.',
    'Leave can be taken in one block or split into up to two blocks, each at least 2 weeks long.',
    'Paid parental leave runs at the same time as any leave provided by the Family and Medical Leave Act (FMLA) or state programs. If a state benefit pays part of the employee’s wage, Nimbus pays the difference so that the employee receives full base pay.',
  ],
};
const TOM_DRAWER: Drawer = {
  title: 'Travel and Expense Policy',
  rows: [['Section', 'Meals'], ['Document', 'fin-pol-001-v3'], ['Version', '3'], ['Status', 'active'], ['Effective', '2025-07-01']],
  chunk: [
    'Travel and Expense Policy > Meals', 'Meals',
    'The daily meal allowance is $75 per day when traveling overnight, covering breakfast, lunch and dinner. On the first and last day of a trip the allowance is 75% of the daily amount. Alcohol is reimbursed only at customer entertainment events approved in advance by a director. Meal expenses are coded to GL-6110.',
  ],
};

/* ---------- small helpers ---------- */
const pop = (frame: number, start: number, fps: number) => {
  const p = spring({frame: frame - start, fps, config: {damping: 13, stiffness: 170, mass: 0.7}});
  return {opacity: interpolate(frame - start, [0, 5], [0, 1], clamp), transform: `translateY(${(1 - p) * 34}px) scale(${0.95 + 0.05 * p})`};
};
const ring = (frame: number, from: number, to: number) => {
  const a = Math.min(6, (to - from) / 2 - 0.01); // short windows (button presses) get a shorter ramp
  return interpolate(frame, [from, from + a, to - a, to], [0, 1, 1, 0], clamp);
};
const dots = (frame: number) => [0, 1, 2].map((i) => 0.35 + 0.65 * Math.max(0, Math.sin((frame - i * 3) * 0.32)));

const Drop: React.FC<{size: number}> = ({size}) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3c3.4 4 6 6.8 6 10a6 6 0 0 1-12 0c0-3.2 2.6-6 6-10z" /><path d="M9.5 13.2a2.6 2.6 0 0 0 2.5 2.4" />
  </svg>
);
const Pill: React.FC<{children: React.ReactNode; level?: boolean}> = ({children, level}) => (
  <span style={{borderRadius: 999, padding: `1px ${9 * S}px`, fontSize: 12 * S, whiteSpace: 'nowrap', fontWeight: level ? 700 : 400,
    background: level ? C.accentSoft : C.surface2, border: level ? '1px solid transparent' : `1px solid ${C.line}`, color: level ? C.accentStrong : C.muted}}>{children}</span>
);
const Mark: React.FC = () => (
  <span style={{width: 30 * S, height: 30 * S, borderRadius: 9 * S, background: C.accentSoft, color: C.accentStrong, display: 'grid', placeItems: 'center', flex: 'none', marginTop: 2 * S}}>
    <Drop size={17 * S} />
  </span>
);

/* ---------- header ---------- */
const Header: React.FC<{user: User; pulse: number}> = ({user, pulse}) => (
  <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: `${10 * S}px ${16 * S}px`, background: C.surface, borderBottom: `1px solid ${C.line}`}}>
    <div style={{display: 'flex', alignItems: 'center', gap: 11 * S}}>
      <span style={{width: 38 * S, height: 38 * S, borderRadius: 11 * S, background: C.accent, color: C.accentFg, display: 'grid', placeItems: 'center'}}><Drop size={22 * S} /></span>
      <div>
        <div style={{fontSize: 18 * S, fontWeight: 700, lineHeight: 1.15, letterSpacing: -0.2}}>Askwell</div>
        <div style={{fontSize: 12.5 * S, color: C.muted}}>Nimbus Logistics knowledge assistant</div>
      </div>
    </div>
    <div style={{display: 'flex', alignItems: 'center', gap: 12 * S, whiteSpace: 'nowrap'}}>
      <span style={{width: 34 * S, height: 34 * S, borderRadius: '50%', background: C.accentSoft, color: C.accentStrong, display: 'grid', placeItems: 'center', fontWeight: 700, fontSize: 13 * S}}>{user.ini}</span>
      <div style={{borderRadius: 10 * S, padding: `${2 * S}px ${8 * S}px`, boxShadow: `0 0 0 ${5 * pulse}px rgba(60,198,182,${0.35 * pulse})`, border: `1px solid ${pulse > 0.05 ? C.accent : 'transparent'}`}}>
        <div style={{fontSize: 11 * S, textTransform: 'uppercase', letterSpacing: '.08em', color: C.faint, fontWeight: 600}}>Viewing as</div>
        <div style={{fontSize: 14 * S, fontWeight: 600}}>{user.label} <span style={{marginLeft: 6}}>{'⌄'}</span></div>
      </div>
      <span style={{fontSize: 12 * S, color: C.muted, display: 'flex', alignItems: 'center', gap: 6 * S}}>
        {user.role}<Pill level>{user.level}</Pill>
        {user.depts.length ? user.depts.map((d) => <Pill key={d}>{d}</Pill>) : <Pill>no departments</Pill>}
      </span>
      <span style={{width: 34 * S, height: 34 * S, borderRadius: 10 * S, border: `1px solid ${C.line}`, background: C.surface, display: 'grid', placeItems: 'center', color: C.muted, fontSize: 16 * S}}>{'◑'}</span>
      <span style={{border: `1px solid ${C.line}`, background: C.surface, borderRadius: 10 * S, padding: `${7 * S}px ${12 * S}px`, fontWeight: 600, fontSize: 13.5 * S}}>+ New chat</span>
    </div>
  </div>
);

/* ---------- welcome ---------- */
const kindStyle = (k: string): React.CSSProperties =>
  k === 'red' ? {background: C.errBg, color: C.err} : k === 'amber' ? {background: C.warnBg, color: C.warn} : k === 'green' ? {background: C.okBg, color: C.ok} : {background: C.accentSoft, color: C.accentStrong};

const Welcome: React.FC<{highlight: number; hi: number; press: number}> = ({highlight, hi, press}) => (
  <div>
    <div style={{fontSize: 30 * S, fontWeight: 700, letterSpacing: -0.6, lineHeight: 1.15, margin: `${6 * S}px 0 ${8 * S}px`}}>What can I help you find?</div>
    <div style={{color: C.muted, maxWidth: '62ch', marginBottom: 22 * S, fontSize: 15 * S, lineHeight: 1.55}}>
      Answers come from Nimbus Logistics documents and always cite their sources. What you see depends on who you are: switch the user above and ask the same question again.
    </div>
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 * S}}>
      {CARDS.map((c, i) => {
        const on = i === highlight ? hi : 0;
        return (
          <div key={c.q} style={{display: 'flex', flexDirection: 'column', gap: 8 * S, padding: 14 * S, borderRadius: 14 * S, background: C.surface,
            border: `1px solid ${on > 0.05 ? C.accent : C.line}`, boxShadow: `0 0 0 ${6 * on}px rgba(60,198,182,${0.3 * on})`, transform: `scale(${1 - 0.03 * (i === highlight ? press : 0)})`}}>
            <span style={{fontWeight: 600, lineHeight: 1.35, fontSize: 15 * S}}>{c.q}</span>
            <span style={{alignSelf: 'flex-start', fontSize: 11 * S, fontWeight: 700, letterSpacing: '.07em', padding: `2px ${8 * S}px`, borderRadius: 999, ...kindStyle(c.kind)}}>{c.tag}</span>
          </div>
        );
      })}
    </div>
  </div>
);

/* ---------- chat pieces ---------- */
const UserBubble: React.FC<{text: string; style: React.CSSProperties}> = ({text, style}) => (
  <div style={{display: 'flex', justifyContent: 'flex-end', ...style}}>
    <div style={{maxWidth: '88%', background: C.accent, color: C.accentFg, padding: `${10 * S}px ${15 * S}px`, borderRadius: `${18 * S}px ${18 * S}px ${4 * S}px ${18 * S}px`, fontSize: 15 * S}}>{text}</div>
  </div>
);

const BotRow: React.FC<{children: React.ReactNode; refused?: boolean; style?: React.CSSProperties}> = ({children, refused, style}) => (
  <div style={{display: 'flex', gap: 10 * S, ...style}}>
    <Mark />
    <div style={{flex: 1, minWidth: 0, background: refused ? C.warnBg : C.surface, border: `1px solid ${C.line}`, borderLeft: refused ? `${4 * S}px solid ${C.warn}` : `1px solid ${C.line}`,
      borderRadius: `${4 * S}px ${18 * S}px ${18 * S}px ${18 * S}px`, padding: `${13 * S}px ${16 * S}px`, fontSize: 15 * S, lineHeight: 1.55}}>{children}</div>
  </div>
);

const Status: React.FC<{frame: number; text: string}> = ({frame, text}) => (
  <div style={{display: 'flex', alignItems: 'center', gap: 10 * S, color: C.muted, fontSize: 14 * S}}>
    <span style={{display: 'inline-flex', gap: 4 * S}}>{dots(frame).map((o, i) => <i key={i} style={{width: 6 * S, height: 6 * S, borderRadius: '50%', background: C.accent, opacity: o, display: 'block'}} />)}</span>
    {text}
  </div>
);

const Cite: React.FC<{lit?: number}> = ({lit = 0}) => (
  <span style={{display: 'inline-grid', placeItems: 'center', minWidth: 20 * S, height: 20 * S, padding: `0 ${5 * S}px`, margin: `0 ${2 * S}px`, borderRadius: 6 * S, background: C.accentSoft, color: C.accentStrong,
    fontSize: 11.5 * S, fontWeight: 700, verticalAlign: 1, lineHeight: 1, boxShadow: `0 0 0 ${5 * lit}px rgba(60,198,182,${0.4 * lit})`, border: `1px solid ${lit > 0.05 ? C.accent : 'transparent'}`}}>1</span>
);

const Streamed: React.FC<{segs: Seg[]; n: number; lit: number}> = ({segs, n, lit}) => {
  let left = n;
  const out: React.ReactNode[] = [];
  segs.forEach((s, i) => {
    const take = Math.max(0, Math.min(s.t.length, left));
    left -= s.t.length;
    out.push(<span key={`t${i}`}>{s.t.slice(0, take)}</span>);
    if (s.cite && take >= s.t.length) out.push(<Cite key={`c${i}`} lit={i === 0 ? lit : 0} />);
  });
  return <div>{out}</div>;
};

const Sources: React.FC<{items: {title: string; sub: string}[]; lit: number}> = ({items, lit}) => (
  <div style={{marginTop: 14 * S, paddingTop: 12 * S, borderTop: `1px dashed ${C.line}`}}>
    <div style={{fontSize: 11 * S, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: C.faint, marginBottom: 8 * S}}>{items.length === 1 ? 'Source' : 'Sources'}</div>
    <div style={{display: 'flex', gap: 8 * S, flexWrap: 'wrap'}}>
      {items.map((it, i) => (
        <div key={it.title} style={{display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: 9 * S, alignItems: 'center', padding: `${8 * S}px ${10 * S}px`, borderRadius: 10 * S, background: C.bg,
          border: `1px solid ${i === 0 && lit > 0.05 ? C.accent : C.line}`, boxShadow: i === 0 ? `0 0 0 ${5 * lit}px rgba(60,198,182,${0.3 * lit})` : 'none'}}>
          <span style={{gridRow: 'span 2', width: 22 * S, height: 22 * S, borderRadius: 7 * S, background: C.accentSoft, color: C.accentStrong, display: 'grid', placeItems: 'center', fontSize: 12 * S, fontWeight: 700}}>{i + 1}</span>
          <span style={{fontSize: 13 * S, fontWeight: 600}}>{it.title}</span>
          <span style={{fontSize: 12 * S, color: C.muted}}>{it.sub}</span>
        </div>
      ))}
    </div>
  </div>
);

const Meta: React.FC<{refused?: boolean; trace: string}> = ({refused, trace}) => (
  <div style={{marginTop: 12 * S, display: 'flex', alignItems: 'center', gap: 6 * S}}>
    {refused && <span style={{borderRadius: 999, padding: `1px ${9 * S}px`, border: `1px solid ${C.warn}`, color: C.warn, fontFamily: MONO, fontSize: 11.5 * S}}>refused: low_score</span>}
    <span style={{marginLeft: 'auto', padding: `2px ${10 * S}px`, borderRadius: 999, border: `1px solid ${C.line}`, color: C.faint, fontFamily: MONO, fontSize: 12 * S}}>Trace {trace}</span>
  </div>
);

const DrawerPanel: React.FC<{d: Drawer; x: number}> = ({d, x}) => (
  <div style={{width: DRAWER_W, flex: 'none', borderLeft: `1px solid ${C.line}`, background: C.surface, padding: `${16 * S}px ${18 * S}px`, transform: `translateX(${x}px)`, fontSize: 13 * S}}>
    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 * S}}>
      <strong style={{fontSize: 15 * S}}>Source 1</strong>
      <span style={{width: 34 * S, height: 34 * S, borderRadius: 10 * S, border: `1px solid ${C.line}`, display: 'grid', placeItems: 'center', color: C.muted, fontSize: 16 * S}}>{'✕'}</span>
    </div>
    <div style={{fontSize: 17 * S, fontWeight: 700, lineHeight: 1.3, marginBottom: 10 * S}}>{d.title}</div>
    <div style={{display: 'grid', gridTemplateColumns: 'auto 1fr', gap: `${6 * S}px ${14 * S}px`, margin: `${12 * S}px 0`}}>
      {d.rows.map(([k, v]) => (<React.Fragment key={k}><span style={{color: C.muted}}>{k}</span><span>{v}</span></React.Fragment>))}
    </div>
    <div style={{background: C.bg, border: `1px solid ${C.line}`, borderRadius: 10 * S, padding: 12 * S, lineHeight: 1.55, display: 'flex', flexDirection: 'column', gap: 9 * S}}>
      {d.chunk.map((p, i) => <div key={i}>{p}</div>)}
    </div>
  </div>
);

const Composer: React.FC<{busy: boolean}> = ({busy}) => (
  <div style={{padding: `${6 * S}px ${16 * S}px ${12 * S}px`}}>
    <div style={{maxWidth: 800 * S, margin: '0 auto', display: 'flex', alignItems: 'center', gap: 8 * S, padding: `${8 * S}px ${8 * S}px ${8 * S}px ${16 * S}px`, background: C.surface, border: `1px solid ${C.lineStrong}`, borderRadius: 20 * S}}>
      <span style={{flex: 1, color: C.faint, fontSize: 15 * S}}>Ask about HR, IT, Operations or Finance...</span>
      <span style={{width: 38 * S, height: 38 * S, borderRadius: '50%', background: busy ? C.fg : C.accent, color: busy ? C.bg : C.accentFg, display: 'grid', placeItems: 'center', fontSize: 16 * S, fontWeight: 700}}>{busy ? '■' : '↑'}</span>
    </div>
    <div style={{maxWidth: 800 * S, margin: `${7 * S}px auto 0`, fontSize: 12 * S, color: C.faint, textAlign: 'center'}}>Enter to send, Shift+Enter for a new line. Demo users only: the user switch is not real sign-in.</div>
  </div>
);

const Caption: React.FC<{from: number; to: number; rightInset: number; children: React.ReactNode; sub?: React.ReactNode}> = ({from, to, rightInset, children, sub}) => {
  const f = useCurrentFrame();
  if (f < from || f > to) return null;
  const op = interpolate(f, [from, from + 8, to - 8, to], [0, 1, 1, 0], clamp);
  const y = interpolate(f, [from, from + 10], [24, 0], {...clamp, easing: Easing.out(Easing.cubic)});
  return (
    <div style={{position: 'absolute', left: 0, right: rightInset, bottom: 170, display: 'flex', justifyContent: 'center', opacity: op, transform: `translateY(${y}px)`, pointerEvents: 'none'}}>
      <div style={{background: 'rgba(5,14,13,0.94)', border: `1px solid ${C.lineStrong}`, color: '#fff', borderRadius: 22, padding: '20px 42px', textAlign: 'center', boxShadow: '0 14px 44px rgba(0,0,0,.45)', maxWidth: 1560}}>
        <div style={{fontSize: 54, fontWeight: 700, letterSpacing: -0.5, lineHeight: 1.2}}>{children}</div>
        {sub && <div style={{fontSize: 30, marginTop: 8, color: C.accentStrong, lineHeight: 1.25}}>{sub}</div>}
      </div>
    </div>
  );
};

/* ---------- the timeline ---------- */
const App: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // who is signed in as
  const user = frame < 216 ? GUEST : frame < 490 ? MARCUS : frame < 630 ? ELENA : TOM;
  const selectPulse = Math.max(ring(frame, 204, 218), ring(frame, 478, 492), ring(frame, 618, 632));

  // scene boundaries: [welcome start, click frame] per user
  const scene = frame < 66 ? 0 : frame < 216 ? 1 : frame < 256 ? 2 : frame < 490 ? 3 : frame < 524 ? 4 : frame < 630 ? 5 : frame < 664 ? 6 : 7;
  const showWelcome = scene === 0 || scene === 2 || scene === 4 || scene === 6;
  const cardIdx = scene === 0 ? 0 : scene === 2 ? 1 : scene === 4 ? 2 : 4;
  const cardRing = scene === 0 ? ring(frame, 48, 66) : scene === 2 ? ring(frame, 238, 256) : scene === 4 ? ring(frame, 506, 524) : ring(frame, 646, 664);
  const cardPress = scene === 0 ? ring(frame, 62, 67) : scene === 2 ? ring(frame, 252, 257) : scene === 4 ? ring(frame, 520, 525) : ring(frame, 660, 665);

  let busy = false;
  let thread: React.ReactNode = null;
  let drawer: Drawer | null = null;
  let drawerAt = 0;

  if (scene === 1) {
    busy = frame < 108;
    thread = (<>
      <UserBubble text={CARDS[0].q} style={pop(frame, 66, fps)} />
      {frame >= 74 && frame < 108 && <BotRow style={pop(frame, 74, fps)}><Status frame={frame} text="Searching documents..." /></BotRow>}
      {frame >= 108 && <BotRow refused style={pop(frame, 108, fps)}><div>{REFUSAL}</div><Meta refused trace="2d227e96" /></BotRow>}
    </>);
  } else if (scene === 3) {
    const n = Math.floor((frame - 310) * 4.4);
    const chipLit = ring(frame, 372, 386);
    busy = frame < 354;
    drawer = frame >= 384 ? MARCUS_DRAWER : null; drawerAt = 384;
    thread = (<>
      <UserBubble text={CARDS[1].q} style={pop(frame, 256, fps)} />
      {frame >= 264 && frame < 310 && <BotRow style={pop(frame, 264, fps)}><Status frame={frame} text={frame < 296 ? 'Searching documents...' : 'Reading 5 sources...'} /></BotRow>}
      {frame >= 310 && (
        <BotRow>
          <Streamed segs={MARCUS_ANSWER} n={n} lit={chipLit} />
          {frame >= 354 && <div style={pop(frame, 354, fps)}><Sources items={[{title: 'Parental Leave Policy', sub: 'Paid Leave Amounts | v1'}]} lit={chipLit} /><Meta trace="bff13c0c" /></div>}
        </BotRow>
      )}
    </>);
  } else if (scene === 5) {
    busy = frame < 566;
    thread = (<>
      <UserBubble text={CARDS[2].q} style={pop(frame, 524, fps)} />
      {frame >= 532 && frame < 566 && <BotRow style={pop(frame, 532, fps)}><Status frame={frame} text="Searching documents..." /></BotRow>}
      {frame >= 566 && <BotRow refused style={pop(frame, 566, fps)}><div>{REFUSAL}</div><Meta refused trace="2579d08c" /></BotRow>}
    </>);
  } else if (scene === 7) {
    const n = Math.floor((frame - 712) * 4.6);
    const chipLit = ring(frame, 768, 782);
    busy = frame < 756;
    drawer = frame >= 780 ? TOM_DRAWER : null; drawerAt = 780;
    thread = (<>
      <UserBubble text={CARDS[4].q} style={pop(frame, 664, fps)} />
      {frame >= 672 && frame < 712 && <BotRow style={pop(frame, 672, fps)}><Status frame={frame} text={frame < 698 ? 'Searching documents...' : 'Reading 5 sources...'} /></BotRow>}
      {frame >= 712 && (
        <BotRow>
          <Streamed segs={TOM_ANSWER} n={n} lit={chipLit} />
          {frame >= 756 && <div style={pop(frame, 756, fps)}><Sources items={[{title: 'Travel and Expense Policy', sub: 'Meals | v3'}, {title: 'Expense Reimbursement FAQ', sub: 'Rules and Limits | v1'}]} lit={chipLit} /><Meta trace="65a81b70" /></div>}
        </BotRow>
      )}
    </>);
  }

  const drawerX = drawer ? interpolate(frame, [drawerAt, drawerAt + 14], [DRAWER_W, 0], {...clamp, easing: Easing.out(Easing.cubic)}) : DRAWER_W;
  const inset = drawer ? DRAWER_W - drawerX : 0;

  return (
    <AbsoluteFill style={{background: C.bg, color: C.fg, fontFamily: FONT, flexDirection: 'column'}}>
      <Header user={user} pulse={selectPulse} />
      <div style={{flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden'}}>
        <div style={{flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column'}}>
          <div style={{flex: 1, padding: `0 ${16 * S}px`, minHeight: 0}}>
            <div style={{maxWidth: 800 * S, margin: '0 auto', padding: `${28 * S}px 0 ${12 * S}px`, display: 'flex', flexDirection: 'column', gap: 18 * S}}>
              {showWelcome ? <Welcome highlight={cardIdx} hi={cardRing} press={cardPress} /> : thread}
            </div>
          </div>
          <Composer busy={busy} />
        </div>
        {drawer && <DrawerPanel d={drawer} x={drawerX} />}
      </div>

      <Caption from={70} to={140} rightInset={inset}>A guest asks about salaries</Caption>
      <Caption from={140} to={205} rightInset={inset} sub="Restricted documents are filtered out before ranking">Restricted? Refused.</Caption>
      <Caption from={262} to={350} rightInset={inset} sub="IT Service Manager, clearance: internal">An employee asks about parental leave</Caption>
      <Caption from={400} to={480} rightInset={inset} sub="Open a citation to read the original">Answered, with the source</Caption>
      <Caption from={530} to={622} rightInset={inset} sub="Shift Supervisor, clearance: internal">A confidential runbook stays confidential</Caption>
      <Caption from={670} to={770} rightInset={inset} sub="Finance Analyst, clearance: confidential">Finance asks about meal expenses</Caption>
      <Caption from={800} to={850} rightInset={inset} sub="Travel and Expense Policy, version 3, active">Always the current version</Caption>
    </AbsoluteFill>
  );
};

const EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame - 852;
  const p = spring({frame: t, fps, config: {damping: 14}});
  return (
    <AbsoluteFill style={{background: C.bg, color: '#fff', fontFamily: FONT, alignItems: 'center', justifyContent: 'center', opacity: interpolate(frame, [848, 862], [0, 1], clamp)}}>
      <div style={{transform: `scale(${0.92 + 0.08 * p})`, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <span style={{width: 150, height: 150, borderRadius: 40, background: C.accent, color: C.accentFg, display: 'grid', placeItems: 'center', marginBottom: 28}}><Drop size={92} /></span>
        <div style={{fontSize: 170, fontWeight: 800, letterSpacing: -4, lineHeight: 1.05}}>Askwell</div>
        <div style={{fontSize: 52, color: C.accentStrong, marginTop: 16, lineHeight: 1.3}}>The knowledge assistant that knows who is asking</div>
        <div style={{fontSize: 34, color: C.muted, marginTop: 34}}>Citations · Versions · Access control</div>
      </div>
    </AbsoluteFill>
  );
};

export const Launch2: React.FC = () => (
  <AbsoluteFill>
    <App />
    <EndCard />
  </AbsoluteFill>
);

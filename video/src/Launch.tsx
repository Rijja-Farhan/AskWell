import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Easing} from 'remotion';

export const DURATION = 750;

// Tokens copied from app/ui/style.css (light theme)
const C = {
  bg: '#f6f7f9', fg: '#1d2330', muted: '#5d6677', card: '#fff', line: '#d9dde5',
  accent: '#2455d6', user: '#e6edff', warn: '#8a5a00', warnBg: '#fff4d6', ok: '#166534', okBg: '#e3f6e8',
};
const FONT = 'system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const S = 1.45;
const T = 1.12; // top bar scale (long badge must fit on one line) // UI scale so text reads at 720p

const SAM = {name: 'Sam Okafor (u05, Temporary Warehouse Associate (contractor))', badge: 'Sam Okafor | Temporary Warehouse Associate (contractor) | clearance: public | departments: none'};
const DANA = {name: 'Dana Whitfield (u06, Chief Operating Officer)', badge: 'Dana Whitfield | Chief Operating Officer | clearance: restricted | departments: HR, IT, Operations, Finance'};
const Q = 'What is the salary range for grade G5?';
const REFUSAL = "I don't know. The company documents I can access do not answer this question. Please check with the relevant department.";
const HOTLINE_Q = 'What is the phone number for the ethics hotline?';
const HOTLINE_A = 'The phone number for the Nimbus Ethics Hotline is 1-800-555-0142.';

const EXAMPLES = ['Salary band G5 (restricted HR)', 'Parental leave (public)', 'Security incident steps (confidential IT)', 'Break-glass access (restricted IT)', 'Travel policy (versions)', 'Guest Wi-Fi (public)'];

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

const Tag: React.FC<{children: React.ReactNode; kind?: 'refused' | 'cached'}> = ({children, kind}) => (
  <span style={{
    borderRadius: 10 * S, padding: `${1 * S}px ${8 * S}px`, border: `1px solid ${kind === 'refused' ? C.warn : kind === 'cached' ? C.ok : C.line}`,
    color: kind === 'refused' ? C.warn : kind === 'cached' ? C.ok : C.muted, background: kind === 'cached' ? C.okBg : 'transparent', fontSize: 12 * S,
  }}>{children}</span>
);

const Chip: React.FC = () => (
  <span style={{padding: `0 ${6 * S}px`, margin: '0 2px', borderRadius: 10 * S, fontSize: 12 * S, color: C.accent, border: `1px solid ${C.accent}`, background: C.card}}>1</span>
);

const Bubble: React.FC<{user?: boolean; refused?: boolean; children: React.ReactNode; meta?: React.ReactNode; style?: React.CSSProperties}> = ({user, refused, children, meta, style}) => (
  <div style={{
    alignSelf: user ? 'flex-end' : 'flex-start', maxWidth: 780 * S, padding: `${10 * S}px ${14 * S}px`, borderRadius: 10 * S,
    border: `1px solid ${C.line}`, background: user ? C.user : refused ? C.warnBg : C.card,
    borderLeft: refused ? `${4 * S}px solid ${C.warn}` : undefined, fontSize: 15 * S, lineHeight: 1.5, ...style,
  }}>
    {children}
    {meta && <div style={{marginTop: 8 * S, display: 'flex', gap: 6 * S, alignItems: 'center', fontSize: 12 * S, color: C.muted}}>{meta}</div>}
  </div>
);

const pop = (frame: number, start: number, fps: number) => {
  const p = spring({frame: frame - start, fps, config: {damping: 11, stiffness: 160, mass: 0.7}});
  return {opacity: interpolate(frame - start, [0, 5], [0, 1], clamp), transform: `translateY(${(1 - p) * 40}px) scale(${0.94 + 0.06 * p})`};
};

const typed = (frame: number, start: number, text: string, cps = 1.0) => text.slice(0, Math.max(0, Math.floor((frame - start) * cps)));

const Caption: React.FC<{from: number; to: number; children: React.ReactNode; sub?: React.ReactNode}> = ({from, to, children, sub}) => {
  const f = useCurrentFrame();
  const op = interpolate(f, [from, from + 8, to - 8, to], [0, 1, 1, 0], clamp);
  const y = interpolate(f, [from, from + 10], [24, 0], {...clamp, easing: Easing.out(Easing.cubic)});
  if (f < from || f > to) return null;
  return (
    <div style={{position: 'absolute', left: 0, right: 0, bottom: 210, display: 'flex', justifyContent: 'center', opacity: op, transform: `translateY(${y}px)`}}>
      <div style={{background: 'rgba(18,21,28,0.92)', color: '#fff', borderRadius: 22, padding: '22px 44px', textAlign: 'center', boxShadow: '0 12px 40px rgba(0,0,0,.25)', maxWidth: 1500}}>
        <div style={{fontSize: 56, fontWeight: 700, letterSpacing: -0.5, lineHeight: 1.2}}>{children}</div>
        {sub && <div style={{fontSize: 32, marginTop: 8, color: '#b9c4e6', lineHeight: 1.25}}>{sub}</div>}
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const isDana = frame >= 240 && frame < 570;
  const isHotline = frame >= 570;
  const user = frame < 240 ? SAM : DANA;

  // composer text
  let composer = '';
  if (frame < 72) composer = typed(frame, 12, Q, 0.8);
  else if (frame >= 246 && frame < 300) composer = typed(frame, 246, Q, 0.8);
  else if (frame >= 575 && frame < 620) composer = typed(frame, 575, HOTLINE_Q, 1.1);

  const flash = interpolate(frame, [334, 336, 352], [0, 0.5, 0], clamp);
  const panelOpen = frame >= 450 && frame < 570;
  const panelX = interpolate(frame, [450, 468], [380 * S, 0], {...clamp, easing: Easing.out(Easing.cubic)});
  const switchPulse = interpolate(frame, [240, 244, 262], [1, 1, 0], clamp);

  return (
    <AbsoluteFill style={{background: C.bg, color: C.fg, fontFamily: FONT, flexDirection: 'column'}}>
      {/* top bar */}
      <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: `${10 * T}px ${16 * T}px`, background: C.card, borderBottom: `1px solid ${C.line}`}}>
        <div style={{fontSize: 18 * T, fontWeight: 700}}>Askwell <span style={{fontWeight: 400, color: C.muted, fontSize: 13 * T, marginLeft: 8}}>Nimbus Logistics knowledge assistant</span></div>
        <div style={{display: 'flex', alignItems: 'center', whiteSpace: 'nowrap', gap: 8 * T, fontSize: 14 * T}}>
          <span style={{color: C.muted}}>Viewing as</span>
          <span style={{border: `1px solid ${switchPulse > 0.01 ? C.accent : C.line}`, boxShadow: switchPulse > 0.01 ? `0 0 0 ${6 * switchPulse}px rgba(36,85,214,.25)` : 'none', borderRadius: 6, padding: `${4 * T}px ${8 * T}px`, background: C.card}}>{user.name} ⌄</span>
          <span style={{background: C.user, borderRadius: 12 * T, padding: `2px ${10 * T}px`, fontSize: 13 * T}}>{user.badge}</span>
          <span style={{border: `1px solid ${C.line}`, borderRadius: 6, padding: `4px ${10 * T}px`}}>New chat</span>
        </div>
      </div>

      <div style={{flex: 1, display: 'flex', minHeight: 0}}>
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', padding: `0 ${16 * S}px ${12 * S}px`, minWidth: 0}}>
          <div style={{flex: 1, display: 'flex', flexDirection: 'column', gap: 12 * S, paddingTop: 16 * S}}>
            {/* Sam thread */}
            {frame < 240 && frame >= 72 && <Bubble user style={pop(frame, 72, fps)}>{Q}</Bubble>}
            {frame < 240 && frame >= 100 && (
              <Bubble refused style={pop(frame, 100, fps)} meta={<><Tag kind="refused">refused: low_score</Tag><Tag>5440 ms</Tag></>}>{REFUSAL}</Bubble>
            )}
            {/* Dana thread */}
            {isDana && frame >= 300 && <Bubble user style={pop(frame, 300, fps)}>{Q}</Bubble>}
            {isDana && frame >= 330 && (
              <Bubble style={{...pop(frame, 330, fps), boxShadow: `0 0 0 ${flash * 14}px rgba(36,85,214,${flash * 0.5})`}} meta={<Tag>Trace</Tag>}>
                Grade G5 ranges from $72,000 to $94,000 <Chip />
              </Bubble>
            )}
            {/* Hotline thread */}
            {isHotline && frame >= 620 && <Bubble user style={pop(frame, 620, fps)}>{HOTLINE_Q}</Bubble>}
            {isHotline && frame >= 636 && <Bubble style={pop(frame, 636, fps)} meta={<Tag>Trace</Tag>}>{HOTLINE_A} <Chip /></Bubble>}
            {isHotline && frame >= 660 && <Bubble user style={pop(frame, 660, fps)}>{HOTLINE_Q}</Bubble>}
            {isHotline && frame >= 668 && <Bubble style={pop(frame, 668, fps)} meta={<><Tag kind="cached">cached</Tag><Tag>Trace</Tag></>}>{HOTLINE_A} <Chip /></Bubble>}
          </div>
          <div style={{display: 'flex', flexWrap: 'wrap', gap: 6 * S, paddingBottom: 8 * S}}>
            {EXAMPLES.map((e, i) => (
              <span key={e} style={{fontSize: 13 * S, border: `1px solid ${i === 0 && isDana && frame < 300 ? C.accent : C.line}`, borderRadius: 6, padding: `${4 * S}px ${10 * S}px`, background: C.card}}>{e}</span>
            ))}
          </div>
          <div style={{display: 'flex', gap: 8 * S, alignItems: 'flex-end'}}>
            <div style={{flex: 1, border: `1px solid ${composer ? C.accent : C.line}`, borderRadius: 6, background: C.card, padding: `${8 * S}px`, fontSize: 15 * S, minHeight: 52 * S, color: composer ? C.fg : C.muted}}>
              {composer || 'Ask a question (Enter to send, Shift+Enter for a new line)'}
              {composer && <span style={{opacity: Math.floor(frame / 8) % 2}}>|</span>}
            </div>
            <span style={{border: `1px solid ${C.line}`, borderRadius: 6, padding: `4px ${10 * S}px`, background: C.card, fontSize: 15 * S, color: C.muted}}>Send</span>
          </div>
        </div>
        {panelOpen && (
          <div style={{width: 360 * S, borderLeft: `1px solid ${C.line}`, background: C.card, padding: `${12 * S}px ${14 * S}px`, transform: `translateX(${panelX}px)`, fontSize: 15 * S}}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontWeight: 600}}>
              <span>Source [1]</span><span style={{border: `1px solid ${C.line}`, borderRadius: 6, padding: '2px 10px', fontWeight: 400}}>Close</span>
            </div>
            <h3 style={{margin: `${12 * S}px 0 ${8 * S}px`, fontSize: 18 * S}}>Salary Bands by Grade 2025</h3>
            <div style={{display: 'grid', gridTemplateColumns: 'auto 1fr', gap: `${4 * S}px ${10 * S}px`, fontSize: 13 * S}}>
              {[['Document', 'hr-tbl-001-v1'], ['Version', '1'], ['Status', 'active'], ['Effective', '2025-01-01']].map(([k, v]) => (
                <React.Fragment key={k}><span style={{color: C.muted}}>{k}</span><span>{v}</span></React.Fragment>
              ))}
            </div>
            <div style={{marginTop: 12 * S, padding: 8 * S, background: C.bg, borderRadius: 6, fontSize: 13 * S, border: `1px solid ${C.line}`}}>
              Access level: <b>restricted</b> (HR)
            </div>
          </div>
        )}
      </div>

      <Caption from={4} to={92}>Same question. Two employees.</Caption>
      <Caption from={108} to={236} sub="Clearance: public">Sam, a contractor: refused</Caption>
      <Caption from={338} to={446} sub="Clearance: restricted, with a citation">Dana, the COO: answered</Caption>
      <Caption from={470} to={566} sub="The model never sees what you can't read">Restricted chunks are filtered before ranking</Caption>
      <Caption from={676} to={742}>Cited. Cached. Honest when it can't answer.</Caption>
    </AbsoluteFill>
  );
};

const EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame - 700;
  const p = spring({frame: t, fps, config: {damping: 14}});
  return (
    <AbsoluteFill style={{background: '#12151c', color: '#fff', fontFamily: FONT, alignItems: 'center', justifyContent: 'center', opacity: interpolate(frame, [696, 712], [0, 1], clamp)}}>
      <div style={{transform: `scale(${0.9 + 0.1 * p})`, textAlign: 'center'}}>
        <div style={{fontSize: 180, fontWeight: 800, letterSpacing: -4, lineHeight: 1.1}}>Askwell</div>
        <div style={{fontSize: 52, color: '#9fb4f0', marginTop: 16, lineHeight: 1.3}}>The knowledge assistant that knows who is asking</div>
        <div style={{fontSize: 34, color: '#9aa3b5', marginTop: 36}}>Citations · Versions · Access control</div>
      </div>
    </AbsoluteFill>
  );
};

export const Launch: React.FC = () => (
  <AbsoluteFill>
    <App />
    <EndCard />
  </AbsoluteFill>
);

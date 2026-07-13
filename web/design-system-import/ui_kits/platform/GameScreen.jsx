const { Button, Badge, PlayingCard, PlayerSeat, ScorePill, TimerBar } = window.Cambia_cc8727;

const OPPONENTS = [
  { name: 'Maple', cards: 4, score: 12 },
  { name: 'Bram', cards: 3, score: 4 },
  { name: 'Sorrel', cards: 4, score: 9 },
];

function OpponentBlock({ name, cards, state }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <PlayerSeat username={name} compact state={state} handSize={cards} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', gap: 5 }}>
        {Array.from({ length: cards }).map((_, i) => <PlayingCard key={i} faceDown size="sm" />)}
      </div>
    </div>
  );
}

function GameScreen({ onExit }) {
  const [phase, setPhase] = React.useState('idle'); // idle | drawn | swapping | called
  const [drawn, setDrawn] = React.useState(null);
  const [discardTop, setDiscardTop] = React.useState({ rank: '4', suit: 'clubs' });
  const [hand, setHand] = React.useState([0, 1, 2, 3]);
  const [peeked, setPeeked] = React.useState(null);
  const [log, setLog] = React.useState([
    'Round 3 begins. Sorrel deals.',
    'Maple draws from the stockpile.',
    'Maple discards 4♣.',
  ]);
  const addLog = (line) => setLog((l) => [...l, line]);

  const draw = () => {
    if (phase !== 'idle') return;
    setDrawn({ rank: '7', suit: 'hearts' });
    setPhase('drawn');
    addLog('You draw from the stockpile.');
  };
  const discardForAbility = () => {
    setDiscardTop({ rank: '7', suit: 'hearts' });
    setDrawn(null);
    setPhase('idle');
    setPeeked(0);
    addLog('You discard 7♥ — Peek Own: you look at your top-left card.');
    setTimeout(() => setPeeked(null), 2500);
  };
  const startSwap = () => { setPhase('swapping'); };
  const swapInto = (i) => {
    if (phase !== 'swapping') return;
    setDiscardTop({ rank: '9', suit: 'clubs' });
    setDrawn(null);
    setPhase('idle');
    addLog('You swap the 7♥ into your hand and discard a 9♣.');
  };
  const snap = () => {
    if (hand.length <= 1) return;
    setHand(hand.slice(0, -1));
    addLog('Snap! You match the discard — one fewer card.');
  };
  const callCambia = () => {
    setPhase('called');
    addLog('You call Cambia. Everyone gets one last turn.');
  };

  const canAct = phase === 'idle';
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 18, padding: 18, width: '100%', maxWidth: 1320, margin: '0 auto', alignItems: 'stretch', flex: 1, minHeight: 0 }}>
      {/* Table */}
      <div style={{
        position: 'relative', display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
        background: 'var(--surface-table)', border: '3px solid var(--outline-ink)',
        borderRadius: 'var(--radius-xl)', boxShadow: 'var(--inset-table)', padding: '16px 20px', minHeight: 620,
      }}>
        {phase === 'called' && (
          <div style={{
            position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)', zIndex: 5,
            padding: '8px 18px', background: 'var(--berry-500)', color: 'var(--text-on-ember)',
            border: '2px solid var(--outline-ink)', borderRadius: 'var(--radius-pill)', boxShadow: 'var(--shadow-piece)',
            fontWeight: 800, whiteSpace: 'nowrap',
          }}>Cambia called — everyone gets one last turn.</div>
        )}
        {/* Opponents */}
        <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-start', paddingTop: phase === 'called' ? 34 : 4 }}>
          {OPPONENTS.map((o) => <OpponentBlock key={o.name} name={o.name} cards={o.cards} state={o.name === 'Maple' ? 'turn' : undefined} />)}
        </div>
        {/* Center piles */}
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-end', gap: 34, margin: '10px 0' }}>
          <div style={{ textAlign: 'center' }}>
            <PlayingCard faceDown size="md" onClick={draw} selected={phase === 'idle'} />
            <div style={{ marginTop: 7, fontSize: 10, fontWeight: 800, letterSpacing: '0.09em', color: 'rgba(243,236,218,0.75)' }}>STOCKPILE · <span style={{ fontFamily: 'var(--font-mono)' }}>31</span></div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <PlayingCard rank={discardTop.rank} suit={discardTop.suit} size="md" />
            <div style={{ marginTop: 7, fontSize: 10, fontWeight: 800, letterSpacing: '0.09em', color: 'rgba(243,236,218,0.75)' }}>DISCARD</div>
          </div>
          {drawn && (
            <div style={{ textAlign: 'center' }}>
              <PlayingCard rank={drawn.rank} suit={drawn.suit} size="md" selected />
              <div style={{ marginTop: 7, fontSize: 10, fontWeight: 800, letterSpacing: '0.09em', color: 'var(--honey-300)' }}>DRAWN</div>
            </div>
          )}
        </div>
        {/* Your area */}
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: 26 }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', gap: 6 }}>
              {hand.map((c, i) => (
                <PlayingCard
                  key={c}
                  faceDown={peeked !== i}
                  rank="8" suit="diamonds"
                  size="md"
                  selected={phase === 'swapping' || peeked === i}
                  onClick={() => swapInto(i)}
                />
              ))}
            </div>
            <PlayerSeat username="Juniper" isYou compact state={phase === 'called' ? 'cambia' : 'turn'} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: 240, paddingBottom: 4 }}>
            {phase === 'idle' && <Button onClick={draw}>Draw from stockpile</Button>}
            {phase === 'idle' && <Button variant="secondary" onClick={() => { setDrawn({ rank: discardTop.rank, suit: discardTop.suit }); setPhase('swapping'); addLog('You take the ' + discardTop.rank + ' from the discard pile.'); }}>Take discard</Button>}
            {phase === 'drawn' && <Button onClick={discardForAbility}>Discard — Peek Own</Button>}
            {phase === 'drawn' && <Button variant="secondary" onClick={startSwap}>Swap into hand</Button>}
            {phase === 'swapping' && (
              <div style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', background: 'rgba(223,174,71,0.12)', border: '1.5px solid var(--honey-600)', color: 'var(--honey-300)', fontWeight: 700, fontSize: 'var(--text-sm)' }}>
                Click one of your cards to swap.
              </div>
            )}
            <Button variant="secondary" onClick={snap}>Snap</Button>
            {phase !== 'called' && <Button variant="cambia" onClick={callCambia}>Call Cambia</Button>}
            <TimerBar totalSec={30} remainingSec={phase === 'called' ? 30 : 19} />
          </div>
        </div>
      </div>

      {/* Rail */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
        <window.Panel title="Circuit · round 3 of 8">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {[{ n: 'Bram', s: 4, sub: '−5' }, { n: 'Sorrel', s: 9, sub: '−2' }, { n: 'Maple', s: 12, sub: null }, { n: 'Juniper', s: 14, sub: null, you: true }].map((p, i) => (
              <div key={p.n} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', width: 16 }}>{i + 1}</span>
                <span style={{ fontWeight: 700, flex: 1, color: p.you ? 'var(--honey-400)' : 'var(--text-primary)' }}>{p.n}{p.you ? ' (you)' : ''}</span>
                {p.sub && <Badge tone="success" mono>{p.sub}</Badge>}
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{p.s}</span>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <ScorePill label="Round" value="3/8" />
            <ScorePill label="Pace" value="~55m" />
          </div>
        </window.Panel>
        <window.Panel title="Game log" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 7 }}>
            {log.map((l, i) => (
              <div key={i} style={{ fontSize: 'var(--text-xs)', lineHeight: 1.45, color: i === log.length - 1 ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)', marginRight: 6 }}>{String(i + 1).padStart(2, '0')}</span>{l}
              </div>
            ))}
          </div>
          <Button size="sm" variant="ghost" onClick={onExit} style={{ marginTop: 12 }}>Leave table</Button>
        </window.Panel>
      </div>
    </div>
  );
}

window.GameScreen = GameScreen;

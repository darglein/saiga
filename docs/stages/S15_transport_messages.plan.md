# S15: Transport and messages

Status: done.

Sections follow the stage plan template of S05.

## Goal

Worlds can talk to each other: a `Transport` interface with channels and reliability, a deterministic in-process
`LoopbackTransport` (delay, seeded loss, a per-link bandwidth limit, fragmentation), registered network messages
(`MessageList`, `world.send`) coalesced into packets and delivered to their handlers in phase 1 in a deterministic
order. No replication yet (S16): this stage is the pipe and the message mechanism that S16 (snapshots, spawns, owner
commands), S22 (live terrain edits), S26 (chunks) and S33 (UDP / Steam transports) build on.

## Depends on

S08 (`World`, phases, commands, events, the aborted step), S07 (`visit`, the binary visitors), S06 (`Rng`, `Hash`,
`SAIGA_LOG`), S10 (profiler counters, `world_status`).

## Rules that apply

R-REG-1 (messages in explicit lists, IDs from an explicit enum; game messages from 64), R-REG-2 (names only at the
boundary), R-DET-1 (loopback delivery and loss are reproducible from the seed; no wall clock), R-PAR-1 (no sends from a
`parallelEach` body), R-SYS-2, R-FILE-1, R-TEST-1, R-TEST-2 (`engine/net/` is performance relevant: the message path
gets a benchmark), R-DOC-1/2, AGENTS.md (export macros, MSVC).

## Scope

### 1. Transport (`engine/net/Transport.h`)

As §13.1, with these decisions:

```cpp
enum class Reliability : uint8_t { Unreliable, Reliable };
enum class Channel : uint8_t { Control, State, ChunkData };
constexpr size_t kMaxPacket = 1200;   // bytes of an unreliable packet and of a fragment incl. its header (§13.4)
constexpr size_t kMaxPacketHeader = 12, kMaxMessageHeader = 14;   // worst-case framing (varints at their maximum)
constexpr size_t kMaxMessagePayload = kMaxPacket - kMaxPacketHeader - kMaxMessageHeader;   // 1174: an unreliable
                                      // message's payload always fits, at any sequence number
constexpr size_t kMaxReliablePacket = 64u << 20;   // 64 MB transport packet, checked on send in every build
constexpr size_t kMaxReliablePayload = kMaxReliablePacket - kMaxPacketHeader - kMaxMessageHeader;   // checked by
                                      // send / sendRaw before anything is read or copied
struct TransportEvent { enum Type : uint8_t { Connected, Disconnected, Packet } type; uint8_t peer; Channel channel;
                        std::vector<uint8_t> packet; };
struct TransportStats   // cumulative since creation
{
    std::array<uint64_t, 3> bytesOut{}, bytesIn{}, packetsOut{}, packetsIn{};   // per channel (payload bytes)
    uint64_t fragmentsOut = 0, fragmentsIn = 0;
    uint64_t droppedLoss = 0;       // unreliable packets lost by the link model (or the network)
    uint64_t droppedBandwidth = 0;  // unreliable packets over the link budget
    uint64_t droppedNoPeer = 0;     // sent to a peer that is not connected
    uint64_t rejectedInput = 0;     // malformed fragments or packets from the wire
};
class SAIGA_ENGINE_API Transport
{
   public:
    virtual ~Transport() = default;
    virtual uint8_t localPeer() const = 0;
    virtual void send(uint8_t peer, Channel channel, const std::vector<uint8_t>& packet, Reliability r) = 0;
    virtual bool poll(TransportEvent& event) = 0;    // non-blocking; false: nothing more this round
    virtual TransportStats stats() const = 0;
};
```

- Reliable packets are delivered exactly once and in send order per (sender, channel); channels are independent (a
  large `ChunkData` packet never delays `Control`). Unreliable packets may be lost, are never duplicated, and in S15
  arrive in order (reordering is S32). An unreliable packet over `kMaxPacket` stops the program in every build (a
  programming error of the sender).
- Sending to a peer that is not connected (never connected, or disconnected within the round) drops the packet and
  counts it (`droppedNoPeer`); no assertion: it is a normal race once peers leave (S16 snapshots, S33).
- When an end goes away, the packets it sent before are delivered before its `Disconnected`.

### 2. `LoopbackTransport` (`engine/net/LoopbackTransport.{h,cpp}`, `engine/net/Fragmenter.{h,cpp}`)

```cpp
struct LoopbackOptions { double dropRate = 0; int delayRounds = 0; uint64_t seed = 1; size_t bytesPerRound = 0; };
// n connected peers in one process (peer 0 is the host); createPair = createNetwork(2)
static std::vector<std::unique_ptr<Transport>> LoopbackTransport::createNetwork(int peers, LoopbackOptions = {});
static std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> LoopbackTransport::createPair(
    double dropRate = 0, int delaySteps = 0, uint64_t seed = 1);   // §13.1's signature
```

- Single-threaded: the hub and its ends are used from one thread (the network thread is S33's).
- Topology: a star, as the S33 transports: clients connect to peer 0 only (a client's `Connected` events and
  `connectedPeers()` contain only the host; a send from a client to another client is a send to a peer that is not
  connected). A shared hub owned by the ends (`shared_ptr`): destroying one end removes it from the hub (its peers
  receive `Disconnected` after its earlier packets); the S33 `disconnect(peer)` test hook is added then. The transport's
  lifetime is independent of the worlds that use it (non-owning `WorldSetup::transport`).
- Rounds: a receiver's round starts with its first `poll` after the previous round ended and ends when `poll` returns
  false. A packet sent before round r starts is visible from round r + `delayRounds`; one sent while a round is being
  polled belongs to the next. With one round per step (the world polls in phase 1), the delay is in the
  receiver's steps (with asynchronous stepping, §17.3, rounds differ from the sender's steps).
  `Connected` for every other peer is the first event each end polls.
- Loss: drawn at send time, one draw per unreliable packet, from a per-link stream `Rng(hash(seed, sender, receiver,
  "loss"))`, so a run is reproducible independent of other links' traffic; S32's reordering gets its own stream so
  S15 loss patterns stay stable. Reliable packets are never dropped.
- Bandwidth (0 = unlimited) per link (sender → receiver; for the host's star this is the per-client budget of S32),
  applied in the delivery round (after the delay) as a token bucket: each round adds `bytesPerRound` to the link's
  budget, capped at `max(bytesPerRound, kMaxPacket)`, so any packet or fragment passes within a few rounds (the cap
  guarantees progress; nothing bypasses the budget). Order per round: pending unreliable packets first (in send order;
  one that does not fit the budget is dropped and counted, `droppedBandwidth`); then reliable data round-robin between
  the channels with pending data, fragment by fragment, each only when the budget covers it; reliable data may not
  spend budget in a round in which an unreliable packet was dropped for budget, so the bucket refills for the next
  unreliable packet and a chunk transfer never starves the snapshots (S26, S32).
- Fragmentation (the loopback's `Fragmenter`; S33 can reuse it for messages above a backend's maximum, the fragment
  layout is not part of the S33 protocol): reliable packets over `kMaxPacket` are split into fragments of at most
  `kMaxPacket` bytes including a 12-byte header (packet sequence, index, count) and reassembled before delivery; at most
  `kMaxReliablePacket` per packet (checked on send in every build). The reassembler validates every header as untrusted
  input: `index < count`, `count <= ceil(kMaxReliablePacket / (kMaxPacket - 12))`, reassembled bytes <=
  `kMaxReliablePacket`, the same `count` for one packet sequence, duplicates and out-of-order fragments, and at most
  `kMaxReliablePacket` bytes and 64 packets in incomplete reassemblies per (peer, channel). The reassembler returns an error for a bad fragment (never
  asserts); the transport counts it (`rejectedInput`) and, since it breaks that channel's ordered stream, removes the
  sending peer's link (both ends receive `Disconnected`). A test-only `LoopbackTransport::injectFragment(from, to,
  channel, bytes)` feeds hostile fragments.

### 3. Messages (`engine/net/MessageList.{h,cpp}`, `engine/net/NetIds.h`, `World`)

```cpp
using MessageId = uint16_t;
enum class EngineMessage : MessageId { /* engine messages, 0..63; S16 adds its own */ FirstGame = 64 };
template <class M> void MessageList::add(Reliability r, void (*handler)(World&, uint8_t fromPeer, const M&));
// M: struct with static constexpr MessageId id, const char* name, and visit (the binary visitors); the channel is
// Control unless M declares static constexpr Channel channel
void registerEngineMessages(MessageList&);   // empty in S15
template <class M> void World::send(uint8_t peer, const M& message);
// Raw messages: a payload the caller serialized itself (S16 snapshots, spawns and owner states: serialized once for all
// clients, zstd blobs; S26 chunks), no visit encoding and no per-byte loop
void MessageList::addRaw(MessageId id, const char* name, Reliability r, Channel channel,
                         void (*handler)(World&, uint8_t fromPeer, const uint8_t* data, size_t size));
void World::sendRaw(uint8_t peer, MessageId id, const uint8_t* data, size_t size);
// VisitBinary.h: decode from a range (the payload inside a packet, no copy)
template <class T> bool fromBytes(T& value, const uint8_t* data, size_t size, std::string* error = nullptr);
```

- Registration (R-REG-1): engine list first, then `WorldSetup::messages`; duplicate IDs or names, a game ID below 64,
  an engine ID at or above 64, and an unreliable type whose minimum encoded size (`MinSizeVisitor`) exceeds
  `kMaxMessagePayload` are fatal at world construction (`SAIGA_EXIT_ERROR`, death tests). The minimum is a lower bound:
  an unreliable message with strings or vectors can still exceed `kMaxMessagePayload` at run time, and `send` /
  `sendRaw` then stop the program (stated in the `MessageList::add` comment); a reliable payload is limited by
  `kMaxReliablePacket`.
- Sending: `world.send` asserts a registered type (every build), a transport, a peer other than the transport's own
  (`transport->localPeer()`; `World::localPeer()` is 0 on every end until S16), and no `parallelEach` body (R-PAR-1,
  `SAIGA_ASSERTS`). During a load's object start (`configure`/`onLoad` after the commit, `loadHookPhase != 0`) a send
  is dropped and logged in every build and asserts with `SAIGA_ASSERTS`: the hooks rebuild local state from the save,
  telling peers about the load is the load listeners' job (S16's `Resync`), and they may send. Sends from `GameObject`
  hooks between steps (`onDestroy` of an API destroy, `onSpawn` of an API spawn) are allowed; the world's destructor
  never touches the transport (unflushed entries are discarded). The payload is encoded at once (a serialization error stops the program, as
  `toBytes`) into the world's outbox, tagged with the current phase and given the next sequence number of this
  sender → receiver pair in send order (u64).
- Flush (`World::flushOutbox()`, public; S16's `stepNetworkOnly` reuses it): at the end of the step (after phase 8);
  a send between steps flushes at once, except inside a batching scope (a depth counter: a world API call, a step
  listener dispatch, a load), which flushes once when it ends (also when it ends with an exception). Messages are coalesced per (peer,
  channel, reliability), keeping send order within a group, into packets of at most `kMaxPacket` bytes: `[base
  sequence varint][count varint]` then per message `[sequence delta varint (to the previous message, the first to the
  base)][id u16][length varint][payload]`; the
  sequences are the send-order ones (across channels), so the receiver's (peer, sequence) order is the sender's send
  order whatever the grouping (a `Spawn` sent before a `Snapshot` is handled first when both arrive in one round). A
  single reliable message larger than `kMaxPacket` is one packet that the transport fragments. A dropped outbox entry
  leaves a gap in the sequences; the receiver uses sequences only for ordering (duplicates or regressions from the wire
  are not validated in S15). §13.4's snapshot payload is at most `kMaxMessagePayload` (S16 and the §13.4 text use it).
- Varints (`engine/net/Varint.h`, reused by S16's zigzag quantization): unsigned LEB128, at most 10 bytes. The reader
  rejects over-long input, overflow bits, `count == 0`, `count` above the remaining bytes / 4 (the smallest message), a wrapping `base +
  delta`, and a length beyond the packet; a bad message rejects the whole packet.
- Receive (`World::receiveMessages()`; reused by `stepNetworkOnly`): at the start of phase 1 the world polls the
  transport until the round ends, updates `connectedPeers()` and the `peersJoined()` / `peersLeft()` lists first (replaced at the next receive, so
  step listeners after the step still see them),
  decodes the packets into a batch kept in the world (after a leftover of an aborted step, which comes first) and runs
  the handlers in (sender peer, sender sequence) order, before the phase's systems. Within a round this order is independent of arrival order; across rounds only the order
  per (channel, reliability) holds (a `Control` message can be handled steps before an earlier-sent `ChunkData`
  message; under a budget an unreliable `State` packet can overtake an earlier reliable one), and
  determinism is a property of the loopback (S26 relies on its own modification sequence numbers). Handlers may write
  components, make structural changes and emit events; the world flushes the commands after each handler (as during a
  flush, so a later handler of the batch can use an entity an earlier one created), and a message counts as consumed
  only together with its applied changes.
- Aborted step (an exception in a phase, `World::abortStep`): the messages handled before the throwing one are
  consumed with their applied changes; the throwing message is consumed too and its partial commands dropped (a poison
  message cannot abort every later step; it is counted in the world's `NetStats::handlerExceptions`); the batch's messages
  not yet handled stay in the world and are handled first in the next step. Events a handler emitted are lost with the
  step's events (`EventQueue::abortStep` drops all of the step's events); a handler whose effect must happen exactly
  once writes components or engine queues (S16 puts `PlayerAction` into the action queue). `peersJoined()` /
  `peersLeft()` are kept until a step completes. The outbox drops the entries of the interrupted phase in place (no
  allocation in `abortStep`) and keeps those of finished phases (like their commands).
- Saves and loads: the kept batch and the outbox are neither saved nor hashed (like commands). A load clears the
  outbox after the object start and before the load listeners (`WorldLoad.cpp`: after `startObjects`; the entries were
  encoded from the replaced state; counted in `NetStats::droppedOnLoad`), so the listeners' sends go out; a refused
  `tryLoad` leaves the outbox untouched. The inbound leftover is kept (the peers' data).
- Malformed input (a packet or message that does not decode, an unknown message ID, a length beyond the packet) is
  dropped and counted (`net/rejected`); it never stops the program (the bytes come from outside the process). The `net`
  log gets at most 8 lines per peer and step for it.
- Peers: `World::connectedPeers()`, `peersJoined()`, `peersLeft()` (this step's changes; not events, so nothing about
  the transport is saved or hashed, and `data/tests/engine/schema.json` does not change). A host can greet a new peer
  in the same step.
- `WorldSetup::transport` (a non-owning `Transport*`, §7.9) and `WorldSetup::messages`. The transport is not saved; a
  load keeps it. `World::localPeer()` stays 0 and `isHost()` true until S16 (roles); handlers' `fromPeer` and
  `world_status.net.peer` are the transport's peer numbers.
- `NetStats` (cumulative in the world, not reset by aborts): messages in and out, bytes in and out, rejected (malformed
  packets, unknown IDs, bad varints), handler exceptions, dropped on load, dropped during object start. Profiler
  counters are the per-step deltas: `net/sent_bytes`, `net/received_bytes`, `net/messages_in`, `net/messages_out`,
  `net/rejected` (§17.1; sends between steps count in the next step's sample; an aborted step's deltas go with its
  sample, `NetStats` keep them; S32 checks the per-client budget against them).

### 4. World API

`world_status` gains `net: {peer, connected: [peers]}` plus the world's `NetStats` and the transport's
`TransportStats` (both cumulative) when the world has a transport; `describe_world` lists the registered messages with ID, reliability and channel.

## Out of scope

Replication, `NetOwner`, roles and `RunOn` filtering, `queueAction` as a message (S16), reordering (S32), UDP / Steam
transports, sessions, handshake and `disconnect(peer)` (S33), chunk transfer (S26), compression (S16 for snapshots),
message versions in the schema hash (S26 adds messages to the handshake's schema table), message tools over MCP
(scenarios call `world.send`).

## Design notes

- A hub of n peers instead of only pairs: S16 tests one client, S32 measures seven, S40 needs a driver and a passenger
  on different clients. The pair API of §13.1 stays as a shortcut.
- Coalescing: one transport packet per message would mean thousands of datagrams per step on S33's UDP; the framing is
  fixed now, before S16 depends on it.
- Rounds instead of time: the loopback has no clock; a world polls once per step, so delays are in steps, as §13.1
  states.
- `queueAction` as a message (§7.10 "S15: a message", §17.2 `input`) moves to S16: before S16 every world is the host
  (`isHost()` is true), so there is no peer to send an action to; S16 registers `PlayerAction` as an engine message
  when it adds roles. The §7.10 and §17.2 markers change to S16.
- Malformed input is not an assertion: the payload is outside the process (S33), and the trust model trusts clients'
  intent, not the wire.

## Tests

`tests/test_engine_net.cpp` (new, registered in `tests/CMakeLists.txt` inside `if (MODULE_ENGINE)`):
- Loopback: `Connected` first; reliable order per channel; channels independent under `bytesPerRound` (a 1 MB
  `ChunkData` packet does not delay a `Control` packet sent after it); `bytesPerRound` below 1200 still delivers one
  fragment per round; unreliable packets over the budget dropped and counted; delay in rounds; seeded loss reproducible
  and only for unreliable packets (two runs give the same delivered set; a different seed a different one; traffic on
  another link does not change a link's pattern); fragmentation at 1200, 1201 and a 4 MB round trip, and `sendRaw` with a size of
  `kMaxReliablePayload + 1` stops before reading the buffer (death test with a small buffer, no 64 MB allocation); with `bytesPerRound` 400, an 1100-byte unreliable
  packet every round and a 100 KB reliable chunk, every third snapshot arrives and the chunk completes; a packet larger than one round's budget passes through the
  token bucket; an unreliable packet over 1200 bytes stops (death test); three peers in a star (clients see only the host; a client-to-client send is dropped as
  no peer);
  destroying an end delivers its earlier packets, then `Disconnected`; sending to a disconnected peer drops and counts;
  stats.
- Fragmenter with hostile headers: index out of range, oversized count, changing count, duplicates, out of order,
  incomplete reassemblies over the cap: the reassembler's errors (unit test), and through
  `injectFragment` the count and the disconnected link (both ends `Disconnected`). Varints: over-long, overflow, `count == 0`,
  wrapping sequences, lengths beyond the packet: the whole packet rejected.
- Messages: registration errors (death tests); coalescing (many small messages in few packets); interleaved channels
  and reliabilities handled in send order; raw messages; an unreliable payload of `kMaxMessagePayload` fits at a
  sequence above 2^56; two worlds over a pair: a message sent in step n is handled in the receiver's step n (delay 0, sender
  stepped first) or n + d; (peer, sequence) order with three peers; a handler's structural change is applied before the next handler (a
  later handler uses the entity an earlier one created) and its event arrives like a system's; a plain send between
  steps flushes at once, sends inside a world API call or a listener dispatch flush once at its end (also when it
  throws); asserts (no transport, own peer, `parallelEach`, during a load; death
  tests with `SAIGA_ASSERTS`); malformed packets and unknown IDs dropped, counted, the log capped; `peersJoined` and
  `peersLeft`; counters; `world_status` and `describe_world`.
- Aborted step: a handler throws (it and the earlier ones consumed, the rest handled in the next step, exactly once);
  a handler emits an event and phase 5 throws (the event is lost, documented); a join seen in an aborted step is still
  in `peersJoined()` in the next one; a system throws after a `send` in its phase (that message is not sent) and after
  a `send` in an earlier phase (that one is). Load: a load clears the outbox before its listeners (counted), a listener's
  send goes out, a refused `tryLoad` keeps the outbox, the inbound leftover is kept; a send from `onLoad` is dropped
  and logged; `NetStats` survive an aborted step.
- World-level determinism: two worlds exchanging messages with loss, run twice: equal `world.hash()` on both ends
  across the runs.
- Benchmark (`tests/test_engine_net_benchmark.cpp`, a separate non-gating executable like the ECS benchmark, printed):
  10^4 small messages per step through a pair (packets per step reported) and one 16 MB raw reliable message with a
  bandwidth limit; reported in the PR (R-TEST-2).

## Done when

- The S15 row of §22: loopback tests for channels, ordering, delay and loss, fragmentation pass (CI).
- `docs/engine_design.md` updated: §13.1 (the network API as implemented: `createNetwork`, rounds, per-link bandwidth,
  `TransportStats`), §13.4 (the snapshot packet inside the message framing), §13.6 (wire format, coalescing, raw messages, delivery
  order and its limits, aborted steps, loads, peers), §7.9 (`messages`, `transport` exist), §7.10 (`send`, the peer lists;
  `queueAction` "S16: a message"), §17.2 (`input`: S16), §8.1 (phase 1 receive with the per-round order limit in row 1, flush after phase 8), §17.1 (net
  counters), §18.2, §3 (a row), §22 (S15 done); `AGENTS.md` (`engine/net/` in the layout, the new test).

## Verification

The CI recipe of AGENTS.md (re-run the configure: the engine sources are globbed); `ctest -R test_engine_net
--output-on-failure`; the full ctest run.

## Files

`src/saiga/engine/net/{Transport.h,NetIds.h,Varint.h,Fragmenter.{h,cpp},LoopbackTransport.{h,cpp},MessageList.{h,cpp}}`,
`src/saiga/engine/ecs/VisitBinary.h` (`fromBytes` from a range), `src/saiga/engine/world/World.{h,cpp}`,
`src/saiga/engine/api/WorldApi.cpp`, `tests/test_engine_net.cpp`, `tests/test_engine_net_benchmark.cpp`, `tests/CMakeLists.txt`, `docs/engine_design.md`,
`AGENTS.md`, this file.

## Design changes

- Bandwidth: with saturating unreliable traffic (snapshots demanding more than the budget) the rule "reliable data
  does not spend in a round in which an unreliable packet was dropped for budget" starved the reliable streams forever;
  every 4th round of a link the reliable streams now go first (a quarter of a saturated link). Found by the planned
  test (a 1100-byte packet every round under 400 bytes per round and a 20 KB chunk).
- Message structs are visit structs, so they need `version` next to `id` and `name` (the binary reader's rule).
- `World::send` stops the program in every build for an unregistered type, a world without a transport and the own
  peer; `parallelEach` is the existing assert.

## Results

- `tests/test_engine_net.cpp`: 25 tests (varints; the loopback: channel order, delay, seeded and per-link loss,
  fragmentation at 1200 / 1201 / 4 MB, the oversized-packet death test, channel independence and the budget below a
  packet, snapshots during a chunk transfer, the star and disconnects, hostile fragments and the injected bad fragment;
  messages: registration errors, delivery, raw messages, coalescing and send order across channels, (peer, sequence)
  order with three peers, a handler's structural change before the next handler, malformed input, aborted steps and
  phases, loads and the outbox, the send death tests incl. `onLoad`, the world API's net fields, determinism with loss).
- `tests/test_engine_net_benchmark.cpp` (non-gating): 10^4 small messages per step through a pair in 1.3 ms per step and
  102 packets per step; a 16 MB raw reliable message at 1 MB per round in 17 steps (52 ms) (RelWithDebInfo, local).
- Full ctest on lavapipe passes except the known local `test_opengl_window` display check.

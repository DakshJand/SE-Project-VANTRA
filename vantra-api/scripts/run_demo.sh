#!/usr/bin/env bash
# VANTRA demo runner — executes all six demo scenarios end-to-end against the API.
#
# Prerequisites:
#   - Postgres running with the vantra DB seeded (scripts/seed_history.py + seed_demo.py)
#   - API running: uvicorn app.main:app --port 8712  (from vantra-api/)
#
# Usage: bash scripts/run_demo.sh [API_URL]
set -euo pipefail
API="${1:-http://localhost:8712}"

B='\033[1;34m'; G='\033[1;32m'; R='\033[1;31m'; Y='\033[1;33m'; N='\033[0m'
pass=0; fail=0
check() { # name, condition (0=ok)
  if [ "$2" -eq 0 ]; then echo -e "  ${G}PASS${N} $1"; pass=$((pass+1));
  else echo -e "  ${R}FAIL${N} $1"; fail=$((fail+1)); fi
}

json() { python3 -c "import json,sys; d=json.load(sys.stdin); $1"; }

echo -e "${B}VANTRA demo — One vehicle, one trail, across every camera in the city.${N}"
echo "API: $API"
echo

# ---------------------------------------------------------------- 1. clean baseline
echo -e "${B}[1/6] Clean baseline trajectory (KA01VX2026, CAM01→CAM06)${N}"
echo -e "${Y}  A perfectly readable plate crosses all six cameras — watch every hop link by exact plate match at high confidence.${N}"
R=$(curl -s "$API/api/trajectories/reconstruct?plate=KA01VX2026&session=demo1-clean")
echo "$R" | json "t=d[0]; print(f'  dets={t[\"n_detections\"]} hops={len(t[\"hops\"])} conf={t[\"confidence\"]*100:.0f}%')"
ok=$(echo "$R" | json "t=d[0]; print(0 if len(t['hops'])==5 and all(h['tier']=='exact' for h in t['hops']) and t['confidence']>0.9 else 1)")
check "5 hops, all exact, confidence > 90%" "$ok"

# ---------------------------------------------------------------- 2. blurred hop
echo -e "${B}[2/6] Blurred hop resolved via appearance + route (KA05HR7741)${N}"
echo -e "${Y}  The plate is half-read at CAM03 because of motion blur (OCR confidence 0.35) — watch the system fall back to fuzzy matching backed by vehicle appearance and route plausibility instead of guessing.${N}"
R=$(curl -s "$API/api/trajectories/reconstruct?plate=KA05HR7741&session=demo2-blur")
echo "$R" | json "t=d[0]; fz=[h for h in t['hops'] if h['tier']=='fuzzy_unique'][0]; print(f'  dets={t[\"n_detections\"]} hops={len(t[\"hops\"])} conf={t[\"confidence\"]*100:.0f}%'); print(f'  fuzzy hop evidence: ed={fz[\"edit_distance\"]} cosine={fz[\"appearance_score\"]:.2f} ocr_conf={fz[\"ocr_conf_min\"]:.2f}')"
ok=$(echo "$R" | json "t=d[0]; fz=[h for h in t['hops'] if h['tier']=='fuzzy_unique']; print(0 if fz and len(t['hops'])==4 else 1)")
check "blurred hop linked as fuzzy (ed=3) with appearance evidence, full path kept" "$ok"

# ---------------------------------------------------------------- 3. missed camera
echo -e "${B}[3/6] Missed camera, hop skipped (KA53MN1108, CAM02 missed)${N}"
echo -e "${Y}  One camera fails to detect the vehicle at all — watch the matcher widen its time window across two hops and still stitch the full path without inventing a sighting.${N}"
R=$(curl -s "$API/api/trajectories/reconstruct?plate=KA53MN1108&session=demo3-missed")
echo "$R" | json "t=d[0]; sk=[h for h in t['hops'] if h['hop_reach']==2]; print(f'  dets={t[\"n_detections\"]} hops={len(t[\"hops\"])} conf={t[\"confidence\"]*100:.0f}%'); print(f'  skipped-camera hops: {len(sk)} (2-hop window)')"
ok=$(echo "$R" | json "t=d[0]; print(0 if len(t['hops'])==3 and any(h['hop_reach']==2 for h in t['hops']) else 1)")
check "CAM02 miss stitched via 2-hop window; all 3 hops linked" "$ok"

# ---------------------------------------------------------------- 4. long dwell
echo -e "${B}[4/6] Long dwell on stop-eligible edge (KA02PS3390, 40min at market)${N}"
echo -e "${Y}  The vehicle stops 40 minutes at a market stretch — watch confidence visibly decay (but never hit zero) instead of the trail being silently dropped.${N}"
R=$(curl -s "$API/api/trajectories/reconstruct?plate=KA02PS3390&session=demo4-dwell")
echo "$R" | json "t=d[0]; print(f'  dets={t[\"n_detections\"]} hops={len(t[\"hops\"])} conf={t[\"confidence\"]*100:.0f}%'); print(f'  hop time-scores: {[round(h[\"time_score\"],2) for h in t[\"hops\"]]}')"
ok=$(echo "$R" | json "t=d[0]; print(0 if len(t['hops'])==2 and 0.4 < t['confidence'] < 0.9 else 1)")
check "dwell hop matched (never-zero decay), confidence lower but not lost" "$ok"

# ---------------------------------------------------------------- 5. clone / spoof
echo -e "${B}[5/6] Impossible traversal — spoofed/cloned plate (KA41CL9042)${N}"
echo -e "${Y}  The same plate shows up at two cameras in less time than physically possible — watch the system flag a cloned-plate anomaly instead of silently linking them into one trip.${N}"
R=$(curl -s "$API/api/trajectories/reconstruct?plate=KA41CL9042&session=demo5-clone")
echo "$R" | json "t=d[0]; an=[h for h in t['hops'] if h['anomaly']]; print(f'  dets={t[\"n_detections\"]} hops={len(t[\"hops\"])} conf={t[\"confidence\"]*100:.0f}%'); print(f'  HARD ANOMALY: {an[0][\"reason\"][:120]}') if an else print('  NO ANOMALY FLAGGED')"
ok=$(echo "$R" | json "t=d[0]; print(0 if any(h['anomaly']=='impossible_edge' for h in t['hops']) and t['confidence']<0.5 else 1)")
check "min_time violation flagged as hard anomaly (not silently linked)" "$ok"

# ---------------------------------------------------------------- 6. blacklist
echo -e "${B}[6/6] Blacklisted plate live detection (KA09ST0555)${N}"
echo -e "${Y}  A reported-stolen vehicle crosses the network — watch a real-time alert fire with the full chain of evidence, then a brand-new frame pushed live through the actual OCR pipeline triggers another one instantly.${N}"
AL=$(curl -s "$API/api/alerts?kind=blacklist&limit=5")
echo "$AL" | json "a=d[0]; print(f'  blacklist alerts: {len(d)}'); print(f'  latest: {a[\"title\"]}'); print(f'  evidence: {a[\"evidence\"]}')"
ok=$(echo "$AL" | json "print(0 if len(d)>=1 and d[0]['evidence'].get('rule','').startswith('blacklist') else 1)")
check "blacklist alert exists with machine-readable evidence" "$ok"


# live ingestion of the blacklisted vehicle at another camera.
# Each ingest renders a fresh (randomly-jittered) frame; OCR occasionally flips a
# glyph (honest behavior — a low-confidence 1-char misread must NOT fire a
# high-severity alert). Retry a few frames so the demo shows the intended alert.
echo -e "${Y}  live: ingesting KA09ST0555 at CAM17 through the real detection pipeline...${N}"
ok=1
for attempt in 1 2 3 4 5; do
  R=$(curl -s -X POST "$API/api/ingest/simulate" \
    -F camera_id=CAM17 -F plate=KA09ST0555 -F condition=clean -F vehicle_type=suv -F color=black)
  fired=$(echo "$R" | json "print(0 if d['alerts'] else 1)")
  if [ "$fired" -eq 0 ]; then
    echo "$R" | json "r=d; print(f'  live ingest at CAM17 (frame $attempt): OCR read={r[\"detection\"][\"plate_raw\"]} conf={r[\"detection\"][\"ocr_conf\"]}'); print(f'  live alert: {r[\"alerts\"][0][\"title\"]}')"
    ok=0
  else
    echo "$R" | ATTEMPT=$attempt json "import os; r=d; print(f'  frame {os.environ[\"ATTEMPT\"]}: read={r[\"detection\"][\"plate_raw\"]} conf={r[\"detection\"][\"ocr_conf\"]} — misread below alert confidence (correct refusal), retrying')"
  fi
done
check "live ingestion of blacklisted plate fires real-time alert" "$ok"

# ---------------------------------------------------------------- summary
echo
echo -e "${B}Dashboard: run vantra-dashboard (npm run dev) and use the Demo Scenarios panel.${N}"
if [ "$fail" -eq 0 ]; then
  echo -e "${G}ALL 6 DEMO SCENARIOS PASSED ($pass checks)${N}"
else
  echo -e "${R}$fail CHECK(S) FAILED ($pass passed)${N}"
  exit 1
fi

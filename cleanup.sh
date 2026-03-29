#!/bin/bash
# Kill any running lab servers and clear cached ChromaDB data.
# Run this if you need a fresh start.

echo "[*] Stopping lab servers..."
pkill -f "rag_server" 2>/dev/null
lsof -ti:8899 | xargs kill 2>/dev/null
sleep 1

echo "[*] Clearing ChromaDB caches..."
rm -rf /tmp/ragdrag-lab-open /tmp/ragdrag-lab-guarded 2>/dev/null

echo "[+] Clean. Ready to start fresh."

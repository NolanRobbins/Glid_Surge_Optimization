#!/bin/bash
# Quick copy script - run from Glid_Surge_Optimization directory

cd /Users/cristianhenao/development/Glid_Surge_Optimization || exit 1

echo "📦 Copying PortTraffic to Glid_Surge_Optimization..."
echo ""

# Create branch
git checkout -b feature/port-traffic 2>/dev/null || git checkout feature/port-traffic

# Copy files
rsync -av --progress \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='.next' \
  --exclude='*.log' \
  --exclude='.DS_Store' \
  /Users/cristianhenao/development/PortTraffic/ \
  ./

echo ""
echo "✅ Copy complete!"
echo ""
echo "Next: npm install && npm run build"

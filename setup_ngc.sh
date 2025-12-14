#!/bin/bash
# =============================================================================
# NGC Setup Script for NVIDIA NIM
# Run this once before using NIM containers
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          NVIDIA NGC Setup for NIM Containers                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if NGC_API_KEY is already set
if [ -n "$NGC_API_KEY" ]; then
    echo -e "${GREEN}✓${NC} NGC_API_KEY already set in environment"
    echo "  Key prefix: ${NGC_API_KEY:0:12}..."
else
    echo -e "${YELLOW}!${NC} NGC_API_KEY not found in environment"
    echo ""
    echo "To get your NGC API key:"
    echo "  1. Go to: https://org.ngc.nvidia.com/setup"
    echo "  2. Sign in or create an account"
    echo "  3. Click 'Generate API Key'"
    echo "  4. Copy the key (starts with 'nvapi-')"
    echo ""
    read -p "Enter your NGC API Key: " NGC_API_KEY
    
    if [ -z "$NGC_API_KEY" ]; then
        echo -e "${RED}✗${NC} No key provided. Exiting."
        exit 1
    fi
    
    # Validate key format
    if [[ ! "$NGC_API_KEY" =~ ^nvapi- ]]; then
        echo -e "${YELLOW}!${NC} Warning: Key doesn't start with 'nvapi-'. Continuing anyway..."
    fi
    
    export NGC_API_KEY
fi

# Login to NGC registry
echo ""
echo -e "${BLUE}[1/3]${NC} Logging into NGC container registry..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Successfully logged into nvcr.io"
else
    echo -e "${RED}✗${NC} Failed to login to NGC registry"
    exit 1
fi

# Create .env file for docker-compose
echo ""
echo -e "${BLUE}[2/3]${NC} Creating .env file..."
ENV_FILE="/home/asus/Desktop/Glid_Surge_Optimization/.env"

if [ -f "$ENV_FILE" ]; then
    # Update existing .env
    if grep -q "NGC_API_KEY" "$ENV_FILE"; then
        sed -i "s|NGC_API_KEY=.*|NGC_API_KEY=${NGC_API_KEY}|" "$ENV_FILE"
    else
        echo "NGC_API_KEY=${NGC_API_KEY}" >> "$ENV_FILE"
    fi
else
    echo "NGC_API_KEY=${NGC_API_KEY}" > "$ENV_FILE"
fi

echo -e "${GREEN}✓${NC} .env file created/updated"

# Add to bashrc for persistence
echo ""
echo -e "${BLUE}[3/3]${NC} Adding to shell profile..."
BASHRC="$HOME/.bashrc"

if grep -q "NGC_API_KEY" "$BASHRC"; then
    sed -i "s|export NGC_API_KEY=.*|export NGC_API_KEY=${NGC_API_KEY}|" "$BASHRC"
else
    echo "" >> "$BASHRC"
    echo "# NVIDIA NGC API Key for NIM containers" >> "$BASHRC"
    echo "export NGC_API_KEY=${NGC_API_KEY}" >> "$BASHRC"
fi

echo -e "${GREEN}✓${NC} Added to ~/.bashrc"

# Pull the NIM image (optional)
echo ""
read -p "Pull Nemotron 49B NIM image now? This may take a while (~30GB). [y/N]: " PULL_IMAGE
if [[ "$PULL_IMAGE" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Pulling NIM image...${NC}"
    docker pull nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest
    echo -e "${GREEN}✓${NC} Image pulled successfully"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  NGC Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "You can now start NIM with:"
echo "  cd /home/asus/Desktop/Glid_Surge_Optimization"
echo "  ./start_with_nemotron.sh nim"
echo ""
echo "Or run docker-compose directly:"
echo "  docker compose -f docker-compose.nim.yml up nemotron-nim"
echo ""

